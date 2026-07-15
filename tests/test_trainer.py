"""Smoke tests for code/trainer image DMD2 helpers.

Run from the project root:

    python tests/test_trainer.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import nn


def setup_import_path():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    code_dir = project_root / "code"

    for path in (
        project_root,
        code_dir,
        code_dir / "datasets",
        code_dir / "samplers",
        code_dir / "trainer",
    ):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


setup_import_path()

from trainer.checkpoint import checkpoint_state, load_checkpoint, save_checkpoint
from trainer.engine import (
    ImageDMD2TrainEngine,
    make_generator_inputs,
    random_class_labels,
    sum_loss_dict,
)
from trainer.evaluator import ImageDMD2SamplerEvaluator, make_image_grid
from trainer.train_loop import train_image_dmd2


def configure_jittor():
    use_cuda = os.environ.get("JITTOR_USE_CUDA", "0") == "1"
    jt.flags.use_cuda = 1 if use_cuda else 0


def to_numpy(x):
    if isinstance(x, jt.Var):
        jt.sync_all()
        return np.asarray(x.numpy())
    return np.asarray(x)


def assert_shape(name, x, expected_shape):
    got = list(x.shape)
    expected = list(expected_shape)
    assert got == expected, f"{name}: got shape {got}, expected {expected}"


class DummyOptimizer:
    def __init__(self):
        self.steps = 0
        self.last_loss = None
        self.loaded = None

    def step(self, loss=None, retain_graph=False):
        _ = retain_graph
        self.steps += 1
        self.last_loss = loss

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.loaded = state
        self.steps = int(state["steps"])


class DummyEMA:
    def __init__(self):
        self.steps = 0

    def update(self, model):
        _ = model
        self.steps += 1

    def state_dict(self):
        return {"steps": self.steps}

    def load_state_dict(self, state):
        self.steps = int(state["steps"])


class DummyFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_channels = 3
        self.img_resolution = 32
        self.label_dim = 10
        self.weight = jt.array([1.0]).float32()

    def execute(self, x, sigma, labels=None):
        _ = sigma
        _ = labels
        return x * 0.0 + self.weight.reshape(1, 1, 1, 1)


class DummyUnifiedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feedforward_model = DummyFeedForward()
        self.last_generator_labels = None
        self.last_real_labels = None
        self.generator_grad_flags = []

    def execute(
        self,
        scaled_noisy_image,
        timestep_sigma,
        labels,
        real_train_dict=None,
        compute_generator_gradient=False,
        generator_turn=False,
        guidance_turn=False,
        guidance_data_dict=None,
    ):
        if generator_turn:
            self.last_generator_labels = labels.stop_grad()
            self.last_real_labels = real_train_dict["real_label"].stop_grad()
            self.generator_grad_flags.append(bool(compute_generator_gradient))
            image = self.feedforward_model(scaled_noisy_image, timestep_sigma, labels)
            loss_dict = {}
            if compute_generator_gradient:
                loss_dict = {
                    "loss_dm": (image ** 2).mean(),
                    "gen_cls_loss": jt.array(100.0).float32(),
                }
            return loss_dict, {
                "generated_image": image.stop_grad(),
                "guidance_data_dict": {
                    "image": image.stop_grad(),
                    "label": labels.stop_grad(),
                    "real_train_dict": real_train_dict,
                },
            }

        if guidance_turn:
            image = guidance_data_dict["image"]
            return {
                "loss_fake_mean": jt.abs(image).mean(),
                "guidance_cls_loss": jt.array(2.0).float32(),
            }, {"faketrain_x0_pred": image.stop_grad()}

        raise AssertionError("expected generator_turn or guidance_turn")


def make_batch(batch_size=2):
    images = jt.ones([batch_size, 3, 32, 32]).float32()
    labels = jt.array(np.eye(10, dtype=np.float32)[:batch_size])
    class_id = jt.arange(batch_size).int32()
    return {
        "image": images,
        "label": labels,
        "class_id": class_id,
    }


def test_sum_loss_dict_and_inputs():
    losses = {
        "a": jt.array(2.0).float32(),
        "b": jt.array(4.0).float32(),
    }
    total = sum_loss_dict(losses, weights={"b": 0.5})
    assert abs(float(total.numpy()[0]) - 4.0) < 1e-6

    batch = make_batch(batch_size=3)
    scaled_noise, sigma, noise = make_generator_inputs(batch, conditioning_sigma=2.0)
    assert_shape("scaled noise", scaled_noise, [3, 3, 32, 32])
    assert_shape("sigma", sigma, [3])
    assert_shape("noise", noise, [3, 3, 32, 32])

    labels, class_id = random_class_labels(batch_size=4, label_dim=10)
    assert_shape("random labels", labels, [4, 10])
    assert_shape("random class ids", class_id, [4])
    assert np.allclose(to_numpy(labels).sum(axis=1), 1.0)


def test_engine_train_step():
    model = DummyUnifiedModel()
    gen_opt = DummyOptimizer()
    guidance_opt = DummyOptimizer()
    ema = DummyEMA()
    engine = ImageDMD2TrainEngine(
        model=model,
        generator_optimizer=gen_opt,
        guidance_optimizer=guidance_opt,
        conditioning_sigma=1.0,
        ema=ema,
    )

    result = engine.train_step(make_batch())
    assert gen_opt.steps == 1
    assert guidance_opt.steps == 1
    assert ema.steps == 1
    assert result["logs"]["step"] == 1
    assert result["logs"]["compute_generator_gradient"] == 1
    assert "generator/loss_dm" in result["logs"]
    assert "guidance/loss_fake_mean" in result["logs"]
    assert abs(float(gen_opt.last_loss.numpy()[0]) - 1.0) < 1e-3
    assert abs(float(guidance_opt.last_loss.numpy()[0]) - 3.0) < 1e-3

    fake_labels = to_numpy(model.last_generator_labels)
    real_labels = to_numpy(model.last_real_labels)
    assert fake_labels.shape == real_labels.shape
    assert np.allclose(fake_labels.sum(axis=1), 1.0)


def test_engine_ratio_skips_generator_update():
    model = DummyUnifiedModel()
    gen_opt = DummyOptimizer()
    guidance_opt = DummyOptimizer()
    ema = DummyEMA()
    engine = ImageDMD2TrainEngine(
        model=model,
        generator_optimizer=gen_opt,
        guidance_optimizer=guidance_opt,
        conditioning_sigma=1.0,
        dfake_gen_update_ratio=2,
        ema=ema,
    )

    first = engine.train_step(make_batch())
    second = engine.train_step(make_batch())

    assert first["logs"]["compute_generator_gradient"] == 1
    assert second["logs"]["compute_generator_gradient"] == 0
    assert gen_opt.steps == 1
    assert guidance_opt.steps == 2
    assert ema.steps == 1
    assert model.generator_grad_flags == [True, False]


def test_train_loop_history():
    model = DummyUnifiedModel()
    engine = ImageDMD2TrainEngine(
        model=model,
        generator_optimizer=DummyOptimizer(),
        guidance_optimizer=DummyOptimizer(),
        conditioning_sigma=1.0,
    )
    loader = [make_batch(), make_batch()]
    history = train_image_dmd2(
        engine=engine,
        train_loader=loader,
        max_steps=3,
        log_interval=0,
        checkpoint_interval=0,
        print_fn=lambda msg: None,
    )
    assert len(history) == 3
    assert history[-1]["step"] == 3


def test_train_loop_writes_record_logs():
    model = DummyUnifiedModel()
    engine = ImageDMD2TrainEngine(
        model=model,
        generator_optimizer=DummyOptimizer(),
        guidance_optimizer=DummyOptimizer(),
        conditioning_sigma=1.0,
    )
    loader = [make_batch(), make_batch()]

    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = os.path.join(tmpdir, "train_metrics.jsonl")
        perf_path = os.path.join(tmpdir, "performance.jsonl")
        history = train_image_dmd2(
            engine=engine,
            train_loader=loader,
            max_steps=2,
            log_interval=0,
            checkpoint_interval=0,
            metrics_logger=metrics_path,
            performance_logger=perf_path,
            print_fn=lambda msg: None,
        )
        assert len(history) == 2
        assert os.path.exists(metrics_path)
        assert os.path.exists(perf_path)
        assert "samples_per_second" in open(perf_path, "r", encoding="utf-8").read()


def test_checkpoint_roundtrip():
    model = DummyUnifiedModel()
    gen_opt = DummyOptimizer()
    guidance_opt = DummyOptimizer()
    ema = DummyEMA()

    state = checkpoint_state(
        model=model,
        generator_optimizer=gen_opt,
        guidance_optimizer=guidance_opt,
        ema=ema,
        step=5,
        extra={"tag": "debug"},
    )
    assert state["step"] == 5
    assert state["extra"]["tag"] == "debug"

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "ckpt.pkl")
        save_checkpoint(path, model, gen_opt, guidance_opt, ema, step=7)
        loaded = load_checkpoint(path, model, gen_opt, guidance_opt, ema)
        assert loaded["step"] == 7
        assert gen_opt.loaded["steps"] == 0


def test_evaluator_grid_and_samples():
    images = np.zeros([5, 4, 4, 3], dtype=np.uint8)
    grid = make_image_grid(images, nrow=3)
    assert_shape("image grid", grid, [8, 12, 3])

    model = DummyUnifiedModel()
    with tempfile.TemporaryDirectory() as tmpdir:
        evaluator = ImageDMD2SamplerEvaluator(
            output_dir=tmpdir,
            batch_size=2,
            nrow=2,
            conditioning_sigma=1.0,
            img_channels=3,
            img_resolution=32,
        )
        path = evaluator.evaluate(model, step=1)
        assert os.path.exists(path)
        assert path.endswith(".svg")


def run_all_tests():
    configure_jittor()

    tests = [
        test_sum_loss_dict_and_inputs,
        test_engine_train_step,
        test_engine_ratio_skips_generator_update,
        test_train_loop_history,
        test_train_loop_writes_record_logs,
        test_checkpoint_roundtrip,
        test_evaluator_grid_and_samples,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        jt.sync_all()
        print(f"[OK]  {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
