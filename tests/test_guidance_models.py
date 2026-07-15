"""Smoke tests for the remaining code/models migration files.

Run from the project root:

    python tests/test_guidance_models.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import nn


def setup_import_path():
    # Make the test runnable from either the project root or tests/.
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    code_dir = project_root / "code"
    models_dir = code_dir / "models"

    for path in (project_root, code_dir, models_dir):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


setup_import_path()

try:
    from models.diffusion import EDMPrecond
    from models.discriminator import BottleneckDiscriminator
    from models.ema import EMAModel, ExponentialMovingAverage
    from models.guidance import EDMGuidance
    from models.guidance import _nan_to_num
    from models.unified_model import EDMUniModel
    from loss.common import stop_grad
except ImportError:
    from diffusion import EDMPrecond
    from discriminator import BottleneckDiscriminator
    from ema import EMAModel, ExponentialMovingAverage
    from guidance import EDMGuidance
    from guidance import _nan_to_num
    from unified_model import EDMUniModel
    from common import stop_grad


def configure_jittor():
    # Default to CPU for portability. Set JITTOR_USE_CUDA=1 to run on GPU.
    use_cuda = os.environ.get("JITTOR_USE_CUDA", "0") == "1"
    jt.flags.use_cuda = 1 if use_cuda else 0


def assert_shape(name, x, expected_shape):
    got = list(x.shape)
    expected = list(expected_shape)
    assert got == expected, f"{name}: got shape {got}, expected {expected}"


def assert_all_finite(name, x):
    jt.sync_all()
    arr = np.asarray(x.numpy())
    assert np.isfinite(arr).all(), f"{name}: contains NaN or Inf"


def make_tiny_edm(label_dim=0):
    # Tiny config used by guidance and unified model smoke tests.
    return EDMPrecond(
        img_resolution=16,
        img_channels=3,
        label_dim=label_dim,
        use_fp16=False,
        sigma_data=0.5,
        model_type="DhariwalUNet",
        augment_dim=0,
        model_channels=16,
        channel_mult=(1, 2),
        channel_mult_emb=2,
        num_blocks=1,
        attn_resolutions=(),
        dropout=0.0,
        label_dropout=0,
    )


class ScaleModel(nn.Module):
    # Small stateful module used for EMA tests.
    def __init__(self, value):
        super().__init__()
        self.weight = jt.array([value]).float32()

    def execute(self, x):
        return x * self.weight


def test_bottleneck_discriminator_forward():
    head = BottleneckDiscriminator(
        in_channels=32,
        hidden_channels=32,
        bottleneck_resolution=8,
    )
    x = jt.randn([2, 32, 8, 8])
    logits = head(x)

    assert_shape("bottleneck discriminator logits", logits, [2, 1])
    assert_all_finite("bottleneck discriminator logits", logits)


def test_nan_to_num_matches_torch_defaults():
    x = jt.array([float("nan"), float("inf"), -float("inf"), 1.0]).float32()
    y = _nan_to_num(x)
    arr = np.asarray(y.numpy())

    assert arr[0] == 0.0
    assert arr[1] > 1e30
    assert arr[2] < -1e30
    assert arr[3] == 1.0


def test_stop_grad_does_not_mutate_live_graph():
    weight = jt.ones([1]).float32()
    x = weight * 2.0
    loss = x.mean()
    _ = stop_grad(x)

    grad = jt.grad(loss, [weight])[0]
    value = float(grad.numpy()[0])
    assert abs(value - 2.0) < 1e-6, (
        f"stop_grad should not detach an already-built loss graph, got grad {value}."
    )


def test_ema_update():
    model = ScaleModel(3.0)
    ema = ExponentialMovingAverage(ScaleModel(1.0), decay=0.5, copy_model=False)
    ema.update(model)

    value = ema.model.weight.numpy()[0]
    assert abs(value - 2.0) < 1e-5, f"EMA update produced {value}, expected 2.0"

    ema_module = EMAModel(ScaleModel(1.0), decay=0.5, copy_model=False)
    ema_module.update(model)
    value = ema_module.model.weight.numpy()[0]
    assert abs(value - 2.0) < 1e-5, f"EMAModel update produced {value}, expected 2.0"


def test_guidance_losses_without_classifier():
    real = make_tiny_edm(label_dim=0)
    fake = make_tiny_edm(label_dim=0)
    guidance = EDMGuidance(
        real_unet=real,
        fake_unet=fake,
        copy_real_to_fake=False,
        gan_classifier=False,
        num_train_timesteps=8,
        min_step_percent=0.0,
        max_step_percent=0.75,
    )
    assert "karras_sigmas_buffer" in guidance.state_dict(), (
        "Karras schedule should be checkpointed as a Jittor state entry."
    )

    x = jt.randn([1, 3, 16, 16])
    labels = None

    gen_loss, gen_log = guidance(
        generator_turn=True,
        generator_data_dict={
            "image": x,
            "label": labels,
        },
    )
    assert "loss_dm" in gen_loss
    assert "dmtrain_grad" in gen_log
    assert_all_finite("guidance loss_dm", gen_loss["loss_dm"])

    fake_loss, fake_log = guidance(
        guidance_turn=True,
        guidance_data_dict={
            "image": x,
            "label": labels,
            "real_train_dict": None,
        },
    )
    assert "loss_fake_mean" in fake_loss
    assert "faketrain_x0_pred" in fake_log
    assert_all_finite("guidance loss_fake_mean", fake_loss["loss_fake_mean"])


def test_guidance_fake_loss_has_fake_unet_gradients():
    guidance = EDMGuidance(
        real_unet=make_tiny_edm(label_dim=0),
        fake_unet=make_tiny_edm(label_dim=0),
        copy_real_to_fake=False,
        gan_classifier=False,
        num_train_timesteps=8,
        min_step_percent=0.0,
        max_step_percent=0.75,
    )

    x = jt.randn([1, 3, 16, 16])
    fake_loss, _ = guidance(
        guidance_turn=True,
        guidance_data_dict={
            "image": x,
            "label": None,
            "real_train_dict": None,
        },
    )

    params = list(guidance.fake_unet.parameters())
    grads = jt.grad(fake_loss["loss_fake_mean"], params)
    total_grad_sq = None
    for grad in grads:
        item = (grad * grad).sum()
        total_grad_sq = item if total_grad_sq is None else total_grad_sq + item

    assert total_grad_sq is not None
    value = float(total_grad_sq.numpy()[0])
    assert value > 0.0, "fake_unet should receive gradients from loss_fake_mean."


def test_guidance_classifier_logits():
    real = make_tiny_edm(label_dim=0)
    fake = make_tiny_edm(label_dim=0)
    guidance = EDMGuidance(
        real_unet=real,
        fake_unet=fake,
        copy_real_to_fake=False,
        gan_classifier=True,
        diffusion_gan=False,
        num_train_timesteps=8,
        min_step_percent=0.0,
        max_step_percent=0.75,
        bottleneck_channels=32,
        bottleneck_resolution=8,
    )

    x = jt.randn([1, 3, 16, 16])
    logits = guidance.compute_cls_logits(x, None)

    assert_shape("guidance classifier logits", logits, [1, 1])
    assert_all_finite("guidance classifier logits", logits)


def test_unified_model_branches():
    guidance = EDMGuidance(
        real_unet=make_tiny_edm(label_dim=0),
        fake_unet=make_tiny_edm(label_dim=0),
        copy_real_to_fake=False,
        gan_classifier=False,
        num_train_timesteps=8,
        min_step_percent=0.0,
        max_step_percent=0.75,
    )
    model = EDMUniModel(
        guidance_model=guidance,
        initialize_generator=True,
    )

    x = jt.randn([1, 3, 16, 16])
    sigma = jt.array([1.0]).float32()

    loss_dict, log_dict = model(
        x,
        sigma,
        labels=None,
        generator_turn=True,
        guidance_turn=False,
        compute_generator_gradient=False,
    )
    assert loss_dict == {}
    assert "generated_image" in log_dict
    assert "guidance_data_dict" in log_dict
    assert_shape("unified generated image", log_dict["generated_image"], [1, 3, 16, 16])

    loss_dict, log_dict = model(
        x,
        sigma,
        labels=None,
        generator_turn=False,
        guidance_turn=True,
        guidance_data_dict=log_dict["guidance_data_dict"],
    )
    assert "loss_fake_mean" in loss_dict
    assert "faketrain_x0_pred" in log_dict
    assert_all_finite("unified loss_fake_mean", loss_dict["loss_fake_mean"])


def run_all_tests():
    configure_jittor()

    tests = [
        test_bottleneck_discriminator_forward,
        test_nan_to_num_matches_torch_defaults,
        test_stop_grad_does_not_mutate_live_graph,
        test_ema_update,
        test_guidance_losses_without_classifier,
        test_guidance_fake_loss_has_fake_unet_gradients,
        test_guidance_classifier_logits,
        test_unified_model_branches,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        jt.sync_all()
        print(f"[OK]  {test.__name__}")

    print("All guidance/unified model tests passed.")


if __name__ == "__main__":
    run_all_tests()
