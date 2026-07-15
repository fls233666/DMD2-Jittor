"""Smoke tests for code/models/diffusion.py.

Run from the project root:

    python tests/test_diffusion.py

Optional GPU run:

    JITTOR_USE_CUDA=1 python tests/test_diffusion.py

These tests intentionally use a tiny EDM network config instead of the full
ImageNet-64 config, so they can be used for quick migration debugging.
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
    from models.diffusion import (
        EDMPrecond,
        EDMTeacherStudent,
        get_edm_network,
        get_imagenet_edm_config,
        get_sigmas_karras,
    )
    import models.diffusion as diffusion_module
except ImportError:
    from diffusion import (
        EDMPrecond,
        EDMTeacherStudent,
        get_edm_network,
        get_imagenet_edm_config,
        get_sigmas_karras,
    )
    import diffusion as diffusion_module


def configure_jittor():
    # Default to CPU for portability. Set JITTOR_USE_CUDA=1 to run on GPU.
    use_cuda = os.environ.get("JITTOR_USE_CUDA", "0") == "1"
    jt.flags.use_cuda = 1 if use_cuda else 0


def to_numpy(x):
    # Convert a Jittor Var to numpy for assertions.
    if isinstance(x, jt.Var):
        jt.sync_all()
        return np.asarray(x.numpy())
    return np.asarray(x)


def assert_shape(name, x, expected_shape):
    got = list(x.shape)
    expected = list(expected_shape)
    assert got == expected, f"{name}: got shape {got}, expected {expected}"


def assert_all_finite(name, x):
    arr = to_numpy(x)
    assert np.isfinite(arr).all(), f"{name}: contains NaN or Inf"


def make_tiny_edm(label_dim=0):
    # Tiny config used by multiple forward tests.
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


class DummyEDM(nn.Module):
    # Minimal network used to test EDMTeacherStudent without building full UNets.
    def execute(self, x, sigma, class_labels=None, **kwargs):
        return x


def test_karras_sigmas_shape_and_order():
    sigmas = get_sigmas_karras(
        n=8,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    )

    assert_shape("karras sigmas", sigmas, [8])
    arr = to_numpy(sigmas)

    assert abs(arr[0] - 80.0) < 1e-3, f"first sigma should be sigma_max, got {arr[0]}"
    assert abs(arr[-1] - 0.002) < 1e-5, f"last sigma should be sigma_min, got {arr[-1]}"
    assert np.all(np.diff(arr) <= 1e-6), "sigmas should be monotonically non-increasing"


def test_imagenet_config_keys():
    config = get_imagenet_edm_config()

    expected = {
        "augment_dim",
        "model_channels",
        "channel_mult",
        "channel_mult_emb",
        "num_blocks",
        "attn_resolutions",
        "dropout",
        "label_dropout",
    }
    assert expected.issubset(config.keys()), f"missing config keys: {expected - set(config.keys())}"

    assert config["model_channels"] == 192
    assert tuple(config["channel_mult"]) == (1, 2, 3, 4)
    assert tuple(config["attn_resolutions"]) == (32, 16, 8)


def test_prepare_class_labels():
    prepare = diffusion_module._prepare_class_labels

    labels = prepare(jt.array([0, 2]), batch_size=2, label_dim=4)
    assert_shape("one-hot labels", labels, [2, 4])

    labels_np = to_numpy(labels)
    expected = np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(labels_np, expected), f"wrong one-hot labels:\n{labels_np}"

    zeros = prepare(None, batch_size=3, label_dim=5)
    assert_shape("default labels", zeros, [3, 5])
    assert np.allclose(to_numpy(zeros), 0), "default class labels should be all zeros"

    unconditional = prepare(None, batch_size=3, label_dim=0)
    assert unconditional is None, "label_dim=0 should return None"


def test_edm_precond_unconditional_forward():
    model = make_tiny_edm(label_dim=0)
    model.eval()

    x = jt.randn([1, 3, 16, 16])
    sigma = jt.array([0.5]).float32()

    y = model(x, sigma)

    assert_shape("EDMPrecond unconditional output", y, [1, 3, 16, 16])
    assert_all_finite("EDMPrecond unconditional output", y)


def test_edm_precond_conditional_forward_with_integer_labels():
    model = make_tiny_edm(label_dim=10)
    model.eval()

    x = jt.randn([2, 3, 16, 16])
    sigma = jt.array([0.5, 1.0]).float32()
    labels = jt.array([3, 7]).int32()

    y = model(x, sigma, class_labels=labels)

    assert_shape("EDMPrecond conditional output", y, [2, 3, 16, 16])
    assert_all_finite("EDMPrecond conditional output", y)


def test_edm_precond_return_bottleneck():
    model = make_tiny_edm(label_dim=0)
    model.eval()

    x = jt.randn([1, 3, 16, 16])
    sigma = jt.array([0.5]).float32()

    bottleneck = model(x, sigma, return_bottleneck=True)

    assert len(bottleneck.shape) == 4, f"bottleneck should be NCHW, got {bottleneck.shape}"
    assert bottleneck.shape[0] == 1, f"wrong bottleneck batch: {bottleneck.shape}"
    assert bottleneck.shape[2] == 8 and bottleneck.shape[3] == 8, (
        f"tiny config bottleneck should be 8x8, got {bottleneck.shape}"
    )
    assert_all_finite("EDMPrecond bottleneck", bottleneck)


def test_get_edm_network_tiny_factory_forward():
    model = get_edm_network(
        dataset_name="debug",
        config_name="tiny",
        resolution=16,
        label_dim=0,
        use_fp16=False,
        sigma_data=0.5,
        model_channels=16,
        channel_mult=(1, 2),
        channel_mult_emb=2,
        num_blocks=1,
        attn_resolutions=(),
    )
    model.eval()

    x = jt.randn([1, 3, 16, 16])
    sigma = jt.array([1.0]).float32()

    y = model(x, sigma)

    assert_shape("get_edm_network tiny output", y, [1, 3, 16, 16])
    assert_all_finite("get_edm_network tiny output", y)


def test_teacher_student_noise_and_forward():
    wrapper = EDMTeacherStudent(
        teacher=DummyEDM(),
        student=DummyEDM(),
        copy_teacher_to_student=False,
        num_train_timesteps=8,
        sigma_min=0.002,
        sigma_max=80.0,
        min_step_percent=0.0,
        max_step_percent=0.75,
    )

    assert_shape("teacher-student karras sigmas", wrapper.karras_sigmas, [8])
    assert "karras_sigmas_buffer" in wrapper.state_dict(), (
        "Karras schedule should be checkpointed as a Jittor state entry."
    )

    x = jt.randn([2, 3, 16, 16])
    timesteps = wrapper.sample_timesteps(batch_size=2)
    assert_shape("sampled timesteps", timesteps, [2, 1, 1, 1])

    sigma = wrapper.timestep_to_sigma(timesteps)
    assert_shape("sigma from timesteps", sigma, [2, 1, 1, 1])

    noisy_x, sigma, noise = wrapper.add_noise(x, timesteps=timesteps)
    assert_shape("noisy x", noisy_x, [2, 3, 16, 16])
    assert_shape("noise", noise, [2, 3, 16, 16])
    assert_shape("sigma from add_noise", sigma, [2, 1, 1, 1])
    assert_all_finite("noisy x", noisy_x)

    student_y = wrapper(x, sigma=sigma, use_teacher=False)
    teacher_y = wrapper(x, sigma=sigma, use_teacher=True)

    assert_shape("student forward", student_y, [2, 3, 16, 16])
    assert_shape("teacher forward", teacher_y, [2, 3, 16, 16])


def test_teacher_student_sampling_upper_bound():
    wrapper = EDMTeacherStudent(
        teacher=DummyEDM(),
        student=DummyEDM(),
        copy_teacher_to_student=False,
        num_train_timesteps=8,
        min_step_percent=0.0,
        max_step_percent=1.0,
    )

    timesteps = wrapper.sample_timesteps(batch_size=256)
    arr = to_numpy(timesteps)
    assert arr.min() >= 0
    assert arr.max() < 8, f"sampled timestep out of range: {arr.max()}"


def run_all_tests():
    configure_jittor()

    tests = [
        test_karras_sigmas_shape_and_order,
        test_imagenet_config_keys,
        test_prepare_class_labels,
        test_edm_precond_unconditional_forward,
        test_edm_precond_conditional_forward_with_integer_labels,
        test_edm_precond_return_bottleneck,
        test_get_edm_network_tiny_factory_forward,
        test_teacher_student_noise_and_forward,
        test_teacher_student_sampling_upper_bound,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        jt.sync_all()
        print(f"[OK]  {test.__name__}")

    print("All diffusion tests passed.")


if __name__ == "__main__":
    run_all_tests()
