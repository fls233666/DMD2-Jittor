"""Smoke tests for code/samplers migration files.

Run from the project root:

    python tests/test_samplers.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import jittor as jt
from jittor import nn


def setup_import_path():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    code_dir = project_root / "code"

    for path in (project_root, code_dir, code_dir / "models", code_dir / "samplers"):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


setup_import_path()

from models.diffusion import EDMPrecond
from models.unified_model import EDMUniModel
from samplers.multistep import EDMMultiStepSampler, edm_sampler, sample_multistep
from samplers.one_step import OneStepSampler, sample_one_step
from samplers.scheduler import (
    EDMScheduler,
    get_edm_timesteps,
    get_sigmas_karras,
    images_to_uint8,
    make_class_labels,
)


def configure_jittor():
    use_cuda = os.environ.get("JITTOR_USE_CUDA", "0") == "1"
    jt.flags.use_cuda = 1 if use_cuda else 0


def to_numpy(x):
    jt.sync_all()
    return np.asarray(x.numpy())


def assert_shape(name, x, expected_shape):
    got = list(x.shape)
    expected = list(expected_shape)
    assert got == expected, f"{name}: got shape {got}, expected {expected}"


def assert_all_finite(name, x):
    arr = to_numpy(x)
    assert np.isfinite(arr).all(), f"{name}: contains NaN or Inf"


def make_tiny_edm(label_dim=0):
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


class IdentityDenoiser(nn.Module):
    def __init__(self, img_channels=3, img_resolution=8, label_dim=0):
        super().__init__()
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.label_dim = label_dim
        self.sigma_min = 0.0
        self.sigma_max = float("inf")

    def round_sigma(self, sigma):
        if not isinstance(sigma, jt.Var):
            sigma = jt.array(sigma)
        return sigma.float32()

    def execute(self, x, sigma, class_labels=None):
        _ = sigma
        _ = class_labels
        return x


class ZeroDenoiser(IdentityDenoiser):
    def execute(self, x, sigma, class_labels=None):
        _ = sigma
        _ = class_labels
        return jt.zeros_like(x)


class GuidanceStub(nn.Module):
    def __init__(self, fake_unet):
        super().__init__()
        self.fake_unet = fake_unet
        self.min_step = 0
        self.max_step = 1
        self.num_train_timesteps = 2

    def execute(self, *args, **kwargs):
        raise RuntimeError("GuidanceStub should not be called in sampler tests.")


def test_scheduler_sigmas_and_labels():
    sigmas = get_sigmas_karras(num_steps=4, sigma_min=0.002, sigma_max=80.0, rho=7.0)
    assert_shape("karras sigmas", sigmas, [4])
    arr = to_numpy(sigmas)
    assert abs(arr[0] - 80.0) < 1e-3
    assert abs(arr[-1] - 0.002) < 1e-5
    assert np.all(np.diff(arr) <= 1e-6)

    steps = get_edm_timesteps(num_steps=4, append_zero=True)
    assert_shape("edm timesteps", steps, [5])
    assert abs(to_numpy(steps)[-1]) < 1e-8

    labels = make_class_labels(batch_size=3, label_dim=5, class_idx=2)
    assert_shape("one-hot labels", labels, [3, 5])
    expected = np.zeros([3, 5], dtype=np.float32)
    expected[:, 2] = 1.0
    assert np.allclose(to_numpy(labels), expected)

    scheduler = EDMScheduler(conditioning_sigma=10.0)
    conditioning = scheduler.conditioning_sigmas(batch_size=2)
    assert np.allclose(to_numpy(conditioning), [10.0, 10.0])


def test_one_step_sampler_direct_model():
    model = make_tiny_edm(label_dim=4)
    model.eval()

    noise = jt.randn([2, 3, 16, 16])
    images, state = sample_one_step(
        generator=model,
        batch_size=2,
        labels=jt.array([1, 3]).int32(),
        conditioning_sigma=1.0,
        noise=noise,
        return_latents=True,
    )

    assert_shape("one-step images", images, [2, 3, 16, 16])
    assert_shape("one-step sigma", state["sigma"], [2])
    assert_shape("one-step labels", state["labels"], [2, 4])
    assert_all_finite("one-step images", images)

    sampler = OneStepSampler(model, conditioning_sigma=1.0)
    wrapped_images = sampler(batch_size=1, class_idx=0)
    assert_shape("one-step sampler images", wrapped_images, [1, 3, 16, 16])


def test_one_step_sampler_unified_model():
    model = make_tiny_edm(label_dim=0)
    unified = EDMUniModel(
        guidance_model=GuidanceStub(model),
        feedforward_model=model,
        initialize_generator=True,
    )

    noise = jt.randn([1, 3, 16, 16])
    images = sample_one_step(
        generator=unified,
        batch_size=1,
        conditioning_sigma=1.0,
        noise=noise,
    )
    assert_shape("unified one-step images", images, [1, 3, 16, 16])
    assert_all_finite("unified one-step images", images)


def test_edm_sampler_identity_is_stable():
    net = IdentityDenoiser(img_channels=3, img_resolution=8, label_dim=0)
    latents = jt.randn([2, 3, 8, 8])
    images = edm_sampler(
        net=net,
        latents=latents,
        num_steps=4,
        sigma_min=0.002,
        sigma_max=1.0,
        solver="heun",
        randn_like=lambda x: jt.zeros_like(x),
    )

    assert_shape("identity edm samples", images, [2, 3, 8, 8])
    assert_all_finite("identity edm samples", images)


def test_multistep_sampler_zero_denoiser():
    net = ZeroDenoiser(img_channels=3, img_resolution=8, label_dim=3)
    latents = jt.randn([2, 3, 8, 8])
    labels = jt.array([0, 2]).int32()

    images = sample_multistep(
        net=net,
        batch_size=2,
        labels=labels,
        latents=latents,
        num_steps=3,
        sigma_min=0.01,
        sigma_max=1.0,
        solver="euler",
        randn_like=lambda x: jt.zeros_like(x),
    )
    assert_shape("zero denoiser samples", images, [2, 3, 8, 8])
    assert_all_finite("zero denoiser samples", images)

    sampler = EDMMultiStepSampler(net, num_steps=3, sigma_min=0.01, sigma_max=1.0, solver="euler")
    wrapped = sampler(batch_size=2, labels=labels, latents=latents, randn_like=lambda x: jt.zeros_like(x))
    assert_shape("wrapped zero denoiser samples", wrapped, [2, 3, 8, 8])


def test_images_to_uint8_shape():
    images = jt.zeros([2, 3, 4, 4])
    converted = images_to_uint8(images)
    assert_shape("uint8-style images", converted, [2, 4, 4, 3])
    assert np.allclose(to_numpy(converted), 128.0)


def run_all_tests():
    configure_jittor()

    tests = [
        test_scheduler_sigmas_and_labels,
        test_one_step_sampler_direct_model,
        test_one_step_sampler_unified_model,
        test_edm_sampler_identity_is_stable,
        test_multistep_sampler_zero_denoiser,
        test_images_to_uint8_shape,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        jt.sync_all()
        print(f"[OK]  {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
