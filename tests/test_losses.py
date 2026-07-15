"""Smoke tests for code/loss migration files.

Run from the project root:

    python tests/test_losses.py
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

    for path in (project_root, code_dir, code_dir / "loss"):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


setup_import_path()

from loss.backward_simulation import make_denoising_step_list, sample_backward_edm
from loss.common import get_x0_from_noise, nan_to_num
from loss.dmd_loss import distribution_matching_loss, dmd_gradient
from loss.gan_loss import (
    clean_classifier_losses,
    generator_realism_loss,
    guidance_realism_loss,
)
from loss.regression_loss import (
    edm_fake_score_denoising_loss,
    noise_prediction_loss,
    x0_loss_from_noise_prediction,
    x0_regression_loss,
)


def configure_jittor():
    use_cuda = os.environ.get("JITTOR_USE_CUDA", "0") == "1"
    jt.flags.use_cuda = 1 if use_cuda else 0


def assert_all_finite(name, x):
    jt.sync_all()
    arr = np.asarray(x.numpy())
    assert np.isfinite(arr).all(), f"{name}: contains NaN or Inf"


def assert_close(name, got, expected, atol=1e-6):
    jt.sync_all()
    got_arr = np.asarray(got.numpy())
    exp_arr = np.asarray(expected.numpy())
    assert np.allclose(got_arr, exp_arr, atol=atol), (
        f"{name}: got {got_arr}, expected {exp_arr}"
    )


def test_nan_to_num_and_dmd_gradient():
    x = jt.array([float("nan"), float("inf"), -float("inf"), 2.0]).float32()
    y = nan_to_num(x)
    arr = np.asarray(y.numpy())
    assert arr[0] == 0.0
    assert arr[1] > 1e30
    assert arr[2] < -1e30
    assert arr[3] == 2.0

    latents = jt.ones([2, 3, 4, 4])
    pred_real = latents * 0.5
    pred_fake = latents * 0.25
    grad = dmd_gradient(latents, pred_real, pred_fake)
    expected = jt.ones_like(latents) * -0.5
    assert_close("dmd gradient", grad, expected)


def test_distribution_matching_loss_dicts():
    latents = jt.ones([1, 3, 4, 4])
    pred_real = latents * 0.5
    pred_fake = latents * 0.25
    noisy = latents + 0.1
    timesteps = jt.zeros([1, 1, 1, 1]).int32()

    loss_dict, log_dict = distribution_matching_loss(
        latents=latents,
        pred_real_image=pred_real,
        pred_fake_image=pred_fake,
        noisy_latents=noisy,
        timesteps=timesteps,
    )

    assert "loss_dm" in loss_dict
    assert "dmtrain_noisy_latents" in log_dict
    assert "dmtrain_timesteps" in log_dict
    assert_all_finite("loss_dm", loss_dict["loss_dm"])


def test_regression_losses():
    pred = jt.array([1.0, 3.0]).float32()
    target = jt.array([0.0, 1.0]).float32()
    assert_close("noise mse", noise_prediction_loss(pred, target), jt.array(2.5).float32())
    assert_close("x0 mse", x0_regression_loss(pred, target), jt.array(2.5).float32())

    fake_x0 = jt.ones([2, 3, 2, 2])
    target_x0 = jt.zeros_like(fake_x0)
    sigma = jt.ones([2, 1, 1, 1])
    loss = edm_fake_score_denoising_loss(fake_x0, target_x0, sigma, sigma_data=0.5)
    assert_close("edm fake score", loss, jt.array(5.0).float32())

    sample = jt.ones([1, 1, 1, 1])
    pred_noise = jt.zeros_like(sample)
    alphas = jt.array([0.25]).float32()
    timesteps = jt.zeros([1]).int32()
    loss, pred_x0 = x0_loss_from_noise_prediction(
        sample=sample,
        pred_noise=pred_noise,
        target_x0=jt.ones_like(sample) * 2,
        alphas_cumprod=alphas,
        timesteps=timesteps,
    )
    assert_close("x0 from noise", pred_x0, get_x0_from_noise(sample, pred_noise, alphas, timesteps))
    assert_close("x0 from noise loss", loss, jt.array(0.0).float32())


def test_gan_losses():
    real_logits = jt.array([[1.0], [2.0]]).float32()
    fake_logits = jt.array([[-1.0], [-2.0]]).float32()

    gen_loss = generator_realism_loss(fake_logits)
    guidance_loss = guidance_realism_loss(real_logits, fake_logits)
    loss_dict, log_dict = clean_classifier_losses(real_logits, fake_logits)

    assert "guidance_cls_loss" in loss_dict
    assert "pred_realism_on_real" in log_dict
    assert "pred_realism_on_fake" in log_dict
    assert_all_finite("generator realism loss", gen_loss)
    assert_all_finite("guidance realism loss", guidance_loss)


class HalfDenoiser(nn.Module):
    def execute(self, x, sigma, labels=None):
        _ = sigma
        _ = labels
        return x * 0.5


def test_backward_simulation_helpers():
    steps = make_denoising_step_list(denoising_timestep=8, num_denoising_step=4)
    assert np.asarray(steps.numpy()).tolist() == [7, 5, 3, 1]

    sigmas = jt.linspace(0.1, 1.0, 8)
    image = jt.ones([2, 3, 4, 4])
    generated, timesteps = sample_backward_edm(
        noisy_image=image,
        model=HalfDenoiser(),
        sigmas=sigmas,
        step_indices=steps,
        selected_step=1,
        noise_fn=lambda ref: jt.zeros_like(ref),
    )

    assert np.asarray(timesteps.numpy()).tolist() == [5, 5]
    assert_close("backward generated image", generated, image * 0.5)


def run_all_tests():
    configure_jittor()

    tests = [
        test_nan_to_num_and_dmd_gradient,
        test_distribution_matching_loss_dicts,
        test_regression_losses,
        test_gan_losses,
        test_backward_simulation_helpers,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        jt.sync_all()
        print(f"[OK]  {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
