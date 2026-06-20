"""Regression and fake-score denoising losses for Jittor DMD2."""

from jittor import nn

try:
    from .common import get_x0_from_noise, mse_loss, stop_grad
except ImportError:
    from common import get_x0_from_noise, mse_loss, stop_grad


def edm_fake_score_denoising_loss(fake_x0_pred, target_latents, timestep_sigma, sigma_data=0.5):
    # EDM fake score loss from main/edm/edm_guidance.py.
    # weights = sigma ** -2 + 1 / sigma_data ** 2
    snrs = timestep_sigma ** -2
    weights = snrs + 1.0 / (sigma_data ** 2)
    return (weights * (fake_x0_pred - target_latents) ** 2).mean()


def noise_prediction_loss(pred_noise, target_noise):
    # Epsilon-prediction MSE used by the official SD guidance branch.
    return mse_loss(pred_noise.float32(), target_noise.float32())


def x0_regression_loss(pred_x0, target_x0):
    # Generic x0/image regression MSE for alignment and ODE-pair tests.
    return mse_loss(pred_x0.float32(), stop_grad(target_x0).float32())


def x0_loss_from_noise_prediction(sample, pred_noise, target_x0, alphas_cumprod, timesteps):
    # Convert epsilon prediction to x0, then compute x0 regression loss.
    pred_x0 = get_x0_from_noise(
        sample=sample,
        model_output=pred_noise,
        alphas_cumprod=alphas_cumprod,
        timestep=timesteps,
    )
    return x0_regression_loss(pred_x0, target_x0), pred_x0


class EDMFakeScoreLoss(nn.Module):
    def __init__(self, sigma_data=0.5):
        super().__init__()
        self.sigma_data = sigma_data

    def execute(self, fake_x0_pred, target_latents, timestep_sigma):
        return edm_fake_score_denoising_loss(
            fake_x0_pred=fake_x0_pred,
            target_latents=target_latents,
            timestep_sigma=timestep_sigma,
            sigma_data=self.sigma_data,
        )


class NoisePredictionLoss(nn.Module):
    def execute(self, pred_noise, target_noise):
        return noise_prediction_loss(pred_noise, target_noise)


class X0RegressionLoss(nn.Module):
    def execute(self, pred_x0, target_x0):
        return x0_regression_loss(pred_x0, target_x0)
