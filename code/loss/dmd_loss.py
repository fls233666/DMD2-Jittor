"""Distribution matching distillation losses for Jittor DMD2."""

import jittor as jt
from jittor import nn

try:
    from .common import mean_abs, mse_loss, nan_to_num, stop_grad, tensor_norm
except ImportError:
    from common import mean_abs, mse_loss, nan_to_num, stop_grad, tensor_norm


def dmd_gradient(latents, pred_real_image, pred_fake_image):
    """Compute the DMD gradient used by the official DMD2 implementation.

    Official PyTorch formula:
        p_real = latents - pred_real_image
        p_fake = latents - pred_fake_image
        grad = (p_real - p_fake) / abs(p_real).mean([1, 2, 3], keepdim=True)
        grad = torch.nan_to_num(grad)
    """

    p_real = latents - pred_real_image
    p_fake = latents - pred_fake_image
    weight_factor = mean_abs(p_real, dims=(1, 2, 3), keepdims=True)
    grad = (p_real - p_fake) / weight_factor
    return nan_to_num(grad)


def distribution_matching_loss(
    latents,
    pred_real_image,
    pred_fake_image,
    noisy_latents=None,
    timesteps=None,
    prefix="dmtrain",
):
    """Return DMD loss and log tensors from real/fake denoised predictions."""

    grad = stop_grad(dmd_gradient(latents, pred_real_image, pred_fake_image))
    target = stop_grad(latents - grad)
    loss = 0.5 * mse_loss(latents, target)

    loss_dict = {
        "loss_dm": loss,
    }
    log_dict = {
        f"{prefix}_pred_real_image": stop_grad(pred_real_image),
        f"{prefix}_pred_fake_image": stop_grad(pred_fake_image),
        f"{prefix}_grad": stop_grad(grad),
        f"{prefix}_gradient_norm": tensor_norm(grad),
    }
    if noisy_latents is not None:
        log_dict[f"{prefix}_noisy_latents"] = stop_grad(noisy_latents)
    if timesteps is not None:
        log_dict[f"{prefix}_timesteps"] = stop_grad(timesteps)

    return loss_dict, log_dict


class DistributionMatchingLoss(nn.Module):
    """Module wrapper for the DMD loss formula."""

    def __init__(self, prefix="dmtrain"):
        super().__init__()
        self.prefix = prefix

    def execute(
        self,
        latents,
        pred_real_image,
        pred_fake_image,
        noisy_latents=None,
        timesteps=None,
    ):
        return distribution_matching_loss(
            latents=latents,
            pred_real_image=pred_real_image,
            pred_fake_image=pred_fake_image,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
            prefix=self.prefix,
        )


def compute_distribution_matching_loss(
    latents,
    labels,
    real_unet,
    fake_unet,
    timestep_sigma,
    noisy_latents=None,
    noise=None,
    timesteps=None,
    prefix="dmtrain",
):
    """Network-facing helper matching EDMGuidance.compute_distribution_matching_loss."""

    if noise is None:
        noise = jt.randn_like(latents)
    if noisy_latents is None:
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

    pred_real_image = real_unet(noisy_latents, timestep_sigma, labels)
    pred_fake_image = fake_unet(noisy_latents, timestep_sigma, labels)

    return distribution_matching_loss(
        latents=latents,
        pred_real_image=pred_real_image,
        pred_fake_image=pred_fake_image,
        noisy_latents=noisy_latents,
        timesteps=timesteps,
        prefix=prefix,
    )
