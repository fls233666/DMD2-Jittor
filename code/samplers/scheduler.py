"""Noise schedules and label helpers for Jittor DMD2 samplers."""

import numpy as np
import jittor as jt


def _as_float32(x):
    if isinstance(x, jt.Var):
        return x.float32()
    return jt.array(x).float32()


def get_sigmas_karras(num_steps, sigma_min=0.002, sigma_max=80.0, rho=7.0):
    # Return EDM/Karras sigmas from sigma_max to sigma_min.
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if num_steps == 1:
        return jt.array([sigma_max]).float32()

    ramp = jt.linspace(0, 1, num_steps).float32()
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    return (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho


def append_zero_sigma(sigmas):
    # Append terminal sigma 0, matching EDM Algorithm 2.
    sigmas = _as_float32(sigmas)
    return jt.concat([sigmas, jt.zeros([1]).float32()], dim=0)


def get_edm_timesteps(
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    net=None,
    append_zero=True,
):
    # Build EDM sampler timesteps, optionally clamped by net sigma limits.
    if net is not None:
        sigma_min = max(float(sigma_min), float(getattr(net, "sigma_min", sigma_min)))
        sigma_max = min(float(sigma_max), float(getattr(net, "sigma_max", sigma_max)))

    sigmas = get_sigmas_karras(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
    )
    if net is not None and hasattr(net, "round_sigma"):
        sigmas = net.round_sigma(sigmas)
    if append_zero:
        sigmas = append_zero_sigma(sigmas)
    return sigmas


def constant_sigma(batch_size, sigma=80.0):
    # Return a [B] sigma tensor for one-step DMD2 sampling.
    return jt.ones([batch_size]).float32() * float(sigma)


def randn_image(batch_size, channels, resolution, sigma=1.0):
    # Sample NCHW Gaussian latents, optionally scaled by sigma.
    return jt.randn([batch_size, channels, resolution, resolution]).float32() * float(sigma)


def labels_to_one_hot(labels, label_dim):
    # Convert integer labels to one-hot labels accepted by EDMPrecond.
    if label_dim == 0:
        return None
    if labels is None:
        return jt.zeros([1, label_dim]).float32()
    if not isinstance(labels, jt.Var):
        labels = jt.array(labels)
    labels = labels.reshape(-1).int32()
    eye = jt.array(np.eye(label_dim, dtype=np.float32))
    return eye[labels]


def make_class_labels(batch_size, label_dim=0, class_idx=None, labels=None):
    # Build class labels for sampling.
    if label_dim == 0:
        return None

    if labels is None:
        if class_idx is None:
            labels = jt.arange(batch_size).int32() % int(label_dim)
        else:
            labels = jt.ones([batch_size]).int32() * int(class_idx)

    return labels_to_one_hot(labels, label_dim)


def images_to_uint8(images, nchw=True):
    # Convert [-1, 1] image tensors to uint8-style NHWC float32 values.
    images = (images * 127.5 + 128.0).clamp(0, 255)
    if nchw:
        images = images.permute(0, 2, 3, 1)
    return images


class EDMScheduler:
    # Small schedule object shared by one-step and multi-step samplers.

    def __init__(
        self,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        conditioning_sigma=80.0,
    ):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.conditioning_sigma = conditioning_sigma

    def timesteps(self, num_steps=18, net=None, append_zero=True):
        return get_edm_timesteps(
            num_steps=num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            net=net,
            append_zero=append_zero,
        )

    def conditioning_sigmas(self, batch_size):
        return constant_sigma(batch_size=batch_size, sigma=self.conditioning_sigma)

    def sample_latents(self, batch_size, channels, resolution):
        return randn_image(
            batch_size=batch_size,
            channels=channels,
            resolution=resolution,
            sigma=1.0,
        )
