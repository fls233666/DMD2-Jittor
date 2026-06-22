"""Shared Jittor helpers for DMD2 loss migration."""

import numpy as np
import jittor as jt
from jittor import nn


def stop_grad(x):
    # Detach a Jittor Var from the gradient graph.
    if x is None:
        return None
    # Var.stop_grad() is in-place in Jittor and can invalidate a loss graph
    # when the same Var is also used for optimization. jt.detach() is not.
    if hasattr(jt, "detach"):
        return jt.detach(x)
    if hasattr(x, "detach"):
        return x.detach()
    if hasattr(x, "stop_grad"):
        return x.stop_grad()
    return x


def nan_to_num(x, nan=0.0, posinf=None, neginf=None):
    # Match torch.nan_to_num defaults closely enough for fp32 training logs/losses.
    if posinf is None:
        posinf = np.finfo(np.float32).max
    if neginf is None:
        neginf = np.finfo(np.float32).min

    if hasattr(jt, "nan_to_num"):
        return jt.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)

    x = jt.where(jt.isnan(x), jt.zeros_like(x) + nan, x)
    x = jt.where(jt.isinf(x) & (x > 0), jt.zeros_like(x) + posinf, x)
    x = jt.where(jt.isinf(x) & (x < 0), jt.zeros_like(x) + neginf, x)
    return x


def softplus(x):
    # Stable softplus wrapper.
    if hasattr(nn, "softplus"):
        return nn.softplus(x)
    return jt.log(1 + jt.exp(-jt.abs(x))) + jt.maximum(x, jt.zeros_like(x))


def sigmoid(x):
    return jt.sigmoid(x)


def mse_loss(x, target):
    return ((x - target) ** 2).mean()


def mean_abs(x, dims=(1, 2, 3), keepdims=True):
    return jt.abs(x).mean(dims=list(dims), keepdims=keepdims)


def tensor_norm(x):
    return jt.sqrt((x ** 2).sum())


def reshape_noise_level(sigma):
    return sigma.reshape(-1, 1, 1, 1).float32()


def get_x0_from_noise(sample, model_output, alphas_cumprod, timestep):
    # Port of main.utils.get_x0_from_noise from the official PyTorch code.
    alpha_prod_t = alphas_cumprod[timestep].reshape(-1, 1, 1, 1)
    beta_prod_t = 1 - alpha_prod_t
    return (sample - beta_prod_t ** 0.5 * model_output) / (alpha_prod_t ** 0.5)
