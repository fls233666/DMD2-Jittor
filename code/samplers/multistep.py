"""Multi-step EDM samplers for Jittor."""

import math
import jittor as jt
from jittor import nn

try:
    from .scheduler import get_edm_timesteps, make_class_labels
except ImportError:
    from scheduler import get_edm_timesteps, make_class_labels


def _to_float(x):
    if isinstance(x, jt.Var):
        return float(x.numpy().reshape(-1)[0])
    return float(x)


def _broadcast_sigma(sigma):
    if not isinstance(sigma, jt.Var):
        sigma = jt.array([sigma]).float32()
    sigma = sigma.float32()
    return sigma.reshape(-1)


def _call_denoiser(net, x, sigma, class_labels=None):
    sigma = _broadcast_sigma(sigma)
    return net(x, sigma, class_labels)


def edm_sampler(
    net,
    latents,
    class_labels=None,
    randn_like=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    S_churn=0.0,
    S_min=0.0,
    S_max=float("inf"),
    S_noise=1.0,
    solver="heun",
):
    # Port of EDM Algorithm 2 from the official PyTorch sampler.
    if solver not in ("euler", "heun"):
        raise ValueError("solver must be 'euler' or 'heun'.")
    if randn_like is None:
        randn_like = jt.randn_like

    t_steps = get_edm_timesteps(
        num_steps=num_steps,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        rho=rho,
        net=net,
        append_zero=True,
    )

    x_next = latents.float32() * t_steps[0]
    for i in range(num_steps):
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]
        x_cur = x_next

        t_cur_value = _to_float(t_cur)
        if S_min <= t_cur_value <= S_max:
            gamma = min(float(S_churn) / num_steps, math.sqrt(2.0) - 1.0)
        else:
            gamma = 0.0

        t_hat = t_cur + gamma * t_cur
        if hasattr(net, "round_sigma"):
            t_hat = net.round_sigma(t_hat)

        noise_scale = jt.sqrt(jt.maximum(t_hat ** 2 - t_cur ** 2, jt.zeros_like(t_cur)))
        x_hat = x_cur + noise_scale * float(S_noise) * randn_like(x_cur)

        denoised = _call_denoiser(net, x_hat, t_hat, class_labels).float32()
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if solver == "heun" and i < num_steps - 1:
            denoised = _call_denoiser(net, x_next, t_next, class_labels).float32()
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


def sample_multistep(
    net,
    batch_size=1,
    labels=None,
    class_idx=None,
    label_dim=None,
    img_channels=None,
    img_resolution=None,
    latents=None,
    num_steps=18,
    sigma_min=0.002,
    sigma_max=80.0,
    rho=7.0,
    solver="heun",
    **sampler_kwargs,
):
    # Generate samples from an EDM denoiser with Euler/Heun updates.
    if img_channels is None:
        img_channels = int(getattr(net, "img_channels", 3))
    if img_resolution is None:
        img_resolution = int(getattr(net, "img_resolution", 64))
    if label_dim is None:
        label_dim = int(getattr(net, "label_dim", 0))

    if latents is None:
        latents = jt.randn([batch_size, img_channels, img_resolution, img_resolution]).float32()
    class_labels = make_class_labels(
        batch_size=batch_size,
        label_dim=label_dim,
        class_idx=class_idx,
        labels=labels,
    )

    with jt.no_grad():
        return edm_sampler(
            net=net,
            latents=latents,
            class_labels=class_labels,
            num_steps=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            solver=solver,
            **sampler_kwargs,
        )


class EDMMultiStepSampler(nn.Module):
    # Module wrapper around EDM Euler/Heun multi-step sampling.

    def __init__(
        self,
        net,
        num_steps=18,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        solver="heun",
    ):
        super().__init__()
        self.net = net
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.solver = solver

    def execute(
        self,
        batch_size=1,
        labels=None,
        class_idx=None,
        latents=None,
        **sampler_kwargs,
    ):
        return sample_multistep(
            net=self.net,
            batch_size=batch_size,
            labels=labels,
            class_idx=class_idx,
            latents=latents,
            num_steps=self.num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            solver=self.solver,
            **sampler_kwargs,
        )
