"""Backward-simulation helpers for DMD2-style multi-step data preparation."""

import jittor as jt

try:
    from .common import reshape_noise_level, stop_grad
except ImportError:
    from common import reshape_noise_level, stop_grad


def make_denoising_step_list(denoising_timestep, num_denoising_step):
    """Create the descending timestep list used by official SD backward simulation."""

    if num_denoising_step <= 0:
        raise ValueError("num_denoising_step must be positive.")
    interval = denoising_timestep // num_denoising_step
    if interval <= 0:
        raise ValueError("denoising_timestep must be >= num_denoising_step.")

    return jt.array(
        list(range(denoising_timestep - 1, 0, -interval)),
    ).int32()


def select_backward_step(num_denoising_step):
    """Sample one shared backward-simulation step index."""

    return jt.randint(low=0, high=num_denoising_step, shape=[1]).int32()


def edm_backward_step(noisy_image, model, sigma, labels=None, next_sigma=None, noise=None):
    """Run one EDM backward-simulation step.

    The EDM model predicts x0 directly. A following noisy state can be produced by
    adding Gaussian noise with next_sigma, mirroring the official SD helper that
    re-noises the predicted clean image at the next timestep.
    """

    clean_image = model(noisy_image, sigma, labels)
    if next_sigma is None:
        return clean_image, clean_image

    if noise is None:
        noise = jt.randn_like(clean_image)
    next_noisy = clean_image + reshape_noise_level(next_sigma) * noise
    return clean_image, next_noisy


def sample_backward_edm(
    noisy_image,
    model,
    sigmas,
    labels=None,
    step_indices=None,
    selected_step=None,
    noise_fn=None,
):
    """EDM variant of official backward simulation for small DMD2 pipelines.

    Args:
        noisy_image: Initial noisy tensor.
        model: Denoiser returning clean x0 from (x, sigma, labels).
        sigmas: Fixed sigma schedule indexed by step_indices.
        labels: Optional class labels.
        step_indices: Descending integer timesteps. Defaults to all sigmas.
        selected_step: Number of denoising transitions to simulate. If None,
            one shared value is sampled, as in the official PyTorch code.
        noise_fn: Optional callable receiving a reference tensor and returning
            noise. Useful for deterministic tests.
    """

    if step_indices is None:
        step_indices = jt.arange(sigmas.shape[0]).int32()

    num_steps = int(step_indices.shape[0])
    if selected_step is None:
        selected_step = int(select_backward_step(num_steps).numpy()[0])
    else:
        selected_step = int(selected_step)

    selected_step = max(0, min(selected_step, num_steps - 1))
    generated_image = noisy_image
    current_noisy = noisy_image

    for i in range(selected_step):
        sigma = sigmas[step_indices[i]]
        next_sigma = sigmas[step_indices[i + 1]]
        noise = noise_fn(generated_image) if noise_fn is not None else None
        generated_image, current_noisy = edm_backward_step(
            noisy_image=current_noisy,
            model=model,
            sigma=sigma,
            labels=labels,
            next_sigma=next_sigma,
            noise=noise,
        )

    return_timesteps = step_indices[selected_step] * jt.ones([noisy_image.shape[0]]).int32()
    return stop_grad(generated_image), return_timesteps
