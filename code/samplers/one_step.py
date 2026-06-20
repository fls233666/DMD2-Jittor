"""One-step DMD2 sampling helpers for Jittor."""

import jittor as jt
from jittor import nn

try:
    from .scheduler import constant_sigma, make_class_labels, randn_image
except ImportError:
    from scheduler import constant_sigma, make_class_labels, randn_image


def infer_image_shape(model, img_channels=3, img_resolution=64):
    # Infer image shape metadata from EDMPrecond/EDMUniModel-like modules.
    if hasattr(model, "img_channels") and hasattr(model, "img_resolution"):
        return int(model.img_channels), int(model.img_resolution)
    if hasattr(model, "feedforward_model"):
        inner = model.feedforward_model
        if hasattr(inner, "img_channels") and hasattr(inner, "img_resolution"):
            return int(inner.img_channels), int(inner.img_resolution)
    return int(img_channels), int(img_resolution)


def infer_label_dim(model, label_dim=0):
    # Infer class label count from known model wrappers.
    if hasattr(model, "label_dim"):
        return int(model.label_dim)
    if hasattr(model, "feedforward_model") and hasattr(model.feedforward_model, "label_dim"):
        return int(model.feedforward_model.label_dim)
    return int(label_dim)


def call_generator(generator, scaled_noise, sigma, labels=None):
    # Call either EDMPrecond-like or EDMUniModel-like generator modules.
    if hasattr(generator, "feedforward_model"):
        return generator.feedforward_model(scaled_noise, sigma, labels)
    return generator(scaled_noise, sigma, labels)


def sample_one_step(
    generator,
    batch_size=1,
    labels=None,
    class_idx=None,
    label_dim=None,
    img_channels=None,
    img_resolution=None,
    conditioning_sigma=80.0,
    noise=None,
    return_latents=False,
):
    # Generate images with the official DMD2 one-step pattern.
    default_channels = 3 if img_channels is None else img_channels
    default_resolution = 64 if img_resolution is None else img_resolution
    channels, resolution = infer_image_shape(
        generator,
        img_channels=default_channels,
        img_resolution=default_resolution,
    )

    if label_dim is None:
        label_dim = infer_label_dim(generator, label_dim=0)
    one_hot_labels = make_class_labels(
        batch_size=batch_size,
        label_dim=label_dim,
        class_idx=class_idx,
        labels=labels,
    )

    if noise is None:
        noise = randn_image(
            batch_size=batch_size,
            channels=channels,
            resolution=resolution,
            sigma=1.0,
        )
    sigma = constant_sigma(batch_size=batch_size, sigma=conditioning_sigma)
    scaled_noise = noise.float32() * float(conditioning_sigma)

    with jt.no_grad():
        images = call_generator(generator, scaled_noise, sigma, one_hot_labels)

    if return_latents:
        return images, {
            "noise": noise,
            "scaled_noise": scaled_noise,
            "sigma": sigma,
            "labels": one_hot_labels,
        }
    return images


class OneStepSampler(nn.Module):
    # Module wrapper around one-step DMD2 sampling.

    def __init__(
        self,
        generator,
        conditioning_sigma=80.0,
        label_dim=None,
        img_channels=None,
        img_resolution=None,
    ):
        super().__init__()
        self.generator = generator
        self.conditioning_sigma = conditioning_sigma
        self.label_dim = label_dim
        self.img_channels = img_channels
        self.img_resolution = img_resolution

    def execute(
        self,
        batch_size=1,
        labels=None,
        class_idx=None,
        noise=None,
        return_latents=False,
    ):
        return sample_one_step(
            generator=self.generator,
            batch_size=batch_size,
            labels=labels,
            class_idx=class_idx,
            label_dim=self.label_dim,
            img_channels=self.img_channels,
            img_resolution=self.img_resolution,
            conditioning_sigma=self.conditioning_sigma,
            noise=noise,
            return_latents=return_latents,
        )
