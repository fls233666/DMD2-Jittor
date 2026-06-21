"""Evaluation helpers for CIFAR-10 DMD2 debug runs."""

import os

import numpy as np
import jittor as jt

try:
    from samplers.one_step import sample_one_step
    from samplers.scheduler import images_to_uint8
    from utils.image import make_image_grid as _make_image_grid
    from utils.image import save_image_grid as _save_image_grid
except ImportError:
    try:
        from ..samplers.one_step import sample_one_step
        from ..samplers.scheduler import images_to_uint8
        from ..utils.image import make_image_grid as _make_image_grid
        from ..utils.image import save_image_grid as _save_image_grid
    except ImportError:
        from one_step import sample_one_step
        from scheduler import images_to_uint8
        from image import make_image_grid as _make_image_grid
        from image import save_image_grid as _save_image_grid


def to_numpy(x):
    if isinstance(x, jt.Var):
        jt.sync_all()
        return np.asarray(x.numpy())
    return np.asarray(x)


def make_image_grid(images, nrow=4):
    # Build a uint8 NHWC grid from a batch of NHWC images.
    return _make_image_grid(images, nrow=nrow)


def save_image_grid(images, path, nrow=4):
    return _save_image_grid(images, path=path, nrow=nrow)


class DebugSamplerEvaluator:
    # Save one-step sample grids during debug training.
    def __init__(
        self,
        output_dir,
        batch_size=16,
        nrow=4,
        labels=None,
        class_idx=None,
        conditioning_sigma=80.0,
        img_channels=None,
        img_resolution=None,
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.nrow = nrow
        self.labels = labels
        self.class_idx = class_idx
        self.conditioning_sigma = conditioning_sigma
        self.img_channels = img_channels
        self.img_resolution = img_resolution

    def evaluate(self, model, step=0):
        generator = model.feedforward_model if hasattr(model, "feedforward_model") else model
        images = sample_one_step(
            generator=generator,
            batch_size=self.batch_size,
            labels=self.labels,
            class_idx=self.class_idx,
            conditioning_sigma=self.conditioning_sigma,
            img_channels=self.img_channels,
            img_resolution=self.img_resolution,
        )
        images = images_to_uint8(images, nchw=True)
        path = os.path.join(self.output_dir, f"samples_{int(step):06d}.svg")
        return save_image_grid(images, path=path, nrow=self.nrow)
