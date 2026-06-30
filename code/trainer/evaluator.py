"""Evaluation helpers for image DMD2 runs."""

import os

import numpy as np
import jittor as jt

try:
    from samplers.one_step import sample_one_step
    from samplers.scheduler import images_to_uint8, randn_image
    from utils.image import make_image_grid as _make_image_grid
    from utils.image import save_image_grid as _save_image_grid
except ImportError:
    try:
        from ..samplers.one_step import sample_one_step
        from ..samplers.scheduler import images_to_uint8, randn_image
        from ..utils.image import make_image_grid as _make_image_grid
        from ..utils.image import save_image_grid as _save_image_grid
    except ImportError:
        from one_step import sample_one_step
        from scheduler import images_to_uint8, randn_image
        from image import make_image_grid as _make_image_grid
        from image import save_image_grid as _save_image_grid


def to_numpy(x):
    if isinstance(x, jt.Var):
        jt.sync_all()
        return np.asarray(x.numpy())
    return np.asarray(x)


def detach(x):
    if x is None:
        return None
    if hasattr(jt, "detach"):
        return jt.detach(x)
    if hasattr(x, "detach"):
        return x.detach()
    return x.stop_grad()


def cleanup_jittor_memory():
    # Best-effort cleanup for Jittor's lazy graph and allocator before/after eval.
    for name in ("sync_all", "gc", "clean_graph"):
        fn = getattr(jt, name, None)
        if not callable(fn):
            continue
        try:
            fn()
        except TypeError:
            continue


def make_image_grid(images, nrow=4):
    # Build a uint8 NHWC grid from a batch of NHWC images.
    return _make_image_grid(images, nrow=nrow)


def save_image_grid(images, path, nrow=4):
    return _save_image_grid(images, path=path, nrow=nrow)


class ImageDMD2SamplerEvaluator:
    # Save one-step sample grids during image DMD2 training.
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
        label_dim=10,
        save_legacy_svg=True,
        chunk_size=None,
    ):
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.nrow = nrow
        self.labels = labels
        self.class_idx = class_idx
        self.conditioning_sigma = conditioning_sigma
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.label_dim = label_dim
        self.save_legacy_svg = save_legacy_svg
        if chunk_size is None:
            chunk_size = int(os.environ.get("DMD2_EVAL_CHUNK_SIZE", "0") or 0)
        self.chunk_size = int(chunk_size) if int(chunk_size) > 0 else int(batch_size)
        self.fixed_noise = None
        self.fixed_class_ids = None

    def _init_fixed_inputs(self, generator):
        if self.fixed_noise is not None:
            return

        channels = self.img_channels
        resolution = self.img_resolution
        if channels is None and hasattr(generator, "img_channels"):
            channels = int(generator.img_channels)
        if resolution is None and hasattr(generator, "img_resolution"):
            resolution = int(generator.img_resolution)
        channels = 3 if channels is None else int(channels)
        resolution = 32 if resolution is None else int(resolution)

        self.fixed_noise = randn_image(
            batch_size=self.batch_size,
            channels=channels,
            resolution=resolution,
            sigma=1.0,
        )
        self.fixed_noise = detach(self.fixed_noise)

        if int(self.label_dim) <= 0:
            self.fixed_class_ids = None
        elif self.labels is not None:
            labels = np.asarray(self.labels, dtype=np.int32).reshape(-1)
            self.fixed_class_ids = detach(jt.array(labels).int32())
        elif self.class_idx is not None:
            labels = np.full([self.batch_size], int(self.class_idx), dtype=np.int32)
            self.fixed_class_ids = detach(jt.array(labels).int32())
        else:
            labels = np.arange(self.batch_size, dtype=np.int32) % int(self.label_dim)
            self.fixed_class_ids = detach(jt.array(labels).int32())

    @staticmethod
    def _slice_labels(labels, start, end):
        if labels is None:
            return None
        if isinstance(labels, jt.Var):
            return labels[start:end]
        labels = np.asarray(labels)
        return labels[start:end]

    @staticmethod
    def _images_to_cpu_uint8(images):
        images = images_to_uint8(images, nchw=True)
        jt.sync_all()
        return np.asarray(images.numpy()).astype(np.uint8)

    def _sample(self, generator, fixed=False):
        if fixed:
            self._init_fixed_inputs(generator)
            return sample_one_step(
                generator=generator,
                batch_size=self.batch_size,
                labels=self.fixed_class_ids,
                conditioning_sigma=self.conditioning_sigma,
                img_channels=self.img_channels,
                img_resolution=self.img_resolution,
                noise=self.fixed_noise,
            )

        return sample_one_step(
            generator=generator,
            batch_size=self.batch_size,
            labels=(
                jt.randint(low=0, high=int(self.label_dim), shape=[self.batch_size]).int32()
                if self.labels is None and self.class_idx is None and int(self.label_dim) > 0
                else self.labels
            ),
            class_idx=self.class_idx,
            conditioning_sigma=self.conditioning_sigma,
            img_channels=self.img_channels,
            img_resolution=self.img_resolution,
        )

    def _sample_numpy_chunked(self, generator, fixed=False):
        chunks = []
        chunk_size = max(1, min(int(self.chunk_size), int(self.batch_size)))

        if fixed:
            self._init_fixed_inputs(generator)

        for start in range(0, int(self.batch_size), chunk_size):
            end = min(start + chunk_size, int(self.batch_size))
            size = end - start

            if fixed:
                noise = self.fixed_noise[start:end]
                labels = self._slice_labels(self.fixed_class_ids, start, end)
                class_idx = None
            else:
                noise = None
                class_idx = self.class_idx
                if self.labels is None and self.class_idx is None and int(self.label_dim) > 0:
                    labels = jt.randint(low=0, high=int(self.label_dim), shape=[size]).int32()
                else:
                    labels = self._slice_labels(self.labels, start, end)

            with jt.no_grad():
                images = sample_one_step(
                    generator=generator,
                    batch_size=size,
                    labels=labels,
                    class_idx=class_idx,
                    conditioning_sigma=self.conditioning_sigma,
                    img_channels=self.img_channels,
                    img_resolution=self.img_resolution,
                    noise=noise,
                )
                chunks.append(self._images_to_cpu_uint8(images))

            del images
            cleanup_jittor_memory()

        if len(chunks) == 1:
            return chunks[0]
        return np.concatenate(chunks, axis=0)

    def evaluate(self, model, step=0):
        generator = model.feedforward_model if hasattr(model, "feedforward_model") else model
        is_training = getattr(generator, "is_training", None)
        was_training = bool(is_training()) if callable(is_training) else bool(
            getattr(generator, "training", False)
        )
        if hasattr(generator, "eval"):
            generator.eval()

        cleanup_jittor_memory()
        fixed_images = self._sample_numpy_chunked(generator, fixed=True)
        random_images = self._sample_numpy_chunked(generator, fixed=False)
        cleanup_jittor_memory()

        fixed_path = os.path.join(self.output_dir, f"fixed_step_{int(step):06d}.png")
        random_path = os.path.join(self.output_dir, f"random_step_{int(step):06d}.png")
        save_image_grid(fixed_images, path=fixed_path, nrow=self.nrow)
        save_image_grid(random_images, path=random_path, nrow=self.nrow)

        legacy_path = None
        if self.save_legacy_svg:
            legacy_path = os.path.join(self.output_dir, f"samples_{int(step):06d}.svg")
            save_image_grid(random_images, path=legacy_path, nrow=self.nrow)

        if was_training and hasattr(generator, "train"):
            generator.train()

        return {
            "fixed": fixed_path,
            "random": random_path,
            "legacy": legacy_path,
        }


DebugSamplerEvaluator = ImageDMD2SamplerEvaluator
