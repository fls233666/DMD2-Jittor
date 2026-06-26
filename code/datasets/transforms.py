"""Image transform helpers for DMD2 Jittor datasets."""

import random

import numpy as np
from PIL import Image, ImageOps


def _setup_size(size):
    if isinstance(size, int):
        return size, size
    if len(size) != 2:
        raise ValueError("size must be an int or a pair of ints.")
    return int(size[0]), int(size[1])


def to_pil_rgb(image):
    # Convert PIL/numpy image inputs to RGB PIL images.
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = image.transpose(1, 2, 0)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)

    if image.dtype != np.uint8:
        if image.size and image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    return Image.fromarray(image).convert("RGB")


def resize(image, size, interpolation=Image.BICUBIC):
    size = _setup_size(size)
    image = to_pil_rgb(image)
    if image.size == (size[1], size[0]):
        return image
    return image.resize((size[1], size[0]), interpolation)


def random_crop(image, size, padding=0):
    size = _setup_size(size)
    image = to_pil_rgb(image)
    if padding > 0:
        image = ImageOps.expand(image, border=int(padding), fill=0)

    width, height = image.size
    crop_h, crop_w = size
    if crop_h > height or crop_w > width:
        raise ValueError("crop size exceeds input image size.")

    top = random.randint(0, height - crop_h)
    left = random.randint(0, width - crop_w)
    return image.crop((left, top, left + crop_w, top + crop_h))


def random_horizontal_flip(image, p=0.5):
    image = to_pil_rgb(image)
    if random.random() < p:
        return image.transpose(Image.FLIP_LEFT_RIGHT)
    return image


def image_to_nchw_float(image, normalize=True):
    # Return float32 CHW image. DMD2 training uses [-1, 1] image latents.
    image = to_pil_rgb(image)
    image = np.asarray(image).astype(np.float32).transpose(2, 0, 1)
    if normalize:
        image = image / 127.5 - 1.0
    else:
        image = image / 255.0
    return image


def one_hot(label, num_classes):
    target = np.zeros([num_classes], dtype=np.float32)
    target[int(label)] = 1.0
    return target


class CIFAR10Transform:
    # CIFAR-10 transform used by debug DMD2 training.
    def __init__(
        self,
        train=True,
        image_size=32,
        augment=True,
        crop_padding=4,
        hflip_prob=0.5,
        normalize=True,
    ):
        self.train = train
        self.image_size = image_size
        self.augment = augment
        self.crop_padding = crop_padding
        self.hflip_prob = hflip_prob
        self.normalize = normalize

    def __call__(self, image):
        image = to_pil_rgb(image)

        if self.train and self.augment:
            image = random_crop(
                image,
                size=self.image_size,
                padding=self.crop_padding,
            )
            image = random_horizontal_flip(image, p=self.hflip_prob)
        else:
            image = resize(image, size=self.image_size)

        return image_to_nchw_float(image, normalize=self.normalize)


def make_cifar10_transform(
    train=True,
    image_size=32,
    augment=True,
    crop_padding=4,
    hflip_prob=0.5,
    normalize=True,
):
    return CIFAR10Transform(
        train=train,
        image_size=image_size,
        augment=augment,
        crop_padding=crop_padding,
        hflip_prob=hflip_prob,
        normalize=normalize,
    )


class ImageClassificationTransform:
    # Generic RGB classification transform for ImageNet-like datasets.
    def __init__(
        self,
        train=True,
        image_size=64,
        augment=True,
        crop_padding=4,
        hflip_prob=0.5,
        normalize=True,
    ):
        self.train = train
        self.image_size = image_size
        self.augment = augment
        self.crop_padding = crop_padding
        self.hflip_prob = hflip_prob
        self.normalize = normalize

    def __call__(self, image):
        image = resize(image, size=self.image_size)

        if self.train and self.augment:
            image = random_crop(
                image,
                size=self.image_size,
                padding=self.crop_padding,
            )
            image = random_horizontal_flip(image, p=self.hflip_prob)

        return image_to_nchw_float(image, normalize=self.normalize)


def make_image_classification_transform(
    train=True,
    image_size=64,
    augment=True,
    crop_padding=4,
    hflip_prob=0.5,
    normalize=True,
):
    return ImageClassificationTransform(
        train=train,
        image_size=image_size,
        augment=augment,
        crop_padding=crop_padding,
        hflip_prob=hflip_prob,
        normalize=normalize,
    )
