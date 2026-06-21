"""Image conversion and grid helpers."""

import base64
import html
import io
import os

import numpy as np
from PIL import Image


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def to_numpy(x):
    if hasattr(x, "numpy"):
        try:
            return np.asarray(x.numpy())
        except Exception:
            pass
    return np.asarray(x)


def normalize_to_uint8(images, nchw=False, value_range="auto"):
    images = to_numpy(images)
    if nchw and images.ndim == 4:
        images = np.transpose(images, [0, 2, 3, 1])

    if images.dtype == np.uint8:
        return images

    images = images.astype(np.float32)
    if value_range == "minus_one_one":
        images = (images + 1.0) * 127.5
    elif value_range == "zero_one":
        images = images * 255.0
    elif value_range == "auto":
        min_value = float(np.nanmin(images))
        max_value = float(np.nanmax(images))
        if min_value >= -1.05 and max_value <= 1.05 and min_value < 0:
            images = (images + 1.0) * 127.5
        elif min_value >= -0.05 and max_value <= 1.05:
            images = images * 255.0

    return np.clip(images, 0, 255).astype(np.uint8)


def make_image_grid(images, nrow=4):
    images = normalize_to_uint8(images)
    if images.ndim == 3:
        images = images[None]
    batch, height, width, channels = images.shape
    nrow = max(int(nrow), 1)
    ncol = (batch + nrow - 1) // nrow
    grid = np.zeros([ncol * height, nrow * width, channels], dtype=np.uint8)
    for index in range(batch):
        row = index // nrow
        col = index % nrow
        grid[row * height:(row + 1) * height, col * width:(col + 1) * width] = images[index]
    return grid


def save_image_grid(images, path, nrow=4, nchw=False, value_range="auto"):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    images = normalize_to_uint8(images, nchw=nchw, value_range=value_range)
    grid = make_image_grid(images, nrow=nrow)
    if str(path).lower().endswith(".svg"):
        save_image_grid_svg(grid, path)
    else:
        Image.fromarray(grid).save(path)
    return path


def save_image_grid_svg(images, path, nrow=None, nchw=False, value_range="auto"):
    # Save a sample grid as SVG with an embedded PNG payload.
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    if nrow is not None:
        images = make_image_grid(
            normalize_to_uint8(images, nchw=nchw, value_range=value_range),
            nrow=nrow,
        )
    else:
        images = normalize_to_uint8(images, nchw=nchw, value_range=value_range)

    if images.ndim == 4:
        images = make_image_grid(images, nrow=4)

    height, width = images.shape[0], images.shape[1]
    buffer = io.BytesIO()
    Image.fromarray(images).save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    title = html.escape(os.path.basename(path))
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{title}">\n'
        f'  <image href="data:image/png;base64,{payload}" width="{width}" height="{height}"/>\n'
        "</svg>\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(svg)
    return path


def list_image_files(image_dir):
    paths = []
    for root, _, filenames in os.walk(image_dir):
        for filename in filenames:
            if filename.lower().endswith(IMAGE_EXTENSIONS):
                paths.append(os.path.join(root, filename))
    return sorted(paths)


def load_image(path, image_size=None):
    image = Image.open(path).convert("RGB")
    if image_size is not None:
        image = image.resize((int(image_size), int(image_size)), Image.BICUBIC)
    return np.asarray(image, dtype=np.uint8)


def load_image_dir(image_dir, image_size=None, max_images=None):
    paths = list_image_files(image_dir)
    if max_images is not None:
        paths = paths[: int(max_images)]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return np.stack([load_image(path, image_size=image_size) for path in paths], axis=0)


def image_dir_to_pixel_features(image_dir, image_size=32, max_images=None):
    images = load_image_dir(image_dir, image_size=image_size, max_images=max_images)
    return images.astype(np.float32).reshape(images.shape[0], -1) / 255.0
