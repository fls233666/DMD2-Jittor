"""Create sample grids from image directories or numpy arrays."""

import argparse
import os
import sys

import numpy as np


def setup_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_dir = os.path.join(project_root, "code")
    for path in (project_root, code_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


setup_paths()

from utils.image import load_image_dir, save_image_grid


def load_samples(path, image_size=None, max_images=None, array_key=None):
    if os.path.isdir(path):
        return load_image_dir(path, image_size=image_size, max_images=max_images)

    data = np.load(path)
    if isinstance(data, np.ndarray):
        images = data
    elif array_key:
        images = data[array_key]
    elif "images" in data:
        images = data["images"]
    elif "arr_0" in data:
        images = data["arr_0"]
    else:
        raise KeyError(f"No image array found in {path}.")

    if max_images is not None:
        images = images[: int(max_images)]
    return images


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Image directory, .npy, or .npz file.")
    parser.add_argument("output", help="Output grid SVG.")
    parser.add_argument("--nrow", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--array-key", default=None)
    parser.add_argument("--nchw", action="store_true")
    parser.add_argument(
        "--value-range",
        choices=("auto", "minus_one_one", "zero_one"),
        default="auto",
    )
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    images = load_samples(
        args.input,
        image_size=args.image_size,
        max_images=args.max_images,
        array_key=args.array_key,
    )
    save_image_grid(
        images,
        path=args.output,
        nrow=args.nrow,
        nchw=args.nchw,
        value_range=args.value_range,
    )
    print(f"saved sample grid: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
