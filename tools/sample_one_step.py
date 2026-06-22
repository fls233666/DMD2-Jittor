"""Generate one-step DMD2 sample grids from a Jittor generator checkpoint."""

import argparse
import os
import pickle
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image


def setup_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_dir = os.path.join(project_root, "code")
    for path in (
        project_root,
        code_dir,
        os.path.join(code_dir, "models"),
        os.path.join(code_dir, "samplers"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    return project_root


PROJECT_ROOT = setup_paths()

import jittor as jt

from models.diffusion import get_edm_network
from models.unified_model import EDMUniModel
from samplers.one_step import sample_one_step
from utils.image import normalize_to_uint8, save_image_grid


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def make_model_args(args):
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        resolution=args.resolution,
        label_dim=args.label_dim,
        use_fp16=False,
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        config_name=args.config_name,
        gan_classifier=args.gan_classifier,
        diffusion_gan=args.diffusion_gan,
        diffusion_gan_max_timestep=args.diffusion_gan_max_timestep,
        num_train_timesteps=args.num_train_timesteps,
        min_step_percent=args.min_step_percent,
        max_step_percent=args.max_step_percent,
    )


def load_checkpoint_object(path):
    with open(path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
    return jt.load(path)


def select_state(obj, state_key=None, target="generator"):
    if state_key:
        for part in state_key.split("."):
            obj = obj[part]
        return obj

    if isinstance(obj, dict):
        if target == "unified" and "model" in obj:
            return obj["model"]
        if "state_dict" in obj:
            return obj["state_dict"]
    return obj


def to_jittor_state(state):
    converted = {}
    for key, value in state.items():
        if isinstance(value, jt.Var):
            converted[key] = value
        else:
            converted[key] = jt.array(np.asarray(value))
    return converted


def build_model(args):
    args_obj = make_model_args(args)
    if args.target == "generator":
        return get_edm_network(args=args_obj)
    if args.target == "unified":
        return EDMUniModel(args=args_obj, initialize_generator=True)
    raise ValueError(f"Unsupported target: {args.target}")


def save_individual_images(images_uint8, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for index, image in enumerate(images_uint8):
        path = os.path.join(output_dir, f"{prefix}_{index:04d}.png")
        Image.fromarray(image).save(path)
        paths.append(path)
    return paths


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Converted generator or training checkpoint.")
    parser.add_argument("--state-key", default=None, help="Optional nested state key, e.g. state_dict or model.")
    parser.add_argument("--target", choices=("generator", "unified"), default="generator")
    parser.add_argument("--allow-random-init", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")

    parser.add_argument("--dataset-name", default="imagenet")
    parser.add_argument("--config-name", default="imagenet")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--label-dim", type=int, default=1000)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--min-step-percent", type=float, default=0.02)
    parser.add_argument("--max-step-percent", type=float, default=0.98)
    parser.add_argument("--gan-classifier", action="store_true")
    parser.add_argument("--diffusion-gan", action="store_true")
    parser.add_argument("--diffusion-gan-max-timestep", type=int, default=1)

    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conditioning-sigma", type=float, default=80.0)
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--labels", default="", help="Comma-separated class ids. Overrides --class-idx.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--output-grid", default=os.path.join(PROJECT_ROOT, "outputs", "grids", "imagenet64_demo.svg"))
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "outputs", "samples", "imagenet64_demo"))
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--prefix", default="sample")
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    if hasattr(jt, "set_global_seed"):
        jt.set_global_seed(args.seed)

    model = build_model(args)
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        state = select_state(
            load_checkpoint_object(args.checkpoint),
            state_key=args.state_key,
            target=args.target,
        )
        model.load_state_dict(to_jittor_state(state))
        print(f"loaded checkpoint: {args.checkpoint}")
    elif not args.allow_random_init:
        raise ValueError("Provide --checkpoint or pass --allow-random-init for smoke sampling.")

    model.eval()
    labels = parse_int_list(args.labels)
    if labels is not None:
        batch_size = len(labels)
    else:
        batch_size = args.batch_size

    images = sample_one_step(
        generator=model,
        batch_size=batch_size,
        labels=labels,
        class_idx=args.class_idx,
        label_dim=args.label_dim,
        img_channels=3,
        img_resolution=args.resolution,
        conditioning_sigma=args.conditioning_sigma,
    )
    jt.sync_all()

    save_image_grid(
        images,
        path=args.output_grid,
        nrow=args.nrow,
        nchw=True,
        value_range="minus_one_one",
    )
    images_uint8 = normalize_to_uint8(images, nchw=True, value_range="minus_one_one")
    save_individual_images(images_uint8, args.output_dir, args.prefix)
    if args.output_npz:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_npz)), exist_ok=True)
        np.savez_compressed(args.output_npz, images=images_uint8)

    print(f"saved sample grid: {args.output_grid}")
    print(f"saved sample images: {args.output_dir}")
    if args.output_npz:
        print(f"saved sample npz: {args.output_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
