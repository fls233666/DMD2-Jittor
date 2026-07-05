"""Generate multi-step EDM sample grids from a Jittor denoiser checkpoint."""

import argparse
import os
import sys

import numpy as np


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
from samplers.multistep import sample_multistep
from sample_one_step import (
    load_checkpoint_object,
    make_model_args,
    parse_int_list,
    save_individual_images,
    to_jittor_state,
)
from utils.image import normalize_to_uint8, save_image_grid


AUTO_STATE_PREFIXES = (
    "feedforward_model",
    "ema.ema_model",
    "guidance_model.real_unet",
    "guidance_model.fake_unet",
)


def select_state(obj, state_key=None):
    if state_key:
        for part in state_key.split("."):
            if not isinstance(obj, dict) or part not in obj:
                raise KeyError(f"Cannot find state-key component {part!r}")
            obj = obj[part]
        return obj

    if isinstance(obj, dict):
        if "state_dict" in obj:
            return obj["state_dict"]
        if isinstance(obj.get("model"), dict):
            return obj["model"]
    return obj


def strip_state_prefix(state, state_prefix="auto"):
    if not isinstance(state, dict):
        raise TypeError("Checkpoint state must be a dict-like state_dict.")

    if state_prefix in (None, "", "none"):
        return state, None

    if state_prefix == "auto":
        for prefix in AUTO_STATE_PREFIXES:
            stripped = _strip_prefix(state, prefix)
            if stripped:
                return stripped, prefix
        return state, None

    stripped = _strip_prefix(state, state_prefix)
    if not stripped:
        raise KeyError(f"No checkpoint keys start with {state_prefix!r}.")
    return stripped, state_prefix


def _strip_prefix(state, prefix):
    prefix = prefix.rstrip(".") + "."
    return {
        key[len(prefix):]: value
        for key, value in state.items()
        if isinstance(key, str) and key.startswith(prefix)
    }


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None, help="Denoiser, teacher, or training checkpoint.")
    parser.add_argument("--state-key", default=None, help="Optional nested state key, e.g. state_dict or model.")
    parser.add_argument(
        "--state-prefix",
        default="auto",
        help=(
            "Prefix to strip from flat training checkpoint keys. Use 'auto' to try "
            "feedforward_model, EMA, real teacher, then fake teacher; use 'none' for generator-only checkpoints."
        ),
    )
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
    parser.add_argument("--num-steps", type=int, default=18)
    parser.add_argument("--solver", choices=("euler", "heun"), default="heun")
    parser.add_argument("--s-churn", type=float, default=0.0)
    parser.add_argument("--s-min", type=float, default=0.0)
    parser.add_argument("--s-max", type=float, default=float("inf"))
    parser.add_argument("--s-noise", type=float, default=1.0)
    parser.add_argument("--class-idx", type=int, default=None)
    parser.add_argument("--labels", default="", help="Comma-separated class ids. Overrides --class-idx.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument(
        "--output-grid",
        default=os.path.join(PROJECT_ROOT, "outputs", "grids", "multistep_demo.svg"),
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "samples", "multistep_demo"),
    )
    parser.add_argument("--output-npz", default="")
    parser.add_argument("--prefix", default="sample")
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    np.random.seed(args.seed)
    if hasattr(jt, "set_global_seed"):
        jt.set_global_seed(args.seed)

    model = get_edm_network(args=make_model_args(args))
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(args.checkpoint)
        state = select_state(load_checkpoint_object(args.checkpoint), state_key=args.state_key)
        state, stripped_prefix = strip_state_prefix(state, state_prefix=args.state_prefix)
        model.load_state_dict(to_jittor_state(state))
        print(f"loaded checkpoint: {args.checkpoint}")
        if stripped_prefix:
            print(f"stripped state prefix: {stripped_prefix}")
    elif not args.allow_random_init:
        raise ValueError("Provide --checkpoint or pass --allow-random-init for smoke sampling.")

    model.eval()
    labels = parse_int_list(args.labels)
    batch_size = len(labels) if labels is not None else args.batch_size

    images = sample_multistep(
        net=model,
        batch_size=batch_size,
        labels=labels,
        class_idx=args.class_idx,
        label_dim=args.label_dim,
        img_channels=3,
        img_resolution=args.resolution,
        num_steps=args.num_steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        solver=args.solver,
        S_churn=args.s_churn,
        S_min=args.s_min,
        S_max=args.s_max,
        S_noise=args.s_noise,
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
