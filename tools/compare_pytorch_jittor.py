"""Compare official PyTorch checkpoints with migrated Jittor models.

The default mode converts a PyTorch checkpoint in memory and checks whether the
result covers the target Jittor model state dict with matching shapes.  An
optional forward check is available for EDM generator checkpoints when both
PyTorch and Jittor are installed in the same environment.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    from .convert_pytorch_ckpt import (
        compare_with_target,
        convert_state_dict,
        format_report,
        load_pytorch_checkpoint,
        load_serialized_state,
        parse_int_list,
        save_converted_state,
        state_shapes,
    )
except ImportError:
    from convert_pytorch_ckpt import (
        compare_with_target,
        convert_state_dict,
        format_report,
        load_pytorch_checkpoint,
        load_serialized_state,
        parse_int_list,
        save_converted_state,
        state_shapes,
    )

try:
    from utils.logger import write_alignment_log
except ImportError:
    write_alignment_log = None


def setup_jittor_paths(project_root: Optional[str] = None) -> str:
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    else:
        project_root = os.path.abspath(project_root)

    for path in (
        project_root,
        os.path.join(project_root, "code"),
        os.path.join(project_root, "code", "models"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)

    return project_root


def make_model_args(args) -> SimpleNamespace:
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


def build_jittor_model(args):
    setup_jittor_paths(args.jittor_root)

    import jittor as jt

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    model_args = make_model_args(args)

    if args.target == "generator":
        from models.diffusion import get_edm_network

        return get_edm_network(args=model_args)

    if args.target == "guidance":
        from models.guidance import EDMGuidance

        return EDMGuidance(args=model_args)

    raise ValueError(f"Unsupported target: {args.target}")


def _to_jittor_state(state_dict: Mapping[str, object]):
    import jittor as jt

    converted = {}
    for key, value in state_dict.items():
        if isinstance(value, jt.Var):
            converted[key] = value
        else:
            converted[key] = jt.array(np.asarray(value))
    return converted


def load_state_into_jittor(model, state_dict: Mapping[str, object]):
    return model.load_state_dict(_to_jittor_state(state_dict))


def _add_pytorch_paths(pytorch_root: Optional[str]) -> None:
    if not pytorch_root:
        return
    pytorch_root = os.path.abspath(pytorch_root)
    for path in (
        os.path.join(pytorch_root, "third_party", "edm"),
        pytorch_root,
    ):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def build_pytorch_generator(args):
    _add_pytorch_paths(args.pytorch_root)

    from third_party.edm.training.networks import EDMPrecond

    return EDMPrecond(
        img_resolution=args.resolution,
        img_channels=3,
        label_dim=args.label_dim,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=args.sigma_data,
        model_type="DhariwalUNet",
        augment_dim=0,
        model_channels=args.model_channels,
        channel_mult=list(parse_int_list(args.channel_mult)),
        channel_mult_emb=4,
        num_blocks=args.num_blocks,
        attn_resolutions=list(parse_int_list(args.attn_resolutions)),
        dropout=0.0,
        label_dropout=0,
    )


def compare_generator_forward(
    raw_pytorch_state: Mapping[str, object],
    converted_state: Mapping[str, object],
    args,
) -> Tuple[float, float]:
    """Run a small EDM generator forward comparison."""

    if args.target != "generator":
        raise ValueError("--forward currently supports --target generator only.")

    import torch
    import jittor as jt

    torch_model = build_pytorch_generator(args)
    torch_model.load_state_dict(raw_pytorch_state, strict=True)
    torch_model.eval()

    jittor_model = build_jittor_model(args)
    load_state_into_jittor(jittor_model, converted_state)
    jittor_model.eval()

    rng = np.random.RandomState(args.seed)
    x_np = rng.randn(args.forward_batch_size, 3, args.resolution, args.resolution).astype(
        np.float32
    )
    sigma_np = np.full([args.forward_batch_size], args.conditioning_sigma, dtype=np.float32)
    class_ids = rng.randint(0, args.label_dim, size=[args.forward_batch_size])
    labels_np = np.eye(args.label_dim, dtype=np.float32)[class_ids]

    with torch.no_grad():
        torch_out = torch_model(
            torch.from_numpy(x_np),
            torch.from_numpy(sigma_np),
            torch.from_numpy(labels_np),
        )
        torch_out_np = torch_out.detach().cpu().numpy()

    with jt.no_grad():
        jittor_out = jittor_model(
            jt.array(x_np),
            jt.array(sigma_np),
            jt.array(labels_np),
        )
        jt.sync_all()
        jittor_out_np = np.asarray(jittor_out.numpy())

    diff = np.abs(torch_out_np - jittor_out_np)
    return float(diff.max()), float(diff.mean())


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--pytorch-ckpt", help="Official PyTorch checkpoint to convert.")
    source.add_argument("--converted-ckpt", help="Already converted Jittor state dict.")
    parser.add_argument(
        "--dump-target-shapes",
        default=None,
        help="Write the target Jittor model state_dict shapes and exit.",
    )
    parser.add_argument("--save-converted", default=None)
    parser.add_argument(
        "--log-path",
        default=None,
        help="Optional alignment log path, e.g. logs/pytorch_jittor_align.log.",
    )
    parser.add_argument("--target", choices=("generator", "guidance"), default="generator")
    parser.add_argument("--jittor-root", default=None, help="DMD2-jittor project root.")
    parser.add_argument("--pytorch-root", default=None, help="DMD2-pytorch project root.")
    parser.add_argument(
        "--input-format",
        choices=("auto", "torch", "pickle"),
        default="auto",
    )
    parser.add_argument("--source-key", default=None)
    parser.add_argument("--state-key", default=None, help="Key in --converted-ckpt.")
    parser.add_argument("--strict", action="store_true")

    parser.add_argument("--dataset-name", default="imagenet")
    parser.add_argument("--config-name", default="imagenet")
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--label-dim", type=int, default=1000)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--channel-mult", default="1,2,3,4")
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument("--model-channels", type=int, default=192)
    parser.add_argument("--attn-resolutions", default="32,16,8")
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--min-step-percent", type=float, default=0.02)
    parser.add_argument("--max-step-percent", type=float, default=0.98)
    parser.add_argument("--gan-classifier", action="store_true")
    parser.add_argument("--diffusion-gan", action="store_true")
    parser.add_argument("--diffusion-gan-max-timestep", type=int, default=1)
    parser.add_argument("--use-cuda", action="store_true")

    parser.add_argument("--forward", action="store_true")
    parser.add_argument("--forward-batch-size", type=int, default=2)
    parser.add_argument("--conditioning-sigma", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_argparser()
    args = parser.parse_args(argv)

    if not args.dump_target_shapes and not args.pytorch_ckpt and not args.converted_ckpt:
        parser.error("one of --pytorch-ckpt, --converted-ckpt, or --dump-target-shapes is required")

    target_model = build_jittor_model(args)
    target_shapes = state_shapes(target_model.state_dict())

    if args.dump_target_shapes:
        output_dir = os.path.dirname(os.path.abspath(args.dump_target_shapes))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.dump_target_shapes, "wb") as handle:
            pickle.dump(target_shapes, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved target shape index: {args.dump_target_shapes}")
        print(f"target keys: {len(target_shapes)}")
        return 0

    raw_state = None

    if args.converted_ckpt:
        converted_state = load_serialized_state(args.converted_ckpt, args.state_key)
        report = compare_with_target(converted_state, target_shapes)
    else:
        raw_state = load_pytorch_checkpoint(
            args.pytorch_ckpt,
            input_format=args.input_format,
            source_key=args.source_key,
            pytorch_root=args.pytorch_root,
        )
        converted_state, report = convert_state_dict(
            raw_state,
            target_shapes=target_shapes,
            img_resolution=args.resolution,
            channel_mult=parse_int_list(args.channel_mult),
            num_blocks=args.num_blocks,
        )

    report_text = format_report(report)
    print(report_text)

    if args.save_converted:
        save_converted_state(args.save_converted, converted_state)
        print(f"saved converted checkpoint: {args.save_converted}")

    summary = {
        "target": args.target,
        "converted_keys": report.converted_keys,
        "unexpected_keys": len(report.unexpected_keys),
        "missing_keys": len(report.missing_keys),
        "shape_mismatches": len(report.shape_mismatches),
    }

    if args.forward:
        if raw_state is None:
            raw_state = load_pytorch_checkpoint(
                args.pytorch_ckpt,
                input_format=args.input_format,
                source_key=args.source_key,
                pytorch_root=args.pytorch_root,
            )
        max_abs, mean_abs = compare_generator_forward(raw_state, converted_state, args)
        print(f"forward max_abs_diff: {max_abs:.8e}")
        print(f"forward mean_abs_diff: {mean_abs:.8e}")
        summary["forward_max_abs_diff"] = max_abs
        summary["forward_mean_abs_diff"] = mean_abs

    if args.log_path:
        if write_alignment_log is None:
            raise RuntimeError("utils.logger.write_alignment_log is not available.")
        write_alignment_log(args.log_path, report_text, summary=summary)
        print(f"saved alignment log: {args.log_path}")

    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
