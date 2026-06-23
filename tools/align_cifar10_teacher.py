"""Multi-env forward alignment for the official CIFAR-10 EDM teacher.

The local PyTorch and Jittor environments are separate, so this tool is split
into subcommands:

1. make-inputs: write deterministic x/sigma/labels to NPZ.
2. torch-forward: run the official PyTorch teacher and save output NPZ.
3. jittor-forward: run the converted Jittor teacher and save output NPZ.
4. compare: compare the two output NPZ files.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np


DEFAULT_TEACHER = (
    "/home/koishi/DMD2/teacher-models/cifar10_teacher/"
    "edm-cifar10-32x32-cond-vp.pkl"
)
DEFAULT_JITTOR_CKPT = (
    "/home/koishi/DMD2/teacher-models/cifar10_teacher/"
    "cifar10_teacher_jittor.pkl"
)


def add_paths(*paths: str) -> None:
    for path in paths:
        path = os.path.abspath(path)
        if path not in sys.path:
            sys.path.insert(0, path)


def make_inputs(args) -> int:
    rng = np.random.RandomState(args.seed)
    x = rng.randn(args.batch_size, 3, 32, 32).astype(np.float32)
    sigma = np.exp(rng.uniform(np.log(0.002), np.log(80.0), size=[args.batch_size])).astype(
        np.float32
    )
    class_ids = rng.randint(0, 10, size=[args.batch_size])
    labels = np.eye(10, dtype=np.float32)[class_ids]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(
        args.output,
        x=x,
        sigma=sigma,
        labels=labels,
        class_ids=class_ids,
    )
    print(f"saved inputs: {args.output}")
    print(f"class_ids: {class_ids.tolist()}")
    return 0


def torch_forward(args) -> int:
    add_paths(
        os.path.join(args.pytorch_root, "third_party", "edm"),
        args.pytorch_root,
    )
    import torch

    data = np.load(args.inputs)
    with open(args.teacher_pkl, "rb") as handle:
        model = pickle.load(handle)["ema"].to(args.device)
    model.eval()

    with torch.no_grad():
        y = model(
            torch.from_numpy(data["x"]).to(args.device),
            torch.from_numpy(data["sigma"]).to(args.device),
            torch.from_numpy(data["labels"]).to(args.device),
        )
        output = y.detach().cpu().numpy()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, output=output)
    print(f"saved PyTorch output: {args.output}")
    print(f"shape: {list(output.shape)} mean: {float(output.mean()):.8g}")
    return 0


def jittor_forward(args) -> int:
    add_paths(
        args.jittor_root,
        os.path.join(args.jittor_root, "code"),
        os.path.join(args.jittor_root, "code", "models"),
    )
    import jittor as jt

    from models.diffusion import get_edm_network

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    data = np.load(args.inputs)
    model = get_edm_network(
        dataset_name="cifar10",
        config_name="cifar10",
        resolution=32,
        label_dim=10,
        use_fp16=False,
        sigma_data=0.5,
    )
    with open(args.jittor_checkpoint, "rb") as handle:
        state = pickle.load(handle)
    model.load_state_dict({
        key: value if isinstance(value, jt.Var) else jt.array(np.asarray(value))
        for key, value in state.items()
    })
    model.eval()

    with jt.no_grad():
        y = model(
            jt.array(data["x"]),
            jt.array(data["sigma"]),
            jt.array(data["labels"]),
        )
        jt.sync_all()
        output = np.asarray(y.numpy())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, output=output)
    print(f"saved Jittor output: {args.output}")
    print(f"shape: {list(output.shape)} mean: {float(output.mean()):.8g}")
    return 0


def compare(args) -> int:
    torch_out = np.load(args.torch_output)["output"]
    jittor_out = np.load(args.jittor_output)["output"]
    if torch_out.shape != jittor_out.shape:
        raise ValueError(f"shape mismatch: {torch_out.shape} vs {jittor_out.shape}")

    diff = np.abs(torch_out - jittor_out)
    rel = diff / np.maximum(np.abs(torch_out), 1e-8)
    stats = {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "max_rel": float(rel.max()),
        "mean_rel": float(rel.mean()),
        "torch_mean": float(torch_out.mean()),
        "jittor_mean": float(jittor_out.mean()),
        "torch_std": float(torch_out.std()),
        "jittor_std": float(jittor_out.std()),
    }

    for key, value in stats.items():
        print(f"{key}: {value:.8g}")

    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        import json

        with open(args.report, "w", encoding="utf-8") as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
        print(f"saved report: {args.report}")

    return 0 if stats["max_abs"] <= args.tolerance else 2


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_parser = subparsers.add_parser("make-inputs")
    make_parser.add_argument("--output", required=True)
    make_parser.add_argument("--batch-size", type=int, default=2)
    make_parser.add_argument("--seed", type=int, default=0)
    make_parser.set_defaults(func=make_inputs)

    torch_parser = subparsers.add_parser("torch-forward")
    torch_parser.add_argument("--inputs", required=True)
    torch_parser.add_argument("--output", required=True)
    torch_parser.add_argument("--teacher-pkl", default=DEFAULT_TEACHER)
    torch_parser.add_argument("--pytorch-root", default="/home/koishi/DMD2/DMD2-pytorch")
    torch_parser.add_argument("--device", default="cpu")
    torch_parser.set_defaults(func=torch_forward)

    jittor_parser = subparsers.add_parser("jittor-forward")
    jittor_parser.add_argument("--inputs", required=True)
    jittor_parser.add_argument("--output", required=True)
    jittor_parser.add_argument("--jittor-checkpoint", default=DEFAULT_JITTOR_CKPT)
    jittor_parser.add_argument(
        "--jittor-root",
        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    )
    jittor_parser.add_argument("--use-cuda", action="store_true")
    jittor_parser.set_defaults(func=jittor_forward)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--torch-output", required=True)
    compare_parser.add_argument("--jittor-output", required=True)
    compare_parser.add_argument("--report", default="")
    compare_parser.add_argument("--tolerance", type=float, default=2e-4)
    compare_parser.set_defaults(func=compare)

    return parser


def main(argv=None) -> int:
    args = create_argparser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
