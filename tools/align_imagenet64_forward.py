"""Cross-environment ImageNet-64 forward alignment for PyTorch and Jittor.

This tool is intentionally split into small modes because the local project
uses separate conda environments for the official PyTorch code and the Jittor
migration.  A typical run is:

1. make-input       create deterministic x/sigma/labels.
2. pytorch-forward  run the official PyTorch generator and save raw output.
3. jittor-forward   run the converted Jittor generator and save raw output.
4. compare          compute raw tensor differences.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from types import SimpleNamespace
from typing import Mapping, Optional, Sequence

import numpy as np


def project_root_from_file() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def add_project_paths(project_root: Optional[str] = None) -> str:
    project_root = project_root_from_file() if project_root is None else os.path.abspath(project_root)
    code_dir = os.path.join(project_root, "code")
    for path in (
        project_root,
        code_dir,
        os.path.join(code_dir, "models"),
        os.path.join(code_dir, "utils"),
        os.path.join(project_root, "tools"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    return project_root


def add_pytorch_paths(pytorch_root: str) -> str:
    pytorch_root = os.path.abspath(pytorch_root)
    for path in (
        pytorch_root,
        os.path.join(pytorch_root, "third_party", "edm"),
    ):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
    return pytorch_root


def ensure_parent(path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def parse_labels(value: str, batch_size: int) -> Optional[np.ndarray]:
    if value is None or value.strip() == "":
        return None
    labels = np.asarray([int(part.strip()) for part in value.split(",") if part.strip()], dtype=np.int64)
    if labels.shape[0] != batch_size:
        raise ValueError(f"--labels must contain exactly {batch_size} ids, got {labels.shape[0]}")
    return labels


def make_one_hot(class_ids: np.ndarray, label_dim: int) -> np.ndarray:
    if np.any(class_ids < 0) or np.any(class_ids >= label_dim):
        raise ValueError(f"class ids must be in [0, {label_dim})")
    return np.eye(label_dim, dtype=np.float32)[class_ids.astype(np.int64)]


def array_summary(value: np.ndarray) -> dict:
    value = np.asarray(value)
    finite = np.isfinite(value)
    summary = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "numel": int(value.size),
        "finite": bool(finite.all()),
    }
    if value.size:
        summary.update(
            {
                "min": float(np.nanmin(value)),
                "max": float(np.nanmax(value)),
                "mean": float(np.nanmean(value)),
                "std": float(np.nanstd(value)),
            }
        )
    return summary


def make_alignment_input(args) -> dict:
    rng = np.random.RandomState(args.seed)
    batch_size = int(args.batch_size)
    noise = rng.randn(batch_size, 3, args.resolution, args.resolution).astype(np.float32)
    sigma = np.full([batch_size], float(args.conditioning_sigma), dtype=np.float32)
    class_ids = parse_labels(args.labels, batch_size)
    if class_ids is None:
        class_ids = rng.randint(0, args.label_dim, size=[batch_size]).astype(np.int64)
    labels = make_one_hot(class_ids, args.label_dim)
    x = noise * float(args.conditioning_sigma)

    payload = {
        "noise": noise,
        "x": x.astype(np.float32),
        "sigma": sigma,
        "labels": labels,
        "class_ids": class_ids.astype(np.int64),
        "seed": np.asarray([args.seed], dtype=np.int64),
        "conditioning_sigma": np.asarray([args.conditioning_sigma], dtype=np.float32),
    }
    ensure_parent(args.input_npz)
    np.savez_compressed(args.input_npz, **payload)
    return {
        "mode": "make-input",
        "input_npz": os.path.abspath(args.input_npz),
        "seed": args.seed,
        "batch_size": batch_size,
        "resolution": args.resolution,
        "label_dim": args.label_dim,
        "conditioning_sigma": args.conditioning_sigma,
        "class_ids": class_ids.tolist(),
        "x": array_summary(payload["x"]),
        "sigma": array_summary(payload["sigma"]),
        "labels": array_summary(payload["labels"]),
    }


def load_input_npz(path: str) -> dict:
    data = np.load(path)
    required = ("x", "sigma", "labels", "class_ids")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"input npz is missing keys: {missing}")
    return {key: data[key] for key in data.files}


def load_pytorch_state(path: str, pytorch_root: str, source_key: Optional[str] = None) -> Mapping[str, object]:
    add_project_paths()
    try:
        from tools.convert_pytorch_ckpt import load_pytorch_checkpoint
    except ImportError:
        from convert_pytorch_ckpt import load_pytorch_checkpoint

    return load_pytorch_checkpoint(
        path,
        input_format="auto",
        source_key=source_key,
        pytorch_root=pytorch_root,
    )


def build_pytorch_imagenet_generator(args):
    add_pytorch_paths(args.pytorch_root)
    from third_party.edm.training.networks import EDMPrecond

    config = dict(
        img_resolution=args.resolution,
        img_channels=3,
        label_dim=args.label_dim,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=args.sigma_data,
        model_type="DhariwalUNet",
        augment_dim=0,
        model_channels=192,
        channel_mult=[1, 2, 3, 4],
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=[32, 16, 8],
        dropout=0.0,
        label_dropout=0,
    )
    return EDMPrecond(**config)


def run_pytorch_forward(args) -> dict:
    import torch

    if args.use_cuda and not torch.cuda.is_available():
        raise RuntimeError("--use-cuda was requested, but torch.cuda.is_available() is false")

    input_data = load_input_npz(args.input_npz)
    state = load_pytorch_state(args.pytorch_ckpt, args.pytorch_root, args.source_key)
    model = build_pytorch_imagenet_generator(args)
    load_result = model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if args.use_cuda else "cpu")
    model = model.to(device).eval()

    x = torch.from_numpy(input_data["x"]).to(device)
    sigma = torch.from_numpy(input_data["sigma"]).to(device)
    labels = torch.from_numpy(input_data["labels"]).to(device)

    with torch.no_grad():
        output = model(x, sigma, labels)
        output_np = output.detach().cpu().numpy().astype(np.float32)

    ensure_parent(args.output_npz)
    np.savez_compressed(
        args.output_npz,
        output=output_np,
        class_ids=input_data["class_ids"],
    )
    return {
        "mode": "pytorch-forward",
        "framework": "pytorch",
        "checkpoint": os.path.abspath(args.pytorch_ckpt),
        "input_npz": os.path.abspath(args.input_npz),
        "output_npz": os.path.abspath(args.output_npz),
        "device": str(device),
        "load_result": str(load_result),
        "output": array_summary(output_np),
    }


def select_state(obj, state_key: Optional[str] = None):
    if state_key:
        for part in state_key.split("."):
            obj = obj[part]
        return obj
    if isinstance(obj, Mapping) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def load_converted_state(path: str, state_key: Optional[str] = None):
    with open(path, "rb") as handle:
        try:
            obj = pickle.load(handle)
        except Exception:
            handle.seek(0)
            add_project_paths()
            import jittor as jt

            obj = jt.load(path)
    return select_state(obj, state_key)


def make_jittor_args(args) -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name="imagenet",
        config_name="imagenet",
        resolution=args.resolution,
        label_dim=args.label_dim,
        use_fp16=False,
        sigma_data=args.sigma_data,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
    )


def to_jittor_state(state: Mapping[str, object]):
    import jittor as jt

    return {
        key: value if isinstance(value, jt.Var) else jt.array(np.asarray(value))
        for key, value in state.items()
    }


def run_jittor_forward(args) -> dict:
    project_root = add_project_paths(args.jittor_root)
    import jittor as jt

    jt.flags.use_cuda = 1 if args.use_cuda else 0
    if hasattr(jt, "set_global_seed"):
        jt.set_global_seed(args.seed)

    from models.diffusion import get_edm_network

    input_data = load_input_npz(args.input_npz)
    model = get_edm_network(args=make_jittor_args(args))
    state = load_converted_state(args.converted_ckpt, args.state_key)
    load_result = model.load_state_dict(to_jittor_state(state))
    model.eval()

    with jt.no_grad():
        output = model(
            jt.array(input_data["x"]),
            jt.array(input_data["sigma"]),
            jt.array(input_data["labels"]),
        )
        jt.sync_all()
        output_np = np.asarray(output.numpy()).astype(np.float32)

    ensure_parent(args.output_npz)
    np.savez_compressed(
        args.output_npz,
        output=output_np,
        class_ids=input_data["class_ids"],
    )
    return {
        "mode": "jittor-forward",
        "framework": "jittor",
        "checkpoint": os.path.abspath(args.converted_ckpt),
        "input_npz": os.path.abspath(args.input_npz),
        "output_npz": os.path.abspath(args.output_npz),
        "project_root": project_root,
        "use_cuda": bool(args.use_cuda),
        "load_result": str(load_result),
        "output": array_summary(output_np),
    }


def load_output(path: str) -> np.ndarray:
    data = np.load(path)
    if "output" not in data:
        raise KeyError(f"{path} does not contain an 'output' array")
    return np.asarray(data["output"], dtype=np.float32)


def compute_diff_report(pytorch_output: np.ndarray, jittor_output: np.ndarray, rtol: float, atol: float) -> dict:
    if pytorch_output.shape != jittor_output.shape:
        raise ValueError(
            f"output shape mismatch: pytorch={pytorch_output.shape}, jittor={jittor_output.shape}"
        )

    diff = np.abs(pytorch_output - jittor_output)
    denom = np.maximum(np.abs(pytorch_output), np.asarray(1e-12, dtype=np.float32))
    rel = diff / denom
    return {
        "shape": list(pytorch_output.shape),
        "numel": int(pytorch_output.size),
        "rtol": float(rtol),
        "atol": float(atol),
        "allclose": bool(np.allclose(pytorch_output, jittor_output, rtol=rtol, atol=atol)),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "median_abs_diff": float(np.median(diff)),
        "rmse": float(np.sqrt(np.mean((pytorch_output - jittor_output) ** 2))),
        "max_rel_diff": float(rel.max()),
        "mean_rel_diff": float(rel.mean()),
        "pytorch_output": array_summary(pytorch_output),
        "jittor_output": array_summary(jittor_output),
        "diff": array_summary(diff),
    }


def format_diff_report(report: Mapping[str, object]) -> str:
    return "\n".join(
        [
            "# ImageNet64 PyTorch-Jittor forward alignment",
            f"shape: {report['shape']}",
            f"numel: {report['numel']}",
            f"allclose(rtol={report['rtol']}, atol={report['atol']}): {report['allclose']}",
            f"max_abs_diff: {report['max_abs_diff']:.8e}",
            f"mean_abs_diff: {report['mean_abs_diff']:.8e}",
            f"median_abs_diff: {report['median_abs_diff']:.8e}",
            f"rmse: {report['rmse']:.8e}",
            f"max_rel_diff: {report['max_rel_diff']:.8e}",
            f"mean_rel_diff: {report['mean_rel_diff']:.8e}",
        ]
    )


def write_json(path: str, data: Mapping[str, object]) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_text(path: str, text: str) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text.rstrip())
        handle.write("\n")


def compare_outputs(args) -> dict:
    pytorch_output = load_output(args.pytorch_output_npz)
    jittor_output = load_output(args.jittor_output_npz)
    report = compute_diff_report(pytorch_output, jittor_output, args.rtol, args.atol)
    report.update(
        {
            "mode": "compare",
            "pytorch_output_npz": os.path.abspath(args.pytorch_output_npz),
            "jittor_output_npz": os.path.abspath(args.jittor_output_npz),
        }
    )
    if args.report_json:
        write_json(args.report_json, report)
    report_text = format_diff_report(report)
    if args.log_path:
        write_text(args.log_path, report_text)
    print(report_text)
    return report


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("make-input", "pytorch-forward", "jittor-forward", "compare"),
    )
    parser.add_argument("--input-npz", default="outputs/alignment/imagenet64_forward/input.npz")
    parser.add_argument("--output-npz", default="outputs/alignment/imagenet64_forward/output.npz")
    parser.add_argument("--pytorch-output-npz", default="outputs/alignment/imagenet64_forward/pytorch_output.npz")
    parser.add_argument("--jittor-output-npz", default="outputs/alignment/imagenet64_forward/jittor_output.npz")
    parser.add_argument("--report-json", default="outputs/alignment/imagenet64_forward/report.json")
    parser.add_argument("--log-path", default="logs/imagenet64_forward_alignment.log")

    parser.add_argument("--pytorch-root", default="../DMD2-pytorch")
    parser.add_argument("--jittor-root", default=None)
    parser.add_argument("--pytorch-ckpt", default="../DMD2-pytorch/checkpoints/imagenet_fid151/pytorch_model.bin")
    parser.add_argument("--converted-ckpt", default="checkpoints/imagenet64_demo/generator_jittor.pkl")
    parser.add_argument("--source-key", default=None)
    parser.add_argument("--state-key", default=None)

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--label-dim", type=int, default=1000)
    parser.add_argument("--conditioning-sigma", type=float, default=80.0)
    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--labels", default="")
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=2e-4)
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = create_argparser().parse_args(argv)

    if args.mode == "make-input":
        summary = make_alignment_input(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.mode == "pytorch-forward":
        summary = run_pytorch_forward(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.mode == "jittor-forward":
        summary = run_jittor_forward(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.mode == "compare":
        report = compare_outputs(args)
        if args.strict and not report["allclose"]:
            return 1
        return 0

    raise ValueError(f"unsupported mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
