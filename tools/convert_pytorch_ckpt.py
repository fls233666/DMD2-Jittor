"""Convert official PyTorch DMD2/EDM checkpoints to Jittor state dicts.

The official EDM code stores modules under ``ModuleDict`` keys such as
``model.enc.64x64_conv.weight``.  The Jittor migration registers the same
modules as normal attributes, e.g. ``model.enc_0_64x64_conv.weight``.  This
tool converts those names and writes a plain pickle state dict that can be
loaded by ``jt.load`` or Python ``pickle.load``.
"""

from __future__ import annotations

import argparse
import os
import pickle
import re
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


CLS_LAYER_MAP = {
    "0": "conv0",
    "1": "norm0",
    "3": "conv1",
    "4": "norm1",
    "6": "conv2",
}


@dataclass
class ConversionReport:
    total_keys: int = 0
    converted_keys: int = 0
    dropped_keys: List[str] = field(default_factory=list)
    unexpected_keys: List[Tuple[str, str]] = field(default_factory=list)
    missing_keys: List[str] = field(default_factory=list)
    shape_mismatches: List[Tuple[str, str, Tuple[int, ...], Tuple[int, ...]]] = field(
        default_factory=list
    )
    duplicate_keys: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.unexpected_keys
            or self.missing_keys
            or self.shape_mismatches
            or self.duplicate_keys
        )


def parse_int_list(value: str) -> Tuple[int, ...]:
    if not value:
        return tuple()
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def build_dhariwal_name_maps(
    img_resolution: int = 64,
    channel_mult: Sequence[int] = (1, 2, 3, 4),
    num_blocks: int = 3,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return official ModuleDict name -> Jittor attribute name maps."""

    channel_mult = tuple(channel_mult)

    enc_map: Dict[str, str] = {}
    enc_index = 0
    for level, _ in enumerate(channel_mult):
        res = img_resolution >> level
        if level == 0:
            name = f"{res}x{res}_conv"
            enc_map[name] = f"enc_{enc_index}_{name}"
            enc_index += 1
        else:
            name = f"{res}x{res}_down"
            enc_map[name] = f"enc_{enc_index}_{name}"
            enc_index += 1

        for block_index in range(num_blocks):
            name = f"{res}x{res}_block{block_index}"
            enc_map[name] = f"enc_{enc_index}_{name}"
            enc_index += 1

    dec_map: Dict[str, str] = {}
    dec_index = 0
    for level, _ in reversed(list(enumerate(channel_mult))):
        res = img_resolution >> level
        if level == len(channel_mult) - 1:
            for name in (f"{res}x{res}_in0", f"{res}x{res}_in1"):
                dec_map[name] = f"dec_{dec_index}_{name}"
                dec_index += 1
        else:
            name = f"{res}x{res}_up"
            dec_map[name] = f"dec_{dec_index}_{name}"
            dec_index += 1

        for block_index in range(num_blocks + 1):
            name = f"{res}x{res}_block{block_index}"
            dec_map[name] = f"dec_{dec_index}_{name}"
            dec_index += 1

    return enc_map, dec_map


def build_song_name_maps(
    img_resolution: int = 32,
    channel_mult: Sequence[int] = (2, 2, 2),
    num_blocks: int = 4,
    encoder_type: str = "standard",
    decoder_type: str = "standard",
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Return official SongUNet ModuleDict name -> Jittor attribute maps."""

    channel_mult = tuple(channel_mult)

    enc_map: Dict[str, str] = {}
    enc_index = 0
    for level, _ in enumerate(channel_mult):
        res = img_resolution >> level
        if level == 0:
            name = f"{res}x{res}_conv"
            enc_map[name] = f"enc_{enc_index}_{name}"
            enc_index += 1
        else:
            name = f"{res}x{res}_down"
            enc_map[name] = f"enc_{enc_index}_{name}"
            enc_index += 1
            if encoder_type == "skip":
                for name in (f"{res}x{res}_aux_down", f"{res}x{res}_aux_skip"):
                    enc_map[name] = f"enc_{enc_index}_{name}"
                    enc_index += 1
            if encoder_type == "residual":
                name = f"{res}x{res}_aux_residual"
                enc_map[name] = f"enc_{enc_index}_{name}"
                enc_index += 1

        for block_index in range(num_blocks):
            name = f"{res}x{res}_block{block_index}"
            enc_map[name] = f"enc_{enc_index}_{name}"
            enc_index += 1

    dec_map: Dict[str, str] = {}
    dec_index = 0
    for level, _ in reversed(list(enumerate(channel_mult))):
        res = img_resolution >> level
        if level == len(channel_mult) - 1:
            for name in (f"{res}x{res}_in0", f"{res}x{res}_in1"):
                dec_map[name] = f"dec_{dec_index}_{name}"
                dec_index += 1
        else:
            name = f"{res}x{res}_up"
            dec_map[name] = f"dec_{dec_index}_{name}"
            dec_index += 1

        for block_index in range(num_blocks + 1):
            name = f"{res}x{res}_block{block_index}"
            dec_map[name] = f"dec_{dec_index}_{name}"
            dec_index += 1

        if decoder_type == "skip" or level == 0:
            if decoder_type == "skip" and level < len(channel_mult) - 1:
                name = f"{res}x{res}_aux_up"
                dec_map[name] = f"dec_{dec_index}_{name}"
                dec_index += 1
            for name in (f"{res}x{res}_aux_norm", f"{res}x{res}_aux_conv"):
                dec_map[name] = f"dec_{dec_index}_{name}"
                dec_index += 1

    return enc_map, dec_map


def map_official_key(
    key: str,
    enc_map: Optional[Mapping[str, str]] = None,
    dec_map: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    """Map one official PyTorch key to the Jittor key.

    Returns ``None`` for deterministic buffers that are intentionally not
    checkpointed in the Jittor modules, such as EDM resampling filters.
    """

    if key.endswith(".resample_filter"):
        return None

    if key == "karras_sigmas":
        return "karras_sigmas_buffer"

    cls_match = re.match(r"^(?P<prefix>.*?cls_pred_branch)\.(?P<layer>\d+)\.(?P<suffix>.+)$", key)
    if cls_match:
        layer = cls_match.group("layer")
        if layer in CLS_LAYER_MAP:
            return (
                f"{cls_match.group('prefix')}.{CLS_LAYER_MAP[layer]}."
                f"{cls_match.group('suffix')}"
            )

    enc_map = enc_map or {}
    dec_map = dec_map or {}
    module_match = re.match(
        r"^(?P<prefix>.*?)(?P<section>enc|dec)\.(?P<name>[^.]+)\.(?P<suffix>.+)$",
        key,
    )
    if module_match:
        section = module_match.group("section")
        name = module_match.group("name")
        name_map = enc_map if section == "enc" else dec_map
        if name in name_map:
            return (
                f"{module_match.group('prefix')}{name_map[name]}."
                f"{module_match.group('suffix')}"
            )

    return key


def shape_of(value) -> Tuple[int, ...]:
    if isinstance(value, (list, tuple)) and all(
        isinstance(dim, (int, np.integer)) for dim in value
    ):
        return tuple(int(dim) for dim in value)
    if hasattr(value, "shape"):
        return tuple(int(dim) for dim in value.shape)
    return tuple(np.asarray(value).shape)


def tensor_to_numpy(value) -> np.ndarray:
    """Convert torch/Jittor/numpy values to a CPU numpy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "contiguous"):
        value = value.contiguous()

    if hasattr(value, "numpy"):
        try:
            return np.asarray(value.numpy())
        except TypeError:
            if hasattr(value, "float"):
                return np.asarray(value.float().numpy())

    return np.asarray(value)


def state_shapes(state_dict: Mapping[str, object]) -> Dict[str, Tuple[int, ...]]:
    return {key: shape_of(value) for key, value in state_dict.items()}


def compare_with_target(
    converted_state: Mapping[str, object],
    target_shapes: Mapping[str, Tuple[int, ...]],
) -> ConversionReport:
    report = ConversionReport(
        total_keys=len(converted_state),
        converted_keys=len(converted_state),
    )
    converted_keys = set(converted_state.keys())
    target_keys = set(target_shapes.keys())

    report.unexpected_keys = [(key, key) for key in sorted(converted_keys - target_keys)]
    report.missing_keys = sorted(target_keys - converted_keys)

    for key in sorted(converted_keys & target_keys):
        source_shape = shape_of(converted_state[key])
        target_shape = tuple(target_shapes[key])
        if source_shape != target_shape:
            report.shape_mismatches.append((key, key, source_shape, target_shape))

    return report


def convert_state_dict(
    source_state: Mapping[str, object],
    target_shapes: Optional[Mapping[str, Tuple[int, ...]]] = None,
    img_resolution: int = 64,
    channel_mult: Sequence[int] = (1, 2, 3, 4),
    num_blocks: int = 3,
    architecture: str = "dhariwal",
    encoder_type: str = "standard",
    decoder_type: str = "standard",
    keep_unmatched: bool = False,
) -> Tuple[OrderedDict, ConversionReport]:
    if architecture == "song":
        enc_map, dec_map = build_song_name_maps(
            img_resolution=img_resolution,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
            encoder_type=encoder_type,
            decoder_type=decoder_type,
        )
    else:
        enc_map, dec_map = build_dhariwal_name_maps(
            img_resolution=img_resolution,
            channel_mult=channel_mult,
            num_blocks=num_blocks,
        )

    converted = OrderedDict()
    report = ConversionReport(total_keys=len(source_state))
    target_key_set = set(target_shapes.keys()) if target_shapes is not None else None

    for source_key, source_value in source_state.items():
        mapped_key = map_official_key(source_key, enc_map=enc_map, dec_map=dec_map)

        if mapped_key is None:
            report.dropped_keys.append(source_key)
            continue

        if target_key_set is not None and mapped_key not in target_key_set:
            report.unexpected_keys.append((source_key, mapped_key))
            if not keep_unmatched:
                continue

        source_array = tensor_to_numpy(source_value)
        source_shape = shape_of(source_array)

        if target_shapes is not None and mapped_key in target_shapes:
            target_shape = tuple(target_shapes[mapped_key])
            if source_shape != target_shape:
                report.shape_mismatches.append(
                    (source_key, mapped_key, source_shape, target_shape)
                )
                continue

        if mapped_key in converted:
            report.duplicate_keys.append((source_key, mapped_key))
            continue

        converted[mapped_key] = source_array

    report.converted_keys = len(converted)
    if target_key_set is not None:
        report.missing_keys = sorted(target_key_set - set(converted.keys()))

    return converted, report


def _add_pytorch_paths(pytorch_root: Optional[str]) -> None:
    if not pytorch_root:
        return

    pytorch_root = os.path.abspath(pytorch_root)
    paths = [
        os.path.join(pytorch_root, "third_party", "edm"),
        pytorch_root,
    ]
    for path in reversed(paths):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)


def _looks_like_state_dict(obj) -> bool:
    if not isinstance(obj, Mapping):
        return False
    if len(obj) == 0:
        return True
    return all(hasattr(value, "shape") or np.isscalar(value) for value in obj.values())


def extract_state_dict(obj, source_key: Optional[str] = None) -> Mapping[str, object]:
    if source_key:
        if not isinstance(obj, Mapping) or source_key not in obj:
            raise KeyError(f"source key {source_key!r} not found")
        obj = obj[source_key]

    if hasattr(obj, "state_dict"):
        obj = obj.state_dict()

    if _looks_like_state_dict(obj):
        return obj

    if isinstance(obj, Mapping):
        for key in ("state_dict", "ema", "model"):
            if key in obj:
                return extract_state_dict(obj[key])

    raise TypeError(f"Cannot extract a state dict from object of type {type(obj)!r}.")


def load_pytorch_checkpoint(
    path: str,
    input_format: str = "auto",
    source_key: Optional[str] = None,
    pytorch_root: Optional[str] = None,
) -> Mapping[str, object]:
    """Load an official PyTorch checkpoint or EDM pkl and return its state dict."""

    _add_pytorch_paths(pytorch_root)

    if input_format == "auto":
        input_format = "pickle" if path.endswith(".pkl") else "torch"

    if input_format == "pickle":
        with open(path, "rb") as handle:
            obj = pickle.load(handle)
    elif input_format == "torch":
        import torch

        try:
            obj = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported input_format: {input_format}")

    return extract_state_dict(obj, source_key=source_key)


def load_serialized_state(path: str, state_key: Optional[str] = None):
    with open(path, "rb") as handle:
        state = pickle.load(handle)
    if state_key:
        state = state[state_key]
    return state


def save_converted_state(
    path: str,
    state_dict: Mapping[str, object],
    metadata: Optional[Mapping[str, object]] = None,
    bundle: bool = False,
) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    payload = state_dict
    if bundle:
        payload = {
            "state_dict": state_dict,
            "metadata": {} if metadata is None else dict(metadata),
        }

    with open(path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def format_report(report: ConversionReport, limit: int = 12) -> str:
    lines = [
        f"total input keys: {report.total_keys}",
        f"converted keys: {report.converted_keys}",
        f"dropped deterministic keys: {len(report.dropped_keys)}",
        f"unexpected keys: {len(report.unexpected_keys)}",
        f"missing keys: {len(report.missing_keys)}",
        f"shape mismatches: {len(report.shape_mismatches)}",
        f"duplicate mapped keys: {len(report.duplicate_keys)}",
    ]

    def add_list(title: str, values: Iterable[object]) -> None:
        values = list(values)
        if not values:
            return
        lines.append(f"{title}:")
        for value in values[:limit]:
            lines.append(f"  {value}")
        if len(values) > limit:
            lines.append(f"  ... {len(values) - limit} more")

    add_list("dropped", report.dropped_keys)
    add_list("unexpected", report.unexpected_keys)
    add_list("missing", report.missing_keys)
    add_list("shape mismatch", report.shape_mismatches)
    add_list("duplicate", report.duplicate_keys)
    return "\n".join(lines)


def create_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Official PyTorch checkpoint or EDM pkl.")
    parser.add_argument("output", help="Output pickle/Jittor-loadable state dict.")
    parser.add_argument(
        "--input-format",
        choices=("auto", "torch", "pickle"),
        default="auto",
        help="Checkpoint reader. Defaults to pickle for .pkl, torch otherwise.",
    )
    parser.add_argument(
        "--source-key",
        default=None,
        help="Optional key inside a checkpoint dict, e.g. ema or state_dict.",
    )
    parser.add_argument(
        "--pytorch-root",
        default=None,
        help="Official DMD2-pytorch root, needed when unpickling EDM pkl files.",
    )
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--channel-mult", default="1,2,3,4")
    parser.add_argument("--num-blocks", type=int, default=3)
    parser.add_argument(
        "--architecture",
        choices=("dhariwal", "song"),
        default="dhariwal",
        help="Underlying official ModuleDict layout to map.",
    )
    parser.add_argument("--encoder-type", default="standard")
    parser.add_argument("--decoder-type", default="standard")
    parser.add_argument(
        "--target-state",
        default=None,
        help="Optional Jittor/converted state dict used for key and shape validation.",
    )
    parser.add_argument(
        "--target-state-key",
        default=None,
        help="Optional key inside --target-state.",
    )
    parser.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep mapped keys that are not present in --target-state.",
    )
    parser.add_argument("--bundle", action="store_true", help="Save metadata wrapper.")
    parser.add_argument("--dry-run", action="store_true", help="Do not write output.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero when validation reports missing/unexpected/mismatched keys.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_argparser()
    args = parser.parse_args(argv)

    channel_mult = parse_int_list(args.channel_mult)
    source_state = load_pytorch_checkpoint(
        args.input,
        input_format=args.input_format,
        source_key=args.source_key,
        pytorch_root=args.pytorch_root,
    )

    target_shapes = None
    if args.target_state:
        target_state = load_serialized_state(args.target_state, args.target_state_key)
        target_shapes = state_shapes(target_state)

    converted, report = convert_state_dict(
        source_state=source_state,
        target_shapes=target_shapes,
        img_resolution=args.resolution,
        channel_mult=channel_mult,
        num_blocks=args.num_blocks,
        architecture=args.architecture,
        encoder_type=args.encoder_type,
        decoder_type=args.decoder_type,
        keep_unmatched=args.keep_unmatched,
    )

    if not args.dry_run:
        metadata = {
            "source": os.path.abspath(args.input),
            "resolution": args.resolution,
            "channel_mult": channel_mult,
            "num_blocks": args.num_blocks,
            "architecture": args.architecture,
            "encoder_type": args.encoder_type,
            "decoder_type": args.decoder_type,
        }
        save_converted_state(
            args.output,
            converted,
            metadata=metadata,
            bundle=args.bundle,
        )

    print(format_report(report))
    if args.strict and not report.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
