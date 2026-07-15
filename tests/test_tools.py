"""Smoke tests for tools checkpoint conversion helpers.

Run from the project root:

    python tests/test_tools.py
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np


def setup_import_path():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    for path in (project_root, project_root / "tools"):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


setup_import_path()

from tools.convert_pytorch_ckpt import (
    build_dhariwal_name_maps,
    compare_with_target,
    convert_state_dict,
    load_serialized_state,
    map_official_key,
    save_converted_state,
    state_shapes,
)
from tools.align_imagenet64_forward import (
    compute_diff_report,
    make_alignment_input,
)


def test_dhariwal_key_mapping():
    enc_map, dec_map = build_dhariwal_name_maps(
        img_resolution=64,
        channel_mult=(1, 2, 3, 4),
        num_blocks=3,
    )

    assert enc_map["64x64_conv"] == "enc_0_64x64_conv"
    assert enc_map["32x32_down"] == "enc_4_32x32_down"
    assert dec_map["64x64_block3"] == "dec_20_64x64_block3"

    assert (
        map_official_key("model.enc.64x64_conv.weight", enc_map, dec_map)
        == "model.enc_0_64x64_conv.weight"
    )
    assert (
        map_official_key("real_unet.model.dec.64x64_block3.skip.bias", enc_map, dec_map)
        == "real_unet.model.dec_20_64x64_block3.skip.bias"
    )
    assert map_official_key("karras_sigmas", enc_map, dec_map) == "karras_sigmas_buffer"
    assert (
        map_official_key("cls_pred_branch.3.weight", enc_map, dec_map)
        == "cls_pred_branch.conv1.weight"
    )
    assert map_official_key("model.enc.32x32_down.conv0.resample_filter", enc_map, dec_map) is None


def test_convert_state_dict_with_target_shapes():
    source = {
        "model.enc.64x64_conv.weight": np.ones([2, 3, 3, 3], dtype=np.float32),
        "model.enc.64x64_conv.bias": np.ones([2], dtype=np.float32),
        "model.enc.32x32_down.conv0.resample_filter": np.ones([1, 1, 2, 2], dtype=np.float32),
        "cls_pred_branch.0.weight": np.ones([4, 4, 4, 4], dtype=np.float32),
        "unused.weight": np.ones([1], dtype=np.float32),
    }
    target = {
        "model.enc_0_64x64_conv.weight": np.zeros([2, 3, 3, 3], dtype=np.float32),
        "model.enc_0_64x64_conv.bias": np.zeros([2], dtype=np.float32),
        "cls_pred_branch.conv0.weight": np.zeros([4, 4, 4, 4], dtype=np.float32),
    }

    converted, report = convert_state_dict(
        source,
        target_shapes=state_shapes(target),
        img_resolution=64,
        channel_mult=(1, 2, 3, 4),
        num_blocks=3,
    )

    assert set(converted) == set(target)
    assert report.converted_keys == 3
    assert report.dropped_keys == ["model.enc.32x32_down.conv0.resample_filter"]
    assert report.unexpected_keys == [("unused.weight", "unused.weight")]
    assert report.missing_keys == []
    assert report.shape_mismatches == []


def test_compare_with_target_reports_mismatch():
    converted = {
        "a": np.zeros([2, 3], dtype=np.float32),
        "extra": np.zeros([1], dtype=np.float32),
    }
    target_shapes = {
        "a": (2, 4),
        "missing": (1,),
    }
    report = compare_with_target(converted, target_shapes)
    assert report.unexpected_keys == [("extra", "extra")]
    assert report.missing_keys == ["missing"]
    assert report.shape_mismatches == [("a", "a", (2, 3), (2, 4))]


def test_state_shapes_accepts_shape_index():
    shape_index = {
        "a": (2, 3),
        "b": [4],
    }
    assert state_shapes(shape_index) == {
        "a": (2, 3),
        "b": (4,),
    }


def test_serialization_roundtrip():
    state = {
        "model.enc_0_64x64_conv.weight": np.ones([2, 3, 3, 3], dtype=np.float32),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "converted.pkl")
        save_converted_state(path, state)
        loaded = load_serialized_state(path)
    assert list(loaded.keys()) == list(state.keys())
    assert np.allclose(loaded["model.enc_0_64x64_conv.weight"], state["model.enc_0_64x64_conv.weight"])


def test_imagenet64_forward_input_and_diff_report():
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.npz")
        args = type(
            "Args",
            (),
            {
                "input_npz": input_path,
                "seed": 123,
                "batch_size": 2,
                "resolution": 8,
                "label_dim": 10,
                "conditioning_sigma": 80.0,
                "labels": "3,7",
            },
        )()
        summary = make_alignment_input(args)
        data = np.load(input_path)

    assert summary["class_ids"] == [3, 7]
    assert data["x"].shape == (2, 3, 8, 8)
    assert data["sigma"].tolist() == [80.0, 80.0]
    assert data["labels"].shape == (2, 10)
    assert np.allclose(data["labels"][0], np.eye(10, dtype=np.float32)[3])

    reference = np.ones([2, 3, 4, 4], dtype=np.float32)
    candidate = reference + np.float32(1e-6)
    report = compute_diff_report(reference, candidate, rtol=1e-4, atol=1e-5)
    assert report["allclose"] is True
    assert report["max_abs_diff"] > 0
    assert report["shape"] == [2, 3, 4, 4]


def run_all_tests():
    tests = [
        test_dhariwal_key_mapping,
        test_convert_state_dict_with_target_shapes,
        test_compare_with_target_reports_mismatch,
        test_state_shapes_accepts_shape_index,
        test_serialization_roundtrip,
        test_imagenet64_forward_input_and_diff_report,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        print(f"[OK]  {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
