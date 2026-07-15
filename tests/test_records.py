"""Smoke tests for experiment record utilities and tools."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def setup_import_path():
    this_file = Path(__file__).resolve()
    project_root = this_file.parents[1]
    code_dir = project_root / "code"
    tools_dir = project_root / "tools"

    for path in (
        project_root,
        code_dir,
        code_dir / "utils",
        tools_dir,
    ):
        path = str(path)
        if path not in sys.path:
            sys.path.insert(0, path)


setup_import_path()

from utils.image import save_image_grid
from utils.logger import JSONLLogger, read_jsonl, write_alignment_log
from utils.metric import fid_from_features, performance_record
from tools.compute_fid import main as compute_fid_main
from tools.plot_metrics import main as plot_metrics_main
from tools.visualize_samples import main as visualize_samples_main


def make_images(count=6, value=0):
    images = np.zeros([count, 8, 8, 3], dtype=np.uint8)
    images[..., 0] = value
    return images


def write_image_dir(path, images):
    os.makedirs(path, exist_ok=True)
    for index, image in enumerate(images):
        Image.fromarray(image).save(os.path.join(path, f"{index:03d}.png"))


def test_jsonl_and_performance_logs():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "train_metrics.jsonl")
        logger = JSONLLogger(path)
        logger.write({"step": 1, "loss": 2.0})
        perf = performance_record(
            step=1,
            batch_size=4,
            data_time=0.1,
            step_time=0.3,
        )
        logger.write(perf)
        records = read_jsonl(path)

    assert len(records) == 2
    assert records[0]["loss"] == 2.0
    assert abs(records[1]["samples_per_second"] - 10.0) < 1e-6


def test_alignment_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "pytorch_jittor_align.log")
        write_alignment_log(path, "missing keys: 0", summary={"ok": True})
        text = open(path, "r", encoding="utf-8").read()
    assert "missing keys: 0" in text
    assert '"ok": true' in text


def test_plot_metrics_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        metrics_path = os.path.join(tmpdir, "metrics.jsonl")
        curve_path = os.path.join(tmpdir, "loss_curve.svg")
        logger = JSONLLogger(metrics_path, include_timestamp=False)
        logger.write({"step": 1, "loss": 2.0, "samples_per_second": 4.0})
        logger.write({"step": 2, "loss": 1.0, "samples_per_second": 5.0})
        plot_metrics_main([metrics_path, curve_path, "--keys", "loss"])
        assert os.path.exists(curve_path)
        assert "<svg" in open(curve_path, "r", encoding="utf-8").read(256)


def test_fid_tool_and_visualization_tool():
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_dir = os.path.join(tmpdir, "pred")
        ref_dir = os.path.join(tmpdir, "ref")
        grid_path = os.path.join(tmpdir, "grid.svg")
        fid_path = os.path.join(tmpdir, "fid.json")

        write_image_dir(pred_dir, make_images(value=20))
        write_image_dir(ref_dir, make_images(value=20))

        visualize_samples_main([pred_dir, grid_path, "--nrow", "3", "--image-size", "8"])
        compute_fid_main([
            "--pred",
            pred_dir,
            "--ref",
            ref_dir,
            "--output",
            fid_path,
            "--image-size",
            "8",
        ])

        assert os.path.exists(grid_path)
        assert "<svg" in open(grid_path, "r", encoding="utf-8").read(256)
        assert os.path.exists(fid_path)
        assert abs(fid_from_features(np.zeros([2, 3]), np.zeros([2, 3]))) < 1e-9


def test_save_image_grid():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "samples.svg")
        save_image_grid(make_images(), path, nrow=3)
        assert os.path.exists(path)
        assert "<svg" in open(path, "r", encoding="utf-8").read(256)


def run_all_tests():
    tests = [
        test_jsonl_and_performance_logs,
        test_alignment_log,
        test_plot_metrics_tool,
        test_fid_tool_and_visualization_tool,
        test_save_image_grid,
    ]

    for test in tests:
        print(f"[RUN] {test.__name__}")
        test()
        print(f"[OK]  {test.__name__}")


if __name__ == "__main__":
    run_all_tests()
