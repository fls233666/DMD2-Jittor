"""Plot loss and performance curves from JSONL/CSV metric logs."""

import argparse
import csv
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

from utils.logger import read_jsonl, write_json


def read_records(path):
    if path.endswith(".jsonl"):
        return read_jsonl(path)

    if path.endswith(".csv"):
        with open(path, "r", encoding="utf-8") as handle:
            return [
                {
                    key: _parse_value(value)
                    for key, value in row.items()
                }
                for row in csv.DictReader(handle)
            ]

    raise ValueError(f"Unsupported metric log format: {path}")


def _parse_value(value):
    if value == "":
        return value
    try:
        return float(value)
    except ValueError:
        return value


def scalar_series(records, names=None):
    if names is None or len(names) == 0:
        ignore = {"time", "step", "batch_size"}
        names = []
        for record in records:
            for key, value in record.items():
                if key in ignore:
                    continue
                if isinstance(value, (int, float)) and key not in names:
                    names.append(key)

    x = [record.get("step", index + 1) for index, record in enumerate(records)]
    series = {}
    for name in names:
        ys = []
        xs = []
        for index, record in enumerate(records):
            value = record.get(name)
            if isinstance(value, (int, float)):
                xs.append(record.get("step", index + 1))
                ys.append(value)
        if ys:
            series[name] = (np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64))
    return x, series


def plot_series(records, output_path, names=None, title=None):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, series = scalar_series(records, names=names)
    if not series:
        raise ValueError("No scalar series found to plot.")

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    for name, (x, y) in series.items():
        plt.plot(x, y, label=name)
    plt.xlabel("step")
    plt.ylabel("value")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_kwargs = {}
    if str(output_path).lower().endswith(".svg"):
        save_kwargs["format"] = "svg"
    else:
        save_kwargs["dpi"] = 150
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    return output_path


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Metric JSONL/CSV log.")
    parser.add_argument("output", help="Output curve SVG.")
    parser.add_argument(
        "--keys",
        default="",
        help="Comma-separated metric names. Defaults to all scalar metrics except step/time.",
    )
    parser.add_argument("--title", default=None)
    parser.add_argument("--summary-json", default=None)
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    records = read_records(args.input)
    keys = [key.strip() for key in args.keys.split(",") if key.strip()]
    output = plot_series(records, args.output, names=keys, title=args.title)

    if args.summary_json:
        _, series = scalar_series(records, names=keys)
        summary = {
            name: {
                "last": float(values[-1]),
                "min": float(values.min()),
                "max": float(values.max()),
                "mean": float(values.mean()),
            }
            for name, (_, values) in series.items()
        }
        write_json(args.summary_json, summary)

    print(f"saved curve: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
