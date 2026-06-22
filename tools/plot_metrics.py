"""Plot loss and performance curves from JSONL/CSV metric logs."""

import argparse
import csv
import html
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
    _, series = scalar_series(records, names=names)
    if not series:
        raise ValueError("No scalar series found to plot.")

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    try:
        import matplotlib
    except ModuleNotFoundError:
        return plot_series_svg(series, output_path, title=title)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def _scale(value, src_min, src_max, dst_min, dst_max):
    if abs(src_max - src_min) < 1e-12:
        return (dst_min + dst_max) * 0.5
    ratio = (value - src_min) / (src_max - src_min)
    return dst_min + ratio * (dst_max - dst_min)


def plot_series_svg(series, output_path, title=None):
    # Lightweight SVG fallback for environments without matplotlib.
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    width, height = 900, 520
    left, right, top, bottom = 80, 30, 60, 80
    plot_left = left
    plot_right = width - right
    plot_top = top
    plot_bottom = height - bottom

    all_x = np.concatenate([x for x, _ in series.values()])
    all_y = np.concatenate([y for _, y in series.values()])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    if abs(y_max - y_min) < 1e-12:
        pad = max(abs(y_min) * 0.1, 1.0)
        y_min -= pad
        y_max += pad

    colors = [
        "#2563eb",
        "#dc2626",
        "#16a34a",
        "#9333ea",
        "#ea580c",
        "#0891b2",
    ]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    if title:
        lines.append(
            f'<text x="{width / 2:.1f}" y="28" text-anchor="middle" '
            f'font-family="sans-serif" font-size="18">{html.escape(title)}</text>'
        )

    lines.extend([
        f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="#333"/>',
        f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="#333"/>',
    ])

    for index in range(6):
        t = index / 5
        y = plot_bottom - t * (plot_bottom - plot_top)
        value = y_min + t * (y_max - y_min)
        lines.append(f'<line x1="{plot_left}" y1="{y:.1f}" x2="{plot_right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(
            f'<text x="{plot_left - 8}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="#555">{value:.4g}</text>'
        )

    for index in range(6):
        t = index / 5
        x = plot_left + t * (plot_right - plot_left)
        value = x_min + t * (x_max - x_min)
        lines.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 5}" stroke="#333"/>')
        lines.append(
            f'<text x="{x:.1f}" y="{plot_bottom + 22}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="11" fill="#555">{value:.4g}</text>'
        )

    for index, (name, (x_values, y_values)) in enumerate(series.items()):
        color = colors[index % len(colors)]
        points = []
        for x_value, y_value in zip(x_values, y_values):
            px = _scale(float(x_value), x_min, x_max, plot_left, plot_right)
            py = _scale(float(y_value), y_min, y_max, plot_bottom, plot_top)
            points.append(f"{px:.2f},{py:.2f}")
        if len(points) == 1:
            px, py = points[0].split(",")
            lines.append(f'<circle cx="{px}" cy="{py}" r="4" fill="{color}"/>')
        else:
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{color}" stroke-width="2"/>'
            )

        legend_y = 52 + index * 18
        lines.append(f'<rect x="{plot_left + 12}" y="{legend_y - 9}" width="10" height="10" fill="{color}"/>')
        lines.append(
            f'<text x="{plot_left + 28}" y="{legend_y}" font-family="sans-serif" '
            f'font-size="12" fill="#222">{html.escape(name)}</text>'
        )

    lines.append(
        f'<text x="{(plot_left + plot_right) / 2:.1f}" y="{height - 24}" '
        f'text-anchor="middle" font-family="sans-serif" font-size="13" fill="#333">step</text>'
    )
    lines.append("</svg>\n")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
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
