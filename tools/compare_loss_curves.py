"""Plot PyTorch and Jittor loss curves in one figure.

Inputs can be JSONL or CSV metric logs.  Each input may use a different loss
key, e.g. official PyTorch wandb exports often use ``loss_dm`` while the
Jittor debug trainer logs ``generator/loss_dm``.
"""

import argparse
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

from tools.plot_metrics import read_records, plot_series_svg  # noqa: E402


def _as_number(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_series(records, key, step_key="step", x_offset=0.0, max_step=None):
    xs = []
    ys = []
    for index, record in enumerate(records):
        y_value = _as_number(record.get(key))
        if y_value is None:
            continue
        x_value = _as_number(record.get(step_key))
        if x_value is None:
            x_value = float(index + 1)
        x_value += float(x_offset)
        if max_step is not None and x_value > float(max_step):
            continue
        xs.append(x_value)
        ys.append(y_value)

    if not xs:
        raise ValueError(f"No numeric values found for key {key!r}.")
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def moving_average(values, window):
    window = int(window)
    if window <= 1 or values.shape[0] < window:
        return values
    kernel = np.ones([window], dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def smooth_series(x, y, window):
    window = int(window)
    if window <= 1 or y.shape[0] < window:
        return x, y
    return x[window - 1 :], moving_average(y, window)


def plot_comparison(
    pytorch_log,
    jittor_log,
    output,
    pytorch_key,
    jittor_key,
    pytorch_label="PyTorch",
    jittor_label="Jittor",
    step_key="step",
    title=None,
    smooth=1,
    max_step=None,
):
    pytorch_records = read_records(pytorch_log)
    jittor_records = read_records(jittor_log)

    pt_x, pt_y = extract_series(
        pytorch_records,
        key=pytorch_key,
        step_key=step_key,
        max_step=max_step,
    )
    jt_x, jt_y = extract_series(
        jittor_records,
        key=jittor_key,
        step_key=step_key,
        max_step=max_step,
    )

    pt_x, pt_y = smooth_series(pt_x, pt_y, smooth)
    jt_x, jt_y = smooth_series(jt_x, jt_y, smooth)

    series = {
        f"{pytorch_label}: {pytorch_key}": (pt_x, pt_y),
        f"{jittor_label}: {jittor_key}": (jt_x, jt_y),
    }

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    try:
        import matplotlib
    except ModuleNotFoundError:
        return plot_series_svg(series, output, title=title)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = os.path.dirname(output)
    if directory:
        os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(9, 5))
    for name, (x_values, y_values) in series.items():
        plt.plot(x_values, y_values, label=name)
    plt.xlabel("step")
    plt.ylabel("loss")
    if title:
        plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_kwargs = {}
    if str(output).lower().endswith(".svg"):
        save_kwargs["format"] = "svg"
    else:
        save_kwargs["dpi"] = 150
    plt.savefig(output, **save_kwargs)
    plt.close()
    return output


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-log", required=True, help="PyTorch JSONL/CSV log.")
    parser.add_argument("--jittor-log", required=True, help="Jittor JSONL/CSV log.")
    parser.add_argument("--output", required=True, help="Output image path, e.g. curves/compare.svg.")
    parser.add_argument("--pytorch-key", default="loss_dm")
    parser.add_argument("--jittor-key", default="generator/loss_dm")
    parser.add_argument("--pytorch-label", default="PyTorch")
    parser.add_argument("--jittor-label", default="Jittor")
    parser.add_argument("--step-key", default="step")
    parser.add_argument("--title", default="PyTorch vs Jittor loss")
    parser.add_argument("--smooth", type=int, default=1, help="Moving-average window.")
    parser.add_argument("--max-step", type=float, default=None)
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    output = plot_comparison(
        pytorch_log=args.pytorch_log,
        jittor_log=args.jittor_log,
        output=args.output,
        pytorch_key=args.pytorch_key,
        jittor_key=args.jittor_key,
        pytorch_label=args.pytorch_label,
        jittor_label=args.jittor_label,
        step_key=args.step_key,
        title=args.title,
        smooth=args.smooth,
        max_step=args.max_step,
    )
    print(f"saved comparison curve: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
