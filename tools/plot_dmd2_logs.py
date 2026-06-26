"""Plot grouped raw and moving-average DMD2 training curves."""

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
    return project_root


PROJECT_ROOT = setup_paths()

from tools.plot_metrics import read_records  # noqa: E402
from utils.logger import write_json  # noqa: E402


GROUPS = {
    "generator_losses": [
        "loss_generator",
        "generator/loss_dm",
        "generator/gen_cls_loss",
    ],
    "guidance_losses": [
        "loss_guidance",
        "guidance/loss_fake_mean",
        "guidance/guidance_cls_loss",
    ],
    "gan_diagnostics": [
        "gan/real_prob_mean",
        "gan/fake_prob_mean",
        "gan/real_acc",
        "gan/fake_acc",
        "gan/total_acc",
        "gan/generator_fooling_rate",
        "gan/real_logits_mean",
        "gan/fake_logits_mean",
        "gan/real_logits_std",
        "gan/fake_logits_std",
    ],
    "gradient_norms": [
        "dmtrain_gradient_norm",
        "generator_grad_norm",
        "guidance_grad_norm",
    ],
}


def _number(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_series(records, key):
    xs = []
    ys = []
    for index, record in enumerate(records):
        value = _number(record.get(key))
        if value is None:
            continue
        step = _number(record.get("step"))
        xs.append(float(index + 1) if step is None else step)
        ys.append(value)
    if not ys:
        return None
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def moving_average(y, window):
    window = int(window)
    if window <= 1 or y.shape[0] < window:
        return y
    kernel = np.ones([window], dtype=np.float64) / float(window)
    return np.convolve(y, kernel, mode="valid")


def smooth_series(x, y, window):
    window = int(window)
    if window <= 1 or y.shape[0] < window:
        return x, y
    return x[window - 1 :], moving_average(y, window)


def plot_group(series, output_path, title):
    if not series:
        return None

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
    try:
        import matplotlib
    except ModuleNotFoundError:
        from tools.plot_metrics import plot_series_svg

        return plot_series_svg(series, output_path, title=title)

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    plt.figure(figsize=(9.5, 5.25))
    for name, (x, y) in series.items():
        plt.plot(x, y, label=name)
    plt.xlabel("step")
    plt.ylabel("value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_kwargs = {"format": "svg"} if str(output_path).lower().endswith(".svg") else {"dpi": 150}
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    return output_path


def build_plots(records, output_dir, windows):
    os.makedirs(output_dir, exist_ok=True)
    summary = {}
    outputs = []

    for group_name, keys in GROUPS.items():
        raw_series = {}
        for key in keys:
            values = extract_series(records, key)
            if values is not None:
                raw_series[key] = values

        if not raw_series:
            summary[group_name] = {"status": "missing", "keys": keys}
            continue

        summary[group_name] = {
            "status": "ok",
            "keys": sorted(raw_series),
            "plots": [],
        }

        raw_path = os.path.join(output_dir, f"{group_name}_raw.svg")
        outputs.append(plot_group(raw_series, raw_path, f"{group_name} raw"))
        summary[group_name]["plots"].append(raw_path)

        for window in windows:
            smoothed = {
                name: smooth_series(x, y, window)
                for name, (x, y) in raw_series.items()
                if y.shape[0] >= int(window)
            }
            if not smoothed:
                continue
            path = os.path.join(output_dir, f"{group_name}_ma{int(window)}.svg")
            outputs.append(plot_group(smoothed, path, f"{group_name} moving average {int(window)}"))
            summary[group_name]["plots"].append(path)

    return [path for path in outputs if path], summary


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="Training metrics JSONL/CSV.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "curves", "dmd2_logs"),
    )
    parser.add_argument(
        "--windows",
        default="50,100,200",
        help="Comma-separated moving-average windows.",
    )
    parser.add_argument("--summary-json", default="")
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    records = read_records(args.log)
    windows = [int(item) for item in args.windows.split(",") if item.strip()]
    outputs, summary = build_plots(records, args.output_dir, windows)

    summary_path = args.summary_json or os.path.join(args.output_dir, "summary.json")
    write_json(summary_path, summary)

    print(f"saved {len(outputs)} plots to {args.output_dir}")
    print(f"saved summary: {summary_path}")
    missing = [name for name, item in summary.items() if item["status"] == "missing"]
    if missing:
        print(f"missing groups: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
