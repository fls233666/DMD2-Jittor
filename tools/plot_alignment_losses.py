"""Create PyTorch-vs-Jittor DMD2 loss alignment curves.

This is a convenience wrapper for presentation artifacts.  It generates one
SVG per important DMD2 loss term and writes a JSON summary with point counts
and basic statistics.
"""

import argparse
import json
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

from tools.plot_metrics import plot_series_svg, read_records, safe_filename  # noqa: E402


CURVES = [
    {
        "name": "generator_loss_dm",
        "title": "Distribution matching loss",
        "key": "generator/loss_dm",
        "filter_generator_step": False,
    },
    {
        "name": "generator_gen_cls_loss",
        "title": "Generator GAN realism loss",
        "key": "generator/gen_cls_loss",
        "filter_generator_step": False,
    },
    {
        "name": "loss_generator",
        "title": "Generator total loss on generator-update steps",
        "key": "loss_generator",
        "filter_generator_step": True,
    },
    {
        "name": "guidance_loss_fake_mean",
        "title": "Fake score denoising loss",
        "key": "guidance/loss_fake_mean",
        "filter_generator_step": False,
    },
    {
        "name": "guidance_guidance_cls_loss",
        "title": "Guidance GAN classifier loss",
        "key": "guidance/guidance_cls_loss",
        "filter_generator_step": False,
    },
    {
        "name": "loss_guidance",
        "title": "Guidance total loss",
        "key": "loss_guidance",
        "filter_generator_step": False,
    },
]


def _number(value):
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_series(
    records,
    key,
    step_key="step",
    step_offset=0.0,
    max_step=None,
    filter_generator_step=False,
):
    xs = []
    ys = []
    for index, record in enumerate(records):
        if filter_generator_step:
            flag = _number(record.get("compute_generator_gradient"))
            if flag != 1.0:
                continue

        y_value = _number(record.get(key))
        if y_value is None:
            continue

        x_value = _number(record.get(step_key))
        if x_value is None:
            x_value = float(index + 1)
        x_value += float(step_offset)

        if max_step is not None and x_value > float(max_step):
            continue

        xs.append(x_value)
        ys.append(y_value)

    if not xs:
        return None
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


def moving_average(values, window):
    window = int(window)
    if window <= 1 or values.shape[0] < window:
        return values
    kernel = np.ones([window], dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def smooth_series(x_values, y_values, window):
    window = int(window)
    if window <= 1 or y_values.shape[0] < window:
        return x_values, y_values
    return x_values[window - 1 :], moving_average(y_values, window)


def series_summary(x_values, y_values):
    return {
        "count": int(y_values.shape[0]),
        "first_step": float(x_values[0]),
        "last_step": float(x_values[-1]),
        "last": float(y_values[-1]),
        "min": float(y_values.min()),
        "max": float(y_values.max()),
        "mean": float(y_values.mean()),
    }


def plot_series(series, output_path, title):
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

    plt.figure(figsize=(9, 5))
    for name, (x_values, y_values) in series.items():
        plt.plot(x_values, y_values, label=name)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_kwargs = {"format": "svg"} if output_path.lower().endswith(".svg") else {"dpi": 150}
    plt.savefig(output_path, **save_kwargs)
    plt.close()
    return output_path


def build_alignment_plots(args):
    pytorch_records = read_records(args.pytorch_log)
    jittor_records = read_records(args.jittor_log)
    os.makedirs(args.output_dir, exist_ok=True)

    selected = {item.strip() for item in args.only.split(",") if item.strip()}
    curves = [curve for curve in CURVES if not selected or curve["name"] in selected or curve["key"] in selected]

    summary = {
        "pytorch_log": os.path.abspath(args.pytorch_log),
        "jittor_log": os.path.abspath(args.jittor_log),
        "jittor_step_offset": float(args.jittor_step_offset),
        "smooth": int(args.smooth),
        "curves": {},
    }
    outputs = []

    for curve in curves:
        pt = extract_series(
            pytorch_records,
            key=curve["key"],
            max_step=args.max_step,
            filter_generator_step=curve["filter_generator_step"],
        )
        jt = extract_series(
            jittor_records,
            key=curve["key"],
            step_offset=args.jittor_step_offset,
            max_step=args.max_step,
            filter_generator_step=curve["filter_generator_step"],
        )

        if pt is None or jt is None:
            summary["curves"][curve["name"]] = {
                "status": "missing",
                "key": curve["key"],
                "pytorch_present": pt is not None,
                "jittor_present": jt is not None,
            }
            continue

        pt_x, pt_y = pt
        jt_x, jt_y = jt
        summary["curves"][curve["name"]] = {
            "status": "ok",
            "key": curve["key"],
            "filter_generator_step": bool(curve["filter_generator_step"]),
            "pytorch": series_summary(pt_x, pt_y),
            "jittor": series_summary(jt_x, jt_y),
        }

        plot_pt_x, plot_pt_y = smooth_series(pt_x, pt_y, args.smooth)
        plot_jt_x, plot_jt_y = smooth_series(jt_x, jt_y, args.smooth)
        suffix = f"_ma{int(args.smooth)}" if int(args.smooth) > 1 else "_raw"
        output_path = os.path.join(args.output_dir, f"{safe_filename(curve['name'])}{suffix}.svg")
        title = curve["title"]
        if int(args.smooth) > 1:
            title = f"{title} (moving average {int(args.smooth)})"
        outputs.append(
            plot_series(
                {
                    "PyTorch": (plot_pt_x, plot_pt_y),
                    "Jittor": (plot_jt_x, plot_jt_y),
                },
                output_path,
                title=title,
            )
        )
        summary["curves"][curve["name"]]["plot"] = output_path

    summary_path = args.summary_json or os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return outputs, summary_path, summary


def create_argparser():
    default_pytorch_log = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "..",
            "DMD2-pytorch",
            "records",
            "cifar10_dmd2_gan_5000_ckpt",
            "train_metrics.jsonl",
        )
    )
    default_jittor_log = os.path.join(
        PROJECT_ROOT,
        "logs",
        "cifar10_dmd2_5000",
        "train_metrics.jsonl",
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-log", default=default_pytorch_log)
    parser.add_argument("--jittor-log", default=default_jittor_log)
    parser.add_argument(
        "--output-dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "curves", "pytorch_jittor_loss_alignment"),
    )
    parser.add_argument("--summary-json", default="")
    parser.add_argument(
        "--smooth",
        type=int,
        default=20,
        help="Moving-average window. Use 1 for raw curves.",
    )
    parser.add_argument(
        "--jittor-step-offset",
        type=float,
        default=-1.0,
        help="Use -1 to align Jittor steps 1..5000 with PyTorch steps 0..4999.",
    )
    parser.add_argument("--max-step", type=float, default=None)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated curve names or metric keys. Empty means all default loss curves.",
    )
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    outputs, summary_path, summary = build_alignment_plots(args)
    missing = [
        name
        for name, item in summary["curves"].items()
        if item.get("status") != "ok"
    ]
    print(f"saved {len(outputs)} alignment loss curves to: {args.output_dir}")
    print(f"saved summary: {summary_path}")
    if missing:
        print(f"missing curves: {', '.join(missing)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
