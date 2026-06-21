"""Compute FID or lightweight pixel-FID for generated results."""

import argparse
import os
import sys


def setup_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_dir = os.path.join(project_root, "code")
    for path in (project_root, code_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


setup_paths()

from utils.image import image_dir_to_pixel_features
from utils.logger import write_json
from utils.metric import (
    compute_feature_stats,
    fid_from_features,
    frechet_distance,
    load_feature_stats,
    load_features,
    save_feature_stats,
)


def features_from_source(path, image_size=32, max_images=None):
    if os.path.isdir(path):
        return image_dir_to_pixel_features(
            path,
            image_size=image_size,
            max_images=max_images,
        )
    return load_features(path)


def compute_fid(args):
    if args.pred_stats:
        pred_mu, pred_sigma = load_feature_stats(args.pred_stats)
    else:
        pred_features = features_from_source(
            args.pred,
            image_size=args.image_size,
            max_images=args.max_images,
        )
        pred_mu, pred_sigma = compute_feature_stats(pred_features)

    if args.ref_stats:
        ref_mu, ref_sigma = load_feature_stats(args.ref_stats)
    else:
        ref_features = features_from_source(
            args.ref,
            image_size=args.image_size,
            max_images=args.max_images,
        )
        ref_mu, ref_sigma = compute_feature_stats(ref_features)

    fid = frechet_distance(pred_mu, pred_sigma, ref_mu, ref_sigma)
    result = {
        "fid": fid,
        "mode": "feature-fid" if (args.pred.endswith(".npz") or args.ref.endswith(".npz")) else "pixel-fid",
        "pred": args.pred,
        "ref": args.ref,
        "image_size": args.image_size,
        "max_images": args.max_images,
        "feature_dim": int(pred_mu.shape[0]),
    }

    if args.save_pred_stats:
        save_feature_stats(args.save_pred_stats, pred_mu, pred_sigma)
        result["pred_stats"] = args.save_pred_stats

    if args.output:
        write_json(args.output, result)

    return result


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", required=True, help="Generated image dir or feature npz.")
    parser.add_argument("--ref", required=True, help="Reference image dir or feature npz.")
    parser.add_argument("--output", default="outputs/fid_results/fid.json")
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--pred-stats", default=None)
    parser.add_argument("--ref-stats", default=None)
    parser.add_argument("--save-pred-stats", default=None)
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    result = compute_fid(args)
    print(f"fid={result['fid']:.6f} mode={result['mode']}")
    if args.output:
        print(f"saved fid result: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
