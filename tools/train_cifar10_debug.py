"""Run a compact CIFAR-10 DMD2 debug training job."""

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np


def setup_paths():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    code_dir = os.path.join(project_root, "code")
    for path in (
        project_root,
        code_dir,
        os.path.join(code_dir, "datasets"),
        os.path.join(code_dir, "models"),
        os.path.join(code_dir, "trainer"),
    ):
        if path not in sys.path:
            sys.path.insert(0, path)
    return project_root


PROJECT_ROOT = setup_paths()

import jittor as jt
from jittor import nn

from datasets.cifar10 import build_cifar10_debug_loader
from models.ema import ExponentialMovingAverage
from models.unified_model import EDMUniModel
from trainer.checkpoint import save_checkpoint
from trainer.engine import DMD2DebugEngine
from trainer.evaluator import DebugSamplerEvaluator
from trainer.train_loop import train_debug


def parse_int_list(value):
    if value is None or value == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def set_seed(seed):
    np.random.seed(int(seed))
    if hasattr(jt, "set_global_seed"):
        jt.set_global_seed(int(seed))
    elif hasattr(jt, "seed"):
        jt.seed(int(seed))


def model_args(args):
    return SimpleNamespace(
        dataset_name="cifar10",
        resolution=args.image_size,
        label_dim=10,
        use_fp16=False,
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        config_name="tiny",
        gan_classifier=args.gan_classifier,
        diffusion_gan=args.diffusion_gan,
        diffusion_gan_max_timestep=args.diffusion_gan_max_timestep,
        num_train_timesteps=args.num_train_timesteps,
        min_step_percent=args.min_step_percent,
        max_step_percent=args.max_step_percent,
    )


def module_parameters(module):
    if module is None or not hasattr(module, "parameters"):
        return []
    return list(module.parameters())


def make_optimizer(params, lr, beta1=0.0, beta2=0.999, optimizer_name="adam"):
    params = list(params)
    if not params:
        raise ValueError("optimizer parameter list is empty")

    name = optimizer_name.lower()
    if name == "adam":
        if hasattr(nn, "Adam"):
            return nn.Adam(params, lr=float(lr), betas=(float(beta1), float(beta2)))
        import jittor.optim as optim

        return optim.Adam(params, lr=float(lr), betas=(float(beta1), float(beta2)))

    if name == "sgd":
        if hasattr(nn, "SGD"):
            return nn.SGD(params, lr=float(lr))
        import jittor.optim as optim

        return optim.SGD(params, lr=float(lr))

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_model(args):
    return EDMUniModel(
        args=model_args(args),
        initialize_generator=True,
    )


def guidance_train_parameters(model):
    params = []
    params.extend(module_parameters(model.guidance_model.fake_unet))
    params.extend(module_parameters(model.guidance_model.cls_pred_branch))
    return params


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "cifar10"))
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--class-subset", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")

    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--conditioning-sigma", type=float, default=80.0)
    parser.add_argument("--dfake-gen-update-ratio", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)

    parser.add_argument("--lr-generator", type=float, default=2e-4)
    parser.add_argument("--lr-guidance", type=float, default=2e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--no-ema", action="store_true")

    parser.add_argument("--sigma-data", type=float, default=0.5)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=80.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--min-step-percent", type=float, default=0.02)
    parser.add_argument("--max-step-percent", type=float, default=0.98)
    parser.add_argument("--gan-classifier", action="store_true")
    parser.add_argument("--diffusion-gan", action="store_true")
    parser.add_argument("--diffusion-gan-max-timestep", type=int, default=1)

    parser.add_argument("--checkpoint-dir", default=os.path.join(PROJECT_ROOT, "checkpoints", "cifar10_debug"))
    parser.add_argument("--sample-dir", default=os.path.join(PROJECT_ROOT, "outputs", "samples", "cifar10_debug"))
    parser.add_argument("--metrics-log", default=os.path.join(PROJECT_ROOT, "logs", "cifar10_debug", "train_metrics.jsonl"))
    parser.add_argument("--performance-log", default=os.path.join(PROJECT_ROOT, "logs", "cifar10_debug", "performance.jsonl"))
    parser.add_argument("--final-checkpoint", default="")
    parser.add_argument("--skip-final-eval", action="store_true")
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    if args.metrics_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_log)), exist_ok=True)
    if args.performance_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.performance_log)), exist_ok=True)

    loader = build_cifar10_debug_loader(
        root=args.data_root,
        train=True,
        download=False,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        image_size=args.image_size,
        augment=not args.no_augment,
        seed=args.seed,
        class_subset=parse_int_list(args.class_subset),
    )

    model = build_model(args)
    generator_optimizer = make_optimizer(
        module_parameters(model.feedforward_model),
        lr=args.lr_generator,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        optimizer_name=args.optimizer,
    )
    guidance_optimizer = make_optimizer(
        guidance_train_parameters(model),
        lr=args.lr_guidance,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        optimizer_name=args.optimizer,
    )
    ema = None if args.no_ema else ExponentialMovingAverage(
        model.feedforward_model,
        decay=args.ema_decay,
    )

    evaluator = None
    if args.eval_interval and args.eval_interval > 0:
        evaluator = DebugSamplerEvaluator(
            output_dir=args.sample_dir,
            batch_size=args.eval_batch_size,
            nrow=args.nrow,
            conditioning_sigma=args.conditioning_sigma,
            img_channels=3,
            img_resolution=args.image_size,
        )

    engine = DMD2DebugEngine(
        model=model,
        generator_optimizer=generator_optimizer,
        guidance_optimizer=guidance_optimizer,
        conditioning_sigma=args.conditioning_sigma,
        ema=ema,
        img_channels=3,
        img_resolution=args.image_size,
        label_dim=10,
        dfake_gen_update_ratio=args.dfake_gen_update_ratio,
        max_grad_norm=args.max_grad_norm,
    )

    history = train_debug(
        engine=engine,
        train_loader=loader,
        max_steps=args.max_steps,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        output_dir=args.checkpoint_dir,
        evaluator=evaluator,
        eval_interval=args.eval_interval,
        metrics_logger=args.metrics_log,
        performance_logger=args.performance_log,
    )

    final_step = history[-1]["step"] if history else 0
    final_checkpoint = args.final_checkpoint
    if not final_checkpoint:
        final_checkpoint = os.path.join(args.checkpoint_dir, "checkpoint_final.pkl")
    save_checkpoint(
        path=final_checkpoint,
        model=model,
        generator_optimizer=generator_optimizer,
        guidance_optimizer=guidance_optimizer,
        ema=ema,
        step=final_step,
        extra={
            "dataset": "cifar10",
            "max_samples": args.max_samples,
            "image_size": args.image_size,
        },
    )

    if evaluator is not None and not args.skip_final_eval:
        if final_step == 0 or final_step % int(args.eval_interval) != 0:
            evaluator.evaluate(model, step=final_step)

    print(f"finished CIFAR-10 debug training: steps={final_step}")
    print(f"metrics log: {args.metrics_log}")
    print(f"performance log: {args.performance_log}")
    print(f"final checkpoint: {final_checkpoint}")
    print(f"sample dir: {args.sample_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
