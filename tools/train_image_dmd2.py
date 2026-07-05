"""Run a compact image DMD2 training job."""

import argparse
import copy
import os
import pickle
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
from datasets.imagenet64_lmdb import build_imagenet64_lmdb_loader
from datasets.tiny_imagenet import build_tiny_imagenet_loader
from models.diffusion import get_edm_network
from models.ema import ExponentialMovingAverage
from models.unified_model import EDMUniModel
from trainer.checkpoint import load_checkpoint, save_checkpoint
from trainer.engine import ImageDMD2TrainEngine
from trainer.evaluator import ImageDMD2SamplerEvaluator
from trainer.train_loop import train_image_dmd2
from utils.performance_monitor import NvidiaSmiMonitor


def parse_int_list(value):
    if value is None or value == "":
        return None
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            result.extend(range(int(start), int(end) + 1))
        else:
            result.append(int(item))
    return result


def set_seed(seed):
    np.random.seed(int(seed))
    if hasattr(jt, "set_global_seed"):
        jt.set_global_seed(int(seed))
    elif hasattr(jt, "seed"):
        jt.seed(int(seed))


def model_args(args):
    return SimpleNamespace(
        dataset_name=args.dataset_name,
        resolution=args.image_size,
        label_dim=args.label_dim,
        use_fp16=args.use_fp16,
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        config_name=args.teacher_config,
        gan_classifier=args.gan_classifier,
        diffusion_gan=args.diffusion_gan,
        diffusion_gan_max_timestep=args.diffusion_gan_max_timestep,
        num_train_timesteps=args.num_train_timesteps,
        min_step_percent=args.min_step_percent,
        max_step_percent=args.max_step_percent,
    )


def load_checkpoint_object(path):
    with open(path, "rb") as handle:
        try:
            return pickle.load(handle)
        except Exception:
            handle.seek(0)
    return jt.load(path)


def select_state(obj, state_key=""):
    if state_key:
        for part in state_key.split("."):
            obj = obj[part]
        return obj

    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"]
    return obj


def to_jittor_state(state):
    converted = {}
    for key, value in state.items():
        if isinstance(value, jt.Var):
            converted[key] = value
        else:
            converted[key] = jt.array(np.asarray(value))
    return converted


def module_parameters(module):
    if module is None or not hasattr(module, "parameters"):
        return []
    return list(module.parameters())


def make_optimizer(params, lr, beta1=0.0, beta2=0.999, optimizer_name="adam", weight_decay=0.0):
    params = list(params)
    if not params:
        raise ValueError("optimizer parameter list is empty")

    name = optimizer_name.lower()
    if name == "adam":
        if hasattr(nn, "Adam"):
            return nn.Adam(params, lr=float(lr), betas=(float(beta1), float(beta2)))
        import jittor.optim as optim

        return optim.Adam(params, lr=float(lr), betas=(float(beta1), float(beta2)))

    if name == "adamw":
        import jittor.optim as optim

        if hasattr(nn, "AdamW"):
            return nn.AdamW(
                params,
                lr=float(lr),
                betas=(float(beta1), float(beta2)),
                weight_decay=float(weight_decay),
            )
        if hasattr(optim, "AdamW"):
            return optim.AdamW(
                params,
                lr=float(lr),
                betas=(float(beta1), float(beta2)),
                weight_decay=float(weight_decay),
            )
        raise ValueError("optimizer=adamw requires a Jittor version with AdamW support")

    if name == "sgd":
        if hasattr(nn, "SGD"):
            return nn.SGD(params, lr=float(lr))
        import jittor.optim as optim

        return optim.SGD(params, lr=float(lr))

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def _set_optimizer_lr(optimizer, lr):
    lr = float(lr)
    if hasattr(optimizer, "set_lr"):
        optimizer.set_lr(lr)
    if hasattr(optimizer, "lr"):
        try:
            optimizer.lr = lr
        except Exception:
            pass
    for group in getattr(optimizer, "param_groups", []) or []:
        if isinstance(group, dict) and "lr" in group:
            group["lr"] = lr
        elif hasattr(group, "lr"):
            group.lr = lr


class ConstantWithWarmupScheduler:
    # Match diffusers' constant_with_warmup scheduler used by PyTorch train_edm.py.
    def __init__(self, optimizer, base_lr, warmup_steps=0):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.step_count = 0
        if self.warmup_steps > 0:
            _set_optimizer_lr(self.optimizer, 0.0)

    def state_dict(self):
        return {
            "base_lr": self.base_lr,
            "warmup_steps": self.warmup_steps,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state):
        self.base_lr = float(state.get("base_lr", self.base_lr))
        self.warmup_steps = int(state.get("warmup_steps", self.warmup_steps))
        self.step_count = int(state.get("step_count", self.step_count))
        _set_optimizer_lr(self.optimizer, self.current_lr())

    def current_lr(self):
        if self.warmup_steps <= 0:
            return self.base_lr
        scale = min(float(self.step_count) / float(self.warmup_steps), 1.0)
        return self.base_lr * scale

    def step(self):
        self.step_count += 1
        _set_optimizer_lr(self.optimizer, self.current_lr())


def make_scheduler(optimizer, base_lr, warmup_steps):
    return ConstantWithWarmupScheduler(
        optimizer=optimizer,
        base_lr=base_lr,
        warmup_steps=warmup_steps,
    )


def build_model(args):
    args_obj = model_args(args)
    real_unet = None
    feedforward_model = None

    if args.real_unet_checkpoint:
        if not os.path.exists(args.real_unet_checkpoint):
            raise FileNotFoundError(args.real_unet_checkpoint)

        real_unet = get_edm_network(args=args_obj)
        raw_state = load_checkpoint_object(args.real_unet_checkpoint)
        state = select_state(raw_state, state_key=args.real_unet_state_key)
        real_unet.load_state_dict(to_jittor_state(state))
        if hasattr(real_unet, "requires_grad_"):
            real_unet.requires_grad_(False)

        if args.init_generator_from_real:
            feedforward_model = copy.deepcopy(real_unet)

    return EDMUniModel(
        args=args_obj,
        initialize_generator=True,
        real_unet=real_unet,
        feedforward_model=feedforward_model,
        copy_real_to_fake=(
            args.init_fake_from_real
            if real_unet is not None
            else True
        ),
    )


def guidance_train_parameters(model):
    params = []
    params.extend(module_parameters(model.guidance_model.fake_unet))
    params.extend(module_parameters(model.guidance_model.cls_pred_branch))
    return params


def cache_frozen_linear_transposes(module):
    if module is None or not hasattr(module, "named_modules"):
        return 0

    count = 0
    for _, child in module.named_modules():
        if hasattr(child, "freeze_transposed_weight"):
            child.freeze_transposed_weight()
            count += 1
    return count


def create_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-name",
        choices=(
            "cifar10",
            "tiny-imagenet",
            "tiny_imagenet",
            "tinyimagenet",
            "imagenet",
            "imagenet64",
            "imagenet64_lmdb",
            "imagenet-64x64",
        ),
        default="cifar10",
        help="Dataset and model family to train.",
    )
    parser.add_argument("--data-root", default=os.path.join(PROJECT_ROOT, "data", "cifar10"))
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--label-dim", type=int, default=10)
    parser.add_argument("--class-subset", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--use-fp16", action="store_true")

    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--eval-chunk-size", type=int, default=None)
    parser.add_argument("--nrow", type=int, default=4)
    parser.add_argument("--conditioning-sigma", type=float, default=80.0)
    parser.add_argument("--dfake-gen-update-ratio", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=None)
    parser.add_argument(
        "--teacher-config",
        choices=("tiny", "cifar10", "tinyimagenet", "imagenet"),
        default="tiny",
        help="Network config for real/fake/generator EDM models.",
    )
    parser.add_argument(
        "--real-unet-checkpoint",
        default="",
        help="Converted Jittor real teacher state dict, e.g. cifar10_teacher_jittor.pkl.",
    )
    parser.add_argument(
        "--real-unet-state-key",
        default="",
        help="Optional nested state key inside --real-unet-checkpoint.",
    )
    parser.add_argument(
        "--init-fake-from-real",
        action="store_true",
        help="Initialize fake score model by copying the loaded real teacher.",
    )
    parser.add_argument(
        "--init-generator-from-real",
        action="store_true",
        help="Initialize generator by copying the loaded real teacher.",
    )
    parser.add_argument(
        "--skip-real-linear-transpose-cache",
        action="store_true",
        help="Skip precomputing W^T caches for frozen real-teacher Linear layers.",
    )

    parser.add_argument("--lr-generator", type=float, default=2e-4)
    parser.add_argument("--lr-guidance", type=float, default=2e-4)
    parser.add_argument("--adam-beta1", type=float, default=0.0)
    parser.add_argument("--adam-beta2", type=float, default=0.999)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam")
    parser.add_argument("--warmup-step", type=int, default=0)
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
    parser.add_argument("--gen-cls-loss-weight", type=float, default=0.0)
    parser.add_argument("--cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--diffusion-gan", action="store_true")
    parser.add_argument("--diffusion-gan-max-timestep", type=int, default=1)

    parser.add_argument("--checkpoint-dir", default=os.path.join(PROJECT_ROOT, "checkpoints", "image_dmd2"))
    parser.add_argument("--sample-dir", default=os.path.join(PROJECT_ROOT, "outputs", "samples", "image_dmd2"))
    parser.add_argument("--metrics-log", default=os.path.join(PROJECT_ROOT, "logs", "image_dmd2", "train_metrics.jsonl"))
    parser.add_argument("--performance-log", default=os.path.join(PROJECT_ROOT, "logs", "image_dmd2", "performance.jsonl"))
    parser.add_argument(
        "--resume-checkpoint",
        default="",
        help="Resume model, optimizers, EMA, and step from a debug checkpoint.",
    )
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help="Resume model/EMA/step but reinitialize optimizers; use after changing trainable parameters.",
    )
    parser.add_argument("--final-checkpoint", default="")
    parser.add_argument("--skip-final-checkpoint", action="store_true")
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument(
        "--enable-gpu-monitor",
        action="store_true",
        help="Sample GPU utilization, power, and nvidia-smi memory into performance.jsonl.",
    )
    parser.add_argument(
        "--gpu-monitor-interval",
        type=float,
        default=0.5,
        help="Seconds between background nvidia-smi samples. Use 0 for one query per logged step.",
    )
    parser.add_argument(
        "--gpu-monitor-index",
        default="",
        help="GPU index or UUID for nvidia-smi. Defaults to DMD2_PERF_GPU_INDEX or first CUDA_VISIBLE_DEVICES entry.",
    )
    return parser


def main(argv=None):
    args = create_argparser().parse_args(argv)
    if args.dataset_name in ("tiny-imagenet", "tiny_imagenet"):
        args.dataset_name = "tinyimagenet"
    if args.dataset_name in ("imagenet64", "imagenet64_lmdb", "imagenet-64x64"):
        args.dataset_name = "imagenet"
    if args.dataset_name == "tinyimagenet" and args.teacher_config == "cifar10":
        raise ValueError("--teacher-config=cifar10 is only valid for --dataset-name=cifar10")
    if args.dataset_name == "imagenet" and args.teacher_config != "imagenet":
        raise ValueError("ImageNet-64 alignment requires --teacher-config=imagenet")
    if args.dataset_name == "cifar10" and args.label_dim != 10:
        raise ValueError("CIFAR-10 requires --label-dim=10")
    if args.dataset_name == "imagenet" and args.label_dim != 1000:
        raise ValueError("ImageNet-64 alignment requires --label-dim=1000")
    if args.max_samples is not None and args.max_samples < 0:
        args.max_samples = None
    jt.flags.use_cuda = 1 if args.use_cuda else 0
    set_seed(args.seed)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    if args.metrics_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.metrics_log)), exist_ok=True)
    if args.performance_log:
        os.makedirs(os.path.dirname(os.path.abspath(args.performance_log)), exist_ok=True)

    class_subset = parse_int_list(args.class_subset)
    if args.dataset_name == "cifar10":
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
            class_subset=class_subset,
        )
    elif args.dataset_name == "tinyimagenet":
        loader = build_tiny_imagenet_loader(
            root=args.data_root,
            train=True,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            image_size=args.image_size,
            augment=not args.no_augment,
            seed=args.seed,
            class_subset=class_subset,
            num_classes=args.label_dim,
        )
    elif args.dataset_name == "imagenet":
        loader = build_imagenet64_lmdb_loader(
            root=args.data_root,
            train=True,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            image_size=args.image_size,
            augment=not args.no_augment,
            seed=args.seed,
            class_subset=class_subset,
            num_classes=args.label_dim,
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset_name}")

    model = build_model(args)
    if args.real_unet_checkpoint:
        print(f"loaded real teacher: {args.real_unet_checkpoint}")
        print(f"teacher config: {args.teacher_config}")
        print(f"init fake from real: {int(args.init_fake_from_real)}")
        print(f"init generator from real: {int(args.init_generator_from_real)}")
    cached_linear_count = 0
    if args.skip_real_linear_transpose_cache:
        print("skipped real teacher linear transpose cache")
    else:
        cached_linear_count = cache_frozen_linear_transposes(model.guidance_model.real_unet)
        if cached_linear_count:
            print(f"cached real teacher linear transposes: {cached_linear_count}")

    generator_optimizer = make_optimizer(
        module_parameters(model.feedforward_model),
        lr=args.lr_generator,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
    )
    guidance_optimizer = make_optimizer(
        guidance_train_parameters(model),
        lr=args.lr_guidance,
        beta1=args.adam_beta1,
        beta2=args.adam_beta2,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
    )
    generator_scheduler = make_scheduler(
        generator_optimizer,
        base_lr=args.lr_generator,
        warmup_steps=args.warmup_step,
    )
    guidance_scheduler = make_scheduler(
        guidance_optimizer,
        base_lr=args.lr_guidance,
        warmup_steps=args.warmup_step,
    )
    ema = None if args.no_ema else ExponentialMovingAverage(
        model.feedforward_model,
        decay=args.ema_decay,
    )

    evaluator = None
    if args.eval_interval and args.eval_interval > 0:
        evaluator = ImageDMD2SamplerEvaluator(
            output_dir=args.sample_dir,
            batch_size=args.eval_batch_size,
            nrow=args.nrow,
            conditioning_sigma=args.conditioning_sigma,
            img_channels=3,
            img_resolution=args.image_size,
            label_dim=args.label_dim,
            chunk_size=args.eval_chunk_size,
        )

    engine = ImageDMD2TrainEngine(
        model=model,
        generator_optimizer=generator_optimizer,
        guidance_optimizer=guidance_optimizer,
        generator_scheduler=generator_scheduler,
        guidance_scheduler=guidance_scheduler,
        conditioning_sigma=args.conditioning_sigma,
        generator_loss_weights={
            "loss_dm": 1.0,
            "gen_cls_loss": args.gen_cls_loss_weight,
        },
        guidance_loss_weights={
            "loss_fake_mean": 1.0,
            "guidance_cls_loss": args.cls_loss_weight,
        },
        ema=ema,
        img_channels=3,
        img_resolution=args.image_size,
        label_dim=args.label_dim,
        dfake_gen_update_ratio=args.dfake_gen_update_ratio,
        max_grad_norm=args.max_grad_norm,
    )

    start_step = 0
    if args.resume_checkpoint:
        state = load_checkpoint(
            args.resume_checkpoint,
            model=model,
            generator_optimizer=None if args.resume_model_only else generator_optimizer,
            guidance_optimizer=None if args.resume_model_only else guidance_optimizer,
            generator_scheduler=None if args.resume_model_only else generator_scheduler,
            guidance_scheduler=None if args.resume_model_only else guidance_scheduler,
            ema=ema,
        )
        start_step = int(state.get("step", 0))
        engine.global_step = start_step
        cached_linear_count = 0
        if not args.skip_real_linear_transpose_cache:
            cached_linear_count = cache_frozen_linear_transposes(model.guidance_model.real_unet)
        print(f"resumed checkpoint: {args.resume_checkpoint}")
        print(f"resume start step: {start_step}")
        print(f"resume model only: {int(args.resume_model_only)}")
        print(f"refreshed real teacher linear transpose cache: {cached_linear_count}")

    performance_monitor = NvidiaSmiMonitor(
        enabled=args.enable_gpu_monitor,
        interval=args.gpu_monitor_interval,
        gpu_index=args.gpu_monitor_index,
    ).start()
    try:
        history = train_image_dmd2(
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
            performance_monitor=performance_monitor,
            start_step=start_step,
        )
    finally:
        performance_monitor.stop()

    final_step = history[-1]["step"] if history else start_step
    final_checkpoint = args.final_checkpoint
    if args.skip_final_checkpoint:
        final_checkpoint = ""
    else:
        if not final_checkpoint:
            final_checkpoint = os.path.join(args.checkpoint_dir, "checkpoint_final.pkl")
        save_checkpoint(
            path=final_checkpoint,
            model=model,
            generator_optimizer=generator_optimizer,
            guidance_optimizer=guidance_optimizer,
            generator_scheduler=generator_scheduler,
            guidance_scheduler=guidance_scheduler,
            ema=ema,
            step=final_step,
            extra={
                "dataset": args.dataset_name,
                "dataset_name": args.dataset_name,
                "max_samples": args.max_samples,
                "image_size": args.image_size,
                "label_dim": args.label_dim,
                "teacher_config": args.teacher_config,
                "real_unet_checkpoint": args.real_unet_checkpoint,
                "resume_checkpoint": args.resume_checkpoint,
                "resume_model_only": args.resume_model_only,
                "init_fake_from_real": args.init_fake_from_real,
                "init_generator_from_real": args.init_generator_from_real,
                "dfake_gen_update_ratio": args.dfake_gen_update_ratio,
                "gan_classifier": args.gan_classifier,
                "gen_cls_loss_weight": args.gen_cls_loss_weight,
                "cls_loss_weight": args.cls_loss_weight,
                "diffusion_gan": args.diffusion_gan,
                "diffusion_gan_max_timestep": args.diffusion_gan_max_timestep,
                "optimizer": args.optimizer,
                "adam_beta1": args.adam_beta1,
                "adam_beta2": args.adam_beta2,
                "weight_decay": args.weight_decay,
                "warmup_step": args.warmup_step,
                "use_fp16": args.use_fp16,
            },
        )

    if evaluator is not None and not args.skip_final_eval:
        if final_step == 0 or final_step % int(args.eval_interval) != 0:
            evaluator.evaluate(model, step=final_step)

    print(f"finished {args.dataset_name} debug training: steps={final_step}")
    print(f"metrics log: {args.metrics_log}")
    print(f"performance log: {args.performance_log}")
    print(f"final checkpoint: {final_checkpoint if final_checkpoint else 'skipped'}")
    print(f"sample dir: {args.sample_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
