"""Training step engine for small DMD2 Jittor experiments."""

import time

import numpy as np
import jittor as jt
from jittor import nn

try:
    from loss.common import stop_grad
except ImportError:
    try:
        from ..loss.common import stop_grad
    except ImportError:
        from common import stop_grad

try:
    from datasets.cifar10 import batch_to_real_train_dict
    from samplers.scheduler import constant_sigma, randn_image
except ImportError:
    try:
        from ..datasets.cifar10 import batch_to_real_train_dict
        from ..samplers.scheduler import constant_sigma, randn_image
    except ImportError:
        from cifar10 import batch_to_real_train_dict
        from scheduler import constant_sigma, randn_image


def as_float(value):
    # Convert scalar Jittor/numpy/Python values to Python float for logs.
    if value is None:
        return None
    if isinstance(value, jt.Var):
        jt.sync_all()
        value = value.numpy()
    value = np.asarray(value)
    if value.size == 0:
        return None
    return float(value.reshape(-1)[0])


def sum_loss_dict(loss_dict, weights=None):
    # Weighted sum over scalar losses in a dict.
    weights = {} if weights is None else weights
    total = None
    for name, loss in loss_dict.items():
        weight = float(weights.get(name, 1.0))
        if weight == 0:
            continue
        item = loss * weight
        total = item if total is None else total + item
    if total is None:
        return jt.array(0.0).float32()
    return total


def optimizer_step(optimizer, loss, max_grad_norm=None):
    # Use manual backward only when gradient clipping is requested.
    if max_grad_norm is None or not hasattr(optimizer, "backward"):
        optimizer.step(loss)
        return None

    if hasattr(optimizer, "zero_grad"):
        optimizer.zero_grad()
    optimizer.backward(loss)
    if hasattr(optimizer, "clip_grad_norm"):
        optimizer.clip_grad_norm(float(max_grad_norm))
    optimizer.step()
    return None


def scheduler_step(scheduler):
    if scheduler is not None and hasattr(scheduler, "step"):
        scheduler.step()


def loss_dict_to_float(loss_dict):
    return {name: as_float(value) for name, value in loss_dict.items()}


def scalar_log_dict(log_dict):
    logs = {}
    for name, value in log_dict.items():
        if isinstance(value, (int, float)):
            logs[name] = float(value)
        elif isinstance(value, jt.Var):
            shape = tuple(value.shape)
            size = int(np.prod(shape)) if shape else 1
            if size == 1:
                logs[name] = as_float(value)
    return logs


def infer_batch_shape(batch):
    image = batch["image"]
    return int(image.shape[0]), int(image.shape[1]), int(image.shape[2])


def infer_label_dim(batch=None, model=None, default=0):
    if batch is not None and batch.get("label") is not None:
        labels = batch["label"]
        if len(labels.shape) == 2:
            return int(labels.shape[1])
    if model is not None:
        if hasattr(model, "label_dim"):
            return int(model.label_dim)
        if hasattr(model, "feedforward_model") and hasattr(model.feedforward_model, "label_dim"):
            return int(model.feedforward_model.label_dim)
    return int(default)


def random_class_labels(batch_size, label_dim):
    # Match official train_edm.py: sample fake labels independently from real data.
    if label_dim == 0:
        return None, None
    class_id = jt.randint(low=0, high=int(label_dim), shape=[batch_size]).int32()
    eye = jt.array(np.eye(int(label_dim), dtype=np.float32))
    return eye[class_id], class_id


def make_generator_inputs(
    batch,
    conditioning_sigma=80.0,
    noise=None,
    img_channels=None,
    img_resolution=None,
):
    # Build one-step generator inputs matching DMD2 sampling/training.
    batch_size, channels, resolution = infer_batch_shape(batch)
    channels = channels if img_channels is None else int(img_channels)
    resolution = resolution if img_resolution is None else int(img_resolution)

    if noise is None:
        noise = randn_image(
            batch_size=batch_size,
            channels=channels,
            resolution=resolution,
            sigma=1.0,
        )
    sigma = constant_sigma(batch_size=batch_size, sigma=conditioning_sigma)
    scaled_noise = noise.float32() * float(conditioning_sigma)
    return scaled_noise, sigma, noise


class DMD2DebugEngine:
    # Coordinates generator/guidance updates for CIFAR-10 debug training.
    def __init__(
        self,
        model,
        generator_optimizer,
        guidance_optimizer,
        conditioning_sigma=80.0,
        generator_loss_weights=None,
        guidance_loss_weights=None,
        generator_scheduler=None,
        guidance_scheduler=None,
        ema=None,
        img_channels=None,
        img_resolution=None,
        label_dim=None,
        random_fake_labels=True,
        dfake_gen_update_ratio=1,
        max_grad_norm=None,
    ):
        self.model = model
        self.generator_optimizer = generator_optimizer
        self.guidance_optimizer = guidance_optimizer
        self.conditioning_sigma = conditioning_sigma
        self.generator_loss_weights = (
            {"loss_dm": 1.0, "gen_cls_loss": 0.0}
            if generator_loss_weights is None
            else generator_loss_weights
        )
        self.guidance_loss_weights = (
            {"loss_fake_mean": 1.0, "guidance_cls_loss": 1.0}
            if guidance_loss_weights is None
            else guidance_loss_weights
        )
        self.generator_scheduler = generator_scheduler
        self.guidance_scheduler = guidance_scheduler
        self.ema = ema
        self.img_channels = img_channels
        self.img_resolution = img_resolution
        self.label_dim = label_dim
        self.random_fake_labels = random_fake_labels
        self.dfake_gen_update_ratio = max(int(dfake_gen_update_ratio), 1)
        self.max_grad_norm = max_grad_norm
        self.global_step = 0

    def train(self):
        if hasattr(self.model, "train"):
            self.model.train()

    def eval(self):
        if hasattr(self.model, "eval"):
            self.model.eval()

    def should_update_generator(self):
        return self.global_step % self.dfake_gen_update_ratio == 0

    def make_fake_labels(self, batch):
        if not self.random_fake_labels:
            return batch.get("label"), batch.get("class_id")

        batch_size = int(batch["image"].shape[0])
        label_dim = infer_label_dim(
            batch=batch,
            model=self.model,
            default=0 if self.label_dim is None else self.label_dim,
        )
        return random_class_labels(batch_size=batch_size, label_dim=label_dim)

    def forward_generator(self, batch, compute_generator_gradient=True):
        image = batch["image"]
        labels, class_id = self.make_fake_labels(batch)
        real_train_dict = batch_to_real_train_dict(batch)
        scaled_noise, sigma, noise = make_generator_inputs(
            batch=batch,
            conditioning_sigma=self.conditioning_sigma,
            img_channels=self.img_channels,
            img_resolution=self.img_resolution,
        )

        loss_dict, log_dict = self.model(
            scaled_noisy_image=scaled_noise,
            timestep_sigma=sigma,
            labels=labels,
            real_train_dict=real_train_dict,
            compute_generator_gradient=compute_generator_gradient,
            generator_turn=True,
            guidance_turn=False,
        )
        log_dict["train_noise"] = stop_grad(noise)
        log_dict["train_sigma"] = stop_grad(sigma)
        log_dict["real_image"] = stop_grad(image)
        if labels is not None:
            log_dict["fake_label"] = stop_grad(labels)
        if class_id is not None:
            log_dict["fake_class_id"] = stop_grad(class_id)
        return loss_dict, log_dict

    def generator_step(self, batch, compute_generator_gradient=True):
        loss_dict, log_dict = self.forward_generator(
            batch=batch,
            compute_generator_gradient=compute_generator_gradient,
        )

        if compute_generator_gradient:
            total_loss = sum_loss_dict(loss_dict, self.generator_loss_weights)
            optimizer_step(
                optimizer=self.generator_optimizer,
                loss=total_loss,
                max_grad_norm=self.max_grad_norm,
            )
        else:
            total_loss = jt.array(0.0).float32()

        scheduler_step(self.generator_scheduler)

        if compute_generator_gradient and self.ema is not None:
            self.ema.update(self.model.feedforward_model)
        return total_loss, loss_dict, log_dict

    def guidance_step(self, guidance_data_dict):
        loss_dict, log_dict = self.model(
            scaled_noisy_image=None,
            timestep_sigma=None,
            labels=None,
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=guidance_data_dict,
        )
        total_loss = sum_loss_dict(loss_dict, self.guidance_loss_weights)
        optimizer_step(
            optimizer=self.guidance_optimizer,
            loss=total_loss,
            max_grad_norm=self.max_grad_norm,
        )
        scheduler_step(self.guidance_scheduler)
        return total_loss, loss_dict, log_dict

    def train_step(self, batch):
        start_time = time.time()
        self.train()

        compute_generator_gradient = self.should_update_generator()
        gen_total, gen_losses, gen_logs = self.generator_step(
            batch=batch,
            compute_generator_gradient=compute_generator_gradient,
        )
        guidance_total, guidance_losses, guidance_logs = self.guidance_step(
            gen_logs["guidance_data_dict"],
        )

        jt.sync_all()
        self.global_step += 1

        logs = {
            "step": self.global_step,
            "time": time.time() - start_time,
            "loss_generator": as_float(gen_total),
            "loss_guidance": as_float(guidance_total),
            "compute_generator_gradient": int(compute_generator_gradient),
        }
        for name, value in loss_dict_to_float(gen_losses).items():
            logs[f"generator/{name}"] = value
        for name, value in loss_dict_to_float(guidance_losses).items():
            logs[f"guidance/{name}"] = value
        for name, value in scalar_log_dict(gen_logs).items():
            logs[name] = value
        for name, value in scalar_log_dict(guidance_logs).items():
            logs[name] = value

        return {
            "loss_generator": gen_total,
            "loss_guidance": guidance_total,
            "generator_losses": gen_losses,
            "guidance_losses": guidance_losses,
            "generator_logs": gen_logs,
            "guidance_logs": guidance_logs,
            "logs": logs,
        }

    def run_no_grad_generator(self, batch):
        with jt.no_grad():
            return self.forward_generator(
                batch=batch,
                compute_generator_gradient=False,
            )


class DMD2DebugEngineModule(nn.Module):
    # Lightweight nn.Module wrapper for code that expects a module-like engine.
    def __init__(self, engine):
        super().__init__()
        self.engine = engine

    def execute(self, batch):
        return self.engine.train_step(batch)
