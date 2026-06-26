"""Diffusion-side model wrappers for the Jittor DMD2/EDM implementation.

This file contains:
- EDMPrecond: EDM preconditioning wrapper around DhariwalUNet.
- get_imagenet_edm_config: ImageNet-64 network configuration.
- get_edm_network: factory function matching the official DMD2 entry.
- get_sigmas_karras: Karras noise schedule used by DMD2 guidance.
- EDMTeacherStudent: a lightweight holder for real/fake EDM networks.

"""

import copy
import numpy as np
import jittor as jt
from jittor import nn

try:
    from .unet import DhariwalUNet, SongUNet
except ImportError:
    from unet import DhariwalUNet, SongUNet


def _as_float32(x):
    # Convert Python values, numpy arrays, or Jittor Vars to float32 Jittor Vars.
    if not isinstance(x, jt.Var):
        x = jt.array(x)
    return x.float32()


def _reshape_sigma(sigma):
    # Convert sigma to [B, 1, 1, 1] for image broadcasting.
    sigma = _as_float32(sigma)
    return sigma.reshape(-1, 1, 1, 1)


def _var_ndim(x):
    # Jittor exposes shape as a list-like object, so len(shape) is enough.
    return len(x.shape)


def _prepare_class_labels(class_labels, batch_size, label_dim):
    # Prepare class labels for class-conditional networks.
    if label_dim == 0:
        return None

    if class_labels is None:
        return jt.zeros([batch_size, label_dim]).float32()

    if not isinstance(class_labels, jt.Var):
        class_labels = jt.array(class_labels)

    # Accept integer labels with shape [B] or [B, 1], and convert to one-hot.
    if _var_ndim(class_labels) == 1:
        eye = jt.array(np.eye(label_dim, dtype=np.float32))
        class_labels = eye[class_labels.int32()]
    elif _var_ndim(class_labels) == 2 and class_labels.shape[1] == 1 and label_dim != 1:
        eye = jt.array(np.eye(label_dim, dtype=np.float32))
        class_labels = eye[class_labels.reshape(-1).int32()]
    else:
        class_labels = class_labels.reshape(-1, label_dim)

    return class_labels.float32()


def _maybe_to_dtype(x, ref):
    # Keep dtype conversion compatible with different Jittor versions.
    if hasattr(x, "to"):
        try:
            return x.to(ref.dtype)
        except Exception:
            return x
    return x


class EDMPrecond(nn.Module):
    # Implement the EDM preconditioning wrapper used by DMD2.
    def __init__(
        self,
        img_resolution,
        img_channels,
        label_dim=0,
        use_fp16=False,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=0.5,
        model_type="DhariwalUNet",
        **model_kwargs,
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.model_type = model_type

        if model_type == "DhariwalUNet":
            model_cls = DhariwalUNet
        elif model_type == "SongUNet":
            model_cls = SongUNet
        else:
            raise NotImplementedError(
                f'Unsupported EDMPrecond model_type={model_type!r}.'
            )

        self.model = model_cls(
            img_resolution=img_resolution,
            in_channels=img_channels,
            out_channels=img_channels,
            label_dim=label_dim,
            **model_kwargs,
        )

    def execute(
        self,
        x,
        sigma,
        class_labels=None,
        force_fp32=False,
        return_bottleneck=False,
        **model_kwargs,
    ):
        # Apply EDM preconditioning and return denoised prediction D_x.
        x = _as_float32(x)
        sigma = _reshape_sigma(sigma)
        class_labels = _prepare_class_labels(
            class_labels=class_labels,
            batch_size=x.shape[0],
            label_dim=self.label_dim,
        )

        # The official PyTorch version optionally runs the UNet in bfloat16.
        # For the first stable Jittor migration, keep the network path in fp32.
        _ = force_fp32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / jt.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / jt.sqrt(self.sigma_data ** 2 + sigma ** 2)
        c_noise = jt.log(sigma) / 4

        F_x = self.model(
            c_in * x,
            c_noise.flatten(),
            class_labels=class_labels,
            return_bottleneck=return_bottleneck,
            **model_kwargs,
        )

        if return_bottleneck:
            return F_x

        F_x = _as_float32(F_x)

        # Official DMD2 uses x[:, :3]; here img_channels is used for safety.
        D_x = c_skip * x[:, : self.img_channels] + c_out * F_x
        return D_x

    def round_sigma(self, sigma):
        # EDM does not quantize sigma.
        return _as_float32(sigma)


def get_imagenet_edm_config():
    # Return the official ImageNet-64 EDM/DhariwalUNet configuration.
    return dict(
        augment_dim=0,
        model_channels=192,
        channel_mult=(1, 2, 3, 4),
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=(32, 16, 8),
        dropout=0.0,
        label_dropout=0,
    )


def get_tiny_imagenet_edm_config():
    # Match the official ImageNet-64 ADM EDM architecture; Tiny-ImageNet only
    # changes label_dim at the EDMPrecond wrapper level.
    return get_imagenet_edm_config()


def get_tiny_edm_config():
    # Return a small config for CPU/GPU smoke tests and CIFAR-like debugging.
    return dict(
        augment_dim=0,
        model_channels=32,
        channel_mult=(1, 2, 2),
        channel_mult_emb=4,
        num_blocks=1,
        attn_resolutions=(16,),
        dropout=0.0,
        label_dropout=0,
    )


def get_cifar10_edm_config():
    # Return the official EDM CIFAR-10 class-conditional DDPM++ configuration.
    return dict(
        augment_dim=9,
        model_channels=128,
        channel_mult=(2, 2, 2),
        channel_mult_emb=4,
        num_blocks=4,
        attn_resolutions=(16,),
        dropout=0.0,
        label_dropout=0,
        embedding_type="positional",
        channel_mult_noise=1,
        encoder_type="standard",
        decoder_type="standard",
        resample_filter=(1, 1),
    )


def get_edm_network(args=None, **kwargs):
    # Build an EDMPrecond network, matching the official get_edm_network style.
    if args is not None:
        dataset_name = getattr(args, "dataset_name", "imagenet")
        resolution = getattr(args, "resolution", 64)
        label_dim = getattr(args, "label_dim", 1000)
        use_fp16 = getattr(args, "use_fp16", False)
        sigma_data = getattr(args, "sigma_data", 0.5)
        config_name = getattr(args, "config_name", "imagenet")
    else:
        dataset_name = kwargs.pop("dataset_name", "imagenet")
        resolution = kwargs.pop("resolution", 64)
        label_dim = kwargs.pop("label_dim", 1000)
        use_fp16 = kwargs.pop("use_fp16", False)
        sigma_data = kwargs.pop("sigma_data", 0.5)
        config_name = kwargs.pop("config_name", "imagenet")

    if dataset_name not in (
        "imagenet",
        "tinyimagenet",
        "tiny_imagenet",
        "tiny_imagenet64",
        "tiny",
        "cifar10",
        "debug",
    ):
        raise NotImplementedError(f"Unsupported dataset_name: {dataset_name}")

    if dataset_name == "cifar10" and config_name != "tiny":
        model_config = get_cifar10_edm_config()
        model_type = "SongUNet"
    elif dataset_name in ("tinyimagenet", "tiny_imagenet", "tiny_imagenet64") and config_name != "tiny":
        model_config = get_tiny_imagenet_edm_config()
        model_type = "DhariwalUNet"
    elif config_name == "tiny" or dataset_name in ("tiny", "cifar10", "debug"):
        model_config = get_tiny_edm_config()
        model_type = "DhariwalUNet"
    else:
        model_config = get_imagenet_edm_config()
        model_type = "DhariwalUNet"

    model_config.update(kwargs)
    model_type = model_config.pop("model_type", model_type)

    return EDMPrecond(
        img_resolution=resolution,
        img_channels=3,
        label_dim=label_dim,
        use_fp16=use_fp16,
        sigma_min=0,
        sigma_max=float("inf"),
        sigma_data=sigma_data,
        model_type=model_type,
        **model_config,
    )


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7.0):
    # Create Karras noise schedule used by the DMD2 guidance network.
    ramp = jt.linspace(0, 1, n).float32()
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas


class EDMTeacherStudent(nn.Module):
    # Lightweight teacher/student holder for DMD2-style guidance experiments.
    def __init__(
        self,
        args=None,
        teacher=None,
        student=None,
        copy_teacher_to_student=True,
        num_train_timesteps=1000,
        sigma_min=0.002,
        sigma_max=80.0,
        rho=7.0,
        min_step_percent=0.02,
        max_step_percent=0.98,
        **network_kwargs,
    ):
        super().__init__()

        if args is not None:
            num_train_timesteps = getattr(args, "num_train_timesteps", num_train_timesteps)
            sigma_min = getattr(args, "sigma_min", sigma_min)
            sigma_max = getattr(args, "sigma_max", sigma_max)
            rho = getattr(args, "rho", rho)
            min_step_percent = getattr(args, "min_step_percent", min_step_percent)
            max_step_percent = getattr(args, "max_step_percent", max_step_percent)

        self.num_train_timesteps = num_train_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.min_step = int(min_step_percent * num_train_timesteps)
        self.max_step = int(max_step_percent * num_train_timesteps)

        if teacher is None:
            teacher = get_edm_network(args=args, **network_kwargs)

        if student is None:
            if copy_teacher_to_student:
                try:
                    student = copy.deepcopy(teacher)
                except Exception:
                    student = get_edm_network(args=args, **network_kwargs)
            else:
                student = get_edm_network(args=args, **network_kwargs)

        self.teacher = teacher
        self.student = student

        # Official code flips the Karras schedule so small sigma appears first.
        self.karras_sigmas_buffer = get_sigmas_karras(
            n=num_train_timesteps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
        )[::-1].stop_grad()

    @property
    def karras_sigmas(self):
        # Return the fixed Karras schedule.
        return self.karras_sigmas_buffer

    def sample_timesteps(self, batch_size, min_step=None, max_step=None):
        # Sample integer timesteps from the training interval.
        min_step = self.min_step if min_step is None else min_step
        max_step = self.max_step if max_step is None else max_step

        # jt.randint high is exclusive, so use max_step + 1.
        timesteps = jt.randint(
            low=min_step,
            high=min(max_step + 1, self.num_train_timesteps),
            shape=[batch_size, 1, 1, 1],
        )
        return timesteps.int32()

    def timestep_to_sigma(self, timesteps):
        # Map integer timestep tensor to sigma tensor.
        return self.karras_sigmas_buffer[timesteps]

    def add_noise(self, x, sigma=None, noise=None, timesteps=None):
        # Add Gaussian noise according to either sigma or sampled timesteps.
        if noise is None:
            noise = jt.randn_like(x)

        if sigma is None:
            if timesteps is None:
                timesteps = self.sample_timesteps(x.shape[0])
            sigma = self.timestep_to_sigma(timesteps)

        sigma = _reshape_sigma(sigma)
        noisy_x = x + sigma * noise
        return noisy_x, sigma, noise

    def teacher_forward(self, x, sigma, class_labels=None, **kwargs):
        # Forward through the teacher/real EDM network.
        return self.teacher(x, sigma, class_labels=class_labels, **kwargs)

    def student_forward(self, x, sigma, class_labels=None, **kwargs):
        # Forward through the student/fake EDM network.
        return self.student(x, sigma, class_labels=class_labels, **kwargs)

    def execute(self, x, sigma=None, class_labels=None, use_teacher=False, **kwargs):
        # Forward through teacher or student. Default is student.
        if sigma is None:
            x, sigma, _ = self.add_noise(x)

        if use_teacher:
            return self.teacher_forward(x, sigma, class_labels=class_labels, **kwargs)

        return self.student_forward(x, sigma, class_labels=class_labels, **kwargs)
