"""EDM guidance model for Jittor DMD2 training."""

import copy
import jittor as jt
from jittor import nn

try:
    from .diffusion import get_edm_network, get_sigmas_karras
    from .discriminator import EDMRealismHead
except ImportError:
    from diffusion import get_edm_network, get_sigmas_karras
    from discriminator import EDMRealismHead

try:
    from loss.common import nan_to_num as _nan_to_num
    from loss.common import sigmoid as _sigmoid
    from loss.common import softplus as _softplus
    from loss.common import stop_grad as _stop_grad
    from loss.dmd_loss import distribution_matching_loss
    from loss.gan_loss import clean_classifier_losses, generator_realism_loss
    from loss.regression_loss import edm_fake_score_denoising_loss
except ImportError:
    from common import nan_to_num as _nan_to_num
    from common import sigmoid as _sigmoid
    from common import softplus as _softplus
    from common import stop_grad as _stop_grad
    from dmd_loss import distribution_matching_loss
    from gan_loss import clean_classifier_losses, generator_realism_loss
    from regression_loss import edm_fake_score_denoising_loss


def _get_arg(args, name, default):
    # Read an argparse-style attribute with a fallback.
    return getattr(args, name, default) if args is not None else default


class EDMGuidance(nn.Module):
    # Wrap real/fake EDM networks and optional bottleneck GAN classifier.
    def __init__(
        self,
        args=None,
        accelerator=None,
        real_unet=None,
        fake_unet=None,
        copy_real_to_fake=True,
        gan_classifier=None,
        diffusion_gan=None,
        diffusion_gan_max_timestep=None,
        num_train_timesteps=None,
        sigma_data=None,
        sigma_min=None,
        sigma_max=None,
        rho=None,
        min_step_percent=None,
        max_step_percent=None,
        cls_pred_branch=None,
        bottleneck_channels=None,
        bottleneck_resolution=None,
        **network_kwargs,
    ):
        super().__init__()

        self.args = args
        self.accelerator = accelerator

        self.sigma_data = _get_arg(args, "sigma_data", 0.5 if sigma_data is None else sigma_data)
        self.sigma_min = _get_arg(args, "sigma_min", 0.002 if sigma_min is None else sigma_min)
        self.sigma_max = _get_arg(args, "sigma_max", 80.0 if sigma_max is None else sigma_max)
        self.rho = _get_arg(args, "rho", 7.0 if rho is None else rho)

        self.gan_classifier = _get_arg(
            args,
            "gan_classifier",
            False if gan_classifier is None else gan_classifier,
        )
        self.diffusion_gan = _get_arg(
            args,
            "diffusion_gan",
            False if diffusion_gan is None else diffusion_gan,
        )
        self.diffusion_gan_max_timestep = _get_arg(
            args,
            "diffusion_gan_max_timestep",
            1 if diffusion_gan_max_timestep is None else diffusion_gan_max_timestep,
        )
        self.num_train_timesteps = _get_arg(
            args,
            "num_train_timesteps",
            1000 if num_train_timesteps is None else num_train_timesteps,
        )

        min_step_percent = _get_arg(
            args,
            "min_step_percent",
            0.02 if min_step_percent is None else min_step_percent,
        )
        max_step_percent = _get_arg(
            args,
            "max_step_percent",
            0.98 if max_step_percent is None else max_step_percent,
        )

        self.min_step = int(min_step_percent * self.num_train_timesteps)
        self.max_step = int(max_step_percent * self.num_train_timesteps)

        if real_unet is None:
            real_unet = get_edm_network(args=args, **network_kwargs)
        if hasattr(real_unet, "requires_grad_"):
            real_unet.requires_grad_(False)
        self.real_unet = real_unet

        if fake_unet is None:
            fake_unet = copy.deepcopy(real_unet) if copy_real_to_fake else get_edm_network(
                args=args,
                **network_kwargs,
            )
        if hasattr(fake_unet, "requires_grad_"):
            fake_unet.requires_grad_(True)
        self.fake_unet = fake_unet

        self.cls_pred_branch = cls_pred_branch
        if self.gan_classifier and self.cls_pred_branch is None:
            if bottleneck_channels is None:
                bottleneck_channels = _infer_bottleneck_channels(self.fake_unet, network_kwargs)
            if bottleneck_resolution is None:
                bottleneck_resolution = _infer_bottleneck_resolution(self.fake_unet, network_kwargs)

            self.cls_pred_branch = EDMRealismHead(
                in_channels=bottleneck_channels,
                hidden_channels=bottleneck_channels,
                bottleneck_resolution=bottleneck_resolution,
            )

        self.karras_sigmas_buffer = get_sigmas_karras(
            n=self.num_train_timesteps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
        )[::-1].stop_grad()

    @property
    def karras_sigmas(self):
        # Return the fixed Karras sigma schedule, small sigma first.
        return self.karras_sigmas_buffer

    def sample_timesteps(self, batch_size, min_step=None, max_step=None, shape=None):
        # Sample timestep indices.
        min_step = self.min_step if min_step is None else min_step
        max_step = self.max_step if max_step is None else max_step
        shape = [batch_size, 1, 1, 1] if shape is None else shape
        return jt.randint(
            low=min_step,
            high=min(max_step + 1, self.num_train_timesteps),
            shape=shape,
        ).int32()

    def timestep_to_sigma(self, timesteps):
        # Map integer timesteps to sigma values.
        return self.karras_sigmas_buffer[timesteps]

    def add_noise(self, latents, timesteps=None, sigma=None, noise=None):
        # Add Gaussian noise to clean images/latents.
        if noise is None:
            noise = jt.randn_like(latents)
        if sigma is None:
            if timesteps is None:
                timesteps = self.sample_timesteps(latents.shape[0])
            sigma = self.timestep_to_sigma(timesteps)

        sigma = sigma.reshape(-1, 1, 1, 1).float32()
        noisy_latents = latents + sigma * noise
        return noisy_latents, sigma, noise

    def compute_distribution_matching_loss(self, latents, labels):
        # Compute the DMD distribution matching loss for generator training.
        batch_size = latents.shape[0]

        with jt.no_grad():
            timesteps = self.sample_timesteps(batch_size=batch_size)
            noise = jt.randn_like(latents)
            timestep_sigma = self.timestep_to_sigma(timesteps)
            noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

            pred_real_image = self.real_unet(noisy_latents, timestep_sigma, labels)
            pred_fake_image = self.fake_unet(noisy_latents, timestep_sigma, labels)

        return distribution_matching_loss(
            latents=latents,
            pred_real_image=pred_real_image,
            pred_fake_image=pred_fake_image,
            noisy_latents=noisy_latents,
            timesteps=timesteps,
        )

    def compute_loss_fake(self, latents, labels):
        # Train the fake score model to denoise generated images.
        batch_size = latents.shape[0]
        latents = _stop_grad(latents)

        noise = jt.randn_like(latents)
        timesteps = self.sample_timesteps(
            batch_size=batch_size,
            min_step=0,
            max_step=self.num_train_timesteps - 1,
        )
        timestep_sigma = self.timestep_to_sigma(timesteps)
        noisy_latents = latents + timestep_sigma.reshape(-1, 1, 1, 1) * noise

        fake_x0_pred = self.fake_unet(noisy_latents, timestep_sigma, labels)
        loss_fake = edm_fake_score_denoising_loss(
            fake_x0_pred=fake_x0_pred,
            target_latents=latents,
            timestep_sigma=timestep_sigma,
            sigma_data=self.sigma_data,
        )

        loss_dict = {
            "loss_fake_mean": loss_fake,
        }
        log_dict = {
            "faketrain_latents": _stop_grad(latents),
            "faketrain_noisy_latents": _stop_grad(noisy_latents),
            "faketrain_x0_pred": _stop_grad(fake_x0_pred),
        }
        return loss_dict, log_dict

    def compute_cls_logits(self, image, label):
        # Compute bottleneck realism logits used by the optional GAN classifier.
        if self.cls_pred_branch is None:
            raise RuntimeError("gan_classifier=True requires cls_pred_branch.")

        if self.diffusion_gan:
            timesteps = jt.randint(
                low=0,
                high=self.diffusion_gan_max_timestep,
                shape=[image.shape[0]],
            ).int32()
            timestep_sigma = self.karras_sigmas[timesteps]
            image = image + timestep_sigma.reshape(-1, 1, 1, 1) * jt.randn_like(image)
        else:
            timesteps = jt.zeros([image.shape[0]]).int32()
            timestep_sigma = self.karras_sigmas[timesteps]

        rep = self.fake_unet(
            image,
            timestep_sigma,
            label,
            return_bottleneck=True,
        ).float32()
        logits = self.cls_pred_branch(rep)
        return logits

    def compute_generator_clean_cls_loss(self, fake_image, fake_labels):
        # Generator-side non-saturating realism loss.
        logits = self.compute_cls_logits(
            image=fake_image,
            label=fake_labels,
        )
        return {
            "gen_cls_loss": generator_realism_loss(logits),
        }

    def compute_guidance_clean_cls_loss(self, real_image, fake_image, real_label, fake_label):
        # Guidance-side real/fake classification loss.
        pred_real = self.compute_cls_logits(
            image=_stop_grad(real_image),
            label=real_label,
        )
        pred_fake = self.compute_cls_logits(
            image=_stop_grad(fake_image),
            label=fake_label,
        )

        return clean_classifier_losses(real_logits=pred_real, fake_logits=pred_fake)

    def generator_forward(self, image, labels):
        # Compute guidance losses needed for the generator update.
        loss_dict = {}
        log_dict = {}

        dm_dict, dm_log_dict = self.compute_distribution_matching_loss(image, labels)
        loss_dict.update(dm_dict)
        log_dict.update(dm_log_dict)

        if self.gan_classifier:
            loss_dict.update(self.compute_generator_clean_cls_loss(image, labels))

        return loss_dict, log_dict

    def guidance_forward(self, image, labels, real_train_dict=None):
        # Compute fake score and optional classifier losses.
        loss_dict, log_dict = self.compute_loss_fake(image, labels)

        if self.gan_classifier:
            if real_train_dict is None:
                raise ValueError("real_train_dict is required when gan_classifier=True.")
            cls_loss_dict, cls_log_dict = self.compute_guidance_clean_cls_loss(
                real_image=real_train_dict["real_image"],
                fake_image=image,
                real_label=real_train_dict["real_label"],
                fake_label=labels,
            )
            loss_dict.update(cls_loss_dict)
            log_dict.update(cls_log_dict)

        return loss_dict, log_dict

    def execute(
        self,
        generator_turn=False,
        guidance_turn=False,
        generator_data_dict=None,
        guidance_data_dict=None,
    ):
        # Dispatch the official DMD2 generator/guidance training branches.
        if generator_turn:
            return self.generator_forward(
                image=generator_data_dict["image"],
                labels=generator_data_dict["label"],
            )

        if guidance_turn:
            return self.guidance_forward(
                image=guidance_data_dict["image"],
                labels=guidance_data_dict["label"],
                real_train_dict=guidance_data_dict.get("real_train_dict"),
            )

        raise NotImplementedError("Specify generator_turn=True or guidance_turn=True.")


def _infer_bottleneck_channels(model, network_kwargs):
    # Infer the encoder bottleneck channel count from DhariwalUNet config.
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "_enc_items") and inner._enc_items:
            last = getattr(inner, inner._enc_items[-1][1])
            if hasattr(last, "out_channels"):
                return last.out_channels

    model_channels = network_kwargs.get("model_channels", 192)
    channel_mult = network_kwargs.get("channel_mult", (1, 2, 3, 4))
    return model_channels * channel_mult[-1]


def _infer_bottleneck_resolution(model, network_kwargs):
    # Infer the spatial size returned by return_bottleneck=True.
    if hasattr(model, "img_resolution") and hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "_enc_items") and inner._enc_items:
            name = inner._enc_items[-1][0]
            res_token = name.split("_", 1)[0]
            if "x" in res_token:
                try:
                    return int(res_token.split("x", 1)[0])
                except ValueError:
                    pass

    resolution = network_kwargs.get("img_resolution", network_kwargs.get("resolution", 64))
    channel_mult = network_kwargs.get("channel_mult", (1, 2, 3, 4))
    return max(int(resolution) >> (len(channel_mult) - 1), 1)
