"""Unified EDM DMD2 model wrapper for Jittor."""

import copy
import jittor as jt
from jittor import nn

try:
    from .guidance import EDMGuidance, _stop_grad
except ImportError:
    from guidance import EDMGuidance, _stop_grad


def _get_arg(args, name, default):
    # Read an argparse-style attribute with a fallback.
    return getattr(args, name, default) if args is not None else default


class EDMUniModel(nn.Module):
    # Wrap feedforward generator and guidance model, matching official DMD2 flow.
    def __init__(
        self,
        args=None,
        accelerator=None,
        guidance_model=None,
        feedforward_model=None,
        initialize_generator=True,
        **guidance_kwargs,
    ):
        super().__init__()

        self.guidance_model = guidance_model
        if self.guidance_model is None:
            self.guidance_model = EDMGuidance(
                args=args,
                accelerator=accelerator,
                **guidance_kwargs,
            )

        self.guidance_min_step = self.guidance_model.min_step
        self.guidance_max_step = self.guidance_model.max_step

        initialize_generator = _get_arg(
            args,
            "initialie_generator",
            initialize_generator,
        )
        if feedforward_model is None:
            if not initialize_generator:
                raise NotImplementedError(
                    "Only support initializing generator from guidance model."
                )
            feedforward_model = copy.deepcopy(self.guidance_model.fake_unet)

        if hasattr(feedforward_model, "requires_grad_"):
            feedforward_model.requires_grad_(True)
        self.feedforward_model = feedforward_model

        self.accelerator = accelerator
        self.num_train_timesteps = self.guidance_model.num_train_timesteps

    def execute(
        self,
        scaled_noisy_image,
        timestep_sigma,
        labels,
        real_train_dict=None,
        compute_generator_gradient=False,
        generator_turn=False,
        guidance_turn=False,
        guidance_data_dict=None,
    ):
        # Run one generator or guidance branch.
        assert (generator_turn and not guidance_turn) or (
            guidance_turn and not generator_turn
        )

        if generator_turn:
            if compute_generator_gradient:
                generated_image = self.feedforward_model(
                    scaled_noisy_image,
                    timestep_sigma,
                    labels,
                )
            else:
                with jt.no_grad():
                    generated_image = self.feedforward_model(
                        scaled_noisy_image,
                        timestep_sigma,
                        labels,
                    )

            if compute_generator_gradient:
                generator_data_dict = {
                    "image": generated_image,
                    "label": labels,
                    "real_train_dict": real_train_dict,
                }
                self.guidance_model.requires_grad_(False)
                loss_dict, log_dict = self.guidance_model(
                    generator_turn=True,
                    guidance_turn=False,
                    generator_data_dict=generator_data_dict,
                )
                self.guidance_model.requires_grad_(True)
            else:
                loss_dict = {}
                log_dict = {}

            log_dict["generated_image"] = _stop_grad(generated_image)
            log_dict["guidance_data_dict"] = {
                "image": _stop_grad(generated_image),
                "label": _stop_grad(labels),
                "real_train_dict": real_train_dict,
            }
            return loss_dict, log_dict

        assert guidance_data_dict is not None
        return self.guidance_model(
            generator_turn=False,
            guidance_turn=True,
            guidance_data_dict=guidance_data_dict,
        )

