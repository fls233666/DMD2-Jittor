"""Exponential moving average helpers for Jittor models."""

import copy
import jittor as jt
from jittor import nn


def _detach_var(value):
    # Avoid Var.stop_grad(), which mutates the Var and can freeze live params.
    if hasattr(jt, "detach"):
        return jt.detach(value)
    if hasattr(value, "detach"):
        return value.detach()
    if hasattr(value, "stop_grad"):
        return value.stop_grad()
    return value


def _assign_var(dst, src):
    # Assign to a Jittor Var while staying compatible with older releases.
    if hasattr(dst, "assign"):
        dst.assign(src)
    elif hasattr(dst, "update"):
        dst.update(src)
    else:
        raise AttributeError("Jittor Var does not support assign/update.")


def copy_params_and_buffers(src_module, dst_module, require_all=False):
    # Copy matching entries from one module state_dict to another.
    src_state = src_module.state_dict()
    dst_state = dst_module.state_dict()

    missing = []
    for name, dst_value in dst_state.items():
        if name not in src_state:
            missing.append(name)
            continue

        src_value = _detach_var(src_state[name])
        _assign_var(dst_value, src_value)

    if require_all and missing:
        raise KeyError(f"Missing source parameters: {missing}")


class ExponentialMovingAverage:
    # Maintain an EMA copy of a Jittor module.
    def __init__(self, model, decay=0.9999, copy_model=True):
        self.decay = decay
        self.model = copy.deepcopy(model) if copy_model else model
        self.model.eval()

        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)

    def update(self, model, decay=None):
        # Update EMA parameters: ema = decay * ema + (1 - decay) * model.
        decay = self.decay if decay is None else decay
        ema_state = self.model.state_dict()
        model_state = model.state_dict()

        for name, ema_value in ema_state.items():
            if name not in model_state:
                continue

            value = _detach_var(model_state[name])

            updated = ema_value * decay + value * (1.0 - decay)
            _assign_var(ema_value, updated)

    def update_by_halflife(self, model, batch_size, ema_halflife_kimg, cur_nimg=0, rampup_ratio=None):
        # Match EDM's half-life schedule used during diffusion training.
        ema_halflife_nimg = ema_halflife_kimg * 1000
        if rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * rampup_ratio)

        decay = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
        self.update(model, decay=decay)
        return decay

    def state_dict(self):
        # Return the EMA model state dict.
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        # Load EMA model parameters.
        return self.model.load_state_dict(state_dict)

    def execute(self, *args, **kwargs):
        # Forward through the EMA model for convenience.
        return self.model(*args, **kwargs)

    __call__ = execute


class EMAModel(nn.Module):
    # Module wrapper variant that can be registered inside larger models.
    def __init__(self, model, decay=0.9999, copy_model=True):
        super().__init__()

        self.decay = decay
        self.model = copy.deepcopy(model) if copy_model else model
        self.model.eval()
        if hasattr(self.model, "requires_grad_"):
            self.model.requires_grad_(False)

    def update(self, model, decay=None):
        # Update EMA parameters.
        decay = self.decay if decay is None else decay
        ema_state = self.model.state_dict()
        model_state = model.state_dict()

        for name, ema_value in ema_state.items():
            if name not in model_state:
                continue
            value = _detach_var(model_state[name])
            _assign_var(ema_value, ema_value * decay + value * (1.0 - decay))

    def execute(self, *args, **kwargs):
        # Forward through the EMA copy.
        return self.model(*args, **kwargs)
