"""Checkpoint helpers for Jittor DMD2 debug training."""

import os

import jittor as jt


def _state_dict(obj):
    if obj is None:
        return None
    if hasattr(obj, "state_dict"):
        return obj.state_dict()
    return obj


def _load_state(obj, state):
    if obj is None or state is None:
        return
    if hasattr(obj, "load_state_dict"):
        obj.load_state_dict(state)


def checkpoint_state(
    model,
    generator_optimizer=None,
    guidance_optimizer=None,
    ema=None,
    step=0,
    extra=None,
):
    return {
        "step": int(step),
        "model": _state_dict(model),
        "generator_optimizer": _state_dict(generator_optimizer),
        "guidance_optimizer": _state_dict(guidance_optimizer),
        "ema": _state_dict(ema),
        "extra": {} if extra is None else extra,
    }


def save_checkpoint(
    path,
    model,
    generator_optimizer=None,
    guidance_optimizer=None,
    ema=None,
    step=0,
    extra=None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = checkpoint_state(
        model=model,
        generator_optimizer=generator_optimizer,
        guidance_optimizer=guidance_optimizer,
        ema=ema,
        step=step,
        extra=extra,
    )
    jt.save(state, path)
    return state


def load_checkpoint(
    path,
    model=None,
    generator_optimizer=None,
    guidance_optimizer=None,
    ema=None,
    strict=True,
):
    if strict and not os.path.exists(path):
        raise FileNotFoundError(path)

    state = jt.load(path)
    _load_state(model, state.get("model"))
    _load_state(generator_optimizer, state.get("generator_optimizer"))
    _load_state(guidance_optimizer, state.get("guidance_optimizer"))
    _load_state(ema, state.get("ema"))
    return state
