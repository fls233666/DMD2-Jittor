"""GAN classifier losses used by DMD2 guidance."""

import jittor as jt
from jittor import nn

try:
    from .common import sigmoid, softplus, stop_grad
except ImportError:
    from common import sigmoid, softplus, stop_grad


def generator_realism_loss(fake_logits):
    # Generator-side non-saturating realism loss: softplus(-D(fake)).
    return softplus(-fake_logits).mean()


def guidance_realism_loss(real_logits, fake_logits):
    # Guidance/classifier real-vs-fake loss from the official DMD2 code.
    return (softplus(fake_logits) + softplus(-real_logits)).mean()


def _std(x):
    x = x.float32()
    return jt.sqrt(((x - x.mean()) ** 2).mean())


def clean_classifier_losses(real_logits, fake_logits):
    # Return classifier loss dict and detached realism probabilities.
    real_prob = sigmoid(real_logits).reshape(-1)
    fake_prob = sigmoid(fake_logits).reshape(-1)
    real_logits_flat = real_logits.reshape(-1).float32()
    fake_logits_flat = fake_logits.reshape(-1).float32()
    real_acc = (real_prob > 0.5).float32().mean()
    fake_acc = (fake_prob <= 0.5).float32().mean()

    loss_dict = {
        "guidance_cls_loss": guidance_realism_loss(real_logits, fake_logits),
    }
    log_dict = {
        "pred_realism_on_real": stop_grad(real_prob),
        "pred_realism_on_fake": stop_grad(fake_prob),
        "gan/real_prob_mean": stop_grad(real_prob.mean()),
        "gan/fake_prob_mean": stop_grad(fake_prob.mean()),
        "gan/real_prob_std": stop_grad(_std(real_prob)),
        "gan/fake_prob_std": stop_grad(_std(fake_prob)),
        "gan/real_logits_mean": stop_grad(real_logits_flat.mean()),
        "gan/fake_logits_mean": stop_grad(fake_logits_flat.mean()),
        "gan/real_logits_std": stop_grad(_std(real_logits_flat)),
        "gan/fake_logits_std": stop_grad(_std(fake_logits_flat)),
        "gan/real_acc": stop_grad(real_acc),
        "gan/fake_acc": stop_grad(fake_acc),
        "gan/total_acc": stop_grad((real_acc + fake_acc) * 0.5),
        "gan/generator_fooling_rate": stop_grad((fake_prob > 0.5).float32().mean()),
    }
    return loss_dict, log_dict


class GeneratorRealismLoss(nn.Module):
    def execute(self, fake_logits):
        return generator_realism_loss(fake_logits)


class GuidanceRealismLoss(nn.Module):
    def execute(self, real_logits, fake_logits):
        return guidance_realism_loss(real_logits, fake_logits)
