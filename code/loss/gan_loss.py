"""GAN classifier losses used by DMD2 guidance."""

from jittor import nn

try:
    from .common import sigmoid, softplus, stop_grad
except ImportError:
    from common import sigmoid, softplus, stop_grad


def generator_realism_loss(fake_logits):
    """Generator-side non-saturating realism loss: softplus(-D(fake))."""

    return softplus(-fake_logits).mean()


def guidance_realism_loss(real_logits, fake_logits):
    """Guidance/classifier real-vs-fake loss from the official DMD2 code."""

    return (softplus(fake_logits) + softplus(-real_logits)).mean()


def clean_classifier_losses(real_logits, fake_logits):
    """Return classifier loss dict and detached realism probabilities."""

    loss_dict = {
        "guidance_cls_loss": guidance_realism_loss(real_logits, fake_logits),
    }
    log_dict = {
        "pred_realism_on_real": stop_grad(sigmoid(real_logits).reshape(-1)),
        "pred_realism_on_fake": stop_grad(sigmoid(fake_logits).reshape(-1)),
    }
    return loss_dict, log_dict


class GeneratorRealismLoss(nn.Module):
    def execute(self, fake_logits):
        return generator_realism_loss(fake_logits)


class GuidanceRealismLoss(nn.Module):
    def execute(self, real_logits, fake_logits):
        return guidance_realism_loss(real_logits, fake_logits)
