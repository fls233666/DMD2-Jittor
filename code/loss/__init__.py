"""Loss modules for the Jittor DMD2 migration."""

from .backward_simulation import (
    edm_backward_step,
    make_denoising_step_list,
    sample_backward_edm,
    select_backward_step,
)
from .dmd_loss import (
    DistributionMatchingLoss,
    compute_distribution_matching_loss,
    distribution_matching_loss,
    dmd_gradient,
)
from .gan_loss import (
    GeneratorRealismLoss,
    GuidanceRealismLoss,
    clean_classifier_losses,
    generator_realism_loss,
    guidance_realism_loss,
)
from .regression_loss import (
    EDMFakeScoreLoss,
    NoisePredictionLoss,
    X0RegressionLoss,
    edm_fake_score_denoising_loss,
    noise_prediction_loss,
    x0_loss_from_noise_prediction,
    x0_regression_loss,
)

__all__ = [
    "DistributionMatchingLoss",
    "EDMFakeScoreLoss",
    "GeneratorRealismLoss",
    "GuidanceRealismLoss",
    "NoisePredictionLoss",
    "X0RegressionLoss",
    "clean_classifier_losses",
    "compute_distribution_matching_loss",
    "distribution_matching_loss",
    "dmd_gradient",
    "edm_backward_step",
    "edm_fake_score_denoising_loss",
    "generator_realism_loss",
    "guidance_realism_loss",
    "make_denoising_step_list",
    "noise_prediction_loss",
    "sample_backward_edm",
    "select_backward_step",
    "x0_loss_from_noise_prediction",
    "x0_regression_loss",
]
