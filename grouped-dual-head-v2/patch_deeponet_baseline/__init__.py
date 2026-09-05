"""Independent, target-free Patch-DeepONet comparison baseline."""

from .features import build_query_descriptors, build_static_features
from .model import PatchDeepONet, PatchDeepONetConfig, resolve_parameter_matched_width
from .training import PatchDeepONetLoss, dense_training_pair, patch_deeponet_loss

__all__ = [
    "PatchDeepONet",
    "PatchDeepONetConfig",
    "PatchDeepONetLoss",
    "build_query_descriptors",
    "build_static_features",
    "dense_training_pair",
    "patch_deeponet_loss",
    "resolve_parameter_matched_width",
]
