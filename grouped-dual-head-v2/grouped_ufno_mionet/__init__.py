"""Grouped single-source U-FNO/MIONet operator.

This package is intentionally independent from ``continuous_wave_operator``.
The public operator keeps each source record separate while allowing a medium
encoding to be reused for records that share the same velocity field.
"""

from .config import DataConfig, ModelConfig, OperatorConfig, TrainConfig
from .contracts import GroupedMacroBatch, SingleSourceRecord, SourceParameters
def __getattr__(name):
    # Keep config/contracts/data imports independent of the optional model
    # stack.  Model construction imports torch layers only when requested.
    if name == "GroupedSingleSourceUFNOMIONetOperator":
        from .model.operator import GroupedSingleSourceUFNOMIONetOperator
        return GroupedSingleSourceUFNOMIONetOperator
    raise AttributeError(name)

__all__ = [
    "DataConfig", "ModelConfig", "OperatorConfig", "TrainConfig",
    "GroupedMacroBatch", "SingleSourceRecord", "SourceParameters",
    "GroupedSingleSourceUFNOMIONetOperator",
]
