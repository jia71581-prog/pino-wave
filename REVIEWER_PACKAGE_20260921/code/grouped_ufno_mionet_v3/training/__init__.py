"""Guarded V3 training and evaluation utilities."""

from .audit import audit_required_gradients
from .checkpoint import CHECKPOINT_FORMAT, load_checkpoint, save_checkpoint_atomic
from .gates import (
    NineRecordGateMetrics,
    OneRecordGateMetrics,
    evaluate_nine_record_gate,
    evaluate_one_record_gate,
    require_one_record_gate,
)
from .trainer import GuardedV3Trainer, PlateauDetector, fresh_full_batch_lbfgs
from .pilot import PilotIdentity, validate_pilot_benchmark, validate_pilot_prerequisite

__all__ = [
    "CHECKPOINT_FORMAT",
    "GuardedV3Trainer",
    "PlateauDetector",
    "OneRecordGateMetrics",
    "NineRecordGateMetrics",
    "PilotIdentity",
    "audit_required_gradients",
    "fresh_full_batch_lbfgs",
    "evaluate_one_record_gate",
    "evaluate_nine_record_gate",
    "load_checkpoint",
    "save_checkpoint_atomic",
    "require_one_record_gate",
    "validate_pilot_prerequisite",
    "validate_pilot_benchmark",
]
