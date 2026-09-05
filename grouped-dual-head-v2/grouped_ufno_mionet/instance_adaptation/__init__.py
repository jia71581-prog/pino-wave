"""Causal, onset-aligned adaptation utilities for the grouped operator."""

from .causal import OnsetWindow, WavefieldAccessAudit, hard_project_observations, onset_window
from .model import OnsetAdaptedOperator

__all__ = ["OnsetAdaptedOperator", "OnsetWindow", "WavefieldAccessAudit", "hard_project_observations", "onset_window"]
