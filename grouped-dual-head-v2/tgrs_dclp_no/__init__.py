"""DCLP-NO numerical-dispersion analytics package.

Analytic finite-difference dispersion symbols (FD2 / FD4 / LWC-84), method-neutral
wavefield and receiver dispersion metrics, and the paired-bootstrap claim gate used to
decide whether a "dispersion suppression" claim is statistically supportable.

Nothing here trains or reads model checkpoints; the modules operate on plain arrays and
scheme labels so that every method (traditional solver, operator baseline, A+1) is scored
through one method-neutral path.
"""
from __future__ import annotations
