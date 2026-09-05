"""vRBA (variational residual-based attention) sampling primitives.

Faithful, self-contained port of the two core update rules from the official
reference (``external/vrba_ref/vrba_sample.py``, Karniadakis group, NPJ AI 2026),
refactored into two *pure* functions so they can be unit tested in isolation and
wired into the g3 A+1 training loop:

* :func:`vrba_frame_weights` -- bounded-EMA temporal attention on the time-frame
  axis.  ``Lambda <- gamma * Lambda + eta * lambda_it`` with
  ``lambda_it = phi * (q / q_max) + (1 - phi)``.  Because ``lambda_it in [1-phi, 1]``
  the steady-state weight is bounded above by ``eta / (1 - gamma)`` (the reference's
  ``Par['Lambda_max']``), which is exactly what keeps this from degenerating into
  the already-falsified ``late_frame_gain`` pathology (memory section 14): the
  attention cannot blow up an arbitrarily low-energy late frame, it saturates.
* :func:`vrba_record_pdf` -- turns per-record importance scores into a sampling PDF
  (normalize by max, mix with a uniform floor for coverage), consumed as
  ``record_weights`` by ``build_full_support_schedule``.

The potential functions ``q`` mirror the reference exactly:

============  ==========================  ================
potential     q(r)                        loss analogue
============  ==========================  ================
``linear``    ``1``                       (uniform)
``sublinear`` ``r ** 0.5``                --
``quadratic`` ``r``                       L2  (default, gentle)
``lp``        ``r ** (p_val - 1)``        Lp
``exponential`` ``beta * exp(beta * r)``  L-inf (aggressive)
``logarithmic`` ``log(beta * r + 1)``     log-safe entropy
============  ==========================  ================

``exponential`` and ``logarithmic`` follow the reference's iteration-annealed /
Newton-solved ``epsilon`` so ``beta = 1 / (epsilon + floor)`` is bounded; the
polynomial potentials are scale-free.  Everything is deterministic given inputs.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

__all__ = [
    "VRBA_POTENTIALS",
    "vrba_frame_weights",
    "vrba_record_pdf",
    "lambda_upper_bound",
]

VRBA_POTENTIALS = (
    "linear",
    "sublinear",
    "quadratic",
    "lp",
    "exponential",
    "logarithmic",
)

_EPS_FLOOR = 1.0e-6
_TINY = 1.0e-20


def lambda_upper_bound(*, gamma: float, eta: float) -> float:
    """Steady-state upper bound of the bounded EMA weight: ``eta / (1 - gamma)``."""

    gamma = float(gamma)
    eta = float(eta)
    if not (0.0 <= gamma < 1.0):
        raise ValueError("gamma must lie in [0, 1)")
    if eta < 0.0:
        raise ValueError("eta must be nonnegative")
    return eta / (1.0 - gamma)


def _newton_epsilon_logarithmic(
    r: torch.Tensor, *, n_newton: int = 20, eps_floor: float = _EPS_FLOOR
) -> torch.Tensor:
    """Solve ``mean(log(r / eps + 1)) = 1`` for eps over the reduction axis (dim 0).

    Direct port of the ``'logarithmic'`` branch of the reference
    ``get_exact_epsilon_torch``.  Returns a scalar tensor (keepdim).
    """

    r_mean = torch.mean(r, dim=0, keepdim=True)
    eps = r_mean / (2.718281828 - 1.0)
    for _ in range(n_newton):
        eps_safe = torch.maximum(eps, torch.tensor(eps_floor, device=r.device, dtype=r.dtype))
        u = r / eps_safe
        val = torch.mean(torch.log(u + 1.0), dim=0, keepdim=True) - 1.0
        d_term = -(1.0 / eps_safe) * u / (u + 1.0)
        grad = torch.mean(d_term, dim=0, keepdim=True)
        eps = eps - val / (grad - eps_floor)
    return torch.maximum(eps, torch.tensor(eps_floor, device=r.device, dtype=r.dtype))


def _potential_weights(
    r: torch.Tensor,
    *,
    potential: str,
    iteration: int,
    c_log: float,
    p_val: float,
) -> torch.Tensor:
    """Unnormalized vRBA potential ``q(r)`` over a 1-D residual vector (reduce dim 0)."""

    if potential == "linear":
        return torch.ones_like(r)
    if potential == "sublinear":
        return torch.pow(r, 0.5)
    if potential == "quadratic":
        return r.clone()
    if potential == "lp":
        return torch.pow(r, max(0.0, float(p_val) - 1.0))
    if potential == "exponential":
        r_max = torch.amax(r, dim=0, keepdim=True)
        log_k = torch.log(torch.tensor(float(iteration) + 2.0, device=r.device, dtype=r.dtype))
        epsilon_q = float(c_log) * r_max / log_k
        beta = 1.0 / (epsilon_q + _EPS_FLOOR)
        return beta * torch.exp(beta * r)
    if potential == "logarithmic":
        epsilon_q = _newton_epsilon_logarithmic(r)
        beta = 1.0 / (epsilon_q + _EPS_FLOOR)
        return torch.log(beta * r + 1.0)
    raise ValueError(f"unknown vRBA potential {potential!r}; choose from {VRBA_POTENTIALS}")


def _as_1d_float_tensor(values, name: str) -> torch.Tensor:
    tensor = values if isinstance(values, torch.Tensor) else torch.as_tensor(values)
    tensor = tensor.detach().to(dtype=torch.float64)
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a 1-D vector, got shape {tuple(tensor.shape)}")
    if tensor.numel() == 0:
        raise ValueError(f"{name} must be non-empty")
    if not torch.all(torch.isfinite(tensor)):
        raise ValueError(f"{name} must be finite")
    return tensor


def vrba_frame_weights(
    frame_residual,
    lambda_prev,
    *,
    gamma: float = 0.999,
    eta: float = 0.01,
    phi: float = 0.9,
    potential: str = "quadratic",
    iteration: int = 0,
    c_log: float = 1.0,
    p_val: float = 4.0,
) -> torch.Tensor:
    """Bounded-EMA temporal attention update on the time-frame axis.

    Parameters
    ----------
    frame_residual : 1-D array/tensor ``[T]``
        Per-frame residual magnitude (nonnegative; abs is applied defensively).
    lambda_prev : 1-D array/tensor ``[T]``
        Previous EMA weights (all zeros is the canonical cold start; the reference
        warm-starts at ``Lambda_max / 2``).
    gamma, eta : float
        EMA retention / injection.  Steady-state bound is ``eta / (1 - gamma)``.
    phi : float
        Fraction of the weight driven by the residual potential; ``1 - phi`` is a
        constant floor so every frame keeps some attention.
    potential : str
        One of :data:`VRBA_POTENTIALS`.
    iteration : int
        Global update index (only used by ``exponential`` annealing).

    Returns
    -------
    torch.Tensor
        Updated EMA weights ``[T]`` (float64), bounded by ``eta / (1 - gamma)`` when
        ``lambda_prev`` was.
    """

    if not (0.0 <= float(gamma) < 1.0):
        raise ValueError("gamma must lie in [0, 1)")
    if float(eta) < 0.0:
        raise ValueError("eta must be nonnegative")
    if not (0.0 <= float(phi) <= 1.0):
        raise ValueError("phi must lie in [0, 1]")
    residual = torch.abs(_as_1d_float_tensor(frame_residual, "frame_residual"))
    if torch.any(residual < 0.0):
        raise ValueError("frame_residual must be nonnegative")
    prev = _as_1d_float_tensor(lambda_prev, "lambda_prev")
    if prev.shape != residual.shape:
        raise ValueError(
            f"frame_residual {tuple(residual.shape)} and lambda_prev {tuple(prev.shape)} must match"
        )

    q = _potential_weights(
        residual, potential=potential, iteration=int(iteration), c_log=c_log, p_val=p_val
    )
    q_max = torch.amax(q, dim=0, keepdim=True)
    q_normalized = q / (q_max + _TINY)
    lambda_it = float(phi) * q_normalized + (1.0 - float(phi))
    return float(gamma) * prev + float(eta) * lambda_it


def vrba_record_pdf(
    record_lambda_scores,
    *,
    potential: str = "quadratic",
    uniform_fraction: float = 0.0,
    phi: float = 1.0,
    iteration: int = 0,
    c_log: float = 1.0,
    p_val: float = 4.0,
) -> np.ndarray:
    """Turn per-record importance scores into a normalized sampling PDF.

    Ports the reference ``update_function_pdf`` (normalize by max, ``phi`` mix) and
    adds the local RAD idiom's explicit ``uniform_fraction`` coverage floor
    (``src/fno_acoustic/train.py::_adaptive_sampling_weights``), so the result can be
    fed directly as ``record_weights`` to ``build_full_support_schedule``.

    Parameters
    ----------
    record_lambda_scores : 1-D array/tensor ``[N]``
        Nonnegative per-record importance (e.g. residual EMA, or summed spatial
        weights).
    potential : str
        One of :data:`VRBA_POTENTIALS`.
    uniform_fraction : float
        Convex mix toward the uniform ``1/N`` distribution; ``1.0`` collapses to a
        uniform PDF (coverage guarantee), ``0.0`` is pure vRBA.
    phi : float
        Reference ``phi`` mix on the max-normalized potential.

    Returns
    -------
    numpy.ndarray
        Probabilities ``[N]`` -- finite, nonnegative, summing to 1.
    """

    if not (0.0 <= float(uniform_fraction) <= 1.0):
        raise ValueError("uniform_fraction must lie in [0, 1]")
    if not (0.0 <= float(phi) <= 1.0):
        raise ValueError("phi must lie in [0, 1]")
    scores = _as_1d_float_tensor(record_lambda_scores, "record_lambda_scores")
    if torch.any(scores < 0.0):
        raise ValueError("record_lambda_scores must be nonnegative")
    n = scores.numel()

    q = _potential_weights(
        scores, potential=potential, iteration=int(iteration), c_log=c_log, p_val=p_val
    )
    q_max = torch.amax(q, dim=0, keepdim=True)
    q_normalized = q / (q_max + _TINY)
    lambda_it = float(phi) * q_normalized + (1.0 - float(phi))
    total = torch.sum(lambda_it)
    if float(total) <= 0.0:
        pdf = torch.full_like(lambda_it, 1.0 / n)
    else:
        pdf = lambda_it / total

    uniform = torch.full_like(pdf, 1.0 / n)
    frac = float(uniform_fraction)
    probs = (1.0 - frac) * pdf + frac * uniform
    probs = probs / probs.sum().clamp_min(_TINY)
    return probs.cpu().numpy()
