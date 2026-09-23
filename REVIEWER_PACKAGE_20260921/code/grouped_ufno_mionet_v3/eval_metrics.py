"""Verbatim in-package copy of the frozen followup metrics module.

Provenance
----------
source        : research/epoch20_followup/metrics.py (master research tree
                grouped-dual-head-v2)
source sha256 : fa13e9a2e6190f75034fc69aa6bfd0775712a68499e9b9aa78904b6fffcc0573
                (whole source file, including its one-line module docstring)
carried body  : the source file from its ``from __future__ import annotations``
                line to EOF, byte for byte -- every algorithm byte and nothing
                else.  Its sha256 is pinned as ``METRICS_BODY_SHA256`` and
                re-derived by ``assert_metrics_provenance()``.
copied        : 2026-09-10 (release_ic8_40m_20260910)
rule          : no algorithm byte was edited.  A Python module has exactly one
                docstring, so the source's one-line docstring was replaced by
                this provenance header; everything from the ``__future__``
                import down is untouched.

Why this file exists
--------------------
``scripts/d0_fullfam_score_v2.py`` did ``sys.path.insert(0,
"research/epoch20_followup")`` followed by ``import metrics``.  That directory
does not exist inside the release, so ``import metrics`` could only resolve to
whatever ``metrics`` module the ambient environment happened to expose (none in
this environment: ``importlib.util.find_spec("metrics")`` is ``None``, which is
why the old failure was silent instead of loud).  The release now carries the
frozen module under an in-package name and imports it as
``grouped_ufno_mionet_v3.eval_metrics`` -- an absolute, package-qualified import
that cannot resolve to a site-packages, cwd, or ``sys.path`` shadow.

See reports/accuracy_refine_v1/EVAL_CLOSURE_CHANGES.md (DEFECT 1).
"""

from __future__ import annotations
import numpy as np

FREQUENCY_EDGES = (0., .005, .015, .03, float("inf"))  # radial cycles/metre

def stat(N, E, P, count, maximum):
    return {"N": float(N), "E": float(E), "P": float(P), "count": int(count),
            "relative_l2": None if E == 0 else float(np.sqrt(N / E)),
            "rmse": float(np.sqrt(N / count)) if count else None,
            "max_abs_error": float(maximum), "amplitude_norm_ratio": None if E == 0 else float(np.sqrt(P / E))}

def masks(x, z):
    xx, zz = np.meshgrid(x, z, indexing="xy")
    left, right, bottom, top = xx <= 200., xx >= 1800., zz >= 1800., zz <= 40.
    side = left | right | bottom
    union = side | top
    return {"all": np.ones_like(union), "side": side, "top": top, "union": union,
            "interior": ~union, "left": left, "right": right, "bottom": bottom,
            "top_line": zz == 0., "near_surface_10_60": (zz >= 10.) & (zz <= 60.)}

def half_weights(nx):
    w = np.full(nx // 2 + 1, 2., dtype=np.float64); w[0] = 1.
    if nx % 2 == 0: w[-1] = 1.
    return w

def checked_fields(prediction, truth, times, x, z):
    p, y = np.asarray(prediction, np.float64), np.asarray(truth, np.float64)
    if p.ndim != 3 or p.shape != y.shape or p.shape != (len(times), len(z), len(x)):
        raise ValueError("physical fields must match [time,z,x] and exact axes")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("nonfinite physical field")
    for axis in (times, x, z):
        axis = np.asarray(axis)
        if axis.ndim != 1 or not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
            raise ValueError("axes must be finite and strictly increasing")
    if len(x) < 2 or len(z) < 2:
        raise ValueError("spatial axes require at least two nodes")
    return p, y

def frequency_metrics(p, y, x, z, future):
    nz, nx = p.shape[-2:]
    dx, dz = float(x[1] - x[0]), float(z[1] - z[0])
    if not np.allclose(np.diff(x), dx, rtol=1e-12, atol=1e-12) or not np.allclose(np.diff(z), dz, rtol=1e-12, atol=1e-12):
        raise ValueError("spatial FFT requires uniform physical axes")
    rho = np.hypot(np.fft.fftfreq(nz, dz)[:, None], np.fft.rfftfreq(nx, dx)[None, :])
    weight = np.broadcast_to(half_weights(nx)[None, :], rho.shape)
    regions = [(rho >= a) & (rho < b) for a, b in zip(FREQUENCY_EDGES[:-1], FREQUENCY_EDGES[1:])]
    sums = np.zeros((4, 3), np.float64)
    for t in future:
        pf, yf = np.fft.rfft2(p[t], norm="ortho"), np.fft.rfft2(y[t], norm="ortho")
        ef = np.fft.rfft2(p[t] - y[t], norm="ortho")
        values = (np.abs(ef) ** 2, np.abs(yf) ** 2, np.abs(pf) ** 2)
        for j, mask in enumerate(regions):
            sums[j] += [np.sum(v[mask] * weight[mask], dtype=np.float64) for v in values]
    spatial = np.array([np.sum((p[future] - y[future]) ** 2), np.sum(y[future] ** 2), np.sum(p[future] ** 2)])
    if not np.allclose(sums.sum(0), spatial, rtol=1e-11, atol=0):
        raise RuntimeError("rFFT Parseval identity failed")
    result = []
    for j, (a, b) in enumerate(zip(FREQUENCY_EDGES[:-1], FREQUENCY_EDGES[1:])):
        N, E, P = sums[j]
        result.append({"radial_cycles_per_m": [a, b if np.isfinite(b) else None],
            "N": float(N), "E": float(E), "P": float(P), "relative_l2": None if E == 0 else float(np.sqrt(N / E)),
            "target_energy_fraction": None if spatial[1] == 0 else float(E / spatial[1]),
            "error_contribution": None if spatial[0] == 0 else float(N / spatial[0]),
            "absolute_band_rmse": float(np.sqrt(N / (len(future) * nx * nz))),
            "full_target_normalized_error": None if spatial[1] == 0 else float(np.sqrt(N / spatial[1]))})
    return {"bands": result, "parseval_verified": True, "window": "full future", "fft": "spatial rfft2 ortho; only last half-axis weighted"}

def evaluate(prediction, truth, times, x, z, onset):
    p, y = checked_fields(prediction, truth, times, x, z)
    n = len(times)
    if onset < 0 or onset + 8 >= n: raise ValueError("invalid IC/future window")
    future = np.arange(onset + 8, n)
    windows = {"full": np.arange(n), "pre_ic": np.arange(onset), "ic8": np.arange(onset, onset + 8),
               "future": future, "last_frame": np.array([n - 1])}
    for name, ix in zip(("future_early", "future_middle", "future_late"), np.array_split(future, 3)):
        windows[name] = ix
    squares = {"N": (p - y) ** 2, "E": y ** 2, "P": p ** 2}
    space = masks(np.asarray(x), np.asarray(z))
    region_frames = {}
    for name, mask in space.items():
        region_frames[name] = {k: value[:, mask].sum(1, dtype=np.float64) for k, value in squares.items()}
        region_frames[name]["max"] = np.sqrt(squares["N"][:, mask].max(1)) if mask.any() else np.zeros(n)
        region_frames[name]["nodes"] = int(mask.sum())
    full = region_frames["all"]
    for key in ("N", "E", "P"):
        if not np.allclose(region_frames["union"][key] + region_frames["interior"][key], full[key], rtol=1e-12, atol=0):
            raise RuntimeError("boundary union/interior decomposition failed")
    future_E, future_N = full["E"][future].sum(), full["N"][future].sum()
    report = {}
    for window, ix in windows.items():
        report[window] = {}
        for region, values in region_frames.items():
            N, E, P = (float(values[k][ix].sum()) for k in ("N", "E", "P"))
            s = stat(N, E, P, len(ix) * values["nodes"], values["max"][ix].max() if len(ix) else 0)
            s.update(target_fraction_of_future=None if future_E == 0 else E / future_E,
                     error_fraction_of_future=None if future_N == 0 else N / future_N)
            report[window][region] = s
    rms = np.sqrt(full["E"] / (len(x) * len(z))); peak = float(rms[future].max())
    frames = []
    for t in range(n):
        item = stat(full["N"][t], full["E"][t], full["P"][t], len(x) * len(z), full["max"][t])
        item.update(index=t, time_s=float(times[t]), target_rms=float(rms[t]),
                    zero_target_energy=bool(full["E"][t] == 0), low_energy=bool(full["E"][t] == 0 or rms[t] < .03 * peak), peak_future_normalized_rmse=None if peak == 0 else item["rmse"] / peak)
        frames.append(item)
    cumulative = {k: np.cumsum(full[k]).tolist() for k in ("N", "E", "P")}
    return {"shape": list(p.shape), "onset": int(onset), "future_frame_count": len(future),
            "windows": report, "per_frame": frames, "cumulative_prefix": cumulative,
            "low_energy_threshold": .03, "low_energy_never_removed": True,
            "regions_node_counts": {k: int(v.sum()) for k, v in space.items()},
            "spectrum": frequency_metrics(p, y, np.asarray(x), np.asarray(z), future)}

def aggregate(records):
    def summarize(items):
        values = [x["metrics"]["windows"]["future"]["all"]["relative_l2"] for x in items]
        finite = [v for v in values if v is not None]
        N = sum(x["metrics"]["windows"]["future"]["all"]["N"] for x in items)
        E = sum(x["metrics"]["windows"]["future"]["all"]["E"] for x in items)
        late = [x["metrics"]["windows"]["future_late"]["all"]["relative_l2"] for x in items]
        return {"records": len(items), "NA_records_retained": len(values) - len(finite),
                "future_record_mean": float(np.mean(finite)) if finite else None,
                "future_median": float(np.median(finite)) if finite else None,
                "future_P95_nearest_rank": float(np.sort(finite)[int(np.ceil(.95 * len(finite))) - 1]) if finite else None,
                "future_pooled_energy_relative_l2": None if E == 0 else float(np.sqrt(N / E)),
                "future_late_record_mean": float(np.mean([v for v in late if v is not None])) if any(v is not None for v in late) else None}
    families = {name: summarize([x for x in records if x["family"] == name]) for name in sorted({x["family"] for x in records})}
    means = [v["future_record_mean"] for v in families.values() if v["future_record_mean"] is not None]
    return {"all_records": summarize(records), "families": families,
            "family_balanced_future_mean": float(np.mean(means)) if means else None,
            "scope": "train development diagnostic, not held-out validation/test certification"}


# ---8<--- release provenance helpers ---8<---
import hashlib
import pathlib as _pathlib

#: sha256 of the *whole* frozen source file
#: research/epoch20_followup/metrics.py in grouped-dual-head-v2.
METRICS_SOURCE_SHA256 = "fa13e9a2e6190f75034fc69aa6bfd0775712a68499e9b9aa78904b6fffcc0573"

#: sha256 of the carried body: the source file from its
#: ``from __future__ import annotations`` line to EOF, byte for byte.
METRICS_BODY_SHA256 = "16e684f1fcfb7c581cdaf0b94b1e35699cc657d85815ec60fb07dfe8dcd86829"

_BODY_BEGIN = "from __future__ import annotations\n"
_HELPERS_BEGIN = "\n\n# ---8<--- release provenance helpers ---8<---\n"


def _frozen_body() -> str:
    text = _pathlib.Path(__file__).read_text(encoding="utf8")
    head, sep, rest = text.partition(_BODY_BEGIN)
    if not sep:
        raise RuntimeError("eval_metrics lost its frozen-body marker")
    body, sep2, _ = rest.partition(_HELPERS_BEGIN)
    if not sep2:
        raise RuntimeError("eval_metrics lost its provenance-helper marker")
    return _BODY_BEGIN + body


def metrics_body_sha256() -> str:
    """sha256 of the frozen source bytes carried by this module."""
    return hashlib.sha256(_frozen_body().encode("utf8")).hexdigest()


def metrics_body_matches_source() -> bool:
    """True when the carried body is byte-identical to the frozen source module."""
    return metrics_body_sha256() == METRICS_BODY_SHA256


def assert_metrics_provenance() -> None:
    """Raise unless this module still carries the frozen metrics implementation."""
    if not metrics_body_matches_source():
        raise RuntimeError(
            "grouped_ufno_mionet_v3.eval_metrics drifted from the frozen metrics "
            f"source (expected body {METRICS_BODY_SHA256}, got {metrics_body_sha256()})"
        )


__all__ = [
    "FREQUENCY_EDGES", "aggregate", "assert_metrics_provenance", "checked_fields",
    "evaluate", "frequency_metrics", "half_weights", "masks", "stat",
    "METRICS_BODY_SHA256", "METRICS_SOURCE_SHA256", "metrics_body_matches_source",
    "metrics_body_sha256",
]
