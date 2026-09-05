# DCLP-NO Dispersion, Baselines, and Statistics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce matched baseline predictions and direct numerical-dispersion evidence over the sealed 480-record test set, including full wavefields and the fixed 37-receiver near-surface line.

**Architecture:** Analytic finite-difference symbols establish the discretization context. A shared evaluator computes field, wavefront, spectrum, phase, and receiver metrics from method-neutral arrays; paired group bootstrap and a machine-readable claim gate determine whether “dispersion suppression” is supportable.

**Tech Stack:** Python 3.13, PyTorch, NumPy, SciPy, h5py, pandas, Matplotlib, pytest, YAML.

---

## File Map

- Create `tgrs_dclp_no/dispersion.py`: FD2/FD4/LWC-84 symbols and phase velocity.
- Create `tgrs_dclp_no/metrics.py`: wavefield, wavefront, spectral, and receiver metrics.
- Create `tgrs_dclp_no/statistics.py`: paired group bootstrap, Holm correction, claim gate.
- Create `tgrs_dclp_no/method_registry.py`: immutable method/artifact identities.
- Create `scripts/analyze_tgrs_fd_dispersion.py`: analytic CSV/JSON/figure.
- Create `scripts/evaluate_tgrs_methods.py`: shared 480-record evaluation.
- Create `scripts/aggregate_tgrs_statistics.py`: tables, intervals, claim decision.
- Create `configs/tgrs_dclp_no/methods.yaml`: accepted method paths and labels.
- Create `configs/tgrs_dclp_no/fno3d_baseline.yaml`: standard FNO control.
- Create `configs/tgrs_dclp_no/factorized_fno_baseline.yaml`: FS-FNO control.
- Add tests under `tests/tgrs_dclp_no/`.

### Task 1: Generalize analytic dispersion to FD2, FD4, and LWC-84

**Files:**
- Create: `tests/tgrs_dclp_no/test_dispersion.py`
- Create: `tgrs_dclp_no/dispersion.py`
- Create: `scripts/analyze_tgrs_fd_dispersion.py`

- [ ] **Step 1: Write continuum-limit and ordering tests**

```python
from __future__ import annotations

from tgrs_dclp_no.dispersion import phase_velocity_ratio


def test_all_schemes_converge_to_unit_phase_velocity():
    for scheme in ("fd2", "fd4", "lwc84"):
        ratio = phase_velocity_ratio(
            scheme=scheme,
            points_per_wavelength=1000.0,
            angle_deg=31.0,
            courant_axis=0.05,
        )
        assert abs(ratio - 1.0) < 1.0e-4


def test_higher_order_space_is_more_accurate_at_eight_points_per_wavelength():
    values = {
        scheme: abs(
            phase_velocity_ratio(
                scheme=scheme,
                points_per_wavelength=8.0,
                angle_deg=45.0,
                courant_axis=0.10,
            )
            - 1.0
        )
        for scheme in ("fd2", "fd4", "lwc84")
    }
    assert values["fd4"] < values["fd2"]
    assert values["lwc84"] < values["fd4"]
```

- [ ] **Step 2: Verify failure**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_dispersion.py
```

Expected: import failure for `tgrs_dclp_no.dispersion`.

- [ ] **Step 3: Implement the analytic symbols**

Create `tgrs_dclp_no/dispersion.py` with:

```python
from __future__ import annotations

import math

import numpy as np


STENCILS = {
    "fd2": (
        np.asarray((1.0, -2.0, 1.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 1.0), dtype=np.float64),
        2,
    ),
    "fd4": (
        np.asarray((-1.0 / 12.0, 4.0 / 3.0, -2.5, 4.0 / 3.0, -1.0 / 12.0)),
        np.arange(-2.0, 3.0),
        2,
    ),
    "lwc84": (
        np.asarray(
            (-1.0 / 560.0, 8.0 / 315.0, -1.0 / 5.0, 8.0 / 5.0,
             -205.0 / 72.0, 8.0 / 5.0, -1.0 / 5.0, 8.0 / 315.0, -1.0 / 560.0)
        ),
        np.arange(-4.0, 5.0),
        4,
    ),
}


def spatial_symbol(scheme: str, theta: float) -> float:
    coefficients, offsets, _ = STENCILS[str(scheme)]
    return float(np.sum(coefficients * np.cos(float(theta) * offsets)))


def phase_velocity_ratio(
    *,
    scheme: str,
    points_per_wavelength: float,
    angle_deg: float,
    courant_axis: float,
) -> float:
    ppw = float(points_per_wavelength)
    if ppw <= 2.0 or float(courant_axis) <= 0.0:
        raise ValueError("points per wavelength and Courant number must be positive")
    angle = math.radians(float(angle_deg))
    kh = 2.0 * math.pi / ppw
    symbol = spatial_symbol(scheme, kh * math.cos(angle)) + spatial_symbol(
        scheme, kh * math.sin(angle)
    )
    q = -float(courant_axis) ** 2 * symbol
    time_order = STENCILS[str(scheme)][2]
    cosine = 1.0 - 0.5 * q
    if time_order == 4:
        cosine += q**2 / 24.0
    if not -1.0 <= cosine <= 1.0:
        raise ValueError("unstable dispersion sample")
    numeric_omega_dt = math.acos(cosine)
    exact_omega_dt = float(courant_axis) * kh
    return numeric_omega_dt / exact_omega_dt
```

- [ ] **Step 4: Implement the analytic CLI**

Sweep:

```python
schemes = ("fd2", "fd4", "lwc84")
ppw_values = np.linspace(4.0, 30.0, 261)
angles = np.arange(0.0, 91.0, 5.0)
courant = {"fd2": 0.25, "fd4": 0.20, "lwc84": 0.10}
```

Write:

- `artifacts/tgrs_dclp_no/dispersion/analytic/phase_velocity.csv`;
- `phase_velocity_summary.json`;
- `fig3a_phase_velocity.pdf`;
- `fig3a_phase_velocity.png`.

The JSON claim string is exactly:

```text
Finite-difference dispersion is quantified; no scheme is claimed dispersion free.
```

- [ ] **Step 5: Run tests and analytic generation**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_dispersion.py
/home/jiayh/miniconda3/bin/python scripts/analyze_tgrs_fd_dispersion.py \
  --output-dir artifacts/tgrs_dclp_no/dispersion/analytic
```

Expected: tests pass and all four artifacts exist.

### Task 2: Implement method-neutral wavefield and receiver metrics

**Files:**
- Create: `tests/tgrs_dclp_no/test_metrics.py`
- Create: `tgrs_dclp_no/metrics.py`

- [ ] **Step 1: Write synthetic lag and wavefront tests**

```python
from __future__ import annotations

import torch

from tgrs_dclp_no.metrics import (
    receiver_phase_metrics,
    wavefront_radius,
)


def test_receiver_lag_recovers_known_two_sample_delay():
    time = torch.arange(64, dtype=torch.float64)
    truth = torch.sin(2.0 * torch.pi * time / 16.0)[None]
    prediction = torch.roll(truth, shifts=2, dims=-1)
    result = receiver_phase_metrics(prediction, truth, dt_s=0.0025)
    assert result["median_lag_samples"] == 2
    assert abs(result["median_lag_s"] - 0.005) < 1.0e-12


def test_wavefront_radius_recovers_a_circular_energy_ring():
    axis = torch.arange(101, dtype=torch.float64) * 10.0
    z, x = torch.meshgrid(axis, axis, indexing="ij")
    radius = torch.sqrt((x - 500.0) ** 2 + (z - 500.0) ** 2)
    field = torch.exp(-0.5 * ((radius - 300.0) / 15.0) ** 2)
    estimated = wavefront_radius(
        field,
        source_x_m=500.0,
        source_z_m=500.0,
        x_m=axis,
        z_m=axis,
    )
    assert abs(estimated - 300.0) <= 10.0
```

- [ ] **Step 2: Verify failure**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_metrics.py
```

Expected: import failure for `tgrs_dclp_no.metrics`.

- [ ] **Step 3: Implement receiver phase metrics**

Create `tgrs_dclp_no/metrics.py`. The receiver input convention is
`[receiver,time]`. Implement:

```python
def _lag_one(prediction: torch.Tensor, target: torch.Tensor) -> int:
    left = prediction.double() - prediction.double().mean()
    right = target.double() - target.double().mean()
    size = int(left.numel())
    correlation = torch.fft.irfft(
        torch.fft.rfft(left, n=2 * size)
        * torch.conj(torch.fft.rfft(right, n=2 * size)),
        n=2 * size,
    )
    signed = int(correlation.argmax())
    if signed > size:
        signed -= 2 * size
    return signed


def receiver_phase_metrics(prediction, target, *, dt_s: float):
    predicted = torch.as_tensor(prediction).double()
    reference = torch.as_tensor(target).double()
    if predicted.shape != reference.shape or predicted.ndim != 2:
        raise ValueError("receiver traces must match [receiver,time]")
    lags = torch.tensor(
        [_lag_one(predicted[index], reference[index]) for index in range(predicted.shape[0])],
        dtype=torch.int64,
    )
    coherence = []
    for index, lag in enumerate(lags.tolist()):
        aligned = torch.roll(predicted[index], shifts=-lag)
        numerator = torch.dot(aligned, reference[index])
        denominator = aligned.norm() * reference[index].norm()
        coherence.append(float(numerator / denominator.clamp_min(1.0e-12)))
    median_lag = int(torch.median(lags))
    return {
        "median_lag_samples": median_lag,
        "median_lag_s": float(median_lag * float(dt_s)),
        "p95_absolute_lag_samples": float(torch.quantile(lags.abs().double(), 0.95)),
        "mean_coherence": float(torch.tensor(coherence).mean()),
    }
```

- [ ] **Step 4: Implement spatial dispersion metrics**

Add:

```python
def wavefront_radius(field, *, source_x_m, source_z_m, x_m, z_m):
    value = torch.as_tensor(field).double()
    x_axis = torch.as_tensor(x_m).double()
    z_axis = torch.as_tensor(z_m).double()
    zz, xx = torch.meshgrid(z_axis, x_axis, indexing="ij")
    radius = torch.sqrt((xx - float(source_x_m)) ** 2 + (zz - float(source_z_m)) ** 2)
    spacing = float(min(x_axis[1] - x_axis[0], z_axis[1] - z_axis[0]))
    bins = torch.round(radius / spacing).long()
    energy = torch.zeros(int(bins.max()) + 1, dtype=torch.float64)
    counts = torch.zeros_like(energy)
    energy.scatter_add_(0, bins.flatten(), value.square().flatten())
    counts.scatter_add_(0, bins.flatten(), torch.ones_like(value).flatten())
    profile = energy / counts.clamp_min(1.0)
    return float(profile.argmax() * spacing)
```

Also add method-neutral functions:

- `fullfield_metrics(prediction, truth, future_indices)`;
- `radial_zero_crossing_error(prediction, truth, source_xy, axes)`;
- `post_front_ringing_ratio(prediction, truth, source_xy, axes)`;
- `dominant_wavenumber_error(prediction, truth)`;
- `komega_ridge_error(prediction, truth, dt_s, dx_m, dz_m)`;
- `extract_near_surface_receivers(field, receiver_zx)`;
- `first_arrival_pick_error(prediction_traces, truth_traces, dt_s)`;
- `frequency_phase_residual(prediction_traces, truth_traces, dt_s)`.

Each function validates finite matching arrays, uses only supplied arrays, and
returns finite Python scalars. The first-arrival threshold is fixed at 5% of
each truth trace’s maximum analytic-envelope amplitude. The ringing region is
the annulus behind the truth wavefront by 2–8 stored cells.

- [ ] **Step 5: Run metric tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_metrics.py
```

Expected: all tests pass, including known lag, radius, zero crossing, and
frequency-phase synthetic cases.

### Task 3: Implement paired statistics and the claim gate

**Files:**
- Create: `tests/tgrs_dclp_no/test_statistics.py`
- Create: `tgrs_dclp_no/statistics.py`

- [ ] **Step 1: Write positive and negative claim tests**

```python
from __future__ import annotations

import numpy as np

from tgrs_dclp_no.statistics import dispersion_claim_gate, paired_group_bootstrap


def test_paired_bootstrap_detects_clear_improvement():
    baseline = np.arange(30, dtype=np.float64) + 10.0
    proposed = baseline - 2.0
    groups = np.asarray([f"g{index}" for index in range(30)])
    result = paired_group_bootstrap(
        proposed,
        baseline,
        groups,
        replicates=2000,
        seed=372,
    )
    assert result["ci95_high"] < 0.0


def test_dispersion_gate_requires_two_field_metrics_and_receiver_agreement():
    passed = dispersion_claim_gate(
        {
            "wavefront_radius_error": {"ci95_high": -0.1},
            "zero_crossing_error": {"ci95_high": -0.2},
            "receiver_lag": {"ci95_high": -0.01},
        }
    )
    failed = dispersion_claim_gate(
        {
            "wavefront_radius_error": {"ci95_high": -0.1},
            "zero_crossing_error": {"ci95_high": 0.2},
            "receiver_lag": {"ci95_high": -0.01},
        }
    )
    assert passed["dispersion_suppression_supported"]
    assert not failed["dispersion_suppression_supported"]
```

- [ ] **Step 2: Implement grouped bootstrap**

```python
def paired_group_bootstrap(proposed, baseline, group_ids, *, replicates, seed):
    proposed = np.asarray(proposed, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    groups = np.asarray(group_ids, dtype=str)
    if proposed.shape != baseline.shape or proposed.shape != groups.shape:
        raise ValueError("paired arrays and groups must match")
    unique = np.unique(groups)
    rng = np.random.default_rng(int(seed))
    observed = float(np.mean(proposed - baseline))
    draws = np.empty(int(replicates), dtype=np.float64)
    by_group = {group: np.flatnonzero(groups == group) for group in unique}
    for index in range(int(replicates)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        positions = np.concatenate([by_group[group] for group in sampled])
        draws[index] = np.mean(proposed[positions] - baseline[positions])
    return {
        "mean_difference": observed,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "replicates": int(replicates),
        "seed": int(seed),
    }
```

- [ ] **Step 3: Implement Holm correction and the claim gate**

`dispersion_claim_gate(results)` counts significant improvements among:

```python
field_metrics = (
    "wavefront_radius_error",
    "zero_crossing_error",
    "dominant_wavenumber_error",
    "komega_ridge_error",
    "post_front_ringing_ratio",
)
receiver_metrics = (
    "receiver_lag",
    "receiver_phase_residual",
    "receiver_coherence_error",
)
```

It returns:

```python
{
    "schema": "dclp_no_claim_gate_v1",
    "dispersion_suppression_supported": field_pass_count >= 2 and receiver_pass_count >= 1,
    "field_pass_count": field_pass_count,
    "receiver_pass_count": receiver_pass_count,
    "permitted_title_prefix": (
        "Dispersion-Controlled"
        if field_pass_count >= 2 and receiver_pass_count >= 1
        else "Physics-Conditioned"
    ),
    "forbidden_phrases": ["dispersion-free", "eliminates numerical dispersion"],
}
```

Holm-adjust the registered primary p-values before marking a metric passed.

- [ ] **Step 4: Run statistics tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_statistics.py
```

Expected: all tests pass.

### Task 4: Finish the parameter-matched Patch-DeepONet baseline

**Files:**
- Use existing `patch_deeponet_baseline/`.
- Use existing `configs/patch_deeponet/parammatched_seed372.yaml`.
- Use existing `scripts/train_patch_deeponet_baseline.py`.
- Write new outputs under `artifacts/tgrs_dclp_no/baselines/patch_deeponet/`.

- [ ] **Step 1: Run the complete baseline test suite**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/patch_deeponet_baseline
```

Expected: all tests pass. Fix only evidenced failures before continuing.

- [ ] **Step 2: Re-run the provenance, dry-run, and resume gates**

```bash
/home/jiayh/miniconda3/bin/python scripts/train_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --mode audit \
  --output-dir artifacts/tgrs_dclp_no/baselines/patch_deeponet/audit
/home/jiayh/miniconda3/bin/python scripts/train_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --mode dry-run \
  --output-dir artifacts/tgrs_dclp_no/baselines/patch_deeponet/dry_run
/home/jiayh/miniconda3/bin/python scripts/train_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --mode resume-probe \
  --output-dir artifacts/tgrs_dclp_no/baselines/patch_deeponet/resume_probe
```

Expected: each `terminal.json` reports `status=complete`.

- [ ] **Step 3: Require a three-family overfit pass**

```bash
/home/jiayh/miniconda3/bin/python scripts/train_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --mode overfit \
  --overfit-updates 400 \
  --output-dir artifacts/tgrs_dclp_no/baselines/patch_deeponet/overfit
```

Pass conditions:

- finite loss;
- non-collapsed prediction RMS;
- better than the zero predictor for every family;
- final full-field relative L2 below the initial value for every family.

If this gate fails, do not launch the 40-epoch run. Preserve the failure and use
the registered global MIONet coarse field as the primary DeepONet-family
baseline. The paper must then label Patch-DeepONet as an optimization-failure
diagnostic, not as the strongest baseline.

- [ ] **Step 4: Run learning-rate pilots only after the overfit pass**

Run 200 updates at `1e-4`, `3e-4`, and `1e-3` in distinct directories. Select
the lowest finite 48-record validation relative L2, breaking ties by lower
gradient-norm variance.

- [ ] **Step 5: Launch and verify the 40-epoch baseline**

Use the selected learning rate and a detached process. Record PID/PGID, command,
working directory, CUDA device, log, and output. Verify two advancing updates,
GPU utilization, and readable `latest.pt` before declaring launch success.

- [ ] **Step 6: Evaluate the selected checkpoint**

```bash
/home/jiayh/miniconda3/bin/python scripts/evaluate_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --checkpoint artifacts/tgrs_dclp_no/baselines/patch_deeponet/run/checkpoints/best.pt \
  --output-dir artifacts/tgrs_dclp_no/baselines/patch_deeponet/evaluation
```

Expected: strict checkpoint identity checks pass. Its legacy nine-receiver
plots are diagnostics only; paper receiver results come from the shared
37-receiver evaluator.

### Task 5: Train standard and factorized FNO controls on the registered VDS

**Files:**
- Modify: `src/fno_acoustic/data.py`
- Create: `tests/tgrs_dclp_no/test_fno_vds_adapter.py`
- Create: `scripts/freeze_tgrs_pino_split.py`
- Create: `configs/tgrs_dclp_no/fno3d_baseline.yaml`
- Create: `configs/tgrs_dclp_no/factorized_fno_baseline.yaml`

- [ ] **Step 1: Add an NTZX axis test**

Create a small HDF5 fixture with wavefield shape `[sample,time,z,x]`. Configure:

```yaml
wavefield_axes: [sample, time, z, x]
velocity_axes: [sample, z, x]
```

Assert `PinoHDF5Dataset[0]["target"]` is `[z,x,time]` and retains an asymmetric
known marker at the same `(z,x)` position.

- [ ] **Step 2: Add explicit NTZX/NZX branches**

In `PinoHDF5Dataset._read_wavefield`:

```python
if axes == ["sample", "time", "z", "x"]:
    raw = np.asarray(dset[sample_index, self.time_indices, :, :])
    return np.transpose(raw, (1, 2, 0))
```

In `_read_velocity`, when axes without sample equal `["z", "x"]`, return the
array without transposition. Add the same explicit branch to `_read_source_map`.
Existing axis behavior remains unchanged.

- [ ] **Step 3: Freeze a split file from the V3 manifest**

`freeze_tgrs_pino_split.py` writes the exact source indices from
`V3DataManifest.indices_by_split` as:

```json
{
  "seed": 372,
  "strategy": "registered_v3_split",
  "grouping_field": "group_id",
  "train": [],
  "val": [],
  "test": [],
  "sample_counts": {
    "train": 2240,
    "val": 480,
    "test": 480,
    "total_used": 3200
  }
}
```

The arrays contain actual registered source indices.

- [ ] **Step 4: Create matched-data FNO configs**

Both configs use:

```yaml
data:
  path: /home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5
  velocity_key: velocity_mps
  wavefield_key: wavefield
  source_map_key: source_map
  frequency_key: source_f0_hz
  amplitude_key: source_amplitude
  time_key: time_s
  velocity_axes: [sample, z, x]
  wavefield_axes: [sample, time, z, x]
  source_map_axes: [sample, z, x]
  input_features: [time, source_map, velocity]
  split_manifest: artifacts/tgrs_dclp_no/manifests/pino_split.json
sampling:
  target_height: 201
  target_width: 201
  max_time_steps: 16
  spatial_method: identity
train:
  device: cuda
  batch_size: 1
  epochs: 40
  learning_rate: 0.0003
  scheduler: cosine
  grad_clip: 1.0
loss:
  relative_l2_weight: 1.0
  mse_weight: 0.1
```

The standard FNO model is:

```yaml
model:
  name: acoustic_fno
  in_features: 3
  out_channels: 1
  modes_x: 16
  modes_z: 16
  modes_t: 8
  width: 16
  n_layers: 4
  padding_ratio: 0.0
  normalization: none
```

The factorized model is:

```yaml
model:
  name: factorized_fno
  in_features: 3
  out_channels: 1
  spatial_modes_x: 32
  spatial_modes_z: 32
  spatial_width: 24
  spatial_layers: 3
  temporal_width: 24
  temporal_kernel: 7
  temporal_layers: 3
  padding_ratio: 0.0625
  normalization: group
  num_groups: 8
  head_hidden: 128
  temporal_chunk_size: 2048
  activation_checkpointing: true
```

Document their parameter counts and that only Patch-DeepONet is
parameter-matched.

- [ ] **Step 5: Run one-batch memory and descent gates**

```bash
/home/jiayh/miniconda3/bin/python scripts/smoke_test_pino.py \
  --config configs/tgrs_dclp_no/fno3d_baseline.yaml \
  --device cuda \
  --output-json artifacts/tgrs_dclp_no/baselines/fno3d/smoke.json
/home/jiayh/miniconda3/bin/python scripts/smoke_test_pino.py \
  --config configs/tgrs_dclp_no/factorized_fno_baseline.yaml \
  --device cuda \
  --output-json artifacts/tgrs_dclp_no/baselines/factorized_fno/smoke.json
```

Require finite forward/backward and peak memory below 23.5 GiB. Reduce only
physical chunk size after an OOM; do not change spatial resolution, selected
frames, or model width without registering a new configuration.

- [ ] **Step 6: Train and select checkpoints**

Run `scripts/train_pino.py` for each config in detached jobs. Select best
checkpoints only by the registered validation split. Record the sampled-time
limitation prominently: these are spectral controls, while DCLP-NO directly
queries all 401 saved times.

### Task 6: Register method artifacts and run the shared evaluator

**Files:**
- Create: `configs/tgrs_dclp_no/methods.yaml`
- Create: `tgrs_dclp_no/method_registry.py`
- Create: `scripts/evaluate_tgrs_methods.py`

- [ ] **Step 1: Create the registry**

The registry contains entries for:

- `fd2_analytic`;
- `fd4_analytic`;
- `lwc84_coarse`;
- `fno3d`;
- `factorized_fno`;
- `patch_deeponet`;
- `global_mionet_coarse`;
- `dclp_no_parent`;
- `dclp_no_lora_rank4`.

Each entry records `kind`, artifact/checkpoint path, SHA-256, parameter count,
adaptation, accessible truth count, and evaluation manifest digest. An absent
or failed method remains in the registry with `eligible_for_primary=false` and
a concrete reason.

- [ ] **Step 2: Implement array-provider adapters**

`evaluate_tgrs_methods.py` normalizes every eligible field-producing method to:

```python
def predict_record(record) -> tuple[torch.Tensor, dict[str, object]]:
    """Return [401,201,201] normalized pressure and immutable method metadata."""
```

The LoRA provider verifies the adapter before reading truth. The parent and
global-coarse providers use the bound epoch-40 checkpoint. Baseline providers
validate checkpoint and split hashes.

- [ ] **Step 3: Evaluate the exact test manifest**

For each record and method, write one JSON row containing:

- full-field and active-energy relative L2;
- early/middle/late future errors;
- low/mid/high spectral errors;
- phase correlation;
- wavefront radius, zero-crossing, dominant-wavenumber, \(k\)-\(\omega\), and
  ringing errors where physically defined;
- 37-receiver L2, RMSE, lag, coherence, first-arrival, envelope, and
  frequency-phase errors;
- inference time and peak memory;
- sample/group/family identifiers;
- checkpoint and adapter hashes.

Use the two-cycle future interval for every learning method, including methods
that do not adapt.

- [ ] **Step 4: Verify completeness**

Require exactly:

```text
480 records × eligible field-producing methods
```

with no duplicate `(sample_id, method)` pair and no nonfinite primary metric.

### Task 7: Aggregate paired statistics and materialize the claim gate

**Files:**
- Create: `scripts/aggregate_tgrs_statistics.py`
- Generate: `artifacts/tgrs_dclp_no/statistics/claim_gate.json`
- Generate: `artifacts/tgrs_dclp_no/statistics/main_table.csv`
- Generate: `artifacts/tgrs_dclp_no/statistics/paired_intervals.csv`

- [ ] **Step 1: Select the strongest baseline without test cherry-picking**

Select the comparison baseline by lowest validation aggregate future
full-field relative L2 among eligible learning baselines. Freeze its name and
artifact digest before reading test aggregates.

- [ ] **Step 2: Compute paired intervals**

Use 10,000 bootstrap replicates, seed 372, group-ID resampling, and the same
record pairs for every method difference. Store raw and Holm-adjusted values.

- [ ] **Step 3: Write the claim decision**

`claim_gate.json` includes:

- strongest baseline identity;
- two or more required full-wavefield dispersion improvements;
- receiver-direction agreement;
- full-field LoRA improvement decision;
- rollback rate;
- permitted title prefix;
- permitted and forbidden wording.

- [ ] **Step 4: Run the full verification**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no
/home/jiayh/miniconda3/bin/python scripts/aggregate_tgrs_statistics.py \
  --metrics artifacts/tgrs_dclp_no/statistics/per_record_metrics.parquet \
  --validation-selection artifacts/tgrs_dclp_no/manifests/baseline_selection.json \
  --output-dir artifacts/tgrs_dclp_no/statistics \
  --bootstrap-replicates 10000 \
  --seed 372
```

Expected: all tests pass; claim gate and tables are finite, paired, and complete.

- [ ] **Step 5: Refresh the source checkpoint**

```bash
/home/jiayh/miniconda3/bin/python scripts/audit_tgrs_dclp_no.py \
  --config configs/tgrs_dclp_no/protocol.yaml \
  --source-root . \
  --output artifacts/tgrs_dclp_no/audit/source_manifest.json
```

Expected: exit code 0.
