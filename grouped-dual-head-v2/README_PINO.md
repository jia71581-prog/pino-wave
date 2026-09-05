# PINO HDF5 Acoustic FNO Adaptation

This branch adapts the original direct 2D acoustic FNO notebook to the local HDF5 dataset
`/home/jiayh/Data/data/pino.hdf5`. The original `Fourier_Acoustic_train.ipynb` is preserved; the new code lives in importable Python modules and CLI scripts.

## Source State

- Repository: `https://github.com/cuiyang512/FNO-Acoustic-Wave-Simulation.git`
- Starting commit: `269674d405cc98a5cc5094148afdab3d65558e5f`
- Adaptation branch: `codex/pino-hdf5-adaptation`

## Actual HDF5 Schema

The schema is recorded in `artifacts/pino_adaptation/dataset_schema.json` and `artifacts/pino_adaptation/dataset_schema.md`.

- Velocity: `/nu`, shape `[2500, 400, 400]`, axes `[sample, x, z]`, float32.
- Wavefield: `/tensor`, shape `[2500, 160, 400, 400]`, axes `[sample, time, x, z]`, float32.
- Canonical wavefield: `[N, H, W, T]` after converting `/tensor`.
- Source map: `/source_mask`, shape `[2500, 400, 400]`, axes `[sample, x, z]`.
- Source indices: `/source_x_idx`, `/source_z_idx`, grid-index units.
- Time: `/t-coordinate`, shape `[160]`, monotonic seconds from `0.0` to `0.4996`.
- Coordinates: `/x-coordinate`, `/y-coordinate`.
- Spacing: `/dx`, constant `5.0`; file attr `dt=0.0004` for full simulation, saved time spacing alternates `0.0028/0.0032` s.
- Frequency/amplitude: `/source_frequency_hz` constant `25.0`, `/source_amplitude` constant `1.0`; no extra conditioning channel is needed.

## Input Features

The direct baseline keeps the notebook mapping:

```text
velocity model + source representation + time coordinate -> selected full spatiotemporal pressure wavefield
```

Configured feature order:

```yaml
input_features: [time, source_map, velocity]
```

The model receives `[B, H, W, T, 3]` and predicts `[B, H, W, T]`.

## Split And Normalization

- Split strategy: fixed-seed random split over complete samples, never over individual time frames.
- Smoke split with `max_samples=8`: train `[0,1,3,5,6,7]`, val `[4]`, test `[2]`.
- Normalization: streaming Welford statistics computed only from train split samples.
- Saved stats: `artifacts/pino_adaptation/normalization_stats.json`.
- Source map normalization: max-to-one.
- Time normalization: zero-to-one over selected saved time values.

## Install

```bash
cd /home/jiayh/Data/FNO-Acoustic-Wave-Simulation
python -m pip install -r requirements-pino.txt
```

## Reproduce Inspection

```bash
python scripts/inspect_pino_hdf5.py \
  --path /home/jiayh/Data/data/pino.hdf5 \
  --json artifacts/pino_adaptation/dataset_schema.json \
  --markdown artifacts/pino_adaptation/dataset_schema.md \
  --sample-dir artifacts/pino_adaptation/dataset_samples
```

## Tests

```bash
python -m pytest -q
```

## Smoke

```bash
python scripts/smoke_test_pino.py \
  --config configs/pino_hdf5_smoke.yaml \
  --device cpu \
  --output-json artifacts/pino_adaptation/smoke_metrics_cpu.json

python scripts/smoke_test_pino.py \
  --config configs/pino_hdf5_smoke.yaml \
  --device cuda \
  --output-json artifacts/pino_adaptation/smoke_metrics_gpu.json
```

AMP was also tested with `artifacts/pino_adaptation/pino_hdf5_smoke_amp_runtime.yaml`. Because the FNO spectral weights are complex-valued, autocast is enabled but `GradScaler` is disabled.

## Full Training Command

This task did not start full training. To launch the baseline later:

```bash
python scripts/train_pino.py \
  --config configs/pino_hdf5.yaml
```

## Evaluation

```bash
python scripts/evaluate_pino.py \
  --config configs/pino_hdf5.yaml \
  --checkpoint artifacts/pino_adaptation/checkpoints/best.pt \
  --split test
```

For smoke checkpoint validation:

```bash
python scripts/evaluate_pino.py \
  --config configs/pino_hdf5_smoke.yaml \
  --checkpoint artifacts/pino_adaptation/checkpoints/best.pt \
  --split val
```

Evaluation metrics and plots use denormalized physical wavefields. Target and prediction share a per-time color range; error uses a separate symmetric range.

## Checkpoint Format

Checkpoints contain state dicts and metadata:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch`
- `global_step`
- `best_val_metric`
- `model_config`
- `data_config`
- `full_config`
- `normalization_stats`
- `split_manifest`
- `source_repo_commit`
- `adaptation_git_commit`

## Known Limits

- The full `400x400x160` direct spatiotemporal FNO is not launched automatically; configs downsample space/time to avoid accidental OOM.
- Baseline config uses `64x64` and up to `100` saved time snapshots; tune this against actual GPU memory before long training.
- AMP works with autocast, but scaler is disabled because PyTorch does not support scaler unscale for complex spectral-weight gradients in this environment.
- No PDE residual, autoregressive rollout, DRP stencil, or attention architecture was added; this remains a direct FNO baseline.
