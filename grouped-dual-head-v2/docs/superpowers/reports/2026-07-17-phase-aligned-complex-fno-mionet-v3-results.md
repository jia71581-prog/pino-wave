# Phase-Aligned Complex-FNO Multi-Input DeepONet V3 Results

Date: 2026-07-17

Branch: `codex/grouped-dual-head-v2`

Implementation worktree:
`/home/jiayh/.config/superpowers/worktrees/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2`

No GitHub push or production launch was performed.

## Outcome

V3 implements a source-isolated acoustic neural operator with two output interfaces:

- arbitrary normalized or physical pressure queries at `(x,z,t)`;
- complete `201×201` normalized or physical wavefield snapshots at requested times.

Receiver observations are absent from model inputs and primary losses. Each record contains
one source only. A velocity model is encoded once and may be reused by multiple independent
source records through `record_to_medium`; their wavefields are never superposed.

The architecture retains the useful V2 contracts: grouped medium reuse, a multiscale local
pyramid plus global rank, source-map and local-medium source features, a direct local query
residual with token attention, query/dense true-target supervision, source/time FiLM dense
conditioning, the exact free-surface factor, and one-time amplitude decoding. V3 replaces
the fixed low-pass FFT behavior with learned complex mode contractions and adds phase-aligned
travel-time/periodic features plus explicit multi-input DeepONet/MIONet fusion.

## Data and identity

- Source VDS: `/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5`
- Allowed families: `uniform`, `layered`, `marmousi`
- Excluded family: `anomaly`
- Filtered train/validation counts: `2240 / 480`
- Manifest digest: `eb776ba2cf782235c23f60d558a1e0bc6b4711d73653faa8b29cc4ed413c6c3b`
- Normalization: `/home/jiayh/Data/data/processed/grouped_v3_normalization.json`
- Normalization train-record count: `2240`
- Pressure scale: `2.7270570059912283e-08`
- Model parameter count: `10,126,581`
- Checkpoint format: `phase_aligned_complex_fno_mionet_v3`

The nine pinned records comprise exactly three records from each allowed family. No anomaly
record is present.

## Gate results

### One-record prerequisite

The identity-bound one-record refinement passed at step 50:

| Metric | Value | Threshold |
|---|---:|---:|
| early relative L2 | 0.01218 | < 0.05 |
| middle relative L2 | 0.02119 | < 0.05 |
| late relative L2 | 0.01827 | < 0.05 |
| query relative L2 | 0.03038 | < 0.05 |
| radial centroid displacement | 0.244 m | < 10 m |

Evidence:

- checkpoint: `artifacts/grouped_ufno_mionet_v3/one_record_finetune/best.pt`
- report: `artifacts/grouped_ufno_mionet_v3/one_record_finetune/passed_report.json`

### Balanced nine-record gate

The initial AdamW phase reached step 2000 without relaxing the gate. Its complete-field
aggregate was already below threshold, but the layered point-query metric remained above
threshold. A parent-hash-bound refinement used learning rate `3e-5`, increased point and
query/dense consistency supervision, and retained dense, complex-spectrum, phase, spatial,
and temporal-difference losses. It passed at refinement step 750.

| Metric | uniform | layered | marmousi | Aggregate |
|---|---:|---:|---:|---:|
| arbitrary-query relative L2 | 0.08019 | 0.09901 | 0.06430 | 0.08117 |
| complete-field relative L2 | 0.05718 | 0.06945 | 0.05303 | 0.05989 |
| late-four-frame relative L2 | 0.05818 | 0.08070 | 0.05539 | — |

All required values are finite, below `0.10`, and below the zero-prediction baseline of
`1.0`. All required gradient groups were present. Record counts were exactly `3/3/3`.

Sealed evidence:

- checkpoint: `artifacts/grouped_ufno_mionet_v3/nine_record_finetune/best.pt`
- checkpoint SHA-256: `99dc25dddc6ebc9b3498c9ad7b0e370ade19a9737d8eabae3764a5afcec081f1`
- passed report: `artifacts/grouped_ufno_mionet_v3/nine_record_finetune/passed_report.json`
- report SHA-256: `e594cb7ef63409ccdfa466ba38fdeb77e6aa2ad2409cd7b83f2475de650e031d`

## Reproducible commands

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/grouped_ufno_mionet_v3

CUDA_VISIBLE_DEVICES=0 /home/jiayh/miniconda3/bin/python -u \
  scripts/overfit_grouped_v3_nine.py \
  --config configs/grouped_v3/nine_record_finetune.yaml \
  --device cuda \
  --artifact-dir artifacts/grouped_ufno_mionet_v3/nine_record_finetune \
  --init-checkpoint artifacts/grouped_ufno_mionet_v3/nine_record_gate/best.pt \
  --one-record-report artifacts/grouped_ufno_mionet_v3/one_record_finetune/passed_report.json
```

Representative evaluation:

```bash
/home/jiayh/miniconda3/bin/python scripts/evaluate_grouped_v3.py \
  --config configs/grouped_v3/one_record_gate.yaml \
  --checkpoint artifacts/grouped_ufno_mionet_v3/nine_record_finetune/best.pt \
  --sample-id train_layered_00883 \
  --output-dir artifacts/grouped_ufno_mionet_v3/nine_record_finetune/evaluation_train_layered_00883 \
  --report-only --device cuda
```

## Output API

After constructing the model, normalizer, velocity tensor, source parameters, and unit-mass
source maps, the same prepared state serves both interfaces:

```python
medium = model.encode_medium(velocity, normalizer)
prepared = model.prepare_sources(
    medium,
    sources,
    source_maps,
    normalizer,
    record_to_medium=record_to_medium,
)

# Off-grid values; coords has shape [records, points, 3] and stores (x, z, t).
point_pressure = model.query_pressure(prepared, coords, chunk_size=2048)

# Complete physical wavefields with shape [records, times, 201, 201].
wavefields = model.predict_wavefield(prepared, times, x_m=x_m, z_m=z_m)

# Bounded-memory streaming for many requested times.
for start, normalized_block in model.iter_dense_normalized(
    prepared, times, x_m=x_m, z_m=z_m, time_block=16
):
    consume(start, normalized_block)
```

The API contains no receiver input. A receiver trace is just a sequence of point queries at
fixed `(x,z)` and requested times.

## Figures and diagnostics

Each representative directory contains `wavefield_comparison.pdf/.png`,
`point_waveforms.pdf/.png`, and `evaluation_report.json`:

- `artifacts/grouped_ufno_mionet_v3/nine_record_finetune/evaluation_train_uniform_00370/`
- `artifacts/grouped_ufno_mionet_v3/nine_record_finetune/evaluation_train_layered_00883/`
- `artifacts/grouped_ufno_mionet_v3/nine_record_finetune/evaluation_train_marmousi_00096/`

The wavefield figures evaluate exact saved frames. The point-waveform figures query all 401
available times only as derived diagnostics; they are not receiver-conditioned inputs or
training targets.

## Known limitations and pilot boundary

- The bounded gates supervise and certify 16 exact saved frames. Although the data layer and
  API support adjacent-frame interpolation and arbitrary requested times, the current gate
  checkpoint does **not** certify continuous-time generalization.
- Full 401-time diagnostic point traces remain materially worse than exact-frame results,
  especially for near-zero target traces where relative L2 is ill-conditioned. A production
  claim would be incorrect at this stage.
- The nine-record checkpoint optimizes balanced multi-record performance and no longer meets
  the earlier uniform single-record memorization threshold `<0.05`; this is not used to judge
  the sealed nine-record gate.
- Straight-ray travel time is an inductive approximation; reflection, refraction, scattering,
  and boundaries are learned residuals.
- FWI, instance adaptation, receiver conditioning, anomaly-family training, full-dataset
  pilot training, production launch, and speed benchmarking against LWC-84 remain out of
  scope.
- Any future pilot must consume the sealed nine-record passed report, enable and separately
  report interpolated-time supervision/validation, preserve one-source-per-record grouping,
  and receive separate user authorization before launch.
