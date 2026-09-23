# Group 1: Wavefield snapshot comparison — method

## What
3x3 panels (rows = early/middle/late physical time, columns = truth / neural
prediction / difference) for two dev records:
- `train_marmousi_00076` (h5 index 2176, onset frame 52, f0 = 11.55 Hz)
- `train_layered_00299` (h5 index 719, onset frame 53, f0 = 11.44 Hz)

Both are the family-median records (by future relL2) of the 9-record gap15
dev evaluation (`future_median` per family in EVALUATION.json).

## Sources
- Truth: `/root/autodl-tmp/data/jiayh/data/acoustic_lwc84_frequency_gap_20260922_v1/combined_dataset_v1.h5`,
  dataset `wavefield`, axis order NTZX, (4067, 401, 201, 201) float32,
  indexed by matching `sample_id`.
- Prediction: `/root/autodl-tmp/staging/scno_homogeneous_longrun_gap15_v1/evaluations/update_00005000/attempt_001/<sid>_prediction.npy`
  (T_future, 201, 201) float32, frame k maps to stored frame onset+8+k.
  Alignment was verified by recomputing the future-window relative L2 and
  matching EVALUATION.json to 1e-6 (marmousi 0.27851, layered 0.19600).

## Caliber
- Frame choice: future window starts at onset+8 (after the 8 IC frames).
  Early = start + 8% of the window (past the immediate IC continuation),
  middle = window midpoint, late = last stored frame (t = 1.000 s).
  Physical times printed on every panel (dt_output = 2.5 ms).
- Colour: `seismic`, vmin = -vmax. Truth and prediction share one symmetric
  scale per row (max |.| over both); the difference column has its own
  symmetric scale per row (stated in NUMBERS.json, `error_scale_pa`).
- Axes: arrays are [z, x] (depth first), plotted WITHOUT transpose,
  extent [0, 2000, 2000, 0] m so depth increases downward.
- Per-frame relative L2 printed in each difference title.

## Limitations
- Dev-monitoring records from the train split (validation/test untouched);
  these figures illustrate error character, not held-out certification.
- The 9-record dev set has 3 records/family; "family median" is over 3.
- Model: scno_homogeneous_longrun_gap15_v1 at update 5000 (the artifact the
  task designated), not the historical anchor 29359.

## Products
- `snapshots_train_marmousi_00076.{png,pdf}`, `snapshots_train_layered_00299.{png,pdf}`
- `NUMBERS.json` (times, per-frame relL2, colour scales)
- Script: `../scripts/make_group1_snapshots.py`
