# Group 2: Near-surface receiver waveforms — method

## What
Receiver time series at z = 40 m (grid row 4; the free surface pins
p(z=0) = 0, so the shallowest informative rows are a few nodes down) and
x = 400 / 800 / 1200 / 1600 m, for the same two records as group 1.

Two figures per record:
- `receivers_<sid>`: truth vs neural operator overlay (left) + residual over
  the future window (right), 4 receivers stacked.
- `receivers_dispersion_<sid>`: truth vs neural vs coarse-grid LWC-84
  (51x51, dx = 40 m, from group 3) — the classical numerical-dispersion
  contrast: the coarse solve arrives late and rings (phase lag, waveform
  broadening), while the neural trace stays in phase but loses late-coda
  amplitude.

## Sources
- Truth and prediction: same as group 1 (same alignment check).
- Coarse LWC traces: `../group3_coarse_dispersion/<sid>_coarse{51,101}_on201.npy`
  (bilinear interpolation of the coarse solve back to 201x201; at these
  receiver nodes the 51x51 values are exact coarse nodes only when the index
  is a multiple of 4 — x=400/800/1200/1600 m and z=40 m all are, so the
  plotted coarse traces are native coarse-node values, not interpolants).

## Quantities (NUMBERS.json)
Per receiver, over the future window (onset+8..400), against the truth trace:
- `relative_l2`: trace relative L2 (float64);
- `lag_ms`: argmax of normalised cross-correlation, search +-100 ms;
  positive = arrival late vs truth (coarse51 shows +2.5..+7.5 ms on
  marmousi — numerical dispersion; the neural lag is 0 at most receivers);
- `peak_xcorr`, `amplitude_ratio` (trace norm / truth norm).

## Limitations
- Receiver relL2 at a single point is far noisier than the field metric;
  e.g. marmousi x=400 m sits behind the largest late-coda structure and all
  methods score poorly there.
- The cross-correlation lag is a whole-trace summary; it under-reports
  dispersion when early arrivals dominate the correlation.
- Train-split dev records; see group 1 limitations.

## Products
- `receivers_train_{marmousi_00076,layered_00299}.{png,pdf}`
- `receivers_dispersion_train_{marmousi_00076,layered_00299}.{png,pdf}`
- `NUMBERS.json`; script `../scripts/make_group2_receivers.py`
