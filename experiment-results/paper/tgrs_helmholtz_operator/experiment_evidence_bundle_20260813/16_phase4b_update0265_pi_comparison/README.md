# Historical-best Phase4b checkpoint versus PI-DeepONet

This directory binds the paper figure for the same-record train-development comparison.

- Proposed checkpoint: Phase4b update 265, SHA-256 `f6efbb81dd1e9baab0eb34b32b125e2cb58cd3b3292ee0db33a29194c84bfea1`.
- Historical selection result: fixed-32-frame G3 relative L2 `0.040691912995691505`; all-401 G3 relative L2 `0.04165225127825517`.
- Same-record PI panel result: mean record relative L2 `0.10599196605053145` for Phase4b and `0.6156354159371821` for PI-DeepONet.
- The corresponding reduction is `82.78332218928962%`.
- Phase4b requires the sigma-2 smoothed-velocity LWC-84 background field. The comparison does not attribute this gain to the learned correction.
- The archived checkpoint was transferred to the repaired manifest with all 320 state tensors matched strictly. Predictions were hashed before future truth was opened.
- The six records and future windows are exactly those in the preregistered r34 PI comparison. They are train-development evidence, not held-out estimates.
- The DFO+RCFA instance-fine-tuning result remains a separate propagator-free row in the manuscript. No RCFA adapter was applied to the structurally incompatible Phase4b parent.

Primary result: `results/phase4b_update0265_vs_pi_train_r35_20260816/result.json`.

Figure audit: `figure_report.json`.

