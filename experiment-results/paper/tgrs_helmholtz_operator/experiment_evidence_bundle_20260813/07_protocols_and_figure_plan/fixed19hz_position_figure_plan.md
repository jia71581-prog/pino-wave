# Marmousi multi-source generalization figure plan

## Main claim

On one input-only selected, held-out Marmousi medium, the frozen operator must reconstruct distinct complete transients for eight source positions at one fixed source frequency, amplitude, onset rule, and wavelet, without systematic late-time phase drift.

## Selection lock

- Dataset: `dataset_v1.h5`, validation split.
- Group: `validation:marmousi:x350.0:z0.0`.
- Rule: among the 30 validation Marmousi groups, rank the input velocity maps by mean squared first-difference energy and take lower median rank 15 of 30, breaking ties by `group_id`.
- This rule used only `split`, `medium_type`, `group_id`, `velocity_mps`, source metadata, and completion/QC metadata. It did not read `wavefield`, predictions, or errors.
- The original five records are not used in the position-generalization figure because their source frequencies vary.
- The controlled panel fixes $f_0=19$ Hz, amplitude 1, $t_0=1.5/f_0$, and the same Ricker wavelet. It varies position only.
- In-range positions are `(400,100)`, `(700,250)`, `(1000,175)`, `(1300,100)`, and `(1600,250)` m. Outside-training-range positions are `(100,175)`, `(1900,175)`, and `(1000,400)` m.
- The quantitative study repeats these eight positions on the complete census of all 30 validation Marmousi velocity slices. Input-only roughness rank 15 remains the preregistered main qualitative panel.

## Main-figure panels

| Panel | Content | Purpose |
|---|---|---|
| a | Marmousi velocity slice with all eight preregistered source positions, marking interpolation and outside-range positions distinctly. | Establish identical medium, fixed source signature, and position diversity. |
| b | For each position, late representative time selected globally before evaluation: reference, prediction, and signed error on common colour limits. | Test distinct wavefront geometry and late coda without per-record cherry-picking. |
| c | Shallow-receiver common-shot gathers at $z=40$ m for all eight positions: reference, prediction, and difference, with identical gain. | Make phase drift and event alignment visible without using the Dirichlet zero-pressure surface. |
| d | Per-source complete, late-window, and high-band relative errors; receiver lag and coherence. | Prevent qualitative overclaiming. |
| e | Metrics versus Euclidean distance to the nearest Marmousi training source position, with interpolation and outside-range roles shown separately. | Establish whether performance degrades outside spatial source support. |

## Supplementary figure

Show reference, prediction, and error for all eight source positions at three globally locked times: an early direct-arrival time, a middle scattering time, and a late-coda time. Never select each position's best-looking time independently.

The locked indices are 80, 240, and 400, corresponding to 0.2, 0.6, and 1.0 s. The receiver line is $z=40$ m (saved-grid index 4), with $x=100,120,\ldots,1900$ m. The pressure at $z=0$ is identically zero under the free-surface Dirichlet condition and is therefore not used as evidence.

## Rendering rules

- Use the same physical extent and aspect ratio for every spatial panel.
- Use one symmetric pressure colour scale per time across all sources and methods. Give errors a separate symmetric scale stated in the legend.
- Mark sources on the velocity map, not over the wavefield panels where they can obscure the direct arrival.
- State normalization/inverse-normalization and whether fields are physical pressure or record-normalized values.
- Do not use “high source-position generalization” from snapshots alone. That wording is allowed only if all 240 fixed-19-Hz controlled records, the source-distance trend, and the velocity-slice cluster uncertainty gate pass.
- If the frozen model fails a family or phase gate, retain the figure as a failure diagnostic and describe the boundary honestly.

## Required generated artifacts

- `prediction_manifest.json`: 240 fixed-frequency predictions plus checkpoint/config/data/protocol digests, written before reference generation.
- `reference_manifest.json`: 240 LWC-84 references plus the stored-record solver reproduction audit and immutable prediction-manifest digest.
- `score_summary.json`: per-record, per-position-role, per-slice, and velocity-slice-cluster uncertainty summaries.
- Vector figure PDF plus a machine-readable plotting command/config.
- Prediction seals created before any target wavefield is opened for evaluation.
