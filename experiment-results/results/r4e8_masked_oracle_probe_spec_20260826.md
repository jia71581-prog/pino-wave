# Probe spec: r4e8_masked_oracle_recompute (train-only, zero training)

Frozen 2026-08-26. This is a **diagnostic probe, not a promotion candidate**. It exists because
`reports/lanes_20260826/*.md` established that the v14 `oracle_gain` thresholds were transplanted
from records other than those scored, so no v15 acceptance threshold can be stated until the
achievable-gain reference is recomputed on the records that will actually be scored.

## Question

For the three smoke-panel records, what is the rank-8 / rank-16 / rank-32 oracle (upper-bound)
gain under a masked, energy-floored metric convention, and is the parent-energy mask a valid
deployment-causal proxy for the truth-energy mask?

## Records (train split only)

`train_uniform_00321`, `train_layered_00564`, `train_marmousi_00385` — the exact v14 smoke panel.

## Required outputs, per (record, rank, convention)

1. Per-frame truth energy `E_t = ||truth_t||^2` and the normalized profile `E_t / max_s E_s`.
2. Truth mask `m_t^truth = 1[E_t >= tau * max_s E_s]`, `tau = 1e-3`.
3. Parent mask `m_t^parent` from parent frame energy with the same `tau`, plus the
   **agreement rate** between the two masks. This is the deployment-causality check: the
   deployed mask may use only parent/observation quantities.
4. Oracle coefficients via the existing weighted-LS path (`weighted_coefficient_target`),
   with the C1 causal mask applied.
5. Gain under all three conventions:
   - (i) `unmasked_per_frame_unsquared_mean` — the convention the v14 gate used
   - (ii) `masked_energy_floored_per_frame` — the proposed v15 convention
   - (iii) `global_energy_rel_l2`
6. `correction_energy_ratio`.

## Pre-stated decision rule (not a promotion gate)

- Convention (ii) rank16 gain >= 0.20 for all three families **and** rank32 adds < 0.05
  => rank is not the bottleneck; v15 proceeds at rank 16 under convention (ii).
- Convention (ii) rank16 gain < 0.20 for any family => report that family's achievable gain
  per-family. Do **not** set a single uniform 0.5 threshold for v15.
- Mask agreement rate < 0.90 for any record => the parent-energy mask is not a valid proxy and
  the masking design must change before v15 is frozen.

## Self-consistency requirement

The reimplementation must reproduce known v14-regime numbers when run in the **old unmasked**
convention, so we know the new code path is faithful and not a second artifact. Reproduce at
least the documented uniform rank16 per-frame figure `-1.639` and layered rank16 capture `0.643`
from `results/r4e7_family_temporal_pod_capacity_train9_v1_20260825.json` and
`results/r4e7_raw_weighted_pod_fresh3_confirmation_v1_20260826.json`, on those files' own records.
Report the reproduced values next to the published ones. A mismatch is a stop condition.

## Hard constraints

- Train split only. **Never** open validation or test_id wavefields.
- Do **not** modify any of the 16 files bound by the frozen v14 preregistration
  (`results/r16_dscp_v14/design_preflight.json` -> `bindings`), and do not touch
  `results/r16_dscp_v14/` at all.
- Delete nothing. Preserve all A3, B2-H, Helmholtz, ASAM and CPADC checkpoints.
- All new artifacts go under `results/r4e8_masked_oracle_probe_20260826/` and must stay under 500 MB
  total. Workspace free space is ~12 GB on a quota that was at 100%; check `df -h .` before writing.
- Do not tune anything on the outcome. This probe measures an upper bound; it does not train.

## Rollback

Nothing to roll back: no training, no checkpoint, no authorization. On failure, delete nothing and
report the exact blocker.

## Verification gate

1. The self-consistency reproduction above matches the published values.
2. A new test file asserts the three conventions on a small synthetic case with hand-computable
   answers, including the near-zero-truth-frame case that the mask is meant to handle.
3. Full `tests/saved_time_phase_operator_v4/test_r16_dscp*.py` still passes 197/197.
