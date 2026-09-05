# Probe spec: r4e9_mask_confined_oracle (train-only, zero training)

Frozen 2026-08-26. Diagnostic probe, not a promotion candidate. Successor to
`results/r4e8_masked_oracle_probe_spec_20260826.md`.

## Why this probe exists

r4e8 measured the oracle upper bound under three metric conventions and produced a conflict that
blocks the proposed v15 objective. At rank 16, on the same record, the fit that optimizes
convention (ii) `masked_energy_floored_per_frame` **destroys** convention (iii)
`global_energy_rel_l2`, and vice versa:

| family | fit arm | (i) unmasked per-frame | (ii) masked floored | (iii) global energy |
|---|---|---|---|---|
| uniform | optimizes (i) | +0.243 | -0.149 | -0.163 |
| uniform | optimizes (ii) | -290.2 | +0.160 | -30.3 |
| layered | optimizes (i) | +0.423 | -0.227 | -0.292 |
| layered | optimizes (ii) | -1.559 | +0.386 | +0.077 |
| marmousi | optimizes (i) | +0.118 | +0.121 | +0.131 |
| marmousi | optimizes (ii) | +0.119 | +0.123 | +0.133 |

marmousi is the only family free of the conflict, and it is the only family whose mask drops
almost nothing (387/391 retained). layered drops 144/380 and conflicts; uniform drops 178/388 and
conflicts worst. The conflict also grows with rank (layered (iii) masked arm: +0.077 at rank 16,
**-1.432** at rank 32).

Reading `scripts/probe_r4e8_masked_oracle.py:470-499`: the masked arm fits coefficients on
retained frames only (`masked_map @ residual_block[keep_device]`) but then applies the correction
to **all** frames (`design_device @ matched_coefficients`). Nothing bounds the correction on
dropped frames, and convention (iii) sums over all frames.

**Hypothesis H:** the (iii) blow-up of the masked arm is located in the mask-dropped frames, and
confining the correction to the mask removes the conflict.

Confinement is deployment-causal: the mask is derived from parent frame energy, which r4e8
measured as a valid proxy for the truth-energy mask (agreement 0.9948 / 0.9895 / 1.0000, all
above the 0.90 minimum).

## What to measure

Same three train records as r4e8 (`train_uniform_00321`, `train_layered_00564`,
`train_marmousi_00385`), ranks 8 / 16 / 32, reusing the r4e8 code path so the numbers are
comparable by construction.

For each (record, rank), report two variants:

- **unconfined** — exactly the r4e8 `masked_matched` arm. Must reproduce the r4e8 numbers
  bit-for-bit; a mismatch is a stop condition.
- **confined** — identical, except the correction is multiplied by the **parent-energy** mask
  `m_t^parent` before it is applied, i.e. the correction is exactly zero on dropped frames.
  Use the parent mask, not the truth mask; using the truth mask here would be a deployment-causality
  violation and must not be the headline variant. Report the truth-mask version only as a
  labelled diagnostic.

For both variants report: all three conventions, the gain against parent under each, and the
**error energy split inside vs outside the mask** (`sum_t in mask err_t` and `sum_t not in mask err_t`,
for parent, unconfined and confined). The split is what tests H directly rather than by elimination.

## Pre-stated decision rule

- **H confirmed and confinement works** if, for all three families at rank 16, the confined variant
  has (ii) gain >= unconfined (ii) gain - 0.02 **and** (iii) gain >= 0. Then the v15 objective is
  masked loss + mask-confined correction, and v15 may be frozen.
- **H confirmed but confinement insufficient** if the outside-mask error split confirms the blow-up
  location yet confined (iii) gain is still < 0 for some family. Then that family cannot be helped
  by this ansatz under the acceptance metric at any measured rank, and v15 must route it to
  abstention with output bit-identical to the parent.
- **H refuted** if the outside-mask split does not account for the (iii) blow-up. Then stop and
  report; do not freeze v15.

Report per-family and per-rank. Do not collapse to a single threshold. Nothing in this probe may be
tuned on its own outcome.

## Hard constraints

- Train split only. Never open validation or test_id wavefields.
- Do not modify any existing file. New code goes in `scripts/probe_r4e9_mask_confined.py`, new tests
  in `tests/saved_time_phase_operator_v4/test_r4e9_mask_confined.py`, artifacts only in
  `results/r4e9_mask_confined_oracle_20260826/`, under 500 MB total.
- Do not touch `results/r16_dscp_v14/` or any of the 16 files bound by the frozen v14
  preregistration. Do not touch `results/r4e8_masked_oracle_probe_20260826/`.
- Delete nothing. Preserve all A3, B2-H, Helmholtz, ASAM and CPADC checkpoints.
- Check `df -h .` before writing; the workspace quota was at 100% earlier today.

## Rollback

Nothing to roll back: zero training, no checkpoint, no authorization. On failure delete nothing and
report the exact blocker.

## Verification gate

1. The unconfined variant reproduces the r4e8 `masked_matched` gains for all 9 (family, rank) points.
2. New tests assert, on a hand-computable synthetic case, that confinement leaves in-mask error
   unchanged and drives out-of-mask error to the parent value exactly.
3. `python -m pytest -p no:cacheprovider tests/saved_time_phase_operator_v4/test_r16_dscp*.py -q`
   still passes 197, and the r4e8 test file still passes.
