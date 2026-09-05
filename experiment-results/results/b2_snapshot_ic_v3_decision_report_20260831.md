# B2 snapshot-IC v3 decision report

Judged at `2026-08-31T09:43:31Z` against the frozen record
`results/b2_snapshot_ic_v3_preregistration_20260831.json`.

## Scope

- Evidence is limited to the 90-record development subset: 30 uniform, 30
  layered, and 30 Marmousi records.
- The same 90 records were used for optimization and epoch evaluation. These
  results measure development-set fit and optimization capacity, not unseen
  record generalization.
- No new complete-validation or `test_id` truth was opened for this judgement.

## Frozen gates

The warp anchor aggregate is `0.1260311091`. The v2 IC8 arm mean is `0.11729`
and its frozen seed band is `0.001`, giving the Gate C threshold `0.11629`.

| Arm | Seed 372 | Seed 733 | Mean | Seed band |
|---|---:|---:|---:|---:|
| A: width 64, 180 epochs | 0.09921735 | 0.10371187 | 0.10146461 | 0.00449452 |
| B: width 128, 180 epochs | 0.12199023 | 0.10991414 | 0.11595219 | 0.01207609 |

- **Gate C, budget recovers trajectory: PASS.** Both Arm A seeds are below
  `0.11629`. Relative to their best values in the first 60 epochs, the two
  lanes gain another `0.01770` and `0.01473`, respectively. The v2 endpoint
  slope was therefore not merely a terminal cosine artifact.
- **Gate D, capacity beats pure budget: FAIL.** Arm B is worse than Arm A for
  both paired seeds. The mean regression is `0.01449`, and the width-128 seed
  band (`0.01208`) is also much larger than Arm A's (`0.00449`). Width 128 is
  rejected as the next lever under this recipe.

All four best epochs are epoch 180. Arm A's last-20-epoch slopes remain small
and negative (`-3.10e-5` and `-1.78e-5` per epoch), but the frozen no-extension
rule is binding. No post-hoc continuation is authorized.

## Secondary diagnostics

- Arm A seed 372 is nonworse than the warp anchor on 88/90 records. The only
  regressions are `validation_layered_00209` (`+0.000472`) and
  `validation_layered_00085` (`+0.000184`).
- The maximum error is dominated by `validation_layered_00047`: `0.50120`
  versus anchor `0.50467`. Its target norm is finite and substantial, so this
  is a real late-time propagation failure rather than a near-zero-denominator
  artifact.
- For Arm A seed 372, mean temporal errors are early `0.08670`, middle
  `0.08920`, and late `0.11880`. Temporal-rFFT low/middle/high-bin errors are
  `0.08744`, `0.20582`, and `0.42200`. The remaining floor is concentrated in
  late time and high temporal frequency.
- Arm A family means averaged over both seeds are approximately uniform
  `0.07908`, layered `0.12328`, and Marmousi `0.10204`. This development result
  does not meet the absolute `0.05` target in any honest promotion sense.

## Lineage deviation

All four immutable `run_identity.json` files point their `preregistration`
field to `results/b2_snapshot_ic_preregistration_20260830.json` (v2), not the
v3 preregistration. The trainer hash, manifest selection hash, lane arguments,
terminal status, and checkpoint hashes agree with the frozen v3 record. The
raw identities are preserved unchanged; the additive closure record is
`results/b2_snapshot_ic_v3_lineage_closure_20260831.json`.

The scientific Gate C/D judgement is retained with a lineage-deviation flag,
but these artifacts are not promoted as a fully reproducible accepted model.

## Decision

Accept the mechanism conclusion that a longer width-64 budget materially
improves development-set fit. Reject width 128 as the next lever. Before any
tail-aware or spectral-loss long training, run the frozen width-64 checkpoints
on unseen, train-split records to determine whether the learned correction
generalizes beyond the 90 records it optimized.
