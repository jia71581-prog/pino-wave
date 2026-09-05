# B2-v10 interim latest-checkpoint online adaptation decision

## Claim scope

This experiment used an immutable snapshot of the active V9 run for an early
train-only algorithm test.  It is not a V9 terminal checkpoint, cannot promote
an online method, and does not authorize validation or `test_id` access.

## Interim parent

- Arm: `physics_cond`, seed 733, best epoch 33.
- Snapshot: `results/b2_v9_interim_snapshot_physics_cond_s733_e33_20260901.pt`.
- Snapshot SHA256:
  `f80ee9d5b7bc980522817857512be24b6ad597752507f878f48ae0407b265572`.
- Same-cache V9 holdout aggregate: 0.1242537422 versus warp-anchor
  0.1308174924.
- The snapshot was copied with identical source-before, source-after, and
  destination hashes while the live V9 job continued.

## Offline residual capacity

Family-specific rank-4 POD modes were fit using the 240 V9 train-fit records.
This is explicitly offline train-truth use.

| Family | Parent relative L2 | Rank-4 oracle relative L2 | Mean oracle gain |
|---|---:|---:|---:|
| Uniform | 0.076991 | 0.070831 | 5.033% |
| Layered | 0.105722 | 0.094638 | 4.724% |
| Marmousi | 0.169488 | 0.156522 | 4.900% |

All capacity gates were positive, but the residual subspace is much weaker than
the approximately 50% capacity observed for the older weak parent.

## Unguarded cycle050 calibration smoke

The online solve read only the causal prefix through 0.5 source cycles after
the Ricker peak.  Candidates were serialized and hashed before future train
truth was opened for scoring.

| Metric | Parent | Adapted |
|---|---:|---:|
| Aggregate relative L2 | 0.124336870 | 0.124624177 |
| Common-tail relative L2 | 0.133062883 | 0.133395415 |

- Relative gain: -0.2311%.
- Accepted online: 24/24.
- Nonworse: 8/24.
- Mean end-to-end runtime: 0.400 s; nearest-rank P95: 0.431 s.
- Verdict: rejected.

The observed-prefix objective accepted every record even though most future
predictions regressed.  Observed frame count was more correlated with future
gain than correction magnitude or POD conditioning.

## Offline-calibrated observation guard

A one-dimensional train-calibration grid selected
`minimum_observed_frames = 40`.  Replaying the already sealed calibration
candidates with exact parent rollback below the threshold produced:

- parent 0.124336870, guarded 0.123776340;
- relative gain +0.4508%;
- 4/24 corrections accepted;
- 23/24 nonworse;
- all three family aggregates improved.

This was calibration evidence only, so the threshold was frozen before opening
the group-disjoint confirmation panel.

## Fresh train confirmation

| Metric | Parent | Guarded adapted |
|---|---:|---:|
| Aggregate relative L2 | 0.104063580 | 0.104301817 |
| Common-tail relative L2 | 0.113617179 | 0.113904108 |
| Uniform | 0.064529127 | 0.064539313 |
| Layered | 0.106162766 | 0.106890401 |
| Marmousi | 0.141498848 | 0.141475736 |

- Relative gain: -0.2289%.
- Accepted online: 5/24; 19/24 exact parent rollback.
- Nonworse: 20/24, because only one of the five accepted corrections improved.
- Largest accepted regression: -5.342% relative gain on a layered record.
- Mean end-to-end runtime: 0.373 s; nearest-rank P95: 0.389 s.
- Validation and `test_id`: unopened.
- Verdict: rejected.

## Decision

Reject `b2_v10b_cycle050_minimum_observed_40_v1` for the interim V9 parent.
Do not tune a new observation threshold, ridge value, or trust radius on the
opened confirmation panel.  The evidence shows that low-rank global residual
modes have positive offline capacity but their causal-prefix coefficient fit
does not transfer reliably to future error.

If online adaptation is revisited after the final V9 parent is selected, change
the conceptual variable: train an observed-prefix-to-future local residual
operator or meta-learn the adaptation path on train tasks.  Do not repeat scalar
calibration of the same rank-4 family POD route.

## Integrity

- Initial attempt failed before data access because `rank` was serialized in
  the wrong preregistration subsection; it is preserved in a disposition file.
- Twenty-two related CPU/static tests passed after the additive guard change.
- Target-5 read-only audit remained passed.
- Active V9 four-GPU training continued without OOM, NaN, or checkpoint loss.
