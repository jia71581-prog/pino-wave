# DeepONet baseline comparison (40M parameter budget)

| Quantity | DeepONet 40M | Ours (IC8 40M) | Caliber |
|---|---|---|---|
| Parameters | 39,975,985 | 40,484,837 | same +/-2% budget (preregistered) |
| Fixed 12-record validation, dense relL2 (best ckpt, step 4488) | 1.000012 | n/a (not our selection metric) | DeepONet's checkpoint-selection batch; ~1.0 = trivial zero predictor |
| 480-record validation census, dense future relL2 (record mean) | not evaluated (run stopped at step 4862/22814) | 0.352 (uniform 0.131 / layered 0.302 / marmousi 0.565) | ours: full census at step 22814 |
| 12-record family-level fit probe (2000 steps), point relL2 | control arm 1.0015-1.1168, never < 1.0 | n/a | packaged-loss probe caliber |

Facts stated as recorded:
- The DeepONet main run was terminated by SIGTERM at step 4862 of a planned 22814; its numbers are NOT a converged endpoint.
- At every recorded validation, DeepONet's dense relative L2 was indistinguishable from the trivial zero predictor (1.0000 +/- 1e-4), on all three families.
- The 2026-09-15 family-level collapse probe found the control arm never beats trivial on 12 records, and the anti-collapse arm is not separable from the control arm (all three effect axes inside its own oscillation band). The failure was attributed to optimisation (H_optim), not capacity (H_capacity) - single-record fits do converge (relL2 0.025 at 2000 steps).
- No same-caliber DeepONet number exists; the table's calibers are the nearest available and are labelled per row.
