# B2-v3 frozen-checkpoint train generalization probe

The preregistered probe
`results/b2_v3_train_generalization_probe_preregistration_20260831.json`
completed on a 30-record, 30-group, train-only panel. No validation or
`test_id` records were opened.

## Frozen decision

The same-panel solve-free warp anchor aggregate was `0.12751960`. The frozen
gate required both width-64 seeds to retain at least half of the v3
development gain (`0.01228325` absolute), improve every family, and be
nonworse on at least 27/30 records.

| Frozen checkpoint | Aggregate | Delta vs anchor | Nonworse | Family gate |
|---|---:|---:|---:|---|
| Arm A seed 372 | 0.12970465 | +0.00218505 | 16/30 | fail, all three regress |
| Arm A seed 733 | 0.12470735 | -0.00281225 | 21/30 | fail, uniform regresses |

Both checkpoints fail all preregistered composite gates. The probe status is
`rejected`.

## Diagnosis

- The v3 development result near `0.10` does not transfer with seed-consistent
  magnitude to unseen records. It primarily demonstrates optimization on the
  90 records used for every training and evaluation epoch.
- Seed 733 retains a small Marmousi benefit (`0.20336` to `0.19297`) but its
  uniform mean worsens (`0.07998` to `0.08255`). Seed 372 regresses in all
  three family means.
- Mean late-time error remains `0.14869` to `0.15473`, and high temporal-rFFT
  error remains approximately `0.433`. The late/high-frequency limitation is
  not confined to the original development panel.

## Decision

Do not authorize tail-weighted, spectral-loss, wider-model, or longer-budget
training from the v3 checkpoints. The only cheaper remaining rescue test is a
parameter-free causal selector that uses the eight deployment-visible frames
to choose between the anchor and the two frozen corrections on a fresh,
group-disjoint train-only panel. It must be preregistered before the fresh
panel's future metrics are opened. Failure of that selector rejects the
current B2 development-fit route and returns research to a train-only
fit/calibration/confirmation design.
