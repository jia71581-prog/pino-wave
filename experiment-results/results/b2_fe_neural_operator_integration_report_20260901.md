# FEM and neural-operator integration for B2 instance adaptation

## Literature mechanisms retained

1. Kapustsin, Kaya, and Richter: enrich a coarse finite-element state with
   learned local fine-scale fluctuations.
2. Yamazaki et al. finite operator learning: use assembled Galerkin weak
   residuals and discrete time integration as an algebraic training objective.
3. Neural-operator element method: place reusable learned fine-scale elements
   inside a variational finite-element framework.
4. FEONet analysis: finite-element matrix conditioning directly affects
   operator-learning convergence.

## Project integration

- Parent: immutable B2-v4 seed-372 checkpoint.
- Online parameters: 64 decoder-channel scaling coefficients; parent weights
  remain frozen.
- Exact rollback: zero coefficients reproduce the B2 parent bit-for-bit.
- Online data: first eight true frames, velocity/source/time metadata, and the
  frozen parent prediction only.
- Physics feature: source-off Q1 bilinear FE weak residual on a 4x coarsened
  stored grid, excluding top and CPML boundary zones.
- Trust region: full correction energy at most 5% of parent energy.
- Discretization disclosure: the dataset generator is 5 m LWC84/CPML followed
  by restriction to 10 m, not Q1 FEM. The weak residual is therefore a feature
  and adaptation objective, never truth supervision.

Implemented files:

- `saved_time_phase_operator_v4/instance_adaptation/fe_weak_residual.py`
- `saved_time_phase_operator_v4/instance_adaptation/b2_fe_weak_adapter.py`
- `scripts/probe_b2_fe_weak_instance_adaptation.py`
- `scripts/run_b2_fe_weak_balanced_panel.py`
- `tests/saved_time_phase_operator_v4/test_b2_fe_weak_adapter.py`

## Evidence

### Static and single-record smoke

- 18 Q1/rollback/existing instance-contract tests passed.
- Deterministic train record `train_uniform_00002`:
  - 64 trainable coefficients;
  - adaptation time 1.63 s;
  - correction ratio 1.02%;
  - parent future relative L2 0.097139;
  - adapted future relative L2 0.096580;
  - relative gain 0.575%;
  - candidate was serialized and hashed before future train truth was opened.

This passed the one-record smoke but did not isolate the FE term.

### Frozen paired 4/4/4 train panel

| Arm | Aggregate | Relative gain vs parent | Nonworse | Accepted online |
|---|---:|---:|---:|---:|
| Frozen parent | 0.135307 | n/a | 12/12 | n/a |
| Observed-only | 0.137921 | -1.932% | 3/12 | 12/12 |
| Observed + FE weak | 0.137764 | -1.815% | 4/12 | 12/12 |

The FE term improved aggregate by 0.114% relative to observed-only and was
better on 6/12 paired records, including meaningful improvements on selected
layered and Marmousi records. However, it still harmed the frozen parent on
aggregate, failed the 11/12 nonworse gate, and regressed the uniform family
relative to observed-only. The preregistered panel verdict is `rejected`.

## Interpretation

The Q1 weak feature contains some useful information, but the current online
objective is not calibrated to future error: all 12 online solves were
accepted while only four were future-nonworse. Strict eight-frame adaptation
therefore cannot safely select unrestricted decoder directions.

Do not promote this online adapter. Retain the FE weak residual only as an
offline feature for the next conceptual route:

1. stream train-only parent residuals and FE weak-defect patches;
2. learn local, travel/family-conditioned fine-scale modes or neural-operator
   elements offline;
3. solve only a small coefficient vector online;
4. train a parent-quality/weak-defect abstention gate;
5. require fresh group-disjoint train confirmation with at least 1% mean gain,
   95% nonworse, every family nonworse, and P95 adaptation below 3 s.

Validation and `test_id` were not opened by this work.
