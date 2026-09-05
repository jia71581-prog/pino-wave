# Parent-anchored block-32 correction: research decision

Date: 2026-09-04

## Decision

Test a lightweight 32-frame block-autoregressive residual corrector on top of an
immutable `PyramidMoECoupledWaveOperator` parent. Do not replace the frequency
parent and do not revive the falsified fixed-coarse-kernel B2-H route.

The candidate carries two corrected residual frames between blocks but receives
the sealed parent prediction for every new 32-frame slab. It is therefore
autoregressive only across 13 block boundaries, rather than across 399 stored
time steps. Its deployment API contains no target, future truth, oracle scale,
family label, validation label, or test label.

## Evidence and mechanism

- The historical B2-H horizon sweep reached learned aggregate relative errors
  1.265, 1.084, 0.978, and 0.881 for horizons 8, 16, 32, and 64; none beat the
  fixed-kernel baseline 0.871. Increasing physical substeps from 2 to 40 changed
  aggregate error by less than 2e-4. The failure was coarse-grid spatial
  dispersion, not time under-resolution.
- Message Passing Neural PDE Solvers frames stable autoregressive training as a
  zero-stability/domain-shift problem and motivates pushforward exposure:
  https://arxiv.org/abs/2202.03376
- PDE-Refiner shows that long rollouts require explicit treatment of weak and
  high-frequency components, but iterative refinement adds inference calls:
  https://openreview.net/forum?id=Qv6468llWS
- SGNO attributes long-rollout drift to one-step accumulation, spectral
  amplitude distortion, and phase misalignment, and constrains the repeatedly
  applied spectral update: https://arxiv.org/abs/2602.18801
- HERO adds history-derived rollout failure supervision without inference-time
  cost: https://arxiv.org/abs/2607.29135. This is reserved for a later ablation
  because adding it to the first architecture test would confound attribution.
- A controlled synthetic-seismogram study identifies phase drift and multi-token
  prediction as dominant issues for autoregressive physical-wave forecasting:
  https://arxiv.org/abs/2606.10868

## Falsifiable hypothesis

Relative to the immutable epoch-23/update-15400 parent, a zero-initialized,
bounded block-32 corrector trained with one detached pushforward block and two
supervised free-running blocks will reduce exact full-401 train-confirmation
future-field relative L2 by at least 5%, without worsening any medium family.

Expected mechanism: joint 32-frame prediction learns local phase/amplitude
closure; two-frame residual state enforces second-order temporal context; parent
re-anchoring bounds drift; temporal-derivative and band-spectral losses expose
phase errors; the nonworse hinge discourages family-level regression.

Failure signal: non-finite values, correction-cap violation, any sealed-split
access, less than 5% group-disjoint train-confirmation improvement, any family
regression, or a measured end-to-end runtime lower bound incompatible with 10x
speedup.

## Evidence ladder and rollback

1. Static API and CPU unit contracts.
2. One-update train-only smoke.
3. 68-record fit with 34-record train calibration.
4. One frozen 34-record group-disjoint train confirmation.
5. Only after acceptance: version and run one complete validation evaluation.

All calibration/confirmation inference and corrector runtime measurements run on
CPU with 16 threads. GPU is reserved for optimization and offline parent-cache
materialization only.

Rollback is exact: omit the corrector or load its zero-initialized state. The
parent checkpoint and every A3/B2-H/Helmholtz/ASAM/CPADC artifact remain intact.
