# Joint pretraining and instance-adaptation design

## Hypothesis

The frozen-parent experiments failed because the parent representation was not
trained to expose useful causal correction directions. Joint episodic training
of a small parent subspace and the instance basis should improve post-adaptation
relative L2 while retaining the parent as an exact zero-correction rollback.

## Variables

- `theta`: selected parent parameters only: temporal latent basis, local-field
  warp/gates, fusion, and dense output projection. The medium encoder remains
  frozen initially.
- `phi`: family-conditioned, travel-aligned residual modes.
- `c_i`: 16 or 32 coefficients solved independently for instance `i`.

## Leakage-safe bilevel objective

For every train episode, the inner solve may use only velocity, source, time,
the two registered onset observations, synthetic bridge frames, source-consistent
LWC defect, and the left/right/bottom CPML interface residual:

`c_i* = argmin_c L_obs + 0.5 L_bridge + L_PDE + 0.1 L_CPML + 1e-4 ||c||^2`

subject to `||B_phi c|| / ||parent|| <= 0.05`. Future train truth is not read
until `c_i*` and the candidate output are sealed.

The outer train-only loss is

`L_outer = L_parent + L_adapted + 0.2 L_late + 0.1 L_spectrum + L_nonworse`

where both field terms are per-record relative L2, and

`L_nonworse = relu(L_adapted - L_parent + 1e-4)`.

Validation and `test_id` are never used in either loop during development.

## Optimization schedule

1. Start from immutable r4 epoch 7.
2. Use two optimizers and two learning-rate scales:
   - parent selected parameters: `5e-7`;
   - residual modes / conditioner: `5e-5`.
3. Alternate three ordinary parent batches with one adaptation episode.
4. Initially stop-gradient through the convex inner solve (first-order bilevel
   approximation); enable implicit differentiation only if the cheap route
   passes disjoint-train confirmation.
5. Clip parent and adapter gradients separately. Save an epoch only when:
   - adapted fixed-panel relative L2 improves;
   - parent-only relative L2 regresses by no more than `0.2%`;
   - every loss and coefficient solve is finite.

## Runtime and rollback

Online deployment solves only `c_i`; no parent weight update occurs online.
Zero coefficients reproduce the jointly trained parent exactly. Reject and use
the original r4 epoch-7 parent if the solve is ill-conditioned, the causal
objective does not improve, or the correction reaches the 5% trust boundary.

## Evidence ladder

1. Unit tests for support/query sealing and parent rollback.
2. One-record joint overfit: at least 5% adapted relative-L2 reduction.
3. Balanced six-record smoke, three outer steps, peak CUDA below 23.5 GiB.
4. Group-disjoint train calibration/confirmation: at least 1% mean adapted
   improvement, 95% nonworse, every family nonworse, P95 adaptation below 3 s.
5. Freeze all parameters and thresholds, then run one complete validation.

No long run or validation opening is authorized before gates 1-3 pass.

