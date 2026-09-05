# V65 LWC-84 multi-fidelity distillation design

## Goal

Replace the ineffective small-correction continuation with a solver-guided pretraining stage that
still produces a pure neural full-wavefield operator at inference.  The accepted output contract
remains one source per record, arbitrary exact stored time, 201 by 201 pressure, pressure-free top,
and CPML on the other three sides.  Receivers are never model inputs.

The promotion target is the independent 480-record exact-time panel: aggregate relative L2 below
10% and each of uniform, layered, and Marmousi below 12%.  A candidate must also strictly improve
the current learned parent on the same panel before replacing a running job.

## Evidence and root cause

The sealed native-grid LWC-84 panel has aggregate error 4.8992%, with family errors 1.7470%,
3.2734%, and 9.3916%.  The current learned parent is near 46.9%.  V63 changes the parent by only
about 0.7% in L2 and its first validation did not improve the parent.  The bottleneck is therefore
the learned base wavefield, not a lack of residual adapter capacity.

The stored dataset cannot support a faithful LWC-84 residual evaluated directly between saved
frames.  Each stored interval skips twenty fine solver steps and the CFS-CPML ADE memory state is
not stored.  A saved-frame finite-difference residual would enforce a different dynamical system.

## Chosen architecture

Use the existing V4 two-output path as an exact-grid auxiliary-distillation operator:

- `u_low_hat`: the existing transferred coarse output, trained against an independently replayed
  LWC-84 numerical teacher `u_num`;
- `u_high_hat`: the existing propagation-conditioned dense output;
- `u_high_hat-u_low_hat` is regularized against `u_high-u_num`; because the replay now uses the
  exact dataset solver contract, this residual is expected to be zero up to reproducibility error
  and acts as a head-consistency constraint rather than a coarse-grid correction;
- inference returns `u_high_hat` and does not run LWC-84.

This retains the medium/source decoupling, MIONet-style branch/trunk fusion, temporal basis,
family routing, and multiscale spectral decoding already present in V2/V4.  It follows the
multi-fidelity operator pattern of learning a low-fidelity map and a nonlinear residual rather
than treating low- and high-fidelity fields as interchangeable labels.

## Numerical teacher cache

Register a deterministic zero-copy teacher view for all 2,240 eligible training records.  The
source HDF5 already contains the output of the exact training solve and save chain, so duplicating
those bytes is neither numerically useful nor storage-efficient:

- reconstruct the frozen-manifest velocity on the 401 by 401 node grid and require its registered
  restriction to be bitwise identical to the HDF5 201 by 201 velocity before solving;
- solve with `dx = dz = 5 m`, `dt = 0.000125 s`, and the original 40 CPML cells;
- save with the same `binomial5_lowpass_then_decimate2` restriction, producing 201 by 201 fields;
- the same target reflection `1e-8`, polynomial order 3, `kappa_max=3`, minimum frequency
  8 Hz, and `alpha_max=pi*8 Hz` are loaded from the frozen dataset config rather than hard-coded;
- CPML remains on left, right, and bottom only;
- pressure-free top;
- the exact dataset source coordinates, frequency, onset, and amplitude;
- 64 registered exact stored-time indices spanning all 401 saved times, including 0 and 400;
- a 262 KB float32 HDF5 VDS with two contiguous record mappings and 64 time mappings, backed by
  the immutable source HDF5.

An independent four-GPU replay audit solves 120 records per GPU with the same 401-grid solver.
The audited uniform and layered fields matched the source HDF5 bitwise at all 64 selected times;
the three-family velocity reconstruction gate was also bitwise exact.  Those replay shards are
retained as evidence, while training uses the zero-copy view and therefore avoids about 23.2 GB of
duplicate storage.  Subsequent continuation returns to the rotating all-time appearance policy.

The teacher view binds the source HDF5, manifest bytes, frozen-config bytes, numerical
contract, time indices, sample IDs, source indices, completion mask, and shard layout.  Cache
generation additionally rejects any disagreement in the 4,003-record census, source metadata,
saved axes, completion mask, reconstructed velocity, or source map.

## Loss

For one selected exact-time block, optimize

`L = L_high + lambda_low L_low + lambda_res L_res + lambda_spec L_spec`,

where each field term is a squared relative-energy objective with a denominator accumulated over
the complete selected time set:

- `L_high = ||u_high_hat - u_high||^2 / E_high`;
- `L_low = ||u_low_hat - u_low||^2 / E_low`;
- `L_res = ||(u_high_hat-u_low_hat) - (u_high-u_low)||^2 / E_res`.

`E_res` is floored by a registered fraction of `E_high` so the small numerical discrepancy cannot
produce unbounded gradients.  The implementation exposes all components in per-update logs.  Hard
causality is applied consistently to both predictions and both targets.

The first exact-grid smoke showed that the former coarse-grid weights
`lambda_low=lambda_res=0.5` were invalid after making `u_num` identical to the training solver:
the second-step dense gradient rose to 586 and validation regressed.  The corrected r2 pilot uses
`lambda_low=0.1`, `lambda_res=0.05`, a five-times smaller dense learning rate, two-epoch warmup,
and a dense-prefix gradient cap of 5.  The numerical terms are now auxiliary consistency losses,
not a claim of a distinct low-fidelity target.
The high-fidelity term always has weight one; the numerical teacher can guide but never replace the
actual 401-to-201 teacher.

## Training and resource policy

1. Finish code and exact-grid replay tests while V63 remains live.
2. Publish the zero-copy numerical-teacher VDS only after the frozen-data and replay gates pass.
3. Run capacity probes with full-forward checkpointing and physical microbatches 2, 3, and 4;
   select the largest size below 23 GiB that has finite gradients.
4. Run a four-epoch fixed-budget V65 pilot with easy-to-hard family curriculum and 16 cached exact
   frames per record per appearance.
5. Evaluate every epoch on the same fixed high-fidelity panel.  At pilot end, require strict
   held-out improvement and no unsafe family regression before any long run.
6. If the safety/improvement gate passes, continue from its best checkpoint with numerical weights annealed to zero and the
   normal rotating exact-time policy over all 401 stored times.  If it fails, retain the prior best
   checkpoint and numerical baseline evidence.

No cache generation or solver call is performed inside the final inference path.  Solver time,
training time, neural inference time, and I/O time remain separately reported.

## Rejected alternatives

- Solver-in-the-loop inference is rejected because it cannot deliver the intended neural speedup.
- A saved-frame PDE residual is rejected because the fine LWC-84 steps and CPML memory variables
  are unavailable.
- A full 401-frame float32 numerical cache is rejected for this pilot because it would exceed the
  safe remaining remote storage budget.

## Acceptance checks

- unit tests prove cache identity, fixed-pool sampling, loss decomposition, residual flooring, and
  no-gradient numerical targets;
- local saved-time regression and Python compilation pass;
- replay smoke uses a 401 by 401 solve, returns 201 by 201 restricted fields, matches the direct
  solver at registered times, and preserves zero pressure at the top row;
- four-GPU smoke has finite loss/gradients, no CPU fallback, peak allocation below 23 GiB, and a
  physical microbatch selected by measured capacity;
- pilot promotion uses high-fidelity validation only and requires strict learned-parent improvement;
  final acceptance additionally requires the aggregate/family thresholds.
