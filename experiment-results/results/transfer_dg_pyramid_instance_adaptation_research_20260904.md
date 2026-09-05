# PyramidMoE instance-adaptation research update (2026-09-04)

## Scope and protocol

This iteration used the immutable update-13650 PyramidMoE checkpoint and the
train-only development record `train_marmousi_00260`. Validation and `test_id`
remained sealed. Online adaptation read only velocity, source information,
the frozen parent prediction, derived DG features, and true frames 17 and 18.
Future train truth was opened only after each candidate had been serialized,
or explicitly inside an oracle-capacity diagnostic that cannot produce a
deployment candidate.

The parent is still being pretrained, so all results below diagnose algorithmic
structure rather than qualify a final adapter checkpoint.

## Measured evidence

### 1. Branch-trunk Transfer-DG online solve

- Effective correction rank: 63.
- Online onset-plus-flux objective: `1.100233 -> 0.952201` (13.45% lower).
- Correction ratio: 5.00%, at the trust cap.
- Future relative L2: `0.720482 -> 0.721413` (0.129% worse).
- Decision: rejected; retain the exact parent prediction.

An analytic scale sweep showed that the best scale along this direction is only
0.0969 of the attempted correction, equivalent to a 0.485% correction ratio.
Even that future-truth oracle changes future relative L2 only to `0.720471`.
The online direction is nearly orthogonal to the true future error.

### 2. Complete branch-trunk span oracle

- Effective rank: 63.
- Trust-capped oracle future relative L2: `0.720482 -> 0.720134`.
- Relative improvement: 0.0483%, below the frozen 1% capacity gate.
- Correction ratio: 2.62%; the 5% cap was not active.
- Decision: reject branch-trunk contributions as the sole correction basis.

This rules out further tuning of onset/flux weights or trust radius within that
span. The high-frequency truncation floor is only `0.02564`, so the poor
capacity is not explained by the retained-64-frequency representation.

### 3. Physical-head channel transformation oracle

The only conceptual change was the correction basis. The 64 highest-energy
channels were selected from the 192 penultimate physical-head channels using
the frozen prediction only.

- Effective rank: 64.
- Trust-capped oracle future relative L2: `0.720482 -> 0.705496`.
- Relative improvement: 2.08%, passing the frozen 1% capacity gate.
- Unrestricted oracle future relative L2: `0.667250` (7.39% improvement), but
  it requires a 31.45% correction ratio and is not a safe deployment update.
- Decision: retain this span for causal coefficient-identification research.

### 4. Physical-head span with unchanged online objective

- Online objective improvement: 0%.
- The core rejected the update and returned the parent bit-exactly.
- Future relative L2 therefore remained `0.720482`.
- Decision: reject the two-onset-plus-current-DG-flux objective as a standalone
  coefficient estimator, even though the underlying span has useful capacity.

## Diagnosis

The limiting factor is now classified as **instance coefficient
identifiability**, not optimizer choice, trust-radius tuning, or branch-trunk
basis capacity.

Two adjacent onset snapshots are intrinsically insufficient to identify a
general wave continuation without a strong prior. This is consistent with the
non-uniqueness results in *The Snapshot Problem for the Wave Equation*
(arXiv:2308.12208). The present DG flux proxy cannot supply the missing phase
and late-time information, especially while the source-consistent saved-grid
Helmholtz forcing remains unbound.

The physical-head result supports structured feature adaptation rather than
full-network online fine-tuning. Recent neural-operator transfer studies report
strong results from neuron-wise linear transformations and branch/trunk-aware
structured adaptation (arXiv:2512.17969; arXiv:2604.03449). LoRA remains useful
as a parameterization, but physics-only LoRA requires a source-consistent PDE
residual; applying it to the current mismatched residual would optimize another
proxy rather than the data generator (arXiv:2504.15933; arXiv:2608.07053).

Historical project evidence also warns against tiny offline basis-training
sets: a prior three-record CPADC run had negative realized improvement and only
0.052% best oracle improvement. The next run must increase group-disjoint train
episode diversity rather than repeat that underidentified regime.

## Selected next algorithm

Use a **meta-learned prior over fixed physical-head channel coefficients**, with
a convex online correction around that prior.

1. Freeze the completed parent checkpoint. Do not bind to the current
   in-progress update-13650 checkpoint for a serious adapter run.
2. On train-fit groups only, select one global set of 64 physical-head channels
   by mean parent-only energy. Fixed channel identities are required so
   coefficient semantics do not change between records.
3. Compute trust-capped oracle coefficients from train future truth for offline
   supervision only.
4. Train a small hypernetwork to predict a coefficient prior and diagonal
   precision from allowed inputs:
   - pooled velocity/medium features;
   - source location, frequency, onset, and amplitude;
   - two parent-onset residual frames projected onto the fixed head modes;
   - normalized DG flux summaries;
   - parent spectral-energy summaries.
5. At deployment, solve the convex system
   `||A_online alpha + b||^2 + ||D(alpha - mu(x))||^2`, then enforce the
   correction trust cap and an online-only abstention gate.
6. The hypernetwork never receives validation/test labels or future deployment
   truth. Future train truth is confined to the offline outer loss.

This combines the measured capacity of physical-head channel transformations
with a learned prior that resolves the two-snapshot ambiguity, while keeping
deployment deterministic and inexpensive.

## Next evidence gates

The serious run must wait for the adaptive PyramidMoE pretraining checkpoint to
be terminal and hash-frozen. Then:

1. Build a new group-disjoint train manifest excluding every development group
   opened in this iteration.
2. Use at least 24 train-fit episodes (8 per target family), not a three-record
   pilot. Reserve separate train calibration and confirmation groups.
3. Require finite training, exact no-future deployment access, and bit-exact
   rollback before accuracy scoring.
4. Pilot gate: positive aggregate future improvement, no family mean regression,
   and at least 75% non-worse records on train calibration.
5. Confirmation gate: same frozen adapter and hyperparameters on disjoint train
   groups. Only after it passes may one complete validation evaluation be opened.

No validation or `test_id` action is authorized by this report.

## Artifacts

- Branch online adaptation: `results/transfer_dg_pyramid_u13650_marmousi_00260_instance_adapt_cpu_20260904/run_v2/`
- Branch-span oracle: `results/transfer_dg_pyramid_u13650_branch_basis_oracle_20260904/run/`
- Physical-head oracle: `results/transfer_dg_pyramid_u13650_head_basis_oracle_20260904/run/`
- Physical-head online adaptation: `results/transfer_dg_pyramid_u13650_head_online_adapt_20260904/run/`

