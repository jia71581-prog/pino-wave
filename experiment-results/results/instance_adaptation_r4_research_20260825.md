# R4 instance-adaptation research decision (2026-08-25)

## Parent and leakage boundary

- Frozen parent: r4 epoch 7, SHA-256 `448035bd0061205c67799eeef3b023a71b49e2db7028b6155078be1a886de789`.
- Fixed train-panel relative L2: `0.28235819335238516`.
- Dominant parent errors: Marmousi `0.581898`, late time `0.745570`, middle spectrum `0.797920`.
- Every experiment below used train records only. Validation and `test_id` remained unopened.

## Evidence obtained

| Candidate | Evidence | Decision |
|---|---:|---|
| Frozen-parent temporal latent subspace, rank 32 | 3/3 records harmed; mean per-record change `-41.81%`; scalar oracle chose zero on all records | Reject basis direction |
| Frozen CPADC R7 basis transfer | Legacy basis lacks current CPML contract | Reject dishonest contract relabeling |
| Modern R5b CPML basis transfer | 7/43 implementation digests drifted | Reject direct transfer |
| New r4 CPADC, rank 32, 12 train records, 10 epochs | Inner solves 100% accepted, but every epoch harmed train patches by about `5-6%`; epoch-10 oracle gain only `0.074%` | Reject current learned-basis parameterization |
| Parent correction-amplitude scalar oracle | Mean capacity `0.874%`; family means uniform `1.008%`, layered `0.272%`, Marmousi `1.341%` | Useful auxiliary mode only |
| Global time-dilation oracle | Mean capacity `0.107%`; four of six records chose zero | Reject |
| One-scalar eikonal travel-time warp oracle | Six of six records chose zero | Reject |
| Onset-conditioned CNN residual meta-adapter | One-record 200-step relative-L2 overfit improved only `0.0198%` | Reject architecture |

The failed candidates were not promoted, and no validation or test data were used to select them.

## Selected next algorithm: travel-aligned empirical residual mode adaptation

The next candidate should learn the residual subspace directly instead of asking a random or parent-internal basis to discover it through a weak online objective.

### Offline train-only construction

1. Freeze the r4 epoch-7 parent.
2. Stream parent residuals `truth - parent` from mutually exclusive train basis records; do not store complete per-record fields on disk.
3. Align residuals in source-centred travel coordinates `xi = t - tau(x,z)` using the registered eikonal cache.
4. Split aligned residuals into early/middle/late and low/middle/high frequency blocks.
5. Fit family-conditioned randomized POD/SVD modes with a C1 causal envelope. Every mode is exactly zero through the two observed onset frames and on the free-surface top row.
6. Preserve the left/right/bottom CPML interface contract in the physics design matrix.

### Online instance solve

For a frozen mode tensor `B`, solve only its coefficients:

`min_c  L_observed + 0.5 L_bridge + L_LWC84_defect + 0.1 L_CPML + 1e-4 ||c||^2`

subject to a correction-energy ratio at most `0.05`. The solve is fixed-size CPU Cholesky/LSPG; parent weights and mode tensors remain frozen. If the causal objective does not improve, the condition number exceeds `1e8`, or the correction hits the trust boundary, use zero coefficients and return the exact parent.

### Required gates

1. **Cross-fitted oracle capacity:** held-out train records, at least `5%` mean residual reduction and at least `1%` in every family.
2. **Sealed disjoint-train online confirmation:** at least `1%` mean improvement, at least `95%` nonworse, every family mean nonworse, adaptation P95 below `3 s`.
3. **Frozen complete validation:** only after modes, coefficient objective, trust radius, and abstention thresholds are immutable.
4. `test_id` is not reopened during development.

## Literature mechanism

- GEPS motivates low-rank rapid adaptation of context parameters rather than full-network fine-tuning.
- PINO supports test-time optimization of a pretrained operator with governing-equation residuals.
- Meta-PDE supports learning a task distribution that enables a few reduced optimization steps on a new PDE instance.
- The local CPADC R7 evidence supports a convex, causal, abstaining online solve, but the basis must be retrained for the selected parent and current implementation contract.

## Resource constraint

Only about 2.2 GB remains on the training volume. The next implementation must stream residuals and save compressed modes and scalar reports only; it must not materialize a second full wavefield corpus.

