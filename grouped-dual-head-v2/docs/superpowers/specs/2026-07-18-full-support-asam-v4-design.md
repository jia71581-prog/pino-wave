# Full-Support Late-ASAM V4 Design

Date: 2026-07-18

## Diagnosis

Fixed-batch L-BFGS reduced its closure objective but worsened held-out error after the first step. The optimizer was functioning; its curvature estimate and descent direction were specialized to only 48 records. A replacement must expose the dense decoder to the full training support and explicitly penalize sharp, support-specific solutions.

## Selected experiment

Restart the AdamW best checkpoint at epoch 36 and freeze the transferred V3 backbone. Apply adaptive sharpness-aware minimization (ASAM) to the dense decoder with AdamW as its base update. ASAM is selected over overlap multi-batch L-BFGS because the observed failure is a growing generalization gap, and over SOAP because matrix preconditioners for the large spectral tensors add substantial implementation and memory risk without directly targeting that gap.

Each update merges four balanced 12-record macros into an effective batch of 48, split as three physical microbatches of 16. The same batch is used for the ascent and descent pass. A schedule of 376 macros (94 updates) covers every one of the 2,240 training records at least once. Eight workers and four-batch prefetch remain enabled.

Use ASAM radius 0.05, adaptive scale epsilon 0.01, AdamW learning rate 2e-5, weight decay 1e-6, record-relative loss, 0.1 spatial-gradient loss, and 0.2 band-limited spectrum loss. Evaluate every 12 updates and at update 94. Atomically save every evaluation checkpoint and retain the validation best.

The short run passes only with at least 2% held-out improvement over 0.5583051 and no family regression above 3%. A non-finite perturbation, peak allocation above 23 GiB, or incomplete 2,240-record coverage terminates the run.

Primary research basis:

- Foret et al., Sharpness-Aware Minimization, ICLR 2021.
- Kwon et al., Adaptive Sharpness-Aware Minimization, ICML 2021.
- Behdin et al., m-Sharpness-Aware Minimization, 2022.
- Late-stage SAM analysis, ICLR 2025.
