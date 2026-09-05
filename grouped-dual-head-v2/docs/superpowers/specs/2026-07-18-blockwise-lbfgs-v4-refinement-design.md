# V4 Blockwise L-BFGS Refinement Design

Date: 2026-07-18

## Evidence and objective

The batch-24 AdamW run improved held-out relative L2 from 59.40% to 55.83%, with its best checkpoint at epoch 36. Epochs 37-48 did not improve the best value; the final ten-epoch validation slope was non-negative while training loss continued to fall. GPU utilization was already 100% at about 345 W, so the failure is an optimization/generalization plateau rather than data starvation or low GPU occupancy.

The next stage must restart from `/data/jiayh/saved_time_v4_spectrum_ablation_batch24/variants/deep_phase_plain/best.pt`, increase the effective batch, preserve exact stored-time targets, and test whether a curvature-aware optimizer can leave the plateau without losing any medium family.

## Alternatives

1. Full-model L-BFGS: rejected. Curvature history over all 18.9M parameters is feasible in raw bytes but expensive, and repeated closure evaluation through the full transferred backbone creates unnecessary activation and line-search cost.
2. Continue AdamW with a different learning rate: retained only as a fallback. The current run already crossed a cosine schedule and plateaued after the best checkpoint.
3. Frozen-backbone blockwise L-BFGS on the V4 dense decoder: selected. It optimizes the newly introduced operator block, keeps the useful V3 representation fixed, bounds history memory, and supports deterministic strong-Wolfe closures.

## Optimizer and batch contract

- Load the exact V4 best checkpoint and verify its manifest/config bindings.
- Freeze every parameter outside `dense_decoder`.
- Use PyTorch L-BFGS with learning rate 0.5, history size 10, `max_iter=4`, `max_eval=6`, tolerance-grad `1e-7`, tolerance-change `1e-9`, and `line_search_fn="strong_wolfe"`.
- A single outer step merges four fixed balanced macro batches into 48 records, then uses three physical microbatches of 16. The frozen-backbone microbatch-12 smoke peaked at only 11.67 GB, leaving enough measured headroom to test 16 safely; the prior 24 GB overflow applied to full-backbone gradient training, not this blockwise stage. Held-out macros remain at their natural physical batch of 12.
- The same cached CPU batches are replayed for every closure evaluation within an outer step. No random target resampling is allowed inside a line search.
- The objective is record-level relative L2 plus 0.1 spatial-gradient loss and 0.2 band-limited complex-spectrum residual loss.
- Run 12 outer steps. Evaluate the same fixed held-out 48-record schedule after every step and atomically save every checkpoint, `latest.pt`, and `best.pt`.

## Gates and observability

The probe passes only if its best validation relative L2 improves by at least 2% over 0.5583050847 and no uniform/layered/Marmousi family regresses by more than 3% relative to the parent metrics. Non-finite loss, a failed line search, identity mismatch, or peak allocated memory above 23 GiB terminates the run.

Every outer step appends one JSON object to `optimizer_steps.jsonl`, including closure count, initial/final loss, validation metrics, learning rate, peak GPU memory, GPU power/utilization snapshot, checkpoint, and gate state. Raw stdout is written to `launcher.log`; users can inspect either file with `tail -F`.

If the 2% gate fails, no L-BFGS long run is authorized. If it passes, the best refined checkpoint becomes the parent for a full-data continuation with the same effective batch contract.
