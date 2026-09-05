# V4 Blockwise L-BFGS Refinement Results

Date: 2026-07-18

## Outcome

The registered 12-step refinement completed successfully at physical microbatch 16 and effective batch 48. CUDA peak allocation was 16.22 GB and training samples reached about 345 W / 100% utilization during closure evaluation.

The parent held-out relative L2 was 0.5583051. The best L-BFGS checkpoint was step 1 at 0.5579976, an improvement of only 0.0551%, with all medium families safe. The registered gate required 2%, so it failed. By step 12, the closure objective had decreased from 0.6995640 to 0.6984346 while held-out error increased to 0.5631445. This is direct evidence of fixed-support overfitting rather than an inability of L-BFGS to decrease its training objective.

No L-BFGS long run is authorized. `best.pt` is retained for audit, but the next optimizer experiment must change training support on every update and cover the complete 2,240-record training split.

Artifacts:

- run log: `/data/jiayh/saved_time_v4_lbfgs_batch48_v3/launcher.log`
- structured steps: `/data/jiayh/saved_time_v4_lbfgs_batch48_v3/run/optimizer_steps.jsonl`
- terminal decision: `/data/jiayh/saved_time_v4_lbfgs_batch48_v3/run/terminal.json`
- best checkpoint: `/data/jiayh/saved_time_v4_lbfgs_batch48_v3/run/best.pt`
