# Proposed method versus LWC-84 runtime

This directory contains a retrospective, compute-only runtime comparison. It was
generated from already completed reports and did not execute a model, solver, or
GPU benchmark.

`proposed_method_vs_lwc84_runtime.json` is the authoritative summary. The
companion CSV contains the manuscript table values. “Ours” denotes the complete
bound parent operator plus CPADC instance fine-tuning. Its end-to-end time includes
input transfer, parent inference, basis generation, fine-tuning of 16 CPU ridge
coefficients, inverse normalization, and materialization of all 401 output frames.
It excludes checkpoint loading, offline pretraining, and disk I/O. The online fit
updates instance coefficients rather than neural-network weights.

On independent `test_id`, instance fine-tuning averages 0.452844 s. Inference
and output materialization average 9.957606 s, giving 10.410450 s end to end.
The synchronized LWC-84 reference averages 20.499601 s. The ratio of means is
1.969137. The conservative fastest-LWC-to-ours-P95 ratio is 1.904215.

Instance fine-tuning is only 4.35% of our mean latency. Removing it entirely
would increase the ratio of means to only 2.058688. A conservative 10x target
requires total latency at or below 2.034137 s, so parent inference and output
materialization, not the ridge solve, are the primary performance bottlenecks.

The result is descriptive, not a certified speed-superiority result. The archived
R7 records predate explicit synchronization and runtime-protocol tags. The methods
were not timed on identical cases, and our absolute accuracy is not matched to the
LWC-84 reference. Both registered 10x speed gates therefore fail.

Reproduce the summary with:

```bash
python scripts/summarize_cpadc_vs_lwc84_runtime.py \
  --validation-summary <cpadc-r7-validation-summary.json> \
  --test-id-summary <cpadc-r7-test-id-summary.json> \
  --traditional-runtime <traditional-lwc84-runtime.json> \
  --output-json proposed_method_vs_lwc84_runtime.json \
  --output-csv proposed_method_vs_lwc84_runtime.csv
```
