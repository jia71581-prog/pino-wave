# V3 Easy-to-Hard Curriculum Design

## Goal

Continue from the best balanced full-data V3 pilot checkpoint and improve generalization by
training in the order uniform → layered → Marmousi slices, while preserving performance on
previous families and retaining the single-source-per-equation data contract.

## Non-goals

- Do not implement FWI.
- Do not condition the forward operator on receiver records.
- Do not superpose sources.
- Do not replace or delete the completed balanced baseline.
- Do not push any branch or artifact to GitHub.

## Starting point and provenance

The curriculum starts only after the 20-epoch balanced pilot has a terminal report. Its parent is
the pilot `best.pt`, not `latest.pt`. The curriculum identity binds the parent checkpoint SHA-256,
the active filtered manifest, the normalization metadata, the curriculum configuration, and the
three stage definitions. A mismatched parent or incomplete baseline prevents startup.

## Stages

1. **Uniform foundation (2 epochs).** Each optimizer step accumulates three four-record uniform
   microbatches. The effective batch is 12 records, but no more than four independent media are
   encoded simultaneously. This avoids the memory increase caused by encoding 12 unrelated
   uniform media at once.
2. **Layered transfer (3 epochs).** The dominant batches contain three four-source layered groups.
   Every fourth optimizer step is a four-record uniform replay microbatch accumulated to the same
   effective batch size. Replay limits forgetting of the uniform foundation.
3. **Marmousi-slice refinement (5 epochs).** Dominant batches use two independent five-source
   Marmousi groups. A deterministic replay cycle inserts uniform and layered microbatches so both
   earlier families remain represented. Sources remain separate records and are never summed.

All stages use a fresh AdamW optimizer at learning rate `2e-5`, weight decay `1e-6`, gradient
clipping `1.0`, exact/interpolated fraction `0.25`, adaptive energy-plus-uniform query sampling,
and the existing dense/query phase-sensitive loss. The medium encoder is not frozen: transfer
between increasing structural complexity is intentional.

## Data and batching contract

The schedule operates on the existing train split only. It selects records deterministically by
family and `group_id`, cycles every record before repetition where possible, and reports actual
family counts for every optimizer step. A microbatch can contain only one physical solution per
record. Grouped records share medium encodings through `record_to_medium`, but their sources,
targets, and losses remain independent.

Uniform uses four-record microbatches with three-step gradient accumulation. Layered uses 12
records from three groups when memory permits. Marmousi uses 10 records from two groups; its loss
is scaled by `12/10` so its optimizer-step magnitude is comparable to the effective batch of 12.
The 30-step stage benchmark may reduce a dominant microbatch only after a CUDA OOM; it may not
remove loss terms or reduce model capacity.

## Validation and checkpoint acceptance

After every curriculum epoch, evaluate the same sealed balanced validation batch used by the
baseline, with exact saved frames and midpoint-interpolated frames reported separately for
uniform, layered, and Marmousi. Save an atomic epoch checkpoint regardless of acceptance. Update
the stage-best link only when all conditions hold:

- all values and required gradients are finite;
- the current target-family exact-plus-interpolated dense/query score improves over the stage
  parent or previous accepted stage checkpoint;
- the aggregate balanced score is no more than 5% worse than the parent;
- each previously learned family's score is no more than 10% worse than the parent.

If a stage produces no accepted checkpoint, the next stage starts from that stage's unchanged
parent. This makes the curriculum additive and prevents a failed specialization from damaging the
balanced baseline.

## Runtime and observability

The curriculum writes to `/data/jiayh/v3_easy_to_hard_curriculum`, with a separate directory per
stage, one checkpoint per epoch, `latest.pt`, accepted `best.pt`, JSONL step metrics, validation
reports, PID, and combined nohup log. CPU workers prefetch complete microbatches with pinned memory.
Each metric row records data wait, transfer, GPU compute, effective records per second, target and
replay family counts, and CUDA peak memory.

## Final evaluation

Compare the accepted final curriculum checkpoint against the balanced baseline best checkpoint on
the same held-out representatives. Produce exact and midpoint full 201×201 wavefield figures,
off-grid arbitrary-point traces, per-family metrics, and inference timing. Promote the curriculum
checkpoint only if it passes the acceptance constraints; otherwise retain the balanced baseline as
the final model and document the curriculum result as negative evidence.

## Stop conditions

Stop the active stage on a provenance mismatch, anomaly-family record, source superposition,
nonfinite target/prediction/loss/gradient, repeated CUDA OOM after returning to the safe
microbatch, query/dense target disagreement, missing epoch checkpoint, or less than 20 GiB free
artifact space.
