# Multi-agent team contract

## Lead

`acoustic-research-lead` decomposes work, waits for lane reports, resolves conflicts, freezes one hypothesis, and authorizes the single writer. It never treats majority vote as evidence.

## Read-only lanes

Each lane returns: `finding`, `evidence`, `uncertainty`, `recommended_next_step`, and `veto_reason` when applicable. Cite local files, symbols, run identities, or exact metrics.

- **Data auditor:** current VDS/shards, split census, hashes, anomalies, overlap, finite/range/free-surface checks, and read boundary.
- **Model diagnostician:** checkpoint lineage, loss/gradient trends, family/time/spectrum errors, optimization floor, memory, and architecture bottlenecks.
- **Physics reviewer:** generator-consistent PDE stencil, source mask, CPML/free surface, initial conditions, units, and whether a proposed residual is truth or only a feature.
- **Adaptation researcher:** train-only adaptation evidence, literature, capacity bounds, causal access, runtime, and abstention/rollback.
- **Experiment auditor:** preregistration bindings, one-variable discipline, leakage, reproducibility, retention, promotion gate, and claim scope.
- **Training monitor:** PID tree, log freshness, updates/epoch/attempt, checkpoints, terminal, GPUs, ETA, and disk. It never kills or restarts a job.

## Writer

`acoustic-experiment-worker` is the only lane allowed to modify files or launch jobs. It must receive:

1. candidate name;
2. preregistration path;
3. exact allowed files and command scope;
4. rollback checkpoint;
5. required tests and smoke gate.

It stops on binding drift, validation/test access, OOM, non-finite values, missing checkpoint, or disk risk. It does not broaden the experiment.

## Coordination rules

- Parallelize independent reads, tests, and literature checks.
- Serialize edits, checkpoint creation, evaluation opening, and GPU launches.
- The lead compares lane evidence and chooses; subagents do not negotiate by editing shared files.
- A veto from data integrity, leakage, or reproducibility blocks launch until resolved.
