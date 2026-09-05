---
name: acoustic-research-orchestrator
description: Coordinate multi-agent, leakage-safe research for the FNO acoustic-wave workspace, from data and checkpoint audits through pretraining, physics losses, instance adaptation, detached GPU experiments, and promotion gates. Use when the user asks to continue, automate, orchestrate, or comprehensively research the current acoustic operator project. Do not use for unrelated PDE repositories or a single self-contained factual question.
---

# Acoustic Research Orchestrator

Work in `/root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2` and obey its closest `CLAUDE.md` (and `AGENTS.md`, which carries the same contract for Codex sessions).

## Start

1. Run `scripts/snapshot_research_state.py` from this skill and inspect active GPU jobs, disk, terminal records, checkpoints, and latest metrics before changing state.
2. Read `/root/.codex/skills/acoustic-operator-research/references/protocol.md`. Read its literature map only when proposing an algorithm, loss, sampler, optimizer, or adapter.
3. Read [references/team.md](references/team.md) when delegation is available. Read [references/research-loop.md](references/research-loop.md) before selecting or launching an experiment.

## Delegate read-heavy lanes

For broad research turns, dispatch the relevant project subagents with the `Task` tool in parallel (one message, multiple calls), wait for all of them, and return concise evidence summaries:

- `acoustic-data-auditor`
- `acoustic-model-diagnostician`
- `acoustic-physics-reviewer`
- `acoustic-adaptation-researcher`
- `acoustic-experiment-auditor`
- `acoustic-training-monitor` when a job is active

Use at most five concurrent research lanes.

If no `Task` tool is exposed, dispatch real subagent processes instead of doing the lanes yourself:

```
/usr/local/bin/claude -p --agent <agent-name> --allowedTools Read Grep Glob < /tmp/lane_prompt.txt > reports/lanes_<date>/<lane>.md 2>&1
```

Run them under `nohup ... &` in a single call for real parallelism and poll `pgrep -fc "claude -p --agent"`. Restricting the toolset to `Read Grep Glob` makes the read-only contract structural instead of advisory. Only if that path is also unavailable may the lanes run in the parent thread, and that fallback must be reported.

## Decide centrally

The lead agent owns synthesis. Select one falsifiable hypothesis and the cheapest evidence tier that can reject it. Record mechanism, primary metric, failure signal, compute budget, exact parent/config/code/data digests, rollback, and split access before a serious run.

Do not let a subagent promote its own proposal. `acoustic-experiment-auditor` must review the preregistration and split boundary before a long launch, including **gate validity**: whether each promotion gate measures what its name claims, whether any reference constant was transplanted from records other than those being scored, and whether any gate can pass trivially. A gate-validity finding voids the candidate verdict through that gate; it is reported as "gate not decidable", never as a result about the candidate.

Acceptance thresholds that depend on an achievable-gain reference cannot be frozen before that reference has been recomputed on the records that will be scored. Sequence the cheap zero-training reference probe *before* the preregistration freeze, not after.

## Mutate through one writer

Only `acoustic-experiment-worker` may edit code/configs or launch a GPU experiment, and only after the lead supplies an exact candidate identity and preregistration path. Never run two write agents concurrently. Preserve user changes and all protected checkpoint families.

For long jobs, launch detached and verify supervisor PID, child ranks, live log, run identity, checkpoint path, terminal state, `nvidia-smi`, and disk headroom. A process alone is not proof of a live run.

## Evidence and promotion

Keep `validation` and `test_id` sealed during development. Online instance adaptation may access only the protocol-approved onset observations and physics-derived quantities. Future train truth is permitted only in explicitly offline train meta-losses or post-seal diagnostics.

Reject candidates that pass only training loss, sampled frames, an oracle diagnostic, or a reused holdout. Promote only through the complete evidence ladder in the acoustic protocol. Lead with measured results, distinguish accepted checkpoints from attempted updates, and never claim the target from projected trends.

## Handoff

Return the current accepted checkpoint, active or terminal run state, measured metrics, rejected hypotheses, next cheapest experiment, artifacts, and any blocker. Keep raw logs out of the main response; link to them instead.
