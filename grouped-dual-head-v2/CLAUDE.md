# Acoustic Operator Research Workspace

- Use Chinese by default unless the user requests otherwise.
- Preserve user changes and all A3, B2-H, Helmholtz, ASAM, and CPADC checkpoints.
- Inspect code, configs, logs, checkpoints, disk, active processes, and GPUs before changing state.
- Treat validation and test_id future wavefields as sealed during algorithm development.
- Never claim a target from training loss, sampled frames, an oracle diagnostic, or projected trends.
- Workspace disk is `/dev/md0` mounted at `/root/autodl-tmp` (380 G, chronically near full). Check `df -h .` there, not `/`.

## Multi-agent research

For broad requests to continue or automate research, use the `acoustic-research-orchestrator` skill and the project subagents under `.claude/agents/`.

- Delegate independent read-only audits in parallel (one message, several `Task` calls, at most five lanes).
- Wait for all requested lane reports before choosing an experiment.
- Use only `acoustic-experiment-worker` for edits or GPU launches, and never run multiple writers concurrently.
- Freeze a preregistration with exact hashes, acceptance, failure signal, budget, and rollback before serious runs.
- Require `acoustic-experiment-auditor` to clear data integrity, leakage, lineage, and reproducibility before a long launch.
- Launch long GPU work detached and verify PID tree, live log, run identity, checkpoint, terminal, `nvidia-smi`, and disk.

Codex 会话已于 2026-08-26 停用,研究全部在 Claude Code 内进行。`AGENTS.md` 与 `.codex/agents/*.toml` 保留为历史记录,不再需要与 `.claude/agents/*.md` 保持同步;Codex 产出的历史结果(r7 续训、CPADC-LSPG-RAD 链、R6-R8 线核验等)仍是有效证据,继续在其基础上推进。
