# Acoustic Operator Research Workspace

> **停用通知(2026-08-26)**:Codex 会话已停用,研究全部在 Claude Code 内进行。本文件与
> `.codex/agents/*.toml` 保留为历史记录,不再与 `.claude/agents/*.md` 同步。活跃契约见 `CLAUDE.md`。
> Codex 产出的历史结果仍是有效证据。

- Use Chinese by default unless the user requests otherwise.
- Preserve user changes and all A3, B2-H, Helmholtz, ASAM, and CPADC checkpoints.
- Inspect code, configs, logs, checkpoints, disk, active processes, and GPUs before changing state.
- Treat validation and test_id future wavefields as sealed during algorithm development.
- Never claim a target from training loss, sampled frames, an oracle diagnostic, or projected trends.

## Multi-agent research

For broad requests to continue or automate research, use `$acoustic-research-orchestrator` and project agents under `.codex/agents/`. Claude Code sessions use the ported equivalents in `.claude/agents/` and `.claude/skills/`; see `CLAUDE.md`.

- Delegate independent read-only audits in parallel.
- Wait for all requested lane reports before choosing an experiment.
- Use only `acoustic_experiment_worker` for edits or GPU launches, and never run multiple writers concurrently.
- Freeze a preregistration with exact hashes, acceptance, failure signal, budget, and rollback before serious runs.
- Require the experiment auditor to clear data integrity, leakage, lineage, and reproducibility before a long launch.
- Launch long GPU work detached and verify PID tree, live log, run identity, checkpoint, terminal, `nvidia-smi`, and disk.
