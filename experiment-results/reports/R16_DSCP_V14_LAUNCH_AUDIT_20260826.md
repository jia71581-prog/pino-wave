# r16_dscp_v14 launch audit (post-remediation)

Date: 2026-08-26. Candidate: `r16_dscp_v14`. Prereg: `results/r16_dscp_v14_preregistration_20260826.json` (status `frozen`).

## Provenance of this audit

The 2026-08-26 07:23 Codex audit thread verified items A-E and terminated while quantifying a **disk VETO**; it never issued a verdict. That VETO was correct: the frozen `space1`/`space4` gate passed with only 91 MiB of margin.

This is the **second audit**, run after the vetoed condition was remediated and nothing else. It is not verdict shopping: the only state change between the two audits is workspace headroom.

Execution mode: **in-thread fallback**. This Claude Code session exposes no `Task` tool, so the `acoustic-experiment-auditor` lane could not be dispatched as a subagent. Per `.claude/skills/acoustic-research-orchestrator/SKILL.md`, the lane was executed in the parent thread and that fallback is reported here. Read-only discipline was kept: no candidate artifact was modified by the audit itself.

## Verdict: CLEAR for GPU0 smoke

| Check | Result |
|---|---|
| Binding drift (16 bound files, recomputed sha256) | 16/16 exact, zero drift, none missing |
| Self-referential hashes (prereg -> preflight -> static_evidence) | all match |
| Checkpoint binding | `best.pt` == `last.pt` == `4281d87d...`, 39271 B, `resume_equal=true`, hardlinked |
| Parent lineage | A3 `epoch_0007.pt` `448035bd...` 339775310 B intact |
| Frozen v13 parent prereg | `249301003a...` intact |
| Full v1-v14 regression, independently re-run | 197 passed, EXIT=0, `/tmp/r16_v14_full_reverify_20260826.log` sha256 `ea6dd011...` |
| Three v13 launch-veto fixes | present and green inside that 197 (production CLI interrupt/resume + reference equivalence; 4-process torchrun rank0 persistence fault; RANK==LOCAL_RANK contract) |
| Workspace space gate (live) | free 12.61 GB vs required 2.215 GB, **margin +10.39 GB** (was +0.096 GB) |
| tmpfs spool gate (live) | 128.8 GB free vs 132 MB (w=1) / 528 MB (w=4) |
| Host cache gate, long role | 742.7 GB available vs 60.5 GB required incl. 16 GiB reserve |
| GPUs | 4x RTX 4090 D idle, 0 MiB used, UUIDs match `identities.gpu` |
| Sealed split boundary | `prep_truth_reads=0`, `test_id_tuning=false`, validation/test futures sealed until stage authorization; both splits 480 records, digests recorded |
| One-variable discipline | `only_change` = production long resume + synchronized persistent-write failure + strict four-rank launch integrity; budget values unchanged from v12 |
| Rollback | preserve parent and every protected checkpoint, no deletion |
| Claim scope | parent speedup is only 1.608x; candidate may not claim 10x or `<=0.05` without sealed evidence |
| Protected checkpoint retention | helmholtz 89, a3 16, cpadc 42, b2h 11 present |

## Remediation record

Workspace headroom was recovered by deleting 120 intermediate `results/wkb*/checkpoints/update_*.pt` files (12.36 GB) from the 20 wkb runs whose `terminal.json` status is `failed`, on explicit user instruction naming that scope.

- Manifest: `reports/WKB_FAILED_RUN_CHECKPOINT_RECLAIM_20260826.json` (per-file path/size/mtime).
- Superset analysis: `reports/WKB_INTERMEDIATE_CHECKPOINT_RECLAIM_20260826.json`.
- 11 intermediate checkpoints are bound as `--parent-checkpoint` / `--exact-init-checkpoint` by other runs' preregistrations and `run_identity.json`. All 11 were excluded and verified present afterwards. A naive "delete all intermediate checkpoints" would have severed those lineages.
- Every one of the 20 runs retains `latest.pt`, `terminal.json`, `run_identity.json`, `metrics.jsonl`, `updates.jsonl` and its preregistration. 8 of them never had a `launch.log` (inline smoke runs use `decision.json`); that absence predates this reclaim.
- 24 non-failed wkb runs untouched, still holding 321 intermediate checkpoints.

## Standing constraints for the promotion chain

Stop immediately on failed scientific or resource gate, NaN/Inf, OOM, binding drift, or unsafe disk. Promote only through smoke -> pilot -> scale-1/scale-4 -> scale-decide -> selected-world long -> final-train-confirm -> validation-once -> test-once. Note that `gates.traditional_speedup_mean_p95_min = 10.0` sits against the recorded finding that the traditional high-order solver is both faster and more accurate on this dataset; the gate, not narrative, decides.
