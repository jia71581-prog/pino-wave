# Pre-launch audit — r6_device_residual_cache_fp32_v1_20260905

Auditor: independent read-only acoustic-experiment-auditor (local Claude subagent), 2026-09-05.
Scope: FP32 single-variable cache-control stage, pre-launch.

**verdict: CLEAR** — launch of the preregistered smoke argv is approved. All seven items PASS; two minor notes below, neither veto-grade.

## Per-item findings

**1. Single-variable scope — PASS**
- `fp32_dataset_contract` (scripts/smoke_r6_device_residual_cache_fp32.py:50-54) copies the frozen `dataset_contract` (scripts/build_r6_device_residual_cache.py:252-267) and mutates exactly two fields: `coarse_norm`/`truth_norm` dtype -> float32. `static_features` stays float16; chunks/lzf/shuffle inherited and re-validated per dataset (smoke script :85-87; test :54-65).
- Solver args identical to the f16 builder (smoke :193-195 vs build :417-425): dt_s=.000625, output_restriction_factor=2, npml=40, c_ref 6750, kappa_max=3, min_freq 8, float32, cuda_graphs. Same `normalized_cache_values`, `static_features`, legacy/device byte parity (smoke :220-221).
- Stronger than declared: `prior_record_parity` (smoke :159-175) requires the fp32 native fields, cast to f16, to reproduce the prior f16 shard bytes exactly, plus equal parent full hash, scalars, and field_scale per record — numerics drift cannot hide.

**2. No sealed-data risk — PASS**
- Truth read only via `TruthGuard.read_truth` (build :143-152): split must be "train", exactly one read, only after the full parent prediction hash is registered (smoke :216-222). `load_roles` rejects any non-train record and enforces fit/development/R29B group disjointness (build :98-104).
- AST-based test forbids any direct `["wavefield"]` subscript in the runner (test :456-461); diagnostics reuse in-memory arrays + cache readback, `additional_source_truth_reads: 0`.
- Panel verified against the hash-pinned R38 manifest: fit = first-per-family at source indices [0, 420, 2100], development = [357, 968, 2375], all `split=train`, byte-identical to `preregistration.json` selection.records.

**3. Preregistration integrity — PASS**
- Recomputed all 33 `bindings` in preregistration.json and all 56 dependency_manifest entries (18 explicit_bindings + 38 closure files, root `scripts/smoke_r6_device_residual_cache_fp32.py`): zero mismatches, including source_h5 `c1684734...`, prior terminal `4aaaa5c6...` (status `cache_representation_rejected`, gate_passed false, prior ratios 0.164-3.058 with 5/6 fails), runtime report `6c221ae7...`.
- Status `fp32_smoke_prepared_pending_audit`; `full_build_authorized`/`training_authorized`/`promotion_authorized` all false.

**4. Runner safety — PASS**
- Atomic terminal (fsync + `os.replace`, `reattest_frozen_fine_grid_r6_train.py:101-113`) on success (:304) and failure (:306-313); refuses to run if `smoke_terminal.json` or `smoke/` exists (:282) and per-role output exists (:181); input hashes verified before and after with equality gate (:286, :293, :295); `cuda:0` only (:260) plus CUDA visibility contract; prior f16 h5 opened `"r"` (:186).

**5. Tests — PASS (executed: 20/20 pass, CPU-only)**
- Real R25 consumers imported directly (test :34-36); `CacheCollection` subset attr check (train_r25 :86), `FitFrameDataset` fit-only (:172). Reads are `np.asarray(..., float32)` — lossless for f32 storage — and tests assert bit-exact equality with pre-write arrays and inequality with f16 round-trips (test :141-178), on data proven non-f16-representable with f16 destroying >25% of the residual (test :127-135).
- Nonzero differentiable corrections in both probes: constant-parameter path with gradient check against an independent numpy loss baseline (test :193-224) and a real `CoarseResidualUNet` with reinitialized nonzero output head and nonzero stem/output grads (test :271-296). `evaluate` checked against an independent native baseline (test :239-268). Only monkeypatch forbids `h5py.File`/`torch.cuda.init` during static verification (test :430-434) — the acceptable kind; nothing fakes formats.

**6. Disk/GPU — PASS**
- The fp32 runner has no build/worker path: `--smoke` required, non-smoke rejected (:260, :277); `build_role_cache`/`shard_records` never invoked. The 49.468 GiB full build is unreachable and unauthorized.
- Expected smoke output ~2x prior f16 shards (12.1 + 77.4 = 89.5 MB) ≈ 180-200 MB. Live free space on /dev/md0: 116,610,539,520 bytes (~108.6 GiB) against a ~2.1e8-byte requirement — no disk veto. Stage dir has no smoke outputs and no terminal. All four GPUs idle (0 MiB).

**7. Consistency traps / gate validity — PASS**
- Gate audited to source (smoke :107-157): all numerators/denominators from the same record; no cross-record ratios; thresholds (E_q<1e-3, ratio<=0.25, R_native floor 1e-5) identical to the f16 gate on the same six records — not transplanted, and not moved toward the observed f16 failures (which reached 3.058). Not trivially passing: E_q=0 is itself the claim, measured from actual HDF5 readback with a separate byte-exact raise (:236-237), and the gate demonstrably fails on f16-equivalent content, zero-truth corruption, and static-byte drift (test :316-349). Degenerate denominators handled; bands diagnostic-only.
- Claim scope locked: `E_q_zero_interpretation: storage_lossless_only_not_learning_benefit` (prereg gate); terminal embeds all authorization flags false and `operator_failure_claimed: false`; no learning metrics in outputs; R25 main hardcodes `expected_subset="holdout"` (train_r25 :534) so the development cache cannot be silently rewired (proven by test :181-190).

## uncertainty
1. Labeling mismatch: prereg `prerequisites.runtime.status: "passed"` while runtime_report.json's own status string is `runtime_identity_passed_cache_prereg_pending_audit`. The file is hash-pinned and its internal gates (`runtime_gate.passed`, `target5_audit.passed`, `time_axis_qc.passed`, `zero_heads.passed`) are all true, so this is a summary-label discrepancy, not drift.
2. The 180 s / 8 GiB budget is enforced post-hoc after both roles (smoke :293-294); a hung process would not self-terminate.
3. Terminal status `"complete"` on pass must not be read as build authorization — the embedded false flags and `pending_debts_not_covered_here` govern.

## recommended_next_step
Launch exactly the frozen argv (`env -u CUBLAS_WORKSPACE_CONFIG CUDA_VISIBLE_DEVICES=0 python scripts/smoke_r6_device_residual_cache_fp32.py --smoke --preregistration results/r6_anchored_r54_device_resident_r1_20260905/cache_stage_fp32_v1/preregistration.json --device cuda:0`) under a launcher-side wall-clock watchdog (~300 s hard kill) to cover the post-hoc budget gap; any full-build authorization requires a new preregistration and the three listed pending debts.

**veto_reason:** none.
