# Research record — FP32 single-variable cache control (2026-09-05)

Candidate: r6_device_residual_cache_fp32_v1_20260905
Stage: results/r6_anchored_r54_device_resident_r1_20260905/cache_stage_fp32_v1/

## Question
The f16 six-record cache smoke was rejected (smoke_terminal.json sha256 4aaaa5c6…, status
cache_representation_rejected: 5/6 records failed E_q/R_native <= 0.25, ratios 0.164–3.058).
Single-variable control: does storing the two wavefields (coarse_norm, truth_norm) as float32 —
static7 unchanged f16, all numerics/parent/scale/data/model/loss unchanged — remove the storage loss?

## Protocol integrity
- Preregistration sha256 03bacfbacd731c371afab1dbd17e7eeef5203f8059939e1b4560173a05fd25f4,
  dependency_manifest sha256 6081ba0d0be886790bf577762197116ce70fce5732b45a8e45f2913af72dc0a5,
  33 bindings + 37-file closure, status fp32_smoke_prepared_pending_audit.
- CPU tests first: tests/test_r6_device_residual_cache_fp32.py — 20 passed (real R25
  CacheCollection/FitFrameDataset/train_loss/evaluate on non-f16-representable f32 data vs independent
  native f32 in-memory baselines; nonzero differentiable corrections with nonzero gradients, both a
  constant-parameter probe and a reinitialized CoarseResidualUNet; explicit development-subset wiring;
  no format-faking monkeypatches). Prior f16 suite: 24 passed, unchanged.
- Independent read-only audit: CLEAR, all 7 items PASS (audit_20260905/prelaunch_audit_report.md).
- Launch: single writer, detached, frozen argv, GNU timeout watchdog 300 s TERM +30 s KILL (per audit),
  GPU0 only.

## Result (smoke_terminal.json sha256 e0a7518d06a8c4102fdfbb108cff12bb445df37244130fe20946c3ad4912cb9e)
status = complete; cache_representation_gate_passed = true; all six records:

| role | source_index | family | R_native | E_q (f16 was) | E_q/R_native (f16 was) | gate |
|---|---|---|---|---|---|---|
| fit | 0 | uniform | 2.06e-4 | 0.0 (2.371e-4) | 0.0 (1.1508) | PASS |
| fit | 420 | layered | 3.9e-5 | 0.0 (1.181e-4) | 0.0 (3.0576) | PASS |
| fit | 2100 | marmousi | 6.91e-4 | 0.0 (2.707e-4) | 0.0 (0.3916) | PASS |
| development | 357 | uniform | 4.11e-4 | 0.0 (2.655e-4) | 0.0 (0.6467) | PASS |
| development | 968 | layered | 1.459e-3 | 0.0 (2.393e-4) | 0.0 (0.1640) | PASS |
| development | 2375 | marmousi | 1.26e-4 | 0.0 (2.063e-4) | 0.0 (1.6416) | PASS |

- Stored f32 bytes reproduce the pre-storage arrays exactly per record (byte compare) and, cast to f16,
  reproduce the prior f16 shard bytes exactly (prior_f16_parity all true; identical parent full hashes) —
  same numerics, storage dtype is the only changed variable.
- Resources: elapsed 30.554 s (budget 180), peak 352,735,232 bytes (= f16 smoke peak, budget 8 GiB).
- Truth ledger: 6 reads, one per record after parent hash; additional_source_truth_reads = 0 everywhere.
- input_hashes_unchanged = true; prior f16 terminal and this preregistration byte-identical after run.
- Outputs: smoke_fit_shard_0.h5 32,127,386 B; smoke_development_shard_0.h5 200,286,192 B
  (2.6x f16 sizes — f32 residual-scale detail compresses worse under LZF, as expected).

## Interpretation (bounded)
- ACCEPTED: fp32 storage of the two wavefields is exactly lossless (E_q = 0) for these six records;
  the f16 rejection was purely a storage-representation failure, not a numerics/protocol defect.
- The measured R_native (3.9e-5 … 1.459e-3) confirms why f16 failed: the R6-parent residual sits at or
  below f16 quantization scale after max-abs normalization.
- NOT claimed: any learning benefit, generalization, or operator improvement. E_q = 0 is storage-only
  (preregistered interpretation). No full build, training, confirmation, validation, test_id, or promotion.

## Costs and remaining debts
- Full fp32 cache projection: 49.468 GiB, launch needs 89.468 GiB free (future budget only; unauthorized).
- Pending debts before any full build: parent_hash replay audit, terminal identity audit, budget enforcement.

## Next cheapest experiment (recommendation, not authorization)
Do NOT build the 49.5 GiB full cache yet. Preregister a small fp32 fit-subset training probe
(e.g., 3-4 GPU-hours: a few hundred fit records cached fp32 on one shard, R25 CoarseResidualUNet,
explicit expected_subset=development evaluation) to test whether a learned correction on lossless
residuals beats the frozen R54 baseline (development mean 1.7747%) at all, before paying full-cache
disk/time. The coarse-201 larger-dt alternative is closed
(audit_20260905/coarse_dt_rejection_evidence_20260905.md): dt 500/625 us violate the frozen qmax<1 QC
(input-side: 690/3203 resp. 1964/3203 core records; v23 historical abort), dt 250 us passes accuracy but
fails runtime (1.4672 s > 0.9 s gate; slower than the R6 fine parent ~0.89 s).
