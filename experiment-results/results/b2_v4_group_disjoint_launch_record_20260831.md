# B2-v4 group-disjoint launch record

Launch verified at `2026-08-31T10:24:07Z`.

## Frozen identity

- Preregistration: `results/b2_v4_group_disjoint_preregistration_20260831.json`
- Preregistration SHA256: `13a3f1cd0894e33d66fabe4627a0fc1d4c143f96f943b8ad4d700951ee9977ef`
- Launch audit: 411/411 checks passed.
- CPU contracts: 15 tests passed.
- Detached screen: `124830.b2_v4_group_disjoint_20260831`
- Supervisor shell PID: `124831`

## Bound caches

- Fit cache: 240 records / 240 groups, 2,618,014,880 bytes,
  SHA256 `d49077526d5c02aac49d9ff62c4aec190d3b2083cecd9a4e75a877766a715336`.
- Calibration cache: 60 records / 60 groups, 654,506,920 bytes,
  SHA256 `5da6cbae5d48418d8bbffbc21160aa6e69e51dfe1256e8fbf2be54c31cdcd08f`.
- Both cache identities bind the same preregistration and report
  `validation_opened=false`, `test_id_opened=false`.
- The frozen confirmation manifest exists, but no confirmation cache or
  metric has been created.

## Live lanes

| Seed | PID | GPU | Epoch 1 wall time | Epoch 1 calibration aggregate | Nonworse |
|---|---:|---:|---:|---:|---:|
| 372 | 126592 | 0 | 182.9 s | 0.15074864 | 0/60 |
| 733 | 126593 | 1 | 181.7 s | 0.13240080 | 3/60 |

The calibration warp-anchor baseline is `0.12054686`; the frozen aggregate
gate is `0.11054686`, with at least 54/60 nonworse records and improvement in
all three families for both seeds. Epoch 1 is an expected zero-head opening
transient and is not a gate result.

Both epoch-1 wall times pass the 360-second abort rule and project to roughly
3.6 hours for 70 epochs. Each lane wrote a bound `run_identity.json`,
`metrics.jsonl`, `best.json`, and `best.pt`. GPU 0/1 each hold approximately
13.3 GB; GPU 2/3 remain free. Disk free after cache creation is approximately
199 GB.

## Next gate

Wait for both 70-epoch terminal records. Judge the two frozen calibration
gates without extension. Build and open the 60-record train confirmation cache
only if both seeds pass; otherwise reject B2-v4 and keep confirmation sealed.
