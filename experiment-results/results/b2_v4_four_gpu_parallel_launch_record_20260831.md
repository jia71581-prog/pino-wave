# B2-v4 four-GPU parallel launch record

## Scheduling decision

At the four-GPU request, the primary seed 372/733 lanes had completed 25–28
of 70 epochs. Their recent single-GPU epoch time was approximately 179.5 s and
their projected remaining wall time was about 2.1 h.

A preregistered two-rank DDP benchmark on GPUs 2–3 completed one full
240-fit/60-calibration epoch in 144.2 s, a 1.245x speedup. Restarting both
primary seeds from epoch zero in a 2x2 layout would require about 2.8 h and
would finish approximately 42 minutes later than preserving the progressed
run. The primary run was therefore not discarded.

## Four-card layout

| GPU | Role | Seed | PID | Preregistration role |
|---|---|---:|---:|---|
| 0 | primary | 372 | 126592 | calibration gate |
| 1 | primary | 733 | 126593 | calibration gate |
| 2 | auxiliary replication | 1049 | 146105 | variance report only |
| 3 | auxiliary replication | 1403 | 146106 | variance report only |

All four processes use the same width-64 IC8 model, 240-record fit cache,
60-record group-disjoint calibration cache, 70 epochs, and single-GPU global
batch semantics. GPU memory is approximately 13.3 GB per card.

Auxiliary preregistration:
`results/b2_v4_aux_seed_preregistration_20260831.json`, SHA256
`78e970c0de577e0ff640876e1f76633110c0dfb4dd9d4d35be71f9b0aa3e068b`.
Auxiliary seeds cannot change the primary gate or authorize confirmation.

## Verified launch state

- Primary lanes: epoch 30/70; projected remaining time about 2.0 h.
- Auxiliary seed 1049 epoch 1: 179.6 s, checkpoint created.
- Auxiliary seed 1403 epoch 1: 177.2 s, checkpoint created.
- Projected time until all four lanes finish: approximately 3.4 h.
- Four GPUs: all active at approximately 95–100% utilization.
- Confirmation cache remains absent; validation and `test_id` remain sealed.
- Disk free: approximately 199 GB.
