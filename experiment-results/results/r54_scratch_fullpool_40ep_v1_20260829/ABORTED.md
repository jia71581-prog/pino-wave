# Aborted 2026-08-29 17:20 (+08)

Reason: throughput, not science. num_workers=0 in the r25 driver serialized
HDF5 decompression with GPU compute; measured 1.64 steps/s -> ~26 h projected
vs the 8.3 h budget. No per-epoch holdout row was ever produced; best.json /
best.pt reflect only the initial identity state plus ~1000 steps of epoch 1.
Superseded by v2 (same seed, same mathematics, worker-parallel loading). See
amendment_v2 in results/r54_scratch_fullpool_preregistration_20260829.json.
