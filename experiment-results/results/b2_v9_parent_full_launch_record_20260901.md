# B2-v9 parent full retraining launch

- Preregistration SHA256: `e72f668fb927f1829c40361a9f80a1c2aa7d63bf40c719ef620a0b2f56811808`.
- Four independent GPU lanes:
  - GPU0 physical_cond seed372, PID 229659;
  - GPU1 physical_cond seed733, PID 229660;
  - GPU2 spectral seed372, PID 229661;
  - GPU3 spectral seed733, PID 229662.
- Fit: 240 causal train records; holdout: 24 fresh group-disjoint train records.
- Hyperparameters: width64, rank32, modes24, depth4, IC8, AdamW lr1e-3,
  microbatch4, 90 epochs, 60 updates/epoch, 5,400 updates total.
- Resource state after epoch 1: approximately 17.3 GiB and 100% utilization
  per GPU; no OOM; 177--180 s/epoch.
- Each lane wrote run identity, metrics, best metadata, and best checkpoint.
- Projected wall time: approximately 4.5 hours.
- Accuracy gate: both seeds of a variant must strictly beat the frozen
  same-cache warp-anchor aggregate; no minimum improvement percentage.
- Validation and test_id remain sealed.
