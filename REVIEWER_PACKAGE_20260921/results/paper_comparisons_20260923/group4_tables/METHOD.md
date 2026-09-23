# Group 4: Runtime table and DeepONet comparison — method

## Runtime table (TABLE_runtime.md)
Assembled from the four frozen artifacts in `../../runtime_20260922/`
(no new timing runs; this group only copies numbers and computes ratios):
- `traditional_lwc84_runtime_29359paper.json` — generation-protocol solve
  (401x401, dx=5 m, dt=1.25e-4 s, restricted to 201x201): mean 22.06 s.
- `classical_lwc84_runtime_accuracy.json` — fine (20.07 s) and native
  201x201 (10.00 s) solves on the paper's figure records, with accuracy.
- `operator_29359_runtime.json` — anchor-29359 standard decode: 3.97 s.
- `operator_29359_runtime_tb16.json` — batched decode (time_block=16): 2.54 s.

All rows: RTX 4090 D, CUDA-synchronised, single instance, output
materialised, disk I/O excluded. Ratios are computed only between
record-matched rows (figure records): 5.06x (tb=1) / 7.90x (tb=16) vs the
fine protocol, 2.52x / 3.93x vs the cheapest classical config. **No
end-to-end 10x claim**: training cost is amortised and excluded, and the
protocol footnotes are part of the table.

Accuracy context on the same records (from the classical artifact and the
29359 panel rows): the native 201x201 classical solve reaches future relL2
0.018 (uniform) / 0.016 (layered) / 0.120 (marmousi) vs anchor 29359's
0.051 / 0.213 / 0.401 — the cheapest classical config is both 2.5x slower
than the operator and (except on marmousi late coda) more accurate; the
honest speed story is the 5-8x vs the generation protocol with the accuracy
gap stated.

## DeepONet table (TABLE_deeponet.md)
Sources (all under `/root/autodl-tmp/staging/deeponet_baseline_20260914/` and
the prereg/synthesis documents listed in NUMBERS.json):
- Preregistered 40M-parameter DeepONet baseline (39,975,985 params vs our
  40,484,837; budget 40,484,837 +/- 2%).
- Main run: planned 22,814 steps, SIGTERM-terminated at step 4,862
  (epoch 26). Fixed 12-record validation batch (its checkpoint-selection
  caliber): aggregate dense relL2 1.000012 (best, step 4488) and 1.000014
  (latest) — indistinguishable from the trivial zero predictor on all three
  families.
- Family-level collapse probe (12 train records, 2000 steps): control arm
  point relL2 1.0015-1.1168, never < 1.0; anti-collapse arm oscillates
  0.97-1.33 and is not separable from control (all three effect axes inside
  its own trailing-10 band). Single-record fits do converge (0.025 at 2000
  steps), so the recorded verdict attributes the failure to optimisation
  (H_optim), not capacity — stated verbatim in the table, no embellishment.

## Caliber mismatch (explicit)
No same-caliber comparison exists: DeepONet never reached the 480-record
census caliber (run stopped early). The table pairs the nearest calibers and
labels each row: DeepONet's fixed 12-record validation (~1.0000) vs our
480-record census mean (0.352; uniform 0.131 / layered 0.302 / marmousi
0.565, checkpoint 22814 from `validation_480_summary.json`). Both are dense
future relative L2 but on different record sets and selection protocols;
the caliber column and the mismatch note carry that caveat.

## Products
- `NUMBERS.json`, `TABLE_runtime.md`, `TABLE_deeponet.md`
- Script: `../scripts/make_group4_tables.py`
