# Runtime comparison (one record -> 401 stored frames, 201x201, RTX 4090 D)

| Method | Mean (s) | Min (s) | n | Records |
|---|---|---|---|---|
| LWC-84 generation protocol (401x401, dx=5 m, dt=1.25e-4 s, restricted to 201x201) | 22.06 | 20.83 | 12 | validation_{uniform,layered,marmousi}_00000 |
| LWC-84 fine (401x401, dx=5 m) on the paper's figure records | 20.07 | 19.88 | 6 | train_uniform_00413, train_layered_00299, train_marmousi_00010 |
| LWC-84 native (201x201, dx=10 m, dt=2.5e-4 s, no restriction) | 10.00 | 9.93 | 6 | train_uniform_00413, train_layered_00299, train_marmousi_00010 |
| Neural operator, anchor 29359, standard decode (time_block=1) | 3.97 | 3.96 | 12 | train_uniform_00413, train_layered_00299, train_marmousi_00010 |
| Neural operator, anchor 29359, batched decode (time_block=16) | 2.54 | 2.54 | 12 | train_uniform_00413, train_layered_00299, train_marmousi_00010 |

Record-matched speed ratios (paper figure records only):

| Comparison | Ratio |
|---|---|
| operator (tb=1) vs fine 401x401 | 5.06x |
| operator (tb=1) vs native 201x201 | 2.52x |
| operator (tb=16) vs fine 401x401 | 7.90x |
| operator (tb=16) vs native 201x201 | 3.93x |

Footnotes (protocol differences):
1. Generation-protocol row times validation_*_00000 records; all other rows time the paper's figure records (train_uniform_00413, train_layered_00299, train_marmousi_00010). Speed ratios reported here use only record-matched rows.
2. The operator amortises training (~GPU-days) and produces all 401 stored frames of ONE record per call; classical rows are one full solve per record. No end-to-end (training-inclusive) speedup is claimed.
3. Operator time excludes reading the 8 IC frames from disk (~0.02 s, reported separately in the source JSON); classical rows need no IC.
4. Same GPU (RTX 4090 D), same timing protocol, single instance, no batching across records in any row.
5. The native 201x201 classical solve is NOT the truth protocol: its accuracy vs stored truth is 0.018-0.120 future relL2 depending on family (classical_lwc84_runtime_accuracy.json); the fine solve is the truth protocol up to velocity-resampling error (1.9e-5 to 5.8e-2).
