# PI-DeepONet train-only development comparison

This directory supports the manuscript's matched comparison between the frozen
r5b parent plus residual-conditioned instance fine-tuning and the completed
full-data PI-DeepONet baseline.

The quantitative source is
`results/r5b_feature_meta_vs_pi_train_all401_r34_result_20260816.json` (SHA-256
`22f9a22cd105995415ec66dcbb15a86bf3dc9c12108f1f1790a67c4667fc2a8b`).
It contains six preregistered train-split records, two per family. Both parent
models had offline training exposure to these records, so the comparison is
development evidence rather than held-out validation or test evidence.

`report.json` binds the r34 result, r33 instance-adaptation summary, r5b
identity, adapter checkpoint, PI-DeepONet checkpoint, configs, manifest, figure,
and compact source-data archive. The rendering script regenerated every stored
time and verified each complete-future relative error against r34 before saving
only the common t=0.60 s snapshots. The accepted Layered update was not bitwise
identical across GPU replays, but its complete-future relative L2 differed from
the sealed r34 value by only `1.01e-8`. Uniform and Marmousi used the exact-parent
fallback and reproduced their sealed tensor hashes.

The mean record-relative L2 values are 0.3014208502 for ours and 0.6156354159
for PI-DeepONet. The r5b parent is 0.3014222828. Thus the train-panel difference
from PI-DeepONet is clear, but the instance update itself is nearly neutral and
failed its preregistered promotion gate. Validation and `test_id` remained
sealed.
