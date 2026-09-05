# CPADC R7 frozen `test_id` confirmation (2026-08-10)

## Decision

The frozen R7 policy passed the complete same-protocol gate on the 480-record
`test_id` split. This is the first CPADC use of these records: the checkpoint,
family thresholds, code, config, sample census, travel cache, and gate were
registered before any `test_id` future wavefield was opened. No parameter was
changed after opening.

This confirms the R7 validation result under a group-disjoint CPADC holdout. It
does not establish that `test_id` was unused by every historical model-building
workflow outside CPADC.

## Frozen protocol

- R7 checkpoint SHA-256:
  `6da9854d4581924de5a74811fb25bc8630f6d34a2ea1e69cecdfbd04308aae2b`
- Preregistration SHA-256:
  `8d55119f034dcfd536696c5c3d5eb3fc880af256ed8527ed96824dcc08935267`
- Complete `test_id` census: 480 records in 180 groups; 90 uniform, 240
  layered, and 150 Marmousi records.
- Validation-group overlap: zero.
- Execution: four mutually exclusive GPU shards of 120 records each, followed
  by one merge and one complete-protocol gate.
- The online adaptation remained sealed from future truth. The derived travel
  cache was bound by both file and content digests and contained only the
  registered `test_id` census.

The fixed gate required at least 1% mean record improvement, at least 90%
nonworse records, every family mean nonworse, complete manifest coverage,
sealed future truth, and at most 5 seconds online adaptation per record.

## Confirmation result

| Metric | R7 validation | Frozen `test_id` | Change |
|---|---:|---:|---:|
| Mean record improvement | 2.7006% | 2.8420% | +0.1414 pp |
| Nonworse fraction | 97.2917% | 97.0833% | -0.2083 pp |
| Accepted fraction | 31.6667% | 32.7083% | +1.0417 pp |
| Maximum online adaptation time | 0.5621 s | 0.6720 s | +0.1099 s |
| Complete same-protocol gate | pass | pass | retained |

All gate checks passed. The `test_id` ratio-of-family-mean error improvements
were 1.6068% for uniform, 5.0238% for layered, and 0.2039% for Marmousi; all
three family means were nonworse.

The policy accepted 157/480 records. There were 14 strictly worse records, all
among accepted corrections, so the nonworse count was 466/480. The largest
single-record degradation was 5.5872%. The aggregate pass must therefore not be
interpreted as per-record monotonic improvement.

## Uncertainty analysis

With seed 20260810 and 100,000 bootstrap replicates:

- Ordinary record bootstrap: mean improvement 2.8420%, 95% interval
  [2.3054%, 3.4030%].
- Family-stratified group-cluster bootstrap, used as the primary interval to
  respect within-medium-group dependence: [2.2611%, 3.4573%].
- Frozen `test_id` minus validation mean improvement: +0.1414 percentage
  points, with group-cluster 95% interval [-0.6493, +0.9522] points.

The holdout effect is positive and its primary interval stays above the 1%
gate threshold. The validation-to-`test_id` difference is not statistically
distinguishable from zero, so there is no detected generalization drop under
this protocol.

## Reproducibility and terminal audit

- Each shard terminal is `complete`, contains exactly 120 records, selects
  `test_id`, preserves future-truth sealing, and reports the frozen checkpoint
  identity.
- Independent re-merge reproduced `instance_manifest.json`, `summary.json`,
  and `terminal.json` byte-for-byte. Their SHA-256 values are respectively
  `bde6c4d950f2cd32708712228aa322a0719261461f56ae4f1982b1d9d80bcf1e`,
  `86061025e92bc36b76f7703c4208b9572c1ebaa5644d84212e3c80064eaac517`,
  and `a79f5e59fcc3336c00d36d9e8a3f2235c5d8f2fe752a9235f87822c24b3f9bb5`.
- All 26 targeted tests for causal correction, sealed evaluation, data guards,
  and instance contracts passed; Python and shell syntax checks passed.
- The controller and all four child PIDs exited. All four GPUs returned to
  0 MiB and 0% utilization with no compute process.

## Research implication

R7 is now the confirmed CPADC policy for this dataset/protocol. The next study
should not tune another R7 variant against this opened `test_id` split. A useful
new direction is to diagnose the 14 harmful accepted corrections using only
causal, deployment-available features, then preregister any proposed safety
rule and evaluate it on a newly generated group-disjoint holdout.
