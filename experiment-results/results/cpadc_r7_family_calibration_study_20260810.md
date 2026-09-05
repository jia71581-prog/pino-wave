# CPADC R7 family-calibrated abstention study (2026-08-10)

## Question and predeclared rule

R6 passed the complete same-protocol gate, but its single global causal-strength
floor was conservative for the layered family. R7 tests one narrow change: keep
the R5 basis, parent, online solve, and deployment gate fixed, but select one
strength floor per medium family using only the 192-record disjoint train
calibration set.

Within each family, R7 selects the floor with the largest train calibration
mean improvement subject to at least 95% nonworse records and nonnegative
family mean improvement. The combined policy must still have at least 90%
nonworse records and 1% mean improvement. No validation target is read by the
online solve or by threshold calibration.

## Immutable inputs

- Parent A3 epoch-1 checkpoint SHA-256:
  `395b83560db3342b1e2331de2d8dde2994640d56a53dfbec8c99ece4bbb177bf`
- R5 CPADC basis checkpoint SHA-256:
  `4c01dea0dafd876106a8798e4962fba6cf25d3635a61c75429b5e7c32d1d09bc`
- R7 calibrated checkpoint SHA-256:
  `6da9854d4581924de5a74811fb25bc8630f6d34a2ea1e69cecdfbd04308aae2b`
- Calibration: 64 disjoint train records per family, 192 total, seed 10372.
- Validation: the complete 480-record manifest, four mutually exclusive
  120-record GPU shards.

Re-running calibration in an independent temporary directory reproduced the
R7 checkpoint byte-for-byte with the same SHA-256.

## Train-only calibration result

| Family | Strength floor | Accepted | Nonworse | Mean record improvement |
|---|---:|---:|---:|---:|
| uniform | 1.995308 | 14.0625% | 95.3125% | 1.0718% |
| layered | 1.674549 | 42.1875% | 95.3125% | 4.2953% |
| marmousi | 2.167699 | 25.0000% | 95.3125% | 0.1976% |
| all | family-specific | 27.0833% | 95.3125% | 1.8549% |

## Complete same-protocol validation

| Metric | R6 global floor | R7 family floors | Change |
|---|---:|---:|---:|
| Mean record improvement | 1.7760% | 2.7006% | +0.9246 pp |
| Nonworse fraction | 96.8750% | 97.2917% | +0.4167 pp |
| Accepted fraction | 27.5000% | 31.6667% | +4.1667 pp |
| Maximum online adaptation time | 0.5875 s | 0.5621 s | -0.0254 s |
| Same-protocol gate | pass | pass | retained |

R7 family mean error improvements were 2.1484% for uniform, 4.5413% for
layered, and 0.2043% for Marmousi. All three family means were nonworse. Every
gate check passed: exact full coverage, future truth sealed during adaptation,
mean improvement, nonworse fraction, runtime, and family safety.

A fixed-seed 20,000-draw record bootstrap estimated the R7 mean improvement as
2.7006% with a 95% interval of [2.1828%, 3.2366%]. The paired R7-minus-R6 gain
was 0.9246 percentage points with a 95% interval of [0.5768, 1.3006] points.

## Verification

- All four shard terminals completed with 120 records and the same R7
  checkpoint identity.
- Independent re-merge exactly reproduced the stored basis identity, gate,
  claim, and terminal status.
- 25 targeted tests covering the causal correction, train-only calibration,
  data guard, sealed evaluation, and instance contracts passed.
- Controller PID 820181 and all validation children exited; all four GPUs were
  at 0 MiB with no compute process after completion.

## Interpretation and limitation

The family-specific rule recovers useful layered corrections without weakening
the aggregate safety result. Marmousi remains the limiting family and its mean
gain is smaller than R6, although it stays positive and passes family safety.

The 480-record validation set was used in earlier R5/R6 studies, so it is not a
pristine one-shot researcher-level holdout. R7 threshold fitting itself is
strictly train-only and the implementation-level online seal is intact, but the
next scientific confirmation should freeze R7 and use a newly generated or
otherwise untouched holdout rather than tune R8 against these 480 records.
