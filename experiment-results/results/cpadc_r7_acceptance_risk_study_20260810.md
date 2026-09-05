# CPADC R7 accepted-correction risk study (2026-08-10)

## Decision

R7 remains the confirmed policy. This study identifies a plausible secondary
safety guard, but the guard is **candidate-only and must not be deployed**: its
feature hypothesis was formed after the frozen `test_id` outcomes had been
inspected. Existing validation and `test_id` results below are retrospective,
not a new confirmation.

The candidate accepts an R7 correction only when both conditions hold:

- coefficient vector L2 norm is at most `0.10821127891540527`;
- learned trust fraction is at least `0.9140290021896362`.

Both quantities are available after the causal online solve and before future
truth is opened. The candidate therefore introduces no truth leakage at
deployment time.

## Diagnostic evidence

The 14 harmful `test_id` acceptances occurred in 13 medium groups, so the
failure was not explained by one duplicated velocity group. Of the inspected
deployment-visible features, two showed the same risk direction on the
disjoint train calibration set and on `test_id`:

| Feature and risk direction | Train harm AUC | `test_id` harm AUC |
|---|---:|---:|
| Larger coefficient L2 norm | 0.7700 | 0.6563 |
| Smaller trust/correction ratio | 0.6098 | 0.8067 |
| Larger unconstrained strength | 0.5866 | 0.5789 |

The causal-objective decrease and condition number were not consistent enough
across the two sets to justify inclusion. In particular, simply raising the
existing family strength floors is not supported by this diagnostic.

## Train-only guard selection

Thresholds were selected using only the same 192 disjoint train calibration
records used by R7. Candidate rules had to retain at least 1% mean record
improvement, at least 95% nonworse records, and a nonnegative mean improvement
in every family. Among feasible rules, selection minimized strict harms, then
maximized mean improvement and acceptance.

| Train evaluation | Accepted | Strict harms | Nonworse | Mean improvement |
|---|---:|---:|---:|---:|
| R7 baseline | 52/192 | 9 | 95.3125% | 1.8549% |
| Candidate, fitted on all train calibration | 29/192 | 0 | 100.0000% | 1.5262% |
| Candidate, 5-fold group-disjoint out-of-fold | 29/192 | 3 | 98.4375% | 1.4095% |

Across the five cross-fit training folds, the selected coefficient cap stayed
between 0.1017682 and 0.1082113. A trust floor was selected in four folds and
lay between 0.9121649 and 0.9142466; one fold needed only the coefficient cap.
This makes the coefficient norm the primary signal and the trust floor a
secondary refinement.

## Retrospective checks only

| Split and policy | Accepted | Strict harms | Nonworse | Mean improvement |
|---|---:|---:|---:|---:|
| Validation R7 | 152/480 | 13 | 97.2917% | 2.7006% |
| Validation candidate | 102/480 | 7 | 98.5417% | 2.2247% |
| `test_id` R7 | 157/480 | 14 | 97.0833% | 2.8420% |
| `test_id` candidate | 108/480 | 5 | 98.9583% | 2.4350% |

The candidate consistently removes harmful corrections but also abstains from
some beneficial corrections. It retains 82.37% of R7's validation mean gain
and 85.68% of its `test_id` mean gain. Every family remains nonworse in these
retrospective results. These numbers motivate a new experiment; they do not
authorize an R8 claim.

## Integrity and reproducibility

- The diagnostic verified the stored SHA-256 of every coefficient tensor it
  loaded and verified that each online solve reported no future-truth access.
- The R7 checkpoint remained unchanged at
  `6da9854d4581924de5a74811fb25bc8630f6d34a2ea1e69cecdfbd04308aae2b`.
- The machine-readable diagnostic SHA-256 is
  `fe924ccd538f231cad35c789b0a711609f46a113ac0f893d8ef7b2dbf5e3c338`.
- The diagnostic explicitly records `claim_authorized=false` and
  `candidate_only_do_not_deploy`.

## Required next confirmation

Before future truth is generated or opened, create and hash a new 480-record
holdout manifest with the same family and group census as the current protocol:
90 independent uniform groups, 60 layered groups with four sources each, and
30 Marmousi crop groups with five sources each. Every group ID, sample ID, and
sample hash must be disjoint from the existing dataset and all CPADC artifacts.

Freeze the candidate thresholds, R7 checkpoint, generator, evaluator, travel
cache construction, exact gate, and one-shot opening policy against that
manifest. The candidate may be promoted only if it passes the absolute R7 gate,
retains at least 75% of frozen R7's paired mean improvement, and reduces strict
harm count by at least 30% on the new holdout.
