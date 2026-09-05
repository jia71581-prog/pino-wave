# B2-v3 causal selector probe decision

The fresh train-only selector probe completed with no validation or `test_id`
access. For each record, the option was fixed using relative L2 on the eight
deployment-visible frames before future frames were scored.

## Frozen result

| Metric | Anchor | Prefix selector | Delta / gate |
|---|---:|---:|---:|
| Aggregate future relative L2 | 0.12956814 | 0.12816949 | -0.00139865; gate required -0.005 |
| Nonworse records | n/a | 26/30 | gate required 27/30 |
| Uniform family | 0.07404898 | 0.07195528 | pass |
| Layered family | 0.09655976 | 0.09701320 | fail |
| Marmousi family | 0.21809568 | 0.21554001 | pass |

The selector chose the anchor on 10 records, seed 372 on 19, and seed 733 on
one. It failed the aggregate, nonworse, and all-family gates, so its terminal
status is `rejected`.

## Mechanism diagnosis

Observed-prefix ranking is weakly informative but not reliable enough for a
deployment gate. The largest selector-induced regression is `+0.02487` on
`train_marmousi_00338`. Mean late-time and high temporal-frequency errors
remain `0.15078` and `0.43119`, respectively. Evaluating two correction models
also doubles their inference work, so the small gain would not justify the
runtime cost even without the frozen gate failures.

## Route decision

Reject the current combination of development-panel B2 checkpoints and
prefix-only abstention. Do not tune a threshold on this opened panel. Any
further B2 experiment must start from a train-only, medium-group-disjoint
fit/calibration protocol with the architecture and loss held fixed. This
directly tests whether broader independent group coverage can replace the
same-panel fitting behavior before introducing a new loss or architecture.
