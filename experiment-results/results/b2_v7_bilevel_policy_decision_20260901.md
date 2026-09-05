# B2-v7 conservative bilevel policy decision

## User-specified pilot gate

No fixed percentage improvement was required. The only accuracy gate was:

`adapted aggregate relative L2 < frozen parent aggregate relative L2`.

Family, nonworse, worst harm, and runtime were report-only diagnostics.

## Offline policy calibration

The offline train-future grid selected:

- E2 prior scale: 0.1;
- FE weak weight: 0.01;
- inner learning rate: 0.03;
- inner steps: 4;
- output scale: 1.0.

On the offline panel this policy improved aggregate relative L2 from 0.12642
to 0.11749.

## Fresh train-group holdout

| Metric | Parent | Adapted |
|---|---:|---:|
| Aggregate | 0.173259 | 0.176559 |
| Uniform | 0.064166 | 0.067853 |
| Layered | 0.229669 | 0.230847 |
| Marmousi | 0.225941 | 0.230976 |

- Relative gain: -1.905%.
- Nonworse: 0/12.
- Strict accuracy gate: FAIL.
- Validation and `test_id`: unopened.

## Decision

Reject `b2_v7_conservative_bilevel_policy_v1`. Offline scalar policy
calibration overfits the fit records and does not align the eight-frame online
objective with future accuracy on new medium groups. Per the user-specified
gate, no further minimum-improvement threshold was involved.

Do not tune damping, FE weight, or step size on this holdout. A new direction
would require changing the information available online (more observations or
receiver traces) or learning a genuinely instance-local correction operator,
not further calibration of the same global family POD modes.
