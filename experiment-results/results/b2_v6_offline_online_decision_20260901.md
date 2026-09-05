# B2-v6 offline pretraining and online self-supervised adaptation decision

## Contract

- Parent: frozen B2-v4 seed-372 checkpoint.
- Offline fit: 24 causal train records, 8 per family.
- Online holdout: 12 causal, medium-group-disjoint train records, 4 per family.
- Offline truth use: residual POD and coefficient-prior fitting only.
- Online truth use: first eight frames only; four candidates were materialized
  before future train truth scoring.
- Validation and `test_id`: unopened.

## Offline result

Family-conditioned rank-4 residual POD passed the frozen 5% capacity gate:

| Family | Mean parent relL2 | Mean projected relL2 | Oracle gain |
|---|---:|---:|---:|
| Uniform | 0.07704 | 0.03601 | 49.76% |
| Layered | 0.12714 | 0.03734 | 49.31% |
| Marmousi | 0.17501 | 0.06367 | 49.82% |

E0 base-seven, E1 physical-15, and E2 B2-latent ridge priors were fitted to
the offline oracle coefficients.

## Online group-disjoint result

| Prior arm | Aggregate | Gain vs parent | Nonworse | Mean adaptation time |
|---|---:|---:|---:|---:|
| Parent | 0.13777 | n/a | 12/12 | n/a |
| Zero prior | 0.13817 | -0.29% | 1/12 | 0.234 s |
| E0 base7 | 0.14745 | -7.03% | 0/12 | 0.199 s |
| E1 physical15 | 0.14860 | -7.86% | 0/12 | 0.203 s |
| E2 B2 latent | 0.14631 | -6.20% | 0/12 | 0.205 s |

All online gates failed except runtime. E1 hit the 5% correction trust boundary
on every record; E0/E2 were also frequently trust-limited. The self-supervised
objective improved on nearly every accepted solve, but future error worsened.

## Decision

Reject `b2_v6_offline_pod_online_selfsupervised_v1`. Large in-panel POD oracle
capacity does not establish transferable modes, and direct ridge prediction of
oracle coefficients overfits eight offline records per family. Richer encoding
does not repair a basis/objective that is misaligned with future error.

Do not increase POD rank, trust radius, or prior capacity on this holdout. The
next defensible conceptual variable is first-order bilevel meta-training:

1. inner loop uses only observed frames and FE/LWC weak features;
2. outer train-only future loss updates the residual modes, encoder projection,
   objective weights, and abstention gate so that inner-objective improvement
   predicts future improvement;
3. parent weights remain frozen initially;
4. fresh group-disjoint train confirmation requires at least 1% mean gain,
   95% nonworse, every family nonworse, and P95 adaptation below 3 s.

This result also reconfirms the strict-eight-frame information/objective wall:
offline capacity is ample, but the online observations do not identify the
correct transferable coefficients without meta-training the adaptation path.
