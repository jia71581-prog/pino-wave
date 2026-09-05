# Reduced CHONKNORIS integration for stored-time acoustic operators

## Scope

This integration adapts Bacho et al., *Operator Learning at Machine Precision*
(arXiv:2511.19980) to the existing V5 two-stage workflow without forming a
dense wavefield Hessian.

The paper learns Cholesky factors of the Tikhonov-regularized Gauss-Newton
operator and uses residual-monotone Newton--Kantorovich iterations.  A dense
factor over a 201 x 201 x time wavefield is infeasible here.  The V5 deployment
adapter has only 33 state variables:

```text
32 latent-delta coordinates + 1 residual gate.
```

We therefore apply CHONKNORIS in that reduced state space.  The parent wave
operator remains frozen and all existing A3, B2-H, and Helmholtz checkpoints
remain untouched.

## Offline stage B

`scripts/train_meta_hypernet.py` can now supervise a
`ReducedCholeskyPredictor` using exact 33 x 33 targets along train-only flow
episodes.  For a residual vector `r(s)` and state Jacobian `J`, the target is

```text
L L^T = J^T J / residual_dimension + lambda I.
```

The predicted matrix is lower triangular with a strictly positive diagonal.
At zero network output it is `sqrt(lambda) I`, the stable Tikhonov/gradient
descent limit.  The exact factor target is detached, avoiding second-order
backpropagation through the wavefield model.

CPU real-data smoke command:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python scripts/train_meta_hypernet.py \
  --config configs/saved_time_v5/meta_hypernet_chonknoris.yaml \
  --output /tmp/meta_hypernet_chonknoris_smoke.pt \
  --device cpu --per-family 1 --epochs 1 --time-points 4 \
  --learning-rate 2e-4 --chonknoris-weight 0.01 \
  --chonknoris-relaxation 0.01 --chonknoris-pool-size 2 --smoke
```

Recommended GPU pilot after the current four-GPU Helmholtz run terminates:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_meta_hypernet.py \
  --config configs/saved_time_v5/meta_hypernet_chonknoris.yaml \
  --output /root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/meta_hypernet_chonknoris/meta_hypernet_chonknoris.pt \
  --device cuda --per-family 4 --epochs 1 --time-points 24 \
  --learning-rate 2e-4 --chonknoris-weight 0.01 \
  --chonknoris-relaxation 0.001 \
  --chonknoris-relaxation 0.01 \
  --chonknoris-relaxation 0.1 \
  --chonknoris-pool-size 4
```

For the balanced four-GPU run, `train_meta_hypernet.py` supports synchronous
`torchrun` execution.  The global episode set must be divisible by the world
size; each rank caches a disjoint shard, gradients are averaged before every
optimizer step, and only rank 0 writes the gathered-history checkpoint:

```bash
scripts/run_chonknoris_meta_balanced_full_ddp4.sh
```

The current full configuration uses 64 train episodes per family, 5 epochs,
48 saved-time targets, and four ranks.  This is 240 synchronized optimizer
steps and 960 effective sample updates.

Because onset-dependent saved-time indices vary across episodes and each exact
33 x 33 Jacobian is sample-specific, `--per-rank-batch-size` uses sequential
gradient accumulation rather than unsafe tensor padding.  A value of 4 gives
an effective global batch of 16 and reduces synchronized optimizer/all-reduce
steps from 240 to 60 while retaining all 960 sample updates.

`--jacobian-vmap-size 2` additionally performs true tensor batching for two
shape-compatible episodes at a time.  It evaluates independent per-instance
Jacobians with `vmap(jacfwd)` and regresses the corresponding batched exact
Cholesky targets.  Mixed 49/50-frame leftovers fall back to compatible
subgroups without padding or dropping an episode.

## Online stage C

With `adaptation_optimizer: chonknoris`, instance adaptation:

1. reads exactly the two guarded onset frames;
2. builds one fixed residual from pooled onset mismatch and sampled,
   self-normalized LWC-84 residual;
3. predicts a positive-definite reduced Cholesky factor;
4. tries the paper's joint 3 x 3 relaxation/step-size search;
5. accepts only a strictly contracting residual step;
6. falls back to an exact 33 x 33 Cholesky if the learned factor cannot contract;
7. applies the existing observed-loss, energy, no-future-data, and rollback gates.

The adaptation artifact records residual, relaxation, step-size, contraction,
and condition-number histories plus learned-factor and exact-fallback counts.

Dry run:

```bash
python scripts/run_meta_instance_adaptation.py \
  --config configs/saved_time_v5/meta_hypernet_chonknoris.yaml \
  --output-dir /tmp/chonknoris_dry_run --device cpu --pilot --dry-run
```

Pilot evaluation after the stage-B checkpoint exists:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_meta_instance_adaptation.py \
  --config configs/saved_time_v5/meta_hypernet_chonknoris.yaml \
  --output-dir /root/autodl-tmp/home/jiayh/Data/FNO-Acoustic-Wave-Simulation/artifacts/meta_hypernet_chonknoris/pilot3 \
  --device cuda --pilot
```

## Claim gate

Do not claim improved accuracy or machine precision from unit/smoke tests.  A
candidate passes only if a same-protocol paired evaluation against the existing
`meta_hypernet.yaml` baseline uses the same parent checkpoint, normalization,
manifest, three-family sample IDs, two-frame access boundary, and evaluator,
and shows:

- lower aggregate future full-field relative L2;
- no material regression in Uniform, Layered, or Marmousi family metrics;
- finite state and strictly sub-unit accepted contraction ratios;
- zero future-truth access before state sealing;
- a reproducible stage-B checkpoint and recorded parent identities.

The paper's machine-precision results use double precision and problem-specific
Newton solvers.  This repository still predicts float32 wavefields, so the
appropriate claim is improved same-protocol accuracy, not machine precision,
unless a separate float64 validation gate demonstrates it.

## 2026-08-07 legacy V41 branch result

The result in this section is bound to the older `saved_time_v41` epoch-1
pilot parent.  It is a valid branch-local comparison but is not a promotion of
the latest Phase4b model and must not be quoted as the project's absolute
accuracy.

The four-GPU stage-B run used 5 epochs, 64 episodes per family (192 total), a
logical global batch of 16, and a physical Jacobian-vmap batch of 2.  It
completed 60 synchronized optimizer steps in 97 seconds.  The promoted
checkpoint SHA-256 is
`99f6e8fb91b7af90c7c0bb2023e9ba7a10d6552d72dcbbcc3954d28d0bcf528b`.

An initial three-sample development gate exposed a Layered regression from
accepting negligible contractions.  Raising the causal
`minimum_relative_improvement` gate to 0.02 rolls those steps back.  The final
gate therefore used a fresh, deterministic nine-sample panel (three per
family), excluding all development samples.  Against the same-protocol
Adam+LBFGS baseline, mean future full-field relative L2 changed from
0.5882978075 to 0.5875242419 (0.1315% relative improvement).  Family means
improved by 0.1797% (Uniform), 0.1354% (Layered), and 0.0782% (Marmousi).
All nine artifacts were sealed, no future truth was used during adaptation,
and the largest accepted contraction ratio was 0.9777648574.  This passes the
gate above and supports only the stated same-protocol float32 accuracy claim.

## Phase4b migration and gate

The Phase4b loader reconstructs the Helmholtz A+1 architecture from its run
identity, strictly loads all 320 checkpoint tensors, and adds the cached
smoothed-velocity background field before adaptation.  Replaying the original
three-record, 32-frame protocol reproduced `0.04069607154747407` exactly.

Stage-B then trained on four GPUs for 5 epochs with 64 episodes per family,
per-rank logical batch 8 (global batch 32), and Jacobian vmap 2.  Host-pinned
parent-field caching and CPU/Gloo history collection keep the Jacobian phase at
about 23--24 GiB per GPU while avoiding an end-of-run NCCL allocation.  The 30
synchronized updates completed in 92 seconds; checkpoint SHA-256 is
`f979f8290742aa5f1ffd3fd42ebac910e0c1686522744508f97b417b0e553e0b`.

The same three held-out instances all triggered the causal rollback gate, so
the sealed candidate fields are bitwise equal to the Phase4b parent.  Fixed-32
aggregate relative L2 remains `0.040696071549719784`; all-401 aggregate relative
L2 is `0.04165605429388125`.  This is safely below 10% and has no regression,
but it has no strict accuracy improvement, so the Phase4b-CHONKNORIS checkpoint
is retained as an unpromoted research artifact.

## References

- Paper: https://arxiv.org/abs/2511.19980
- Reference implementation: https://github.com/ArasBacho/CHONKNORIS
