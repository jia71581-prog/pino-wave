# TGRS high-impact optimization audit

Date: 2026-08-12

Scope: `paper/tgrs_helmholtz_operator/manuscript.tex` and the active
`grouped-dual-head-v2` experimental line.

## Executive decision

The paper should be rebuilt around one transferable scientific question:

> Can a neural operator represent an entire transient wavefield with one
> query-invariant temporal-frequency state, and can that state be corrected
> from deployment-available observations without opening future truth?

The generic claim "physical background plus learned scattered field" is no
longer sufficiently distinctive by itself. Huang and Alkhalifah (GJI 2025)
already embed source/frequency information in a homogeneous background and
learn a scattered frequency-domain field; Ma and Alkhalifah (JGR 2026) add a
physics-informed CNO for the same broad decomposition. The defensible novelty
is instead the combination of:

1. one query-invariant bank that synthesizes the complete transient time axis;
2. explicit phase/arrival carriers and a measurable representation ceiling;
3. a sealed two-observation adaptation contract with abstention;
4. complete, group-disjoint, exact-time accuracy and end-to-end runtime gates.

This positioning is broader and more reusable than another FNO variant. It can
be cited by work on transient operator representations, hybrid solvers,
test-time adaptation, trustworthy scientific ML, and seismic surrogates.

## Current evidence status

### Claims that are currently supported

- The temporal-frequency synthesis is query-invariant by construction: its
  predicted state is independent of the requested output time and the time axis
  is rendered by a fixed inverse transform.
- A zero-training representation oracle shows that 64 temporal frequencies and
  the registered spatial bandwidth can reconstruct the three development
  examples below 5% relative L2.
- The three-record G3 experiment is useful as a mechanistic pilot. It is not a
  population-level generalization result.
- CPADC R7 is reproducible and leakage-safe under its own protocol. On 480
  validation and 480 independent `test_id` records it improves the old parent
  by 2.7006% and 2.8420%, respectively, with 97.2917% and 97.0833% nonworse
  records. Its absolute future-field errors are roughly 4.8--6.2 by family, so
  it is a safety/adaptation result, not evidence of an accurate solver.
- The train-only DRP candidate improves mean Marmousi error by 12.689% and the
  high-frequency subset by 23.982%, but wins only 13/24 records (54.167%). It
  fails the preregistered 62.5% paired-win gate and must remain rejected.

### Claims that are not currently authorized

- Relative L2 at or below 0.05 overall and in every family on complete frozen
  validation and independent `test_id`.
- Mean and P95 end-to-end speedup at or above 10x after including the physical
  background, parent inference, adaptation, synchronization, and output
  materialization.
- A 20x accuracy advantage over FNO3D, U-NO, factorized FNO, or DeepONet. The
  existing comparison has only three development records and all baselines
  collapse to the zero predictor; it is a failed-baseline diagnostic, not a
  fair comparative result.
- Population-level claims from the current G3 triplet or bootstrap confidence
  intervals obtained from those three records.
- Superiority of the DRP stencil or any claim based on its mean improvement
  alone.

## Highest-risk reviewer objections

1. **Closest prior art is missing.** The original related-work section does not
   discuss the 2025 GJI scattered-wavefield operator, the 2024 GJI deep
   Helmholtz neural operator, the 2026 PICNO paper, or WaveBench. Without these,
   the physical-background contribution appears rediscovered.
2. **The main comparison is not publication-grade.** Three held-out records do
   not establish generalization. Baselines at relative L2 approximately one
   require overfit sanity checks, tuned learning curves, parameter/compute
   matching, and independent implementations or official checkpoints.
3. **The title says fast, but the draft explicitly makes no speed claim.** A
   same-device, synchronized, end-to-end benchmark is mandatory. The smoothed
   background solve is part of inference and cannot be excluded.
4. **The manuscript mixes incompatible numbers.** The G3 table reports
   aggregate 0.076 while the baseline table reports 0.053; their record sets
   and checkpoints are not identified in the paper. Every table row needs a
   manifest, checkpoint hash, code/config digest, record count, and metric
   definition.
5. **The statistical unit is unclear.** Multiple sources from one velocity
   model are dependent. Confidence intervals must resample independent medium
   groups and preserve family strata.
6. **The physical-background novelty is overstated.** The paper must state
   precisely that the distinct component is complete transient synthesis and
   its deployment contract, not scattered-field learning itself.
7. **No external benchmark or downstream task is present.** A WaveBench transfer
   experiment or an FWI/RTM downstream study would materially improve reach and
   citation potential.

## Falsifiable primary hypothesis

**Hypothesis.** A query-invariant temporal-frequency parent, followed by a
fixed-step causal correction using only two onset observations, reaches
relative L2 no greater than 0.05 overall and for uniform, layered, and Marmousi
families on complete group-disjoint validation and `test_id`, while preserving
mean and nearest-rank P95 end-to-end speedup of at least 10x over the fastest
synchronized same-GPU LWC84 reference.

**Mechanism.** The analytic carrier shares phase across every output time; the
parent predicts medium-dependent amplitudes; the causal low-dimensional
correction removes instance-specific phase/amplitude bias; abstention prevents
unsafe corrections.

**Primary metric.** Adapted future-field relative L2 on all stored times after
the two observations, aggregated by record and reported overall and by family.

**Failure signals.** Any family exceeds 0.05, mean or P95 speedup falls below
10x, future truth is accessed before serialization, manifest coverage is
incomplete, or code/config/checkpoint identities differ from preregistration.

**Compute budget.** Complete the already-running A3-R4 Muon parent and its
supervised detached instance-finetuning pipeline. Do not launch a competing GPU
run while those four GPUs are occupied. Use train-only diagnostics before one
frozen validation opening; open `test_id` once only after the method is frozen.

**Parent identity.** The candidate parent is the accepted checkpoint produced
by `temporal_latent_a3_rank32_warm_marmousi1_4m_v2_r4_muon`; its SHA-256 must be
bound after training completes. The current best is epoch 6 with a 48-record
fixed gate aggregate 0.358712, so the target has not been reached.

**Rollback.** Preserve all A3, B2-H, Helmholtz, ASAM, and CPADC checkpoints.
If the frozen validation gate fails, report the family/time/spectrum failure and
return to train-only evidence. Do not retune on `test_id`.

## Required experiment matrix

### P0: required before submission

| Question | Minimum evidence | Promotion condition |
|---|---|---|
| Does the full method meet accuracy? | Complete 480-record frozen validation and independent 480-record `test_id`; exact 401 stored times | Overall and every family at most 0.05 on both splits |
| Is it actually fast? | Same GPU, synchronized warm/cold runs, at least 30 repetitions; include all online components | Mean and nearest-rank P95 speedup at least 10x |
| Are baselines fair? | FNO3D, U-NO, factorized FNO, DeepONet or GNO, plus nearest frequency-domain background/scattering method; matched data and metric | Every baseline first passes a small-set overfit sanity gate; disclose parameters, FLOPs, memory, and tuning budget |
| Does query invariance matter? | Replace fixed synthesis by a parameter-matched time-query decoder | Paired group-cluster interval supports lower late-time and phase error |
| Does each physical component matter? | Background only, learned residual only, carrier only, full model | One conceptual change per ablation; full method improves every family or abstains safely |
| Is evaluation leakage-safe? | Mutation test of all future truth plus access log | Coefficients, gates, and serialized adapted output are byte-identical |

### P1: makes the paper substantially more citable

- Add WaveBench time-varying and time-harmonic transfer results using the public
  splits and official FNO/U-Net checkpoints.
- Add four OOD axes separately: frequency, velocity topology, source geometry,
  and spatial/domain resolution. Avoid one mixed OOD score that hides failure.
- Add at least one downstream geophysical task: FWI gradient agreement,
  inversion convergence under a fixed budget, or RTM image similarity.
- Release the data generator, immutable manifests, exact-time evaluator,
  checkpoint, environment lock, and one-command reproduction of every main
  table. IEEE explicitly encourages shared data and executable code.
- Report learning curves against number of independent velocity groups, not
  merely number of source records.
- Include failure cases and abstention-risk curves. Scientific users care about
  worst-family and harmful-correction behavior, not only mean error.

## Recommended paper structure

1. Introduction: the missing abstraction is a trustworthy complete-transient
   operator, not simply a faster per-frequency predictor.
2. Related work: separate frequency-domain Helmholtz surrogates, transient
   operators, hybrid/scattered-field learning, and deployment-time adaptation.
3. Method: query-invariant state, carrier, physical background, causal
   correction, abstention, and complexity.
4. Evaluation contract: split construction, truth seal, identities, metrics,
   runtime boundary, and statistical unit.
5. Results: complete accuracy/runtime first; fair baselines second; ablations
   and diagnostics third; downstream task fourth.
6. Limitations: 2-D acoustic setting, finite bandwidth, observation assumption,
   background cost, and failure families.
7. Reproducibility statement: public artifacts and exact commands.

## Submission rule

Do not submit the current numerical claims as final. A submission-ready abstract
must be generated from the frozen validation and `test_id` terminal artifacts,
not copied from pilot tables. If the absolute 5%/10x gate fails, the paper can
still be publishable as a representation-and-reliability study, but the title,
abstract, and conclusion must drop the fast/high-accuracy solver claim and the
evaluation must demonstrate a broader benchmark or downstream value.

## Literature verified on 2026-08-12

- Huang and Alkhalifah, "Learned frequency-domain scattered wavefield
  solutions using neural operators," *Geophysical Journal International*,
  2025, DOI: 10.1093/gji/ggaf113.
- Zou et al., "Deep Neural Helmholtz Operators for 3-D Elastic Wave Propagation
  and Inversion," *Geophysical Journal International*, 2024,
  DOI: 10.1093/gji/ggae342.
- Ma and Alkhalifah, "An Effective Physics-Informed Neural Operator Framework
  for Predicting Wavefields," *JGR: Machine Learning and Computation*, 2026,
  DOI: 10.1029/2025JH000899.
- Liu et al., "WaveBench: Benchmarking Data-driven Solvers for Linear Wave
  Propagation PDEs," *Transactions on Machine Learning Research*, 2024.

