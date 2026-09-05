# IEEE TGRS Manuscript Design: DCLP-NO

**Date:** 2026-07-24  
**Status:** Approved section by section in chat; written specification awaiting final user review  
**Editable project:** `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/project`  
**Read-only pristine snapshot:** `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/project_pristine`  
**Bound parent artifact:** `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long`

## 1. Objective and Submission Position

The objective is to turn the completed acoustic-wave research into a
submission-ready IEEE Transactions on Geoscience and Remote Sensing manuscript
centered on a local-propagation neural operator and a strictly leakage-free,
two-snapshot LoRA instance-adaptation protocol.

The working title is:

> **Dispersion-Controlled Local-Propagation Neural Operator With Two-Snapshot
> LoRA Adaptation for Acoustic Wavefield Modeling**

The working acronym is **DCLP-NO**.

The paper is acoustic only. It must not introduce, imply, or claim experiments
on elastic waves, shear waves, VTI elasticity, or multicomponent displacement.
The geoscience remote-sensing relevance is the acceleration of repeated
acoustic forward modeling for seismic wavefield prediction, near-surface
receiver synthesis, and downstream subsurface imaging or inversion workflows.
The manuscript may motivate inversion as a downstream application, but it must
not claim full-waveform-inversion results unless such experiments are actually
added and verified.

The primary scientific story is:

1. global low-rank and globally truncated spectral representations smear sharp
   translating wavefronts and introduce ringing or phase errors;
2. DCLP-NO supplies dense travel-time, retarded-time, causal, phase, and
   multi-scale envelope features to a spatially preserving local field
   generator;
3. a global MIONet term retains smooth long-range context, while a local
   U-Net residual restores sharp propagation structure and a factorized
   spectral corrector removes remaining structured error;
4. at deployment, exactly two complete wavefield snapshots are permitted for
   instance-specific LoRA adaptation;
5. all later wavefields and all near-surface receiver traces remain unseen and
   are used only for evaluation;
6. dispersion suppression is supported only by direct phase, wavefront, and
   receiver evidence with paired uncertainty estimates.

## 2. Evidence Already Available and Evidence Still Required

### 2.1 Bound pretrained parent

The only primary parent checkpoint is:

`pretraining/gate4_long/run/best.pt`

Registered identity:

- epoch: 40;
- global step: 2800;
- model format: `phase_aligned_complex_fno_mionet_v3`;
- checkpoint tensors: 364;
- checkpoint SHA-256:
  `0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f`;
- trainable parameters: approximately 32.294 million;
- training records: 2240;
- fixed full-time validation panel: 48 records by 401 saved times.

The current verified epoch-40 validation evidence is:

| Metric | Value |
|---|---:|
| Aggregate full-time relative L2 | 0.27130275 |
| Uniform relative L2 | 0.16321997 |
| Layered relative L2 | 0.29935535 |
| Marmousi relative L2 | 0.30013583 |
| Global-coarse aggregate relative L2 | 0.30605304 |
| Relative gain over global coarse | 11.3543% |
| Phase correlation | 0.76721071 |
| Early-time relative L2 | 0.13836 |
| Middle-time relative L2 | 0.23175 |
| Late-time relative L2 | 0.40426 |
| Low-band spectral relative L2 | 0.26849 |
| Mid-band spectral relative L2 | 0.53191 |
| High-band spectral relative L2 | 0.84054 |

These numbers establish that local propagation improves the global coarse field,
but they are not sufficient by themselves for a strong TGRS submission. In
particular, late-time and high-frequency errors remain large. The paper must
therefore add strict held-out testing, strong baselines, direct dispersion
measurements, two-snapshot LoRA evaluation, ablations, statistical uncertainty,
and efficiency results.

The earlier CLFC two-frame smoke result regressed by approximately 1.50%.
It must be reported as a failed preliminary route or omitted from the main
method history; it must never be presented as evidence for LoRA success.

### 2.2 Numerical reference

The physical reference is the existing two-dimensional acoustic solver:

- 401 by 401 physical grid at 5 m spacing;
- 201 by 201 stored grid at 10 m spacing after binomial-5 anti-alias
  restriction;
- 401 stored times from 0 to 1 s;
- fine solver time step 0.000125 s;
- stored time step 0.0025 s;
- LWC-84 spatial discretization and fourth-order time stepping in the physical
  domain;
- pressure-free top boundary;
- CFS-CPML on the left, right, and bottom boundaries.

The manuscript must distinguish three different questions:

1. numerical dispersion of coarse-grid finite-difference schemes;
2. approximation error of a neural operator relative to the fine-grid
   LWC-84 reference;
3. learned-model wavefront or phase error that resembles dispersive smearing.

The teacher solver is not itself claimed to be dispersion free. Its dispersion
is quantified, and it serves as the registered high-resolution numerical
reference.

## 3. Data Contract and Evaluation Splits

The active dataset is:

`/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5`

The registered travel-time data are:

`/home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5`

The main in-distribution acoustic split excludes the anomaly family and uses:

| Split | Records | Families |
|---|---:|---|
| Train | 2240 | uniform, layered, Marmousi |
| Validation | 480 | uniform, layered, Marmousi |
| Test-ID | 480 | uniform, layered, Marmousi |

Three canonical out-of-distribution cases may be reported as an additional
qualitative and diagnostic panel, but they do not replace the full held-out
test-ID results.

The following identities are frozen before experiments:

- dataset file identity and schema;
- split indices and medium-group membership;
- saved-time axis;
- spatial coordinates;
- travel-time file identity;
- pressure normalization;
- source-parameter normalization;
- receiver coordinates;
- parent and baseline checkpoint hashes;
- all random seeds.

All primary statistics are computed on the 480-record test-ID split. Validation
truth may select global hyperparameters and checkpoints, but test future truth
must not influence architecture selection, LoRA hyperparameters, adaptation
acceptance, stopping, rollback, or figure-case selection.

## 4. DCLP-NO Architecture

### 4.1 Inputs

For velocity \(v(x,z)\), source descriptor
\(\boldsymbol{s}=(x_s,z_s,f_0,t_0,A)\), and a requested registered saved time
\(t\), DCLP-NO predicts one pressure field
\(\hat p(x,z,t)\). It directly queries any stored time and does not roll out
autoregressively.

The spatial propagation bundle contains 12 dense channels:

1. retarded time \(\tau=t-t_0-T(x,z)\);
2. travel time \(T(x,z)\);
3. source distance;
4. path velocity;
5. endpoint velocity;
6. differentiable causal gate;
7. \(\sin(2\pi f_0\tau)\);
8. \(\cos(2\pi f_0\tau)\);
9. four Gabor-like envelopes at different propagation scales.

These features are derived from velocity, source information, time, and the
registered eikonal/ray travel-time field. They never use target pressure.

### 4.2 Medium and source encoders

The medium encoder is a learned-complex FNO pyramid with successively reduced
spectral modes, currently 20, 16, 12, and 8. It supplies spatial features at
multiple resolutions.

The source encoder produces:

- a dense source map aligned with the spatial grid;
- a global hidden descriptor used for FiLM modulation;
- embeddings of source frequency, onset time, amplitude, and location.

The amplitude path remains explicit to respect linear pressure scaling over the
registered amplitude range.

### 4.3 Hybrid global-local field

The predicted normalized field before hard boundary factors is

\[
\hat p_{\mathrm{raw}}
=
\hat p_{\mathrm{global}}
+\lambda_{\mathrm{loc}}\Delta p_{\mathrm{local}}
+\lambda_{\mathrm{spec}}\Delta p_{\mathrm{spec}}.
\]

Here:

- \(\hat p_{\mathrm{global}}\) is the global MIONet coarse field;
- \(\Delta p_{\mathrm{local}}\) is a zero-initialized local-propagation U-Net
  residual;
- \(\Delta p_{\mathrm{spec}}\) is the factorized spectral dense correction;
- the correction scales are learned under the fixed parent configuration.

The local field receives the medium pyramid, the 12 propagation channels, the
source map, source FiLM state, and the saved-time embedding. Its odd-grid-safe
multiscale path follows the 201 to 101 to 51 to 26 to 13 resolution sequence
and reconstructs the 201 by 201 field with skip connections. Local convolutions
represent short-wavelength front shape; the coarse levels supply the receptive
field needed for long-range propagation.

The final output applies:

- a structural causal factor to suppress pressure before first arrival;
- the registered pressure-free top-boundary factor;
- the existing pressure normalization and source-amplitude scaling.

### 4.4 Architectural interpretation

The paper must avoid claiming that convolutions inherently remove dispersion.
The testable mechanism is narrower:

- travel-time and retarded-time fields place the front;
- explicit sine/cosine phase channels reduce the burden of learning oscillatory
  phase from absolute coordinates;
- Gabor envelopes separate front support at several spatial scales;
- a translation-equivariant local residual reconstructs sharp local geometry
  without relying only on a global low-rank product;
- the spectral corrector retains efficient long-range correction.

This mechanism motivates the dispersion hypothesis, but only the experiments
in Section 7 may validate it.

## 5. Two-Snapshot LoRA Instance Adaptation

### 5.1 Observation protocol

Each deployment instance exposes exactly two complete pressure snapshots:

\[
t_a=t_0+\frac{2}{f_0},\qquad
t_b=t_0+\frac{4}{f_0}.
\]

Each requested time is snapped deterministically to the nearest registered
saved time, with ties resolved toward the earlier time. The resulting indices
are stored in the instance manifest.

The adapter may read:

- velocity and deterministic velocity features;
- source parameters and source map;
- coordinates and registered time axis;
- travel-time and propagation features;
- the two complete target fields at \(t_a\) and \(t_b\);
- frozen-parent predictions and internal features.

It may not read:

- any target pressure frame other than the two registered observation frames;
- any near-surface receiver trace;
- any metric computed with \(t>t_b\);
- any test future truth during acceptance, rollback, early stopping, or
  hyperparameter selection.

All reported temporal generalization metrics use only \(t>t_b\).

### 5.2 Adapter placement and parameterization

All parent weights remain frozen. Low-rank adapters are inserted only into the
selected time-, phase-, and source-conditioned linear or 1 by 1 projection
layers in:

- the local-field branch;
- the factorized dense corrector.

For a frozen projection \(W\), adaptation uses

\[
W' = W + \frac{\alpha}{r}BA,
\]

where rank \(r=4\) is primary, \(A\) receives a small deterministic
initialization, and \(B\) is initialized to zero. Therefore the initial adapted
model is exactly the parent.

Rank 2 and rank 8 are pre-registered ablations. The main paper reports adapter
parameter count, percentage of parent parameters, optimization time, peak
memory, and inference overhead.

The adapters do not modify the medium FNO encoder or the global MIONet coarse
branch in the primary method. Layer-target ablations may test these choices,
but they cannot replace the pre-registered main configuration after observing
test future truth.

### 5.3 Two-frame objective

Only the two observed fields contribute to the instance loss:

\[
\mathcal L_{\mathrm{2snap}}
=
\mathcal L_{\mathrm{field}}
+\lambda_g\mathcal L_{\nabla}
+\lambda_s\mathcal L_{\mathrm{spectrum/phase}}
+\lambda_c\lVert\Delta\theta_{\mathrm{LoRA}}\rVert_2^2.
\]

The terms are:

- normalized full-field reconstruction over \(t_a,t_b\);
- spatial-gradient mismatch over both observed frames;
- a stabilized amplitude-spectrum and phase-sensitive loss over both frames;
- LoRA correction regularization.

No saved-frame PDE residual is claimed because the stored 2.5 ms frames do not
contain the fine solver steps or CPML memory variables required to reproduce
the reference update faithfully.

The optimizer, learning rate, step count, rank, loss weights, clipping, and
acceptance thresholds are selected once on validation data and frozen before
test-ID evaluation. An instance may roll back only for pre-registered observed
loss, nonfinite-value, correction-norm, causality, or boundary violations. It
may never roll back because its unseen future score is poor.

### 5.4 Leakage-proof execution

Adaptation and future evaluation are separate processes:

1. a guarded data view exposes exactly the two observation indices;
2. the adapter writes an immutable state, access log, gate decision, and
   SHA-256 digest;
3. the future evaluator verifies the sealed state;
4. only then does it open \(t>t_b\) truth.

A mutation-invariance test replaces all future target frames and requires
identical:

- adapter weights;
- optimization trace;
- accessed indices;
- accept or rollback decision;
- state digest.

Near-surface receivers are evaluation only and must not be constructed from
truth until the adapter state is sealed.

## 6. Receiver Geometry and Wavefield Visualization Protocol

The registered receiver line is:

\[
z_r=20~\mathrm{m}, \qquad
x_r=100,150,\ldots,1900~\mathrm{m}.
\]

This produces 37 receivers one stored-grid row below the pressure-free top
boundary. The \(z=0\) row is not used because pressure is structurally zero
there.

Receiver metrics include:

- normalized trace L2;
- normalized RMSE;
- maximum normalized cross-correlation and lag;
- first-arrival pick error;
- dominant-phase error in registered frequency bands;
- amplitude-envelope error.

Receiver figures use fixed representative offsets selected without test-error
inspection, plus a full 37-receiver gather difference panel. All methods use
identical receiver interpolation, time window, normalization, and axes.

The main snapshot panel includes:

- the two supervised frames \(t_a,t_b\), clearly marked as observed;
- an early future frame
  \(t_c=t_b+2/f_0\), snapped to the registered axis;
- a late future frame
  \(t_d=t_b+0.75(T_{\max}-t_b)\), snapped to the registered axis.

Truth, prediction, and error share registered color-limit rules. Predictions
must not receive per-method contrast enhancement. Figure cases are selected
from a deterministic manifest before final test metrics are opened.

## 7. Experimental Design

### 7.1 Main comparisons

The primary table compares:

1. coarse-grid second-order finite difference;
2. coarse-grid fourth-order finite difference;
3. coarse-grid or reference-resolution LWC-84 as explicitly labeled;
4. standard FNO;
5. factorized FNO;
6. parameter-matched Patch-DeepONet or MIONet;
7. the registered global MIONet coarse field;
8. DCLP-NO pretrained parent;
9. DCLP-NO with two-snapshot LoRA.

The parameter-matched DeepONet/MIONet baseline must be trained independently
from random initialization on the same training split and update budget. The
existing failed or zero-update DeepONet artifacts are diagnostic evidence only
and cannot serve as the primary published baseline.

Every learning baseline receives the same target-free physical inputs that its
architecture supports, the same train/validation/test split, and a documented
training budget. Any difference in parameter count, sampled queries, or
training frames is disclosed.

### 7.2 Structural ablations

The pre-registered architecture ablations are:

- global MIONet coarse field only;
- local field replacing the coarse field;
- global coarse plus local residual;
- no travel-time or retarded-time features;
- no sine/cosine phase and Gabor-envelope features;
- no structural causal gate;
- no factorized spectral corrector.

These ablations answer whether gains come from local propagation, explicit
physics-conditioned coordinates, phase representation, causality, or generic
extra capacity.

### 7.3 LoRA ablations

The pre-registered adaptation ablations are:

- no adaptation;
- head-only fine-tuning;
- full fine-tuning;
- LoRA rank 2, 4, and 8;
- LoRA in local-field layers only;
- LoRA in dense-corrector layers only;
- LoRA in both selected groups;
- adjacent first-post-onset frames versus the fixed two-period/four-period
  protocol;
- clean observations and additive observation noise at 20 dB and 10 dB SNR.

All ablations obey the same two-complete-snapshot information budget. Full
fine-tuning is an adaptation baseline, not the proposed deployment method.

### 7.4 Primary accuracy metrics

For all future frames \(t>t_b\), report:

- full-field relative L2;
- active-energy relative L2;
- early, middle, and late future relative L2;
- framewise relative-L2 median and 95th percentile;
- low-, middle-, and high-band spectral relative L2;
- phase correlation;
- structural similarity as a secondary perceptual metric;
- family-wise metrics for uniform, layered, and Marmousi.

### 7.5 Direct dispersion evidence

Dispersion evidence has three linked levels.

**A. Discretization analysis**

- reproduce modified-wavenumber and normalized phase-velocity curves for FD2,
  FD4, and LWC-84;
- sweep propagation angle and points per wavelength;
- state explicitly that the analysis quantifies rather than eliminates
  reference-solver dispersion.

**B. Wavefield propagation analysis**

- radial wavefront-position error in homogeneous media;
- zero-crossing displacement;
- dominant spatial-wavenumber error;
- two-dimensional or radial \(k\)-\(\omega\) ridge deviation;
- post-front ringing energy;
- phase error versus propagation distance and time.

**C. Near-surface receiver analysis**

- cross-correlation phase lag;
- first-arrival timing error;
- trace coherence;
- frequency-dependent phase residual;
- receiver-gather difference energy.

The main dispersion comparison is DCLP-NO and DCLP-NO+LoRA against the strongest
learning baseline under identical data and receiver geometry. FD curves provide
physical context but are not substituted for neural-model comparisons.

### 7.6 Statistical analysis

Primary method comparisons are paired by test record. Report:

- mean, median, and interquartile range;
- paired bootstrap 95% confidence intervals with medium-group resampling;
- effect size and absolute difference;
- family-wise results;
- Holm correction for the small registered family of primary dispersion
  hypotheses.

Seeds, bootstrap replicates, paired record IDs, and confidence-interval
implementation are stored in the experiment manifest. Isolated hand-picked
examples never establish the central claim.

### 7.7 Efficiency

Report:

- trainable and total parameters;
- parent training budget;
- adapter parameter count;
- per-instance LoRA optimization time;
- single-frame and full-trajectory inference latency;
- peak GPU memory;
- storage per parent and per adapter;
- speedup relative to each finite-difference configuration, with hardware,
  precision, batch size, I/O inclusion, and warm-up stated.

## 8. Claim Gates

### 8.1 Dispersion claim

The manuscript may state that the method **suppresses numerical-dispersion-like
phase and wavefront errors** only if:

1. DCLP-NO or DCLP-NO+LoRA beats the strongest learning baseline on at least two
   direct dispersion metrics from Section 7.5B;
2. paired 95% confidence intervals for those improvements do not cross zero;
3. receiver lag or coherence metrics in Section 7.5C improve in the same
   direction;
4. the result is not confined to one displayed example or one medium family.

It must never claim that dispersion is eliminated. It must not attribute
fine-grid reference accuracy to the learned model.

If the gate fails, the paper title changes to:

> **Physics-Conditioned Local-Propagation Neural Operator With Two-Snapshot
> LoRA Adaptation for Acoustic Wavefield Modeling**

and the dispersion discussion becomes a diagnostic result rather than the
central claim.

### 8.2 LoRA claim

The paper may claim successful two-snapshot instance adaptation only if:

- aggregate future test error improves over the frozen parent;
- paired uncertainty supports the improvement;
- rollback frequency and all regressions are reported;
- the two-frame access and future-mutation audits pass;
- near-surface receiver results do not show a contradictory phase degradation.

If the gate fails, LoRA remains a negative or robustness study and is not
described as an improvement.

### 8.3 Accuracy claim

No claim such as “state of the art,” “high fidelity,” or “superior
generalization” is allowed without a matched, current baseline and full
held-out evidence. The current 0.2713 validation relative L2 alone does not
meet this gate.

## 9. Paper Structure

The main manuscript follows this structure:

1. **Introduction**
   - geophysical forward-modeling cost and phase fidelity;
   - shortcomings of global low-rank or purely spectral field generation;
   - local propagation and two-snapshot deployment setting;
   - concise contributions.
2. **Problem Formulation and Numerical Reference**
   - acoustic wave equation;
   - boundary and source conventions;
   - LWC-84 reference, storage restriction, and data splits;
   - distinction between solver and surrogate dispersion.
3. **Dispersion-Controlled Local-Propagation Neural Operator**
   - medium and source encoding;
   - propagation-feature construction;
   - global-local hybrid field;
   - causal and free-surface constraints.
4. **Two-Snapshot LoRA Instance Adaptation**
   - observation-time protocol;
   - adapter placement and objective;
   - leakage guard, sealing, and rollback.
5. **Experimental Protocol**
   - baselines, splits, receiver geometry, metrics, statistics, and efficiency.
6. **Results**
   - main test accuracy;
   - direct dispersion evidence;
   - two-snapshot future prediction;
   - wavefield and receiver comparisons;
   - ablations and cost.
7. **Discussion**
   - mechanism, failure cases, high-frequency and late-time limitations;
   - dependence on travel-time quality;
   - acoustic-only scope.
8. **Conclusion**
   - evidence-matched summary without claims beyond the gates.

The abstract is written last and contains only values from sealed final
artifacts. The introduction contributions are phrased as:

1. a hybrid global-local neural operator with dense propagation conditioning;
2. a strict two-complete-snapshot LoRA deployment protocol;
3. a dispersion evaluation spanning analytic discretization, full wavefields,
   and near-surface receivers;
4. a leakage-audited benchmark with uncertainty and efficiency reporting.

## 10. Figure and Table Plan

### 10.1 Figures

**Graphical abstract — method and outcome**

- left: velocity/source and two observation snapshots;
- center: DCLP-NO plus rank-4 LoRA;
- right: future sharp wavefront and near-surface receiver traces;
- minimal text, 660 by 295 aspect for the IEEE graphical-abstract version.

**Fig. 1 — DCLP-NO architecture**

- top-level end-to-end flow;
- inset for the learned-complex FNO medium pyramid;
- inset for the 12-channel propagation bundle;
- inset for the odd-grid local U-Net;
- explicit global coarse, local residual, spectral corrector, causal gate, and
  free-surface factor;
- frozen and LoRA-adapted layers distinguished by line style and color.

**Fig. 2 — numerical and observation setup**

- domain, source, pressure-free top, three-sided CPML;
- 5 m physical and 10 m stored grids;
- receiver line at 20 m depth;
- timeline showing \(t_a,t_b\) observed and \(t>t_b\) blind.

**Fig. 3 — dispersion evidence**

- modified-wavenumber or normalized phase-velocity curves;
- wavefront-radius and zero-crossing errors;
- \(k\)-\(\omega\) or radial spectral-ridge deviation;
- receiver phase lag versus offset.

**Fig. 4 — main quantitative results**

- aggregate and family-wise paired distributions;
- future-error versus time;
- phase or spectral error;
- 95% confidence intervals.

**Fig. 5 — wavefield snapshots**

- truth, strongest baseline, frozen DCLP-NO, and two-snapshot LoRA;
- observed \(t_a,t_b\) and blind future \(t_c,t_d\);
- matched color limits and error maps;
- uniform, layered, and Marmousi cases.

**Fig. 6 — near-surface receiver comparison**

- representative waveform overlays;
- full receiver gathers;
- residual gathers;
- lag, coherence, and spectral-phase summaries.

**Fig. 7 — ablation and efficiency**

- architectural component ablations;
- LoRA rank and layer placement;
- noise robustness;
- error-cost or error-memory Pareto view.

The visual language is “Classic Academic plus Modern Minimal.” Use an
Okabe-Ito-compatible color palette, direct labels, restrained grids, vector
PDF/EPS for diagrams and curves, and 300–600 dpi raster output for wavefields.
Every figure must remain interpretable in grayscale and for common
color-vision deficiencies.

### 10.2 Tables

- **Table I:** dataset, numerical solver, training, and observation protocol;
- **Table II:** main held-out accuracy and dispersion metrics;
- **Table III:** LoRA, structural ablation, and efficiency results;
- detailed family, OOD, noise, and hyperparameter tables move to the
  supplement when needed for the page target.

## 11. Literature and Citation Standard

The literature review must cover primary sources for:

- Fourier neural operators and factorized spectral operators;
- DeepONet and multi-input operator networks;
- neural operators for seismic or acoustic wave propagation;
- finite-difference numerical dispersion and LWC-style optimized stencils;
- Eikonal travel-time conditioning;
- LoRA and parameter-efficient adaptation;
- phase-, spectrum-, and receiver-domain evaluation.

All references are verified against the primary paper, DOI, publisher page, or
official preprint record. No citation is added solely from a search snippet.
The final bibliography contains no unverifiable placeholder references.

The manuscript uses the current official IEEE journal template and TGRS author
guidance. The main paper targets approximately 10 printed pages, with extended
ablations and implementation details in supplementary material. Page count is
an editing target, not a reason to omit evidence necessary for reproducibility.

## 12. Execution and Artifact Layout

No file under `project_pristine` or the bound pretraining artifact may be
modified. New work is isolated under:

```text
project/
  paper/tgrs_dclp_no/
    main.tex
    references.bib
    sections/
    figures/
    tables/
    supplement/
    cover_letter/
    README.md
  artifacts/tgrs_dclp_no/
    audit/
    baselines/
    parent/
    lora/
    dispersion/
    receivers/
    statistics/
    figures/
    manifests/
    logs/
```

The root-level downloaded `artifacts/` tree may contain baseline runs, but no
existing artifact is overwritten. Each run stores:

- resolved configuration;
- command, working directory, environment, hardware, and seed;
- dataset, split, checkpoint, and source-tree identities;
- update and evaluation logs;
- best and last checkpoints where training is used;
- terminal state;
- raw per-record metrics;
- aggregate tables;
- figure-source arrays.

## 13. Execution Order and Resource Safety

The work proceeds in this order:

1. provenance and leakage audits;
2. manuscript and figure-source scaffolding;
3. CPU-only analytic dispersion calculations;
4. plotting and statistical unit tests;
5. LoRA and baseline unit tests;
6. one-record CPU or GPU forward/backward smoke;
7. three-family small-panel experiments;
8. validation-only global hyperparameter selection;
9. full 480-record held-out evaluation;
10. paired statistics and claim-gate audit;
11. final figures, manuscript, supplement, and cover letter;
12. LaTeX compile, PDF visual inspection, reference verification, and artifact
    manifest.

The currently occupied RTX 3090 is not preempted. CPU work is completed first.
GPU experiments start only when sufficient memory is available. Long runs use a
detached process host and are considered successfully launched only after live
PID, advancing log, GPU utilization, and writable checkpoint verification.

## 14. Verification Requirements

Before final completion, verify:

- dataset, split, saved-time, travel-time, and checkpoint identities;
- exact two-frame access contract;
- future-mutation invariance;
- receiver-exclusion audit;
- LoRA zero-initialization identity;
- frozen-parent weight identity before and after adaptation;
- causality and free-surface preservation;
- unit consistency and array-axis order;
- finite forward, backward, adaptation, and evaluation;
- deterministic case and receiver manifests;
- baseline parameter counts and training budgets;
- analytic dispersion reproduction;
- metric definitions on synthetic signals with known phase shifts;
- paired bootstrap reproducibility;
- no unresolved placeholder text;
- no unsupported claims or elastic-wave language;
- all figures at final dimensions with legible labels;
- color and grayscale accessibility;
- successful IEEE LaTeX compilation;
- visual inspection of every PDF page;
- reference-to-bibliography consistency and DOI/source verification;
- SHA-256 artifact manifest for manuscript, figures, tables, and primary
  checkpoints.

Completion means a verified submission package, not merely a draft or a running
experiment.

## 15. Explicit Non-Goals and Failure Reporting

- no elastic-wave extension;
- no receiver-supervised adaptation;
- no more than two complete target snapshots per adapted instance;
- no future-truth model selection;
- no replacement of failed samples after evaluation;
- no claim that LWC-84 or DCLP-NO eliminates dispersion;
- no use of the failed CLFC smoke as positive evidence;
- no use of incomplete DeepONet attempts as the strongest baseline;
- no hiding of late-time, high-frequency, rollback, or family-specific
  failures.

If a pre-registered gate fails, the result is reported honestly and the title,
abstract, and contribution language are reduced accordingly.

## 16. Provenance Limitation

The editable continuation workspace contains no `.git` directory. Therefore
this specification cannot be committed locally without inventing repository
history or receiving a repository restored by the user. Provenance is instead
preserved through this dated specification, immutable bound artifacts,
source-tree hashes, checkpoint hashes, resolved configurations, access logs,
and final SHA-256 manifests.
