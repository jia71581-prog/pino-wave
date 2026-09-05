# Kakeya-Inspired Anisotropic Phase-Space Representation — Design

Date: 2026-07-31
Project: `FNO-Acoustic-Kakeya-20260731` (2D acoustic operator, author Yang Cui)
Base: Option B Local Propagation Field (LPF) leading branch (0.47 → 0.29; late 0.43).
Status: **DRAFT skeleton.** Code anchors are finalized against the source; the literature
positioning / incremental-novelty section (§4) is a placeholder pending the background
literature scan (task #2).

## 0. TL;DR of the idea

Standard FNO representations are **isotropic** (global Fourier disk truncation) or worse,
**axis-separable** (our factorized spectral corrector). Wave energy in phase space is
**anisotropic**: it concentrates on tubes/wave-packets that follow light rays, at the
**parabolic scaling** δ × δ^{1/2} (the 1/2 power comes from light-cone curvature). This is
exactly the geometry the Kakeya / Restriction / local-smoothing chain is built on, and the
curvelet / wave-atom frame (Candès–Donoho) is its concrete, computable realization —
Fourier integral operators (the wave propagator is one) are **sparse** in curvelet frames.

**Hypothesis under test:** replacing/augmenting the isotropic-or-separable spectral
representation with a **directional, parabolic-scaled, wavefront-adapted** representation
attacks the two open bottlenecks recorded in memory:
- **#1 "coarse-field uniform representation"** — clean concentric wavefronts have sharp,
  *oriented* spectral ridges that isotropic/separable truncation smears (Gibbs).
- **late-time multi-wave representation** — several wavefronts = several groups of packets
  at distinct (direction, frequency); a directional frame separates them by construction.

This is a **representation / architecture** contribution, not a proof-technique transfer.
Wang–Zahl's induction-on-scales / polynomial-method machinery is *not* imported; only the
shared wave-packet geometry is.

## 1. Evidence this is the right lever (from the code + prior verdicts)

Two places in the current leading branch carry the isotropic/separable assumption:

### 1a. Coarse-field generator — `saved_time_phase_operator_v4/local_field.py`
`LocalPropagationFieldGenerator._UNet` (local_field.py:65-112) is a **purely isotropic**
3×3-conv U-Net. Its convolutions have no orientation selectivity; multi-scale pooling gives
a global receptive field but every kernel is direction-agnostic. It renders a ~5-cell front
sharply in shape but has no structural bias toward the front's *local orientation*.

Reusable hook already present: the 12-channel propagation bundle from
`features.py:dense_propagation_features` (retarded τ, causal gate, sin/cos phase, **4 Gabor
envelopes**) is injected at local_field.py:245. **The Gabor envelopes are a proto-wave-atom
input** — a natural anchor for a parabolic-scaled directional block.

### 1b. Residual spectral corrector — `saved_time_phase_operator_v4/spectral.py`
`AxisFactorizedComplexSpectralConv2d` (spectral.py:19-107) is **axis-separable**: it rffts
along x and z *independently* and truncates each axis at `modes` (spectral.py:49-83). This
is *worse* than isotropic disk truncation for oblique wavefronts — a front propagating at
~45° is truncated on **both** axes simultaneously. This is the cleanest possible instance of
the failure mode wave-packet theory describes: energy concentrated on an oblique tube is
maximally mismatched to an axis-aligned tensor basis.

`LowRankCoupledSpectralResidual2d` (spectral.py:178+) is a CP-factorized full-2D-rFFT
response — full 2D grid, but still an isotropic global basis (no directional wedges).

**One-line diagnosis:** the representation places energy on an isotropic/axis grid; the
physics places it on oriented parabolic tubes. Close that gap.

## 2. Hard constraints inherited (do not break)

From `2026-07-23-local-propagation-coarse-field-design.md` §2 and the V4 contract:
- Output must stay `coarse [records, count, H, W]` normalized pressure — decoder, free-
  surface factor `tanh(z/20)²`, residual corrector, scoring all depend on this exact tensor.
- Arbitrary saved-time query (any of 401 stored indices), no autoregressive rollout.
- Grid 201×201, dx=dz=10 m; sharpest feature ≈ 4.7 cells; front moves ≈ 2 cells/frame.
- Memory: full-dataset training on 24 GB 4090s; keep the flag-gated non-breaking pattern
  (off → today's behavior byte-for-byte; all existing runs/tests unaffected).
- Physics scale for parabolic scaling: min wavelength ≈ 47 m ≈ 4.7 cells sets the finest
  angular/frequency resolution worth representing.

## 3. Candidate integration points (two, independently switchable)

Both are flag-gated additions, mirroring the LPF `dense_local_field=` pattern
(operator.py wiring, ProbeVariant flag, diagnose CLI toggle).

### Point A — Directional block inside the LPF U-Net (`local_field.py`)
Add an oriented, parabolic-scaled block at one or more U-Net scales, fed by the medium
pyramid + the Gabor bundle. Options for the directional primitive (TBD by §4 lit scan):
- steerable / oriented convolution filters (fixed or learned orientation banks);
- a curvelet-style wedge decomposition of the coarse-scale spectrum;
- a learned tube-adapted latent basis (direction × scale channels).
Keeps translation-equivariance; adds orientation-equivariance/selectivity.

### Point B — Wedge (directional-frequency) truncation in the residual corrector (`spectral.py`)
Replace axis-separable truncation with a **polar-wedge** truncation on the 2D rFFT: partition
frequency plane into `n_angles` angular wedges × `n_scales` radial bands with **parabolic
scaling** (wedge angular width ∝ (radius)^{-1/2}), learn a per-wedge complex response. This is
the curvelet tiling of frequency space, implemented as a masked 2D-FFT operator.

### Latent capacity allocation (the real design question)
Where to spend parameters/channels: direction (n_angles) vs scale (n_scales) vs channel
width. Physics prior: finest angular resolution set by ~4.7-cell wavelength; late-time
multi-wave needs enough angular bins to separate crossing fronts. To be pinned in §5 once
§4 tells us what's already known to work.

## 4. Literature positioning & incremental novelty

> **Provenance:** all pivotal IDs **verified 2026-07-31** directly against the arXiv API
> (`export.arxiv.org`; arxiv.org/google were blocked, API reachable). SNO title/abstract
> confirmed verbatim; Candès–Demanet, FIONet, PDNO, Multiwavelet-NO, Group-Equiv-FNO,
> Isotropic-FNO, Physics-Aligned-Canonical-FNO all resolve to the stated titles.
> Exact-phrase searches `"curvelet neural operator"` and `"wave atom"+"neural operator"`
> **returned zero results** (sanity control `curvelet`+`neural network` returned hits, so
> the null is real, not a broken query). The white space is confirmed.
>
> SNO abstract (verbatim, verified): *"…we introduce the Shearlet Neural Operator (SNO)…
> replaces the Fourier transform with a shearlet-based representation… Across seven benchmark
> PDE families, including strongly anisotropic advection, anisotropic diffusion, and nonlinear
> conservation laws with straight, curved, interacting, spiral, and polygonal shock
> structures…"* — **no wave equation, no local propagator, no Kakeya/FIO-sparsity theory.**

**The single closest work is the Shearlet Neural Operator (SNO, arXiv 2604.25181, Apr 2026):**
it replaces FNO's Fourier transform with a shearlet transform — a parabolic-scaling (δ×δ^{1/2}),
directional, multiscale frame — motivated exactly as we motivate this: isotropic global Fourier
is the wrong representation for anisotropic/front structure. It occupies our intended
architectural slot. But its gaps are precisely our increment:
- SNO targets **generic PDEs** (advection, aniso-diffusion, shocks) — **not the wave equation**,
  where the FIO-sparsity theorem actually bites.
- SNO is a **global transform**, not a **local propagation** operator (our validated Option B).
- SNO has **no** connection to the Kakeya / restriction / local-smoothing chain or to the
  Candès–Demanet FIO-sparsity theorem that justifies *why* the parabolic frame is right for waves.

**Theory spine (cite as justification, not competitors):**
- Candès–Demanet 2004 (math/0407210): the wave propagator (an FIO) is **optimally sparse in
  curvelets** via parabolic scaling. This is the theorem the whole program rests on.
- Kakeya ⟹ Restriction ⟹ local smoothing: shared wave-packet/tube machinery; **zero footprint**
  in the operator-learning literature — genuinely unclaimed as an inductive-bias motivation.

**Alternative-basis NO family (baselines to beat / position against):**
- WNO (Wavelet NO, 2205.02191) — isotropic wavelets, no directionality. Natural baseline.
- Multiwavelet NO (2109.13459) — multiscale, non-directional.
- SNO (2604.25181) — shearlet, directional+parabolic. **The must-beat / must-differentiate.**
- PDNO (2201.11967) — learned pseudo-differential symbol σ(x,ξ); a genuine phase-space
  parametrization but ΨDO (wrong operator class for waves; FIO ≠ ΨDO). Cite for phase-space framing.
- FIONet (2006.05854) — FIO-geometry inductive bias for wave imaging; argues CNN
  translation-equivariance is the *wrong* bias for waves (echoes our critique), but learns the
  canonical transform, not a curvelet frame; inversion not forward propagation.

**White space (our increment):** *parabolic-scaling phase-space frame (curvelet / wave-atom)
+ acoustic wave equation + **local** propagation operator + explicit Kakeya/wave-packet/
FIO-sparsity theory.* Nobody. Wave-atoms and curvelets are **absent** from operator learning
despite having the strongest FIO-sparsity theorems.

**Locked novelty statement (1 sentence):** *A local wave-propagation neural operator whose
spectral layer is a parabolic-scaling directional frame (curvelet/wave-atom) rather than
isotropic Fourier, justified by the Candès–Demanet FIO-sparsity theorem and the wave-packet
geometry underlying the Kakeya/local-smoothing chain — differentiated from SNO by (i) the
curvelet/wave-atom frame with its wave-specific sparsity theorem, (ii) the wave-equation target,
and (iii) fusion into a validated local propagation field.*

## 4b. Numerical evidence collected so far (this session)

`docs/kakeya/probe_truncation_anisotropy.py` — measured relative-L2 reconstruction of a
localized wave-packet front at matched retained complex DOF (8443) on the 201×201 grid:

| front            | axis-separable (current corrector) | isotropic disk | wedge (energy-oracle upper bound) |
|------------------|-----------------------------------:|---------------:|----------------------------------:|
| oblique 0°/90°   | 0.0002                             | 0.0000         | 0.0000                            |
| oblique 30°/60°  | 0.0203                             | 0.0017         | 0.0000                            |
| **oblique 45°**  | **1.0000** (total failure)         | 0.0020         | 0.0000                            |
| **circular**     | **0.5054**                         | 0.0000         | 0.0000                            |

**Reading (honest):** (1) The current `AxisFactorizedComplexSpectralConv2d` axis-separable
truncation *completely fails* on 45° fronts and loses half the energy on circular fronts — the
uniform-family worst case from memory. (2) **Most of that gap closes by going isotropic/full-2D
alone**, which needs no Kakeya. (3) The wedge column is an energy-greedy **oracle** (upper
bound), so the *real* curvelet increment over isotropic is smaller and lives in regimes this
single-packet probe cannot see: **multi-wavefront superposition, curved marmousi fronts, and
sparsity under a fixed parameter budget.** This directly shapes the staged plan below.

## 5. Selection, gates, and rollout

### 5a. Parabolic-tiling probe result (2026-07-31) — decision-relevant NEGATIVE

`docs/kakeya/probe_parabolic_tiling.py` builds three **energy-normalized tight frames**
(POU residual ~1e-16, full-reconstruction rel-L2 ~1e-16 — self-checks pass, probe is
trustworthy): plain **Fourier**, **isotropic** dyadic radial bands, and **parabolic** radial×
angular tiling with wedge count growing like 2^{j/2} (true curvelet/Kakeya geometry: more
orientations at finer scales). Metric: nonlinear m-term rel-L2 at **aggressive** budgets
(0.5–5% of N² DOF), on both narrowband cos fronts and **broadband Ricker fronts** (the
correct model of the dataset's f0∈[8,30] Hz snapshots).

Outcome: **parabolic never wins on the physically-realistic broadband fields.** On Ricker
oblique-45 @5%: Fourier 0.0009, isotropic 0.168, parabolic 0.168. On Ricker multi-wave and
curved arc, plain Fourier wins decisively at every budget. Narrowband fields: isotropic beats
parabolic; parabolic only ties on the single oblique front.

**Honest conclusion (two layers):**
1. **Falsified:** the easy justification — "a curvelet spectral layer represents our wave
   *snapshots* more sparsely than isotropic/Fourier, so it will be more accurate" — does not
   hold for our fields. Snapshot-level parabolic sparsity is absent at 201×201.
2. **Not falsified (probe can't test it):** the actual Candès–Demanet theorem is about the
   **wave propagator (FIO) matrix** being sparse in curvelets — a property of the *operator*,
   not of single snapshots. The neural operator learns a *mapping*; both probes measure
   snapshot m-term approximation, which is the wrong object for the FIO-sparsity claim. So the
   probe removes the cheap argument but cannot kill the operator-sparsity hypothesis.

**Implication for the plan:** the Kakeya/curvelet direction is now a *high-risk operator-level
bet*, not a representation win we can bank. The only cheap, probe-backed, certain gain is the
axis-separable → isotropic/full-2D corrector fix (§4b). Recommend decoupling:
- **Track 1 (certain, low-risk):** replace `AxisFactorizedComplexSpectralConv2d`'s axis
  truncation with an isotropic/full-2D truncation. Probe-backed for uniform + oblique fronts.
  Doubles as the isotropy ablation baseline for any later directional test.
- **Track 2 (research bet, defer or gate hard):** a curvelet/wave-atom *operator* layer,
  justified by FIO sparsity — but only worth building if a *operator-level* probe (does the
  learned propagator become near-diagonal in curvelets?) shows promise, since the
  snapshot-level probe says no. Do NOT build on the snapshot-sparsity rationale.

### 5b. Gates (when a track is chosen)
Reuse the LPF gate ladder (G1 smoke → G2 overfit probe at parameter-comparable size, target
agg < 0.10 & every family < 0.12 → G3 short pilot must beat 0.29 → G4 long → G5 verify).
Pre-register the isotropy-vs-anisotropy ablation at matched parameter count so any gain is
attributable to the mechanism, not added capacity (§6).

### 5c. Track 1 implementation status (2026-07-31) — DONE, with a critical config finding

Implemented `IsotropicComplexSpectralConv2d` (`spectral.py`): true 2-D rFFT + isotropic disk
mask `|k|<modes`, two-corner FNO layout. Flag-gated into `FactorizedComplexResidualBlock` /
`FactorizedComplexResidualStack` via `isotropic_spectral=False` (default) → **byte-for-byte
unchanged behavior**; existing runs/tests unaffected. G1 smoke
(`docs/kakeya/test_isotropic_spectral_g1.py`) passes: shapes, finite fwd/bwd, default-off
axis-layer equivalence, retained-DOF accounting.

**CRITICAL finding for G2 — equal `modes` is NOT a fair swap.** The axis-separable layer
retains a '+'-shaped region (~2·m·N ≈ **8443** coeffs at m=24, N=201); the isotropic disk
`|k|<m` retains only ~π·m²/4 ≈ **943** coeffs at m=24 — a **9× capacity cut**. A naive
equal-`modes` swap therefore *under-provisions* the isotropic layer and it clips oblique peaks
(a 64-grid fit showed isotropic *worse* purely from this DOF starvation, NOT a method failure).
**The isotropic layer needs `modes ≥ 73` to match the axis DOF at 201×201.** G2 MUST size the
isotropic `modes` to match-or-exceed axis DOF, else it will produce a false-negative. The
matched-DOF probe (`probe_truncation_anisotropy.py`) remains the real evidence that isotropic
is the right geometry; this layer is the trainable realization of it.

Wired the flag through `ProbeVariant` / operator / decoder / stack / diagnose CLI
(`--isotropic-spectral`), and gave the isotropic layer a **low-rank (CP) weight**
(`coupling_rank=8`, params linear in `modes`) so the DOF-matched disk (m=73) fits memory and
stays parameter-fair — at rank 8 it has FEWER params (~6.5k/layer) than the axis layer
(~1.2M/layer), so any win would be attributable to geometry, not capacity.

### 5d. Track 1 G2 result (2026-07-31) — FALSIFIED

Ran the 3-record overfit probe, 200 updates, DOF-matched (axis m=24 vs isotropic m=73
low-rank), `results/kakeya_g2_clean/`:

| | axis baseline | isotropic (low-rank) | Δ |
|---|---|---|---|
| best_fixed aggregate rel-L2 | **0.3699** | **0.3686** | 0.35% (noise) |

The convergence curves are **point-for-point identical** (@50 0.6898/0.6900, @100
0.5283/0.5287, @200 0.3699/0.3686). Uniform — the clean circular front the isotropic disk
should help most — was if anything slightly *worse*; late/phase terms unchanged.

**Verdict: isotropic spectral truncation gives no measurable gain over axis-separable at the
operator level. Track 1 is falsified.** This confirms the prediction from the probe phase:
Track 1 fixes the *spatial* representation, but this model's real bottleneck is *temporal
propagation / wavefront placement* (the uniform-0.47 / late-0.43 shortfall in memory), which a
spatial-frequency basis cannot touch no matter how well it reconstructs a single frame.

**Kakeya direction is now doubly falsified:** snapshot sparsity (§5a parabolic probe) AND
operator-level isotropic truncation (§5d G2). See `[[fno-acoustic-kakeya-direction]]`.

G2 memory/ops notes: the 3-record microbatch OOMs at width128/depth8/rank112 on a 24 GB card —
use `--microbatch-records 1` (gradient-accumulation-equivalent, ~3× less memory) +
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, and pin each arm to its own GPU
(`CUDA_VISIBLE_DEVICES`); the diagnose script defaults to cuda:0, so co-locating arms OOMs.

## 6. Risks / falsification hooks

- **Added capacity, not anisotropy, explains any gain** → ablate with an isotropic block of
  equal parameter count (must under-perform the directional one to claim the mechanism).
- Directional block may under-fit strong-gradient media (marmousi curved fronts) — same risk
  the LPF doc flagged; mitigate with medium-pyramid conditioning.
- Wedge FFT masking cost / memory on 201×201 — bound n_angles×n_scales; measure vs baseline.
- If neither point beats 0.29 at matched capacity → Kakeya framing is a dead end for *this*
  bottleneck; record the negative result (consistent with the memory discipline).

## 7. Closure & the only surviving path (Track 2 gate)

**Status 2026-07-31: the Kakeya direction is closed at the representation and operator-spectral
levels.** Two independent falsifications:
1. §5a — parabolic-tiling (curvelet) frames give no snapshot sparsity advantage on broadband
   Ricker wavefields; plain Fourier wins.
2. §5d — an isotropic (low-rank, DOF-matched, parameter-fair) spectral corrector trains
   identically to the axis-separable one; no operator-level gain.

The root reason both fail is the same and worth stating plainly: **spatial-frequency geometry
is the wrong axis.** This operator's bottleneck lives in *time* — wavefront placement and
late-time multi-wave evolution (memory: uniform 0.47, late 0.43) — not in how a single frame's
spatial spectrum is tiled. Any purely-spatial basis swap is orthogonal to the disease. This is
consistent with the parallel finding that the surviving <5% direction is a *temporal-frequency*
Helmholtz construction (`[[fno-acoustic-temporal-frequency-helmholtz]]`), not a spatial one.

**The one thing not yet falsified — and its hard gate.** Candès–Demanet is a theorem about the
*propagator operator* (the FIO matrix is near-diagonal in curvelets), which neither probe
tested (both measured single-frame m-term approximation — the wrong object). If Track 2 is ever
revisited, it MUST start with an **operator-level probe**, not another architecture build:

> **Track 2 entry gate (do this first, or do not start):** take the trained propagator
> (e.g. the frame-to-frame map, or the learned dense-block operator), express it in a curvelet
> frame, and measure whether its matrix is actually sparse / near-diagonal on *this* data at
> *this* resolution (201×201, ppw≈4.7). Only if the operator is empirically curvelet-sparse
> does a curvelet operator layer have a mechanism to exploit. If it is not sparse — likely,
> given the coarse 201×201 grid truncates the fine-scale curvelet tail the theorem needs — then
> Track 2 is falsified too and the Kakeya program is fully closed. Budget this probe (a few
> hours, no training) before any curvelet/wave-atom layer implementation.

**What is preserved regardless:** the `IsotropicComplexSpectralConv2d` layer (dense + low-rank),
its flag wiring, and the G1/G2 harness all remain in the tree, default-off and byte-compatible.
They are a ready-made isotropy ablation baseline should any future directional-spectral work
need one — the negative result is reusable infrastructure, not dead code.
