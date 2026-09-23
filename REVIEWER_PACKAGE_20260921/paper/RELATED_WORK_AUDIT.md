# Related-work audit: the accuracy routes

Compiled 2026-09-21. Every entry below was retrieved from arXiv, OpenAlex or Crossref during
this audit. Evidence level is marked per entry: **[abstract]** means the abstract was read via
API, **[full]** means the HTML full text was retrieved and read. Nothing here is written from
memory, and nothing is included that was not read at one of those two levels.

This file exists because the project has a specific failure history with citations: an earlier
round quoted a paper's reference list as precedent and later found that list had itself been
LLM-hallucinated and then corrected by its authors. The rule adopted since is that a claim
enters a paper only with a verifiable DOI or arXiv ID plus a location.

---

## Part 1 — The three accuracy questions

The three targets are **complex media**, **long-time stability**, and **numerical
dispersion**. What follows is what the literature actually establishes on each, and what it
implies for the route we are considering.

### 1.1 Coarse-grid plus learned correction: the accounting is favourable and published

| Work | ID | Reported result | Level |
|---|---|---|---|
| Bar-Sinai et al., *Learning data-driven discretizations for PDEs*, PNAS 2019 | `10.1073/pnas.1814058116` (562 citations) | Neural networks estimate spatial derivatives, optimised end to end to satisfy the equations on a low-resolution grid. Integrates nonlinear equations in **1 spatial dimension** at resolutions **4× to 8× coarser** than standard finite differences. | [abstract] |
| Kochkov et al., *Machine learning–accelerated CFD*, PNAS 2021 | `10.1073/pnas.2101784118` | For 2-D turbulence, DNS and LES: accuracy equal to baseline solvers at **8–10× finer resolution per spatial dimension**, giving **40–80× computational speedups**. States the method **remains stable during long simulations** and generalises to forcing and Reynolds numbers outside training, explicitly contrasted with black-box ML. | [abstract] |

These two settle the cost question in principle, and they settle it in our favour: the speedup
is quoted *net*, and the stability claim is made for long rollouts. This is the accounting that
our own project lacks, because our formulation competes with the fine solver directly rather
than standing on top of a cheap one.

Two caveats that must travel with these numbers. Bar-Sinai is **1-D**. Kochkov is
**Navier–Stokes**, an advection-dominated first-order-in-time system; the acoustic wave
equation is second-order hyperbolic, so neither the stability argument nor the speedup factor
transfers without demonstration.

### 1.2 Numerical dispersion: one direct precedent, and it defines our gap

The closest published work to the route under consideration is:

> Siahkoohi, Louboutin, Herrmann, *Neural network augmented wave-equation simulation*,
> arXiv:**1910.00925** (2019). **[abstract]**

It treats an inaccurate finite-difference Laplacian as "incomplete physics", exploits the
one-to-one similarity between timestepping and CNNs, and **intersperses CNNs between
low-fidelity timesteps**, correcting the wavefield several times during propagation to limit
numerical dispersion from a poor Laplacian discretisation.

That is the same mechanism we were considering. Its self-stated scope is what leaves a gap:
it is a **proof of concept** that corrects dispersion **with the velocity model held fixed**,
varying only source locations to generate training and testing pairs. So the precedent exists
for the mechanism but not for the generalisation that matters to us: **varying heterogeneous
media**. A method that works at fixed velocity is not evidence about Marmousi-class media,
where the medium is the input.

A second, independent probe of the same space:

> *Evaluating Operators for Acoustic Wave Simulation Correction*, arXiv:**2606.08711** (2026).
> **[full]**

Instantiates the Deep FDM framework on 2-D anisotropic acoustic propagation, pairing a
fourth-order FD proxy against a pseudo-spectral reference over 27,000 heterogeneous velocity
fields, and benchmarks twelve correction architectures under 10-fold cross-validation. It
calls itself "the first systematic benchmark for correcting numerical dispersion artifacts in
2D anisotropic acoustic wave simulation, a setting previously unaddressed in the literature."
Two findings from the full text matter to us: the FNO wins but "the absolute margin over CNN2d
is modest", and **all pipelines incorporating a PCR initial guess substantially outperform
those operating directly**. Critically, it operates on **traces**, not full wavefields.

**Search-derived negative evidence.** Three independent query angles returned nothing:
`"coarse grid" AND seismic AND "deep learning"` → **0 hits**;
`"numerical dispersion" AND "deep learning"` → **0 hits**;
`"finite difference" AND correction AND wavefield` → **1 hit**, which is 1910.00925 above.
`"learned correction" AND "wave propagation"` → 1 hit, on ultrasound CT imaging
(arXiv:2502.09546), not time-domain wavefields. Absence of search hits is weak evidence and is
recorded as such, not as proof that no precedent exists.

**Net assessment of the gap:** time-domain **full-wavefield** learned dispersion correction
across **varying strongly heterogeneous media** appears unoccupied. The two nearest works are
fixed-velocity (1910.00925) and trace-domain (2606.08711).

### 1.3 Long-time stability: the diagnosis in the literature matches our measurement

| Work | ID | Claim | Level |
|---|---|---|---|
| PDE-Refiner | arXiv:**2308.05732** | Large-scale analysis of rollout strategies identifies **neglect of non-dominant spatial frequency information, often high frequencies, as the primary pitfall** limiting stable accurate rollouts; fixes it with diffusion-style multistep refinement. | [abstract] |
| PDESpectralRefiner | arXiv:**2506.10711** | On harder PDEs, diffusion refinement can **over-degrade** high frequencies; releases the equal-weight-per-frequency constraint and adjusts in spectral space (blurring diffusion, v-prediction). | [abstract] |
| SGNO | arXiv:**2602.18801** | Structured spectral evolution for autoregressive rollouts; **GMean100 reduced by a median of 74.8%**, per-task 13.6–92.9%, **gains strongest on dispersive and transport-dominated tasks**. Code released. | [abstract] |
| TF-SNO | arXiv:**2606.21189** | Many spectral operators **apply a shared spectral response across rollout stages**, mismatching time-varying spectra in non-stationary systems; adds learnable time-frequency gating. | [abstract] |

**Why this matters for us specifically.** PDE-Refiner's identified pitfall is precisely what we
measure: our prediction carries **less** above-mode-16 energy than the truth in all three
families (0.1669 vs 0.1871 uniform; 0.0465 vs 0.0598 layered; 0.2100 vs 0.2299 marmousi). And
TF-SNO's criticism names a concrete defect in our decoder: we apply **one shared set of
spectral kernels across all frames**, while the coda is exactly where the spectral content
drifts.

**Hard compatibility limit on SGNO.** Its abstract states it is "designed for periodic linear
and semilinear evolution PDEs with **Fourier multiplier linear dynamics**", evaluated on
APEBench tasks in that regime. Our problem is **variable-coefficient** (the velocity model is
the input) with **CPML absorbing boundaries and a free surface** — not periodic, not a Fourier
multiplier. The 74.8% figure therefore **must not be quoted as applicable to us**.

### 1.4 Complex media

> *Hybrid operator learning of wave scattering maps in high-contrast media*,
> arXiv:**2602.11197** (2026-01). **[abstract]**

Decomposes the scattering operator into a **smooth background propagation** learned by an FNO
producing globally coupled tokens, plus a **high-contrast scattering correction** learned by a
vision transformer via attention. Reports substantially improved phase and amplitude accuracy
against standalone FNOs or transformers on high-frequency Helmholtz problems with strong
contrast, with favourable accuracy-parameter scaling.

This is the single most consequential entry in this audit, for two opposite reasons, both
recorded in Part 3.

### 1.5 Cost accounting: status

Kochkov (1.1) reports net speedup and is the strongest available precedent that the accounting
can close. For the **wave-equation** case specifically, no work retrieved in this audit reports
a defensible end-to-end cost account of coarse-solve-plus-correction against the fine solve.
1910.00925 is a proof of concept and makes no such claim; 2606.08711 benchmarks accuracy across
architectures rather than net cost. **Recorded as: not established for our setting.**

---

## Part 2 — Verification of quantitative claims already in the package

| Claim as it stood | Status | Evidence |
|---|---|---|
| Multigrid honest gain interval | **CORRECTED** earlier and re-affirmed here: the interval must be quoted against parameter-matched baselines separately (vs BFNO 104M, vs AFNO 58.6M, vs FNO 734M), never as a single "7–10×" | recorded in `EVIDENCE_INDEX.md` §0 row 6 |
| The two cited papers' tables are byte-identical | **WEAKENED** to the supported wording: one paper states the same USCT task is also benchmarked in the other | `EVIDENCE_INDEX.md` §0 row 7 |
| MIFNO reports relative L2 | **NOT RE-VERIFIED IN THIS AUDIT.** The package's claim is that it uses Kristeková GOF, reports surface wavefields only, and has 3.4M parameters. This audit did not retrieve MIFNO, so those three points remain as previously recorded and are **not** independently confirmed here. | — |
| FNO original Burgers/Darcy error magnitudes | **NOT RE-VERIFIED IN THIS AUDIT** | — |
| WNO accuracy and cost claims | **NOT RE-VERIFIED IN THIS AUDIT**; the cost figure was already deleted as unsourced | `EVIDENCE_INDEX.md` §0 |

Two rows are explicitly left unverified rather than quietly asserted. Anyone citing them must
return to the original source first.

---

## Part 3 — Novelty assessment and guidance for the paper

### 3.1 The direct threat

**arXiv:2602.11197 is a partial novelty threat to our mechanism narrative.** It publishes the
decomposition "smooth background plus learned high-contrast correction" for wave scattering in
strongly heterogeneous media, which is structurally the architecture whose correction branch we
report as collapsed. Two things must therefore change in how we write:

1. We **cannot** present coarse-plus-correction as novel. It is published, and it is reported
   to work in the high-contrast regime.
2. Our orthogonality result becomes **more** interesting rather than less, but only if framed
   correctly: they make the decomposition work with an **attention** correction pathway,
   whereas ours uses a **spectral** pathway and we measure it to be near-orthogonal to the
   residual (median cosine 0.0406, best achievable ratio 0.9984). The honest framing is a
   negative result about *one instantiation* of a decomposition that others have made work by
   different means — which incidentally points at the attention pathway as the substantive
   difference.

Note their setting is **frequency-domain Helmholtz**, ours is **time-domain full wavefield**.
That is a real distinction and should be stated, but it does not license claiming the
decomposition as ours.

### 3.2 What remains defensible

No work retrieved in this audit reports any of the following, which are the paper's actual
contributions:

- A **frequency-resolved and time-banded error-mass decomposition** used as a diagnostic, with
  the exact identity `full² = Σ s_b r_b²` verified to machine precision and used to derive a
  hard improvement ceiling.
- The **orthogonality identity** `min_s ‖e+s·r‖/‖e‖ = √(1−cos²(r,−e))` applied as a decision
  rule that bounds what a residual branch can buy independently of its gain.
- **Per-frame memorisation** quantified by transfer to disjoint frames of the same record
  (0.9841) against supervision density (0.5170 at 51 frames).
- A systematic **elimination protocol** with decision rules frozen before measurement, on
  single-shot full-wavefield operators over Marmousi-class media.

### 3.3 Instructions for the paper text

- **Do not** cite SGNO's 74.8% as applicable; state its periodic / Fourier-multiplier scope
  when citing it at all.
- **Do** cite 2602.11197 in Related Work and concede the decomposition explicitly.
- **Do** cite PDE-Refiner for the high-frequency-neglect diagnosis, since our own
  above-mode-16 measurement independently reproduces its signature.
- **Do** cite 1910.00925 as the nearest precedent on learned dispersion correction and state
  its fixed-velocity scope, which is what leaves our gap open.
- **Do not** cite Bar-Sinai or Kochkov as wave-equation precedent; they are 1-D and
  Navier–Stokes respectively. Cite them only for the coarse-grid-plus-learning accounting.
- **Do not** claim that no precedent exists for learned dispersion correction on full
  wavefields in varying media. The supportable wording is that this audit did not find one,
  with the three zero-hit queries recorded.
