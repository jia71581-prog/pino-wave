# Source-Aware Temporal-Frequency Helmholtz Operator — Design

Date: 2026-07-31
Project: `FNO-Acoustic-Kakeya-20260731` (2D acoustic operator, author Yang Cui)
Base: Option B Local Propagation Field branch (0.47 → 0.29). Successor direction after the
Kakeya line closed (see `2026-07-31-kakeya-anisotropic-phase-space-design.md`; the disease is
*temporal*, not spatial-frequency geometry).
Status: DESIGN, backed by three zero-training probes on real data (all positive).

## 0. What we are solving — target domain vs internal representation

**We solve the TIME-DOMAIN acoustic wave equation WITH a source.** Deliverable, scoring, and
all comparisons stay in the time domain: given (velocity field, source), output p(x,z,t) over
the 401 stored frames. This never changes.

The frequency domain here is an **internal representation, not the solve target** — exactly
like doing convolution via FFT: you want the time-domain result, you route through frequency
because it is cheaper/cleaner there, and you inverse-FFT back. Nobody calls FFT-convolution
"solving a frequency-domain problem." Same here.

```
input:  c(x,z), source (x_s, z_s, f0, t0) + stored wavelet w(t)
                    │
 [internal repr]  time→freq:  P̂(x,ω) = ŵ(ω) · Ĝ(x; x_s, ω, c)
                    │            └ source factor (KNOWN)   └ network learns this
 inverse FFT (401)  ▼
output: p(x,z,t)  ← time-domain sourced wavefield — the deliverable
                    │
 scoring ──────────┘  all in the time domain, identical to today
```

## 1. The exact source factorization (why this is principled, not a trick)

Time-domain sourced acoustic wave equation:
    (1/c²) ∂_tt p − ∇²p = δ(x−x_s) w(t; f0, t0).
Fourier in time (the equation is LINEAR in the source) gives the inhomogeneous Helmholtz eq:
    (∇² + ω²/c²) P̂(x,ω) = −δ(x−x_s) ŵ(ω),
so **exactly**:
    P̂(x,z,ω) = ŵ(ω) · Ĝ(x,z; x_s, ω, c).

Consequences for the four source parameters (x_s, z_s, f0, t0):
- **f0** enters ONLY through the wavelet spectrum ŵ(ω). In this dataset ŵ is even stored
  (`source_wavelet [4003,401]`), so ŵ = FFT_t(wavelet) is KNOWN per record — **zero learning.**
- **t0** is a pure linear phase e^{−iωt0} inside ŵ — **zero learning.**
- The network learns ONLY Ĝ(x; x_s, ω, c): depends on velocity and source *position*, not on
  f0/t0. Two of four source parameters are analytically stripped.

This is impossible in the time domain, where the source is convolved into and entangled with
the wavefield.

## 2. Zero-training evidence (three probes, real data, all positive)

### Probe #1 — `docs/kakeya/probe_source_freq_helmholtz.py` (Q1/Q2/Q3)
Per family (uniform/layered/anomaly/marmousi), 2 records each:
- **Q1 source factorization conditioning:** ŵ has no zeros in the field's 99%-energy band
  (Wcond 3–13); dividing P̂/ŵ is numerically stable.
- **Q2 K-frequency reconstruction (the <5% oracle re-verified WITH the source):** keep K
  complex frequency slices, inverse-FFT to 401 frames. **All families reach <5% by K≈48–64**,
  including marmousi (#2100 K48=3.38%, K64=2.78%); low-freq layered hits <5% by K16.
- **Q3 eikonal phase model:** phase of Ĝ/ŵ is **linear in ω** at energy-dominant pixels,
  slope = traveltime T(x). **R² > 0.96 all families, marmousi up to 0.998** — a single
  eikonal phase A(x)e^{iωT(x)} suffices (the feared multipath scramble does NOT dominate here).

### Probe #2 — `docs/kakeya/probe_amplitude_learnability.py` (M1/M2)
The only thing left to learn is the complex amplitude Ĝ_k(x). Is it low-complexity?
- **M1 spatial smoothness:** raw |Ĝ_k| is **86–98% low-wavenumber energy** all families — the
  amplitude is smooth (unlike the sharp time-domain wavefront that caused Gibbs / the uniform
  0.47 floor). A low-mode / conv generator can render it.
- **M2 cross-frequency low rank:** the K=64 amplitude fields, stacked, have **rank@95% ≤ 6**
  (anomaly/marmousi rank = 1). The frequency fields share a tiny common basis → a few basis
  fields + per-frequency mixing suffice; parameter budget is small.
- **M3 (de-phasing gain) — inconclusive, retracted:** the hypothesis "explicit eikonal
  de-phasing makes the amplitude smoother" FAILED (gain <1 all families). Root cause is the
  probe's per-pixel phase-slope T estimate being noisy, not physics. **Conclusion: raw Ĝ_k is
  already smooth and low-rank; no explicit de-phasing module is needed** — a simplification.

**Net:** every load-bearing assumption is data-supported. The network must produce a handful of
smooth, low-rank complex amplitude fields per frequency; source and time-shift are analytic;
inverse-FFT returns the time-domain sourced field. This is the strongest-evidenced direction in
the project's history and directly dissolves the spatial "sharp wavefront placement" floor by
moving it into smooth phase.

## 3. Architecture — SourceAwareHelmholtzOperator

Output contract unchanged: p [records, T=401, H, W] time-domain (or any queried saved-time
index via the same inverse transform evaluated at that t).

Forward:
1. **Frequency selection.** Fix K≈64 angular frequencies ω_k (data-driven: the band carrying
   99% wavelet energy; can be per-record or a fixed union band). Store ŵ(ω_k) per record from
   the wavelet (analytic, no params).
2. **Green's amplitude generator** G_θ(c, x_s) → {Ĝ_k(x)}_{k=1..K}, complex [K,H,W]. Given M2,
   parametrize as a **low-rank factorization**: R≤8 shared complex basis fields B_r(x) (from a
   conditioned conv/U-Net over velocity, reusing the Option B LPF backbone) + a small
   per-frequency complex mixing matrix M[K,R] conditioned on ω_k and source position. This is
   the entire learnable core and it is tiny (R basis fields + K×R mixings).
3. **Optional eikonal phase prior.** Multiply B_r or Ĝ_k by e^{iω_k T(x)} using the project's
   existing `RayTravelTime` T(x) as a *fixed* carrier, so the generator only learns the smooth
   residual amplitude (Q3 says one phase suffices). Gate this behind a flag — M3 showed naive
   de-phasing can hurt, so make it opt-in and ablate.
4. **Source assembly (analytic):** P̂_k(x) = ŵ(ω_k) · Ĝ_k(x). Pure known scalar per (record,k).
5. **Inverse transform to time (fixed, differentiable):** p(x,t) = Σ_k P̂_k(x) e^{iω_k t}
   (real part; a fixed inverse-DFT matrix [K,T]). Any saved-time index is one row of this
   matrix — arbitrary-time query for free, no autoregression.

Why this respects every hard constraint:
- Time-domain output, arbitrary saved-time query, no rollout — inverse-DFT gives all frames or
  any single frame in one shot.
- Reuses the Option B LPF conv backbone for the basis generator and the existing eikonal T.
- Memory: K×R complex fields, R small; far below the full 3D space-time tensor that was rejected.

## 4. Gates (LPF ladder, pre-registered)
- **G1 smoke:** shapes; finite fwd/bwd; inverse-DFT of analytic P̂_k reproduces a held wavefield
  to the Q2 K-truncation floor (sanity: the fixed transform is correct).
- **G2 overfit probe:** 3-record uniform/layered/marmousi, same scoring/target as the capacity
  ladder (agg<0.10, every family<0.12). Compare against the LPF 0.29. **Memory/ops:** use
  `--microbatch-records 1` + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, pin to one GPU
  (lessons from the Kakeya G2 run).
- **G3 short pilot** must beat LPF 0.29 on held-out; **G4 long; G5 verify twice.**

## 5. Risks / falsification hooks
- **Rank/phase generalization:** M2 rank≈1 and Q3 R²≈1 are measured per-record from the true
  field. The open question is whether G_θ can *predict* those smooth low-rank amplitudes from
  (c, x_s) alone. Probe #3 (next) should test cross-record amplitude predictability, or go
  straight to the G2 overfit which answers it directly.
- **Marmousi multipath tail:** Q3 R² is high on energy-dominant pixels; weak late multiples may
  live in the residual. If G2 marmousi stalls, add a second phase branch Σ_j A_j e^{iωT_j}
  (two arrivals) before abandoning.
- **Fixed vs per-record frequency band:** a fixed union band wastes freqs on narrowband
  records; a per-record band complicates batching. Decide at G2; log which was used.
- If G2 does not beat 0.29 at matched params → the amplitude is not *predictable* even though it
  is *simple*; record the negative result. Smoothness/low-rank of the target ≠ learnability of
  the map (the same distinction that closed Kakeya).
