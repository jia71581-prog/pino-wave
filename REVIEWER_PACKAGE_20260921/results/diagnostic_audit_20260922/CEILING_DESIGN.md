# R14 — variant T forward oracle ceiling: frozen design

Written and frozen **before** any number from this probe was computed.
Role: **DIAGNOSTIC, first-order decision aid. Not a gate, not a falsification.**

---

## 1. What is being measured

The dense decoder returns (`model/dense.py`, anchor tree
`research/coda_round6_density_20260919/snapshot`):

    y = coarse + s * r,            s = correction_scale  (anchor 29359: 1.0425e-04)
    prediction = y * surf,         surf = free_surface_factor(z) = tanh(z/20)^2

With defect `e` and branch output `r`, the R6 alignment probe measured the rank-1
optimum

    min_s ||e + s r|| / ||e||  =  sqrt(1 - cos^2(r, -e))

(median cos = +0.0406, median best ratio = 0.9984, 144 frames).

This probe computes the **exact B-dimensional generalisation**: split `r` by radial
wavenumber into `r_1..r_B`, then solve, **independently per frame**,

    min_{g_1..g_B} || e + sum_b g_b r_b ||        (closed form / lstsq)
    best_ratio    = || e + sum_b g_b r_b || / ||e||

`g` is free to differ per frame, i.e. free in `t` with no parameterisation at all,
which is strictly looser than variant T's `g = 1 + A phi(t)` with a fixed `phi` and a
single linear layer. `g_b` also absorbs `s`, so a shut gate cannot depress this number.

### Band decomposition (hard self-check)

Per frame: `Rk = rfft2(r, norm='ortho')` in float64, masks over
`rho = hypot(fftfreq(nz,dz)[:,None], rfftfreq(nx,dx)[None,:])` (cyc/m, the **same**
`rho` construction as `grouped_ufno_mionet_v3/eval_metrics.py:frequency_metrics`),
each masked spectrum `irfft2`-ed back. The masks partition the rfft2 half-plane and
`rho` is invariant under the Hermitian partner `(kz,kx) -> (-kz,-kx)`, so

    sum_b r_b == r     to float64 round-trip precision

is an identity, and it is **asserted** (abort above 1e-10 relative).

## 2. Two spaces, both reported

| space | defect `e` | columns |
|---|---|---|
| `decoder` (R6-compatible) | `coarse - truth` | `r_b` |
| `deployed` (loss space) | `coarse*surf - truth` | `r_b * surf` |

`decoder` exists for continuity with the 0.9984 self-check. `deployed` is the space the
training loss is actually computed in, so **`deployed` carries the verdict**.
`surf` is a depth taper that differs from 1 only in the top ~5 rows of 201, so the two
are expected to be close; if they are not, see rule D below.

`r` is captured **twice**: (a) exactly, as the output of the `decoder.output` conv via a
forward hook (this is `correction` before `s`); (b) by the R6 recipe `(y - coarse)/s`.
With `||s*r*surf||/||coarse*surf|| ~ 9.3e-06` (R9), recipe (b) loses ~1e-2 relative
precision to float32 cancellation, so (a) is primary and the max relative deviation
between them is recorded. The 0.9984 self-check is evaluated on **both**.

## 3. Bin schemes

`nested` (primary) — a strict refinement hierarchy, so `best_ratio` must be monotone
non-increasing in B and that monotonicity is a free self-check:

| B | edges (cyc/m) |
|---|---|
| 1 | 0, inf |
| 2 | 0, .015, inf |
| 4 | 0, .005, .015, .03, inf  — **exactly `eval_metrics.FREQUENCY_EDGES`** |
| 8 | each B=4 band bisected; the open band `[.03, inf)` is bisected at `(.03+rho_max)/2` |
| 16 | each B=4 band split into 4 equal parts, same treatment of the open band |

`equalwidth` (secondary robustness, B=4 and B=16 only) — equal width in `|k|` over
`[0, rho_max]`, which is the binning **variant T itself uses**
(`time_spectral_gate.radial_bin_maps`), to test whether the ceiling depends on the
scheme rather than on B.

## 4. Calibre discipline (the two historical errors this project paid for)

- **GLOBAL** (`active_horizon_s`, `dense_time_steps`, `ic_frames`) read from
  `l2_lwc84_ddp4_20260915_v1/formal/run_identity.json -> effective_config`, never from
  `config.py` defaults. Hard assertion: `active_horizon_s != 0.60` (the code default) and
  `== 1.0`. The YAML that builds the model is cross-asserted against it.
- **PER-RECORD** `source_t0_s = record.source_parameters[3]`, never `time_s[0]` and never
  a constant. Onset via `evaluation.resolve_onset_frame`, future window via
  `evaluation.future_window` — the production functions, not a hand-rolled `arange`.
- Hard assertion: **per-family numbers must not be identical** and the 12 onsets must not
  all be equal; identical per-family output is the signature of a pinned `t0`.
- Frame set cross-asserted frame-for-frame against `r6_alignment/ALIGNMENT_PROBE.json`.

Evaluation window is the **full future window** (`onset+ic_frames .. end`), which is what
`evaluation.future_window` and `eval_metrics.evaluate` use and what R6 used.
`active_horizon_s` is a *training-time sampler* calibre; it is read, asserted and recorded
but does **not** truncate this evaluation window. The frame index at which the horizon
ends is recorded per record so the distinction is auditable.

## 5. Frozen decision rule

Primary statistic: **per-family median of `best_ratio`, `deployed` space, `nested` scheme,
at B=4 and B=16**, over each family's 48 frames.

- **A — NO ARM.** If all three family medians at **both** B=4 and B=16 are `>= 0.99`
  (i.e. the whole band x time diagonal-gain family buys `< 1 %`), variant T is judged
  **not worth an arm**, same mechanism class as the attention variant. The report must
  state which of the two causes it is, decided by the fitted gains:
  - if `median |g_b|` is `>> s = 1.04e-04` (say `> 10 s`) the gate factor `s` is shut but
    the **direction is still nearly orthogonal**, so reopening `s` would not help either;
  - if `median |g_b| ~ s`, the gate is already where the optimum is.
- **B — WORTH AN ARM.** If any family median at B=16 is `< 0.99`, report the gain at each
  B and conclude an arm is worth pre-registering, with a step-count order of magnitude.
  Strength requires the gain in `>= 2` families or in marmousi (the binding family).
- **C — MIDDLE.** Anything else is reported as middle, unrounded.
- **D — DOWNGRADE.** If `decoder` and `deployed` family medians differ by `> 0.005` at
  B=4 or B=16, the verdict is downgraded to **C** and the discrepancy is flagged, because
  then the surface taper, not the band structure, is driving the number.

### Self-check gates (failure => artifact stamped VOID, no verdict)

1. `decoder`-space, B=1, overall median best ratio in **[0.9979, 0.9989]** (R6's 0.9984
   +/- 5e-4), under at least the R6 recipe for `r`.
2. `max_b ||sum_b r_b - r|| / ||r|| < 1e-10` over every frame and every scheme.
3. Per-frame monotonicity of `best_ratio` along `nested` B = 1,2,4,8,16: no increase
   beyond 1e-6.
4. 12 onsets not all equal; per-family medians not all identical.
5. Model parameters bitwise unchanged (no optimizer step, forward only).

## 6. Boundary declaration (must be carried into every quotation of these numbers)

- This oracle is the ceiling of **a band x time diagonal gain family acting on the FINAL
  correction output spectrum**. The real variant T gate sits at an **intermediate** layer,
  with further spectral blocks, a 1x1 output conv and nonlinearities after it. **The two
  are not the same function family**: the real gate can produce effects the output-side
  family cannot (they are mixed by the later layers), and the output-side family can
  produce effects the real gate cannot. Therefore this is a **first-order decision
  diagnostic, NOT a strict upper bound and NOT a falsification.** A "no arm" verdict here
  means "the evidence does not justify the cost", never "proved impossible".
- Variant T's gate is also per-channel (`g(c, bin, t)`, width 128) and acts only on the
  **retained** modes (radial cutoff 44); this oracle is channel-free and acts on the whole
  rfft2 plane. Neither dominates the other.
- **This is a diagnostic, not a gate judgement.** Single-record / single-frame ratios must
  **never** be placed beside family-level gate means
  (`research/RULED_OUT_ROUTES_20260919.md` 5.1, same prohibition).
- **train** split, and the marmousi family contains 50 m translated twin records (the
  split leakage disclosed in the paper). The word **generalisation** must not be used of
  any number here.
- Every ratio is fitted and evaluated on the same 40401 pixels. With `B <= 16` parameters
  against 40401 pixels the in-sample optimism is `~ B/P <= 4e-4` in squared terms, which
  biases the ceiling **downwards** (optimistically), so it cannot manufacture a "no arm"
  verdict — it can only manufacture a false "worth an arm".

## 7. Scope of execution

Forward only. No optimizer step, no `--resume`, no training arm, nothing written outside
`/root/autodl-tmp/staging/r14_variantT_ceiling_20260921/`. The code tree is read-only and
is `coda_round6_density_20260919/snapshot` — deliberately **not**
`coda_round7_hinge_20260919/snapshot`, which another agent is editing. Config, per-record
`source_t0_s` with its source field name, and tree file sha256s are embedded in the output
JSON.
