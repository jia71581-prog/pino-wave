# Local-Propagation Coarse-Field Replacement — Architecture Options & Selection

Date: 2026-07-23
Project: `FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2` (2D acoustic operator, author Yang Cui)
Author of this doc: autonomous architecture phase following the capacity-ladder verdict.
Status: options + selection. Integration hooks finalized against the code map (see §8).

## 1. Why we are changing the architecture (evidence, not intuition)

The capacity ladder (`results/capacity_ladder/VERDICT.md`, 12 rungs) established, on the
deterministic 3-record overfit probe with identical scoring:

- **Decoder side is not the bottleneck.** dense_depth 8→24, spectral_rank 112→224 give
  ~0 gain (315M params is *worse*); dense_modes 32→64→101 gives only a tiny real gain
  (best_fixed 0.13736→0.13474, ~0.0026 abs / ~1.9% rel; all_saved 0.13381→0.13292) —
  far short of the 0.10 target, marmousi essentially frozen at 0.165.
- **Backbone width is the only sizeable lever, and it saturates** at ~0.12–0.13 (w64→w192
  = 0.169→0.125; w192→w256 only −0.004), still above target.
- **The floor is the coarse MIONet field.** With the warm-started front-end the decoder
  correction contributes 0% (coarse == corrected == 0.18); width helps only by improving
  the coarse field. `uniform` coarse error is 0.47–0.51 regardless of width — the *most
  regular* wavefield (clean concentric circles) is the coarse field's *worst* case.
- **Visualization (w192, uniform):** true = clean expanding circular front; coarse MIONet =
  smeared front + global spurious concentric ringing (Gibbs); the wide decoder only partly
  cleans it. Raising decoder spectral modes did not remove the ringing.

This is independently corroborated by the author's own 2026-07-02 deep-research analysis:
- Root Cause #1 (spectral truncation): "**uniform media most affected because clean
  wavefronts have sharp spectral ridges that get truncated**."
- Root Cause #4 (global pooling): DeepONet global-pool branches can only produce
  spatially-uniform fields → "**abandon global-pooling branch designs**; use
  spatial-preserving (convolutional); the temporal basis should be spatially-conditioned."
- Appendix C: Dense-3D "works ≤64×64"; **Factorized 2D+1D and Multi-Resolution (low-res
  global + high-res local CNN) = "Recommended"**; Autoregressive = "slow inference";
  DeepONet global-pool = "Abandoned".

**Diagnosis in one line:** the coarse field is a *global low-rank product* (MIONet /
DeepONet) with a *truncated global spectral* backbone. That structure cannot place a sharp,
**translation-invariant** traveling wavefront at an arbitrary location without Gibbs
ringing, because it must encode absolute wavefront position in a global low-rank basis. The
fix must restore **locality + translation-equivariance** in the field generator.

## 2. Hard constraints the new module must respect (from the data + V4 contract)

Empirical data contract (`dataset_v1.h5`, acoustic_lwc84_2km_401x401_to_201_v1):
- `wavefield [4003, 401, 201, 201]` float32, axis order NTZX.
- 401 stored times, **0–1 s, dt = 2.5 ms uniform**; grid 201×201, **dx = dz = 10 m**, 0–2000 m.
- families uniform/layered/anomaly/marmousi (probe scores uniform/layered/marmousi).
- vmax 7963 / vmin 1421 m/s; f0 ∈ [8, 30] Hz; t0 ∈ [0.05, 0.19] s; amplitude ≡ 1.

Physics scales that dictate the receptive-field design:
- Sharpest spatial feature ≈ min wavelength vmin/f0max ≈ 47 m ≈ **4.7 cells** (ppw ≈ 4.7).
- Wavefront motion per saved step ≈ vmax·dt/dx ≈ **2 cells/frame**.
- Wavefront radius over 1 s reaches the whole domain (≫201 cells) → **the front can sit
  anywhere**: the generator needs a *global* receptive field to place it, but *local*
  translation-equivariant structure to render its ~5-cell-wide shape sharply.

Contract constraints (`saved-time-v4-design.md`):
- **Arbitrary saved-time query is a hard requirement** (public API queries any of the 401
  stored indices; interpolated-time accuracy is explicitly out of scope, judged at stored
  indices only).
- **Full-3D space-time FNO was rejected** (16.2M output values/record → activation memory;
  a single-frame query would waste the whole trajectory).
- **Autoregressive / Koopman rollout was rejected twice** — it accumulates error over long
  horizons and does not naturally provide arbitrary-time output.
- Target: held-out relative L2 < 10 %, every family < 12 %, late-time < 15 %.
- Memory: full-dataset training on 24 GB 4090s; current micro=12 full-model peak ≈ 20 GB.

## 3. Option 1 — Local Propagation Field (LPF) coarse operator  **[SELECTED]**

Replace the global-low-rank coarse MIONet field with a **spatially-preserving,
translation-equivariant, time-conditioned field generator** producing the 201×201 coarse
field for a requested saved-time index directly (no rollout).

Structure (a conditioned U-Net / multi-scale conv, *not* a global product):
- **Backbone:** multi-scale conv U-Net over the 201×201 grid (e.g. 201→101→51→26 down,
  symmetric up, skip connections). Local 3×3/5×5 convs render ~5-cell fronts sharply with
  no spectral truncation → no Gibbs; pooling gives the global receptive field needed to
  place a front anywhere. Optionally a *local* (windowed) spectral block at the coarsest
  scale for smooth far-field / long-range DC, but **no global low-rank pooling**.
- **Conditioning (keep everything the current model already computes):**
  - medium pyramid features (velocity encoder) sampled per scale — spatial, translation-
    equivariant;
  - the existing **local propagation bundle** as *input channels* on the grid: travel time
    T(x,z), retarded time τ = t − t0 − T, causal gate around τ=0, sin/cos(2πf0τ),
    multi-scale Gabor envelopes — this injects the physically-correct wavefront *location*
    and *phase* as dense fields, so the conv net only has to learn the local *shape*;
  - source conditioning via FiLM from the source latent (x_s, z_s, f0, t0), and explicit
    multiply by source_amplitude (linearity; amplitude≡1 here but keep the structure);
  - **time conditioning** via the saved-index embedding + continuous-time features (as V4
    already does), so a single forward renders one requested frame.
- **Residual corrector kept:** the existing factorized spectral corrector stays as a
  *residual* on top of the LPF coarse field (it is cheap and gave a small real gain). The
  LPF field replaces only the coarse term that the ladder proved is the floor.
- **Physics gates (from continuous-wave-operator design):** causal gate enforces p≈0 before
  first arrival; optional free-surface gate p(z=0)=0. These are structural, not learned.

Why this is the evidence-maximal, minimal-risk choice:
- It is exactly the author's "Multi-Resolution / spatial-preserving local CNN"
  recommendation applied to the *specific* module the ladder isolated as the floor.
- It **honors the user's "causal local-propagation" priority** (locality, causal gate,
  translation-equivariance are the physically-meaningful essence) **while respecting the
  hard arbitrary-saved-time contract** that made pure autoregression a non-starter (time-
  conditioned direct prediction, not frame recurrence → no rollout error, any frame in one
  forward).
- It reuses ~all existing plumbing: front-end encoders, local-propagation bundle, data
  loader, the deterministic 3-record probe, the scoring accumulator, the full-support
  trainer. The change is contained to the coarse-field module + its construction flag.

Risks: (a) U-Net over 201×201 per requested frame adds activation memory — mitigate with
activation checkpointing and microbatch=1 in the probe (already standard here); (b) the
translation-equivariant conv may under-fit long-range/curved fronts in strong-gradient
media (marmousi) — mitigate with the coarsest-scale windowed-spectral block + medium-pyramid
conditioning; (c) placing the front relies on the travel-time bundle being accurate — it is
the eikonal field the author already validated.

## 4. Option 2 — Windowed 3D space-time UFNO  **[strong control / fallback]**

Process a *short window* of K consecutive saved frames jointly with a 3D UFNO (3D spectral
conv + a 3D U-Net local path for high frequencies), conditioned on velocity + source. Query
a saved index by selecting its window. Windowing (K≈8–16, not 401) bounds the activation
memory that got full-3D rejected, while the U-Net local path supplies the sharp-front
capacity the global spectral part lacks.

Pros: enforces short-horizon temporal coherence directly; `UFNO/ufno3d.py` may be reusable.
Cons: heavier memory than Option 1; window boundaries need handling; still an indirect fix
to the "sharp front" problem (relies on the U-Net path). Kept as the **strong control** to
run if Option 1's overfit probe stalls, and as the "3D time" arm the user asked to keep.

## 5. Option 3 — Autoregressive local stepper  **[fallback; author-rejected]**

Predict frame t+dt from frame t (+ velocity + source) with a local conv operator; train
with teacher forcing + short rollout + a PDE/energy consistency loss between adjacent saved
frames. Most physically-grounded (finite-difference-like; per-step RF only ~5 cells since
the front moves ~2 cells/step), strongest causality.

Cons (why it is a fallback, not the pick): reintroduces **rollout error accumulation over up
to 401 steps** and **does not give arbitrary-time output** (a single late-frame query must
roll from t=0) — the two reasons the author rejected it twice. Only revisit if both Option 1
and Option 2 saturate above target, and even then likely as a *refiner* over Option 1's
direct prediction rather than a from-scratch stepper.

## 6. Selection

**Option 1 (Local Propagation Field coarse operator).** It is the direct structural fix to
the isolated floor, maximally reuses validated plumbing, satisfies every hard contract
constraint, and honors the user's local-propagation priority. Option 2 is the pre-registered
strong control/fallback; Option 3 is a later refiner if needed. This ordering is allowed to
be overturned by the overfit-probe evidence (Gate 2).

## 7. Gates (stop-gates, not success theater)

1. **G1 smoke:** shapes for single/batched sources; nonzero receiver signal after first
   arrival and ≈0 before (causal gate works); finite forward/backward; correct physical
   units (pressure normalization round-trips); handles partial-saved-time sampling.
2. **G2 overfit probe:** same deterministic uniform/layered/marmousi triplet, same scoring
   and target as the capacity ladder (`diagnose_capacity_ladder_overfit.py` path), at a
   **parameter-comparable** size to the w128 baseline. Pass = agg < 0.10 AND every family
   < 0.12. Fail → diagnose (front placement? shape? conditioning? loss?) and iterate the
   module/loss/time-conditioning; do not stop at a failure report.
3. **G3 full-dataset short pilot:** independent validation set, same protocol/census as the
   current baseline; require the validation curve to *clearly* drop and beat the v63=0.47 /
   width-saturation reference, with no leakage/OOM/NaN. "Process started" ≠ success.
4. **G4 long training:** 4-GPU, screen/nohup setsid/torchrun that survives the terminal;
   persist frozen config + data manifest digest + seed + code snapshot + best/last
   checkpoints + logs. Disk: reclaim capacity-ladder intermediate `.pt` first (best+last
   retention), keep checkpoints bounded.
5. **G5 verify twice:** PID + children alive, 4-GPU mem/util, log steps increasing, first
   metrics/checkpoint readable; write RUN_STATUS.md with PID/cmd/cwd/log/config/ckpt paths.

## 8. Integration points (against the code map)

**What produces the coarse field today.** `PhaseAlignedMIONetFusion.forward`
(`grouped_ufno_mionet_v3/model/fusion.py:89-125`) computes the global low-rank product
`product = velocity_rank * source_rank * travel_rank * trunk_rank` (fusion.py:110) →
`coarse = coarse_scale * product.sum(-1)/sqrt(rank)` (fusion.py:111). It is *point-wise*:
V4 calls it with the whole grid flattened to points. In V4 the coarse field surfaces in
`SavedTimePhaseOperatorV4._dense_block_with_anchor_increment_and_routing`
(`saved_time_phase_operator_v4/operator.py:181-191`):
`fused = self.fusion(...); coarse = fused.normalized_pressure.reshape(records, count, H, W)`.

**How it is consumed.** `PropagationConditionedDenseDecoder` (`decoder.py:86`) lifts it
(`flat = flat + self.coarse_lift(coarse.reshape(records*count,1,H,W))`, decoder.py:301) and
adds it back as the anchor: `anchor = coarse + correction_scale*correction + temporal +
expert_correction` (decoder.py:341-346). Scoring (`_evaluate_triplet`) takes
`dense_normalized_with_coarse(...) -> (corrected, coarse)`, prediction=corrected,
reference=coarse.

**The already-computed physics conditioning I will reuse.** `dense_propagation_features`
(`saved_time_phase_operator_v4/features.py:12`) yields a 12-channel `[record, time, 12, z, x]`
grid field (retarded τ = t−t0−T, distance, path/endpoint velocity, causal sigmoid gate,
sin/cos phase, 4 Gabor envelopes) — already injected in the decoder (decoder.py:302-313).
Medium pyramid: `pyramid[0] = [medium, width, 201, 201]` (+ downsampled levels). Source:
`map_field [record, 64, 201, 201]`, `hidden [record, 64]`. Saved-time embedding:
`nn.Embedding(saved_time_count, width)`.

**Plan (flag-gated, non-breaking).**
1. New module `LocalPropagationFieldGenerator` (`saved_time_phase_operator_v4/local_field.py`):
   grid-native conditioned U-Net. Inputs per requested frame: medium pyramid features,
   the 12-channel propagation bundle, source `map_field`+FiLM(`hidden`), saved-time
   embedding. Output: coarse field `[records, count, H, W]` in normalized pressure, with the
   causal gate applied structurally. No global pooling; local convs + multiscale down/up.
2. Wire into `SavedTimePhaseOperatorV4.__init__` behind a constructor flag
   (`dense_local_field=...`), construct only when on. In the dense block, when on, set
   `coarse = self.local_field(...)` instead of the reshaped fusion product; keep the
   factorized spectral corrector as the residual on top. When off → today's behavior exactly
   (all existing runs/tests unaffected).
3. Extend `ProbeVariant` (`saved_time_phase_operator_v4/probe.py`) with the flag and thread
   it through `_model` (`scripts/train_saved_time_v4_probe.py`), so the capacity-ladder probe
   builds it via the same path and scores it identically.
4. Add a CLI flag to `scripts/diagnose_capacity_ladder_overfit.py` to toggle the local field,
   so Gate 2 runs are parameter-comparable to the w128 baseline under identical scoring.

**Units/contract:** output normalized pressure = Pa/(pressure_scale·amplitude); keep the
free-surface factor `tanh(z/20)²` (operator.py:212-213); query only stored saved-time
indices (no interpolation) — all preserved by producing the same `coarse [records,count,H,W]`
tensor the decoder already expects.
