# Deep Research Analysis: FNO Acoustic Wave Simulation

Date: 2026-07-02
Project: `/data/jiayh/FNO-Acoustic-Wave-Simulation`
Data: `/home/jiayh/Data/FNO-Acoustic-Wave-Simulation`

## Executive Summary

After comprehensive analysis of all experiment artifacts, source code, failure analyses, and metric patterns across 50+ experiment runs, I have identified **five interconnected root causes** that collectively explain why the project cannot currently train a successful 400×400×160 acoustic wavefield predictor. Below I present the evidence, root cause analysis, and concrete solutions.

---

## 1. The Spectral Truncation Crisis (Root Cause #1)

### Evidence: Resolution-dependent mode coverage collapse

| Resolution | H/2 | modes_x=16 coverage | Best val L2 | komega_high (uniform) |
|---|---|---|---|---|
| 32×32 | 16 | **100%** | 0.488 | 1.29 |
| 64×64 | 32 | **50%** | 0.444 | **5.07** |
| 128×128 | 64 | **25%** | 0.650 (plateau) | N/A (stuck) |
| 400×400 | 200 | **8%** | 1.001 (zero-pred) | N/A (failed) |

### Root Cause

The FNO's SpectralConv3d only learns weights for `modes_x × modes_z × modes_t` low-frequency Fourier coefficients. At 32×32, all frequencies are covered. At 400×400, **92% of spatial frequencies have zero learnable spectral weights**. The model relies entirely on the 1×1×1 Conv3d skip connection for high frequencies, which has no spatial mixing capacity.

### Why This Wasn't Obvious

The 54-frame 400×400 FNO training achieved val L2 = 0.0665. This worked because:
1. 54 time frames at stride ~3 means only early-time, low-frequency content
2. The full 160-frame evaluation reveals the high-frequency failure
3. Uniform media is most affected because clean wavefronts have sharp spectral ridges that get truncated

### Solution

We need **frequency-aware architecture** that doesn't rely solely on truncated spectral convolutions:

**Option A: Multi-Resolution FNO (MR-FNO)**
- Process at multiple spatial resolutions in parallel
- Low-res path: full spectral coverage (e.g., 32×32 with modes=16)
- High-res path: lightweight spatial convolutions for high-frequency residuals
- Fuse with learnable weights

**Option B: Factorized Spectral-Spatial Architecture**
- 2D spatial FNO (H×W) + 1D temporal transformer/RNN
- Spatial FNO can use modes=64 at 400×400 (still 16% coverage but 2D)
- Temporal processing is full-resolution
- Memory: O(H×W×width + T×width) instead of O(H×W×T×width)

**Option C: Adaptive Mode Expansion**
- Start with modes=16 and gradually expand to modes=32, 48, 64
- Use checkpoint migration to preserve low-frequency knowledge
- Requires memory optimizations (gradient checkpointing, mixed precision)

---

## 2. The Uniform komega_high Anomaly (Root Cause #2)

### Evidence

At 32×32×160 best checkpoint:
- uniform komega_high = 1.29
- marmousi komega_high = 0.82

At 64×64×160 best checkpoint:
- uniform komega_high = **5.07** (4× worse!)
- marmousi komega_high = 1.02 (only 1.25× worse)

### Root Cause: Resolution-Mismatch Aliasing Artifact

This is NOT a genuine high-frequency prediction failure. It's a **resolution mismatch artifact**:

1. At 32×32, the spatial downsampling low-pass filters BOTH reference and prediction, hiding high-frequency errors
2. At 64×64, the reference recovers sharp wavefronts (high spatial frequencies), but the FNO with modes=16 cannot represent them
3. Uniform media produce the sharpest, most spectrally-concentrated wavefronts → most affected
4. Marmousi's complex scattering naturally distributes energy across frequencies → less affected

### Verification Method

Compare 32×32 prediction upsampled to 64×64 vs native 64×64 prediction:
- If upsampled-32 has lower komega_high than native-64, the effect is confirmed as a resolution artifact
- This also means the 32×32 model is NOT genuinely better — it's hiding error through low-pass filtering

### Solution

- Report **resolution-normalized komega metrics**: band-limit the reference to the model's Nyquist frequency before computing spectral errors
- Use **multi-resolution evaluation**: evaluate all models at a common low resolution for fair comparison

---

## 3. The Dense 3D Hidden Tensor Memory Wall (Root Cause #3)

### Evidence

| Config | Hidden tensor | Memory | Status |
|---|---|---|---|
| 32×32×160, width=32 | [1,32,32,160,32] = 5.2M floats | ~2.6 GB | Works |
| 64×64×160, width=32 | [1,64,64,160,32] = 21M floats | ~4.0 GB | Works |
| 128×128×160, width=32 | [1,128,128,160,32] = 84M floats | ~10 GB | Works |
| 400×400×160, width=32 | [1,400,400,160,32] = 819M floats | **OOM (needs ~48 GB)** | Fails |
| 400×400×160, width=16, head=16 | [1,400,400,160,16] = 410M floats | ~23.4 GB | OOM+zero-pred |

### Root Cause

The dense 3D FNO stores activations as `[B, width, H, W, T]`. At 400×400×160, this is impossible at width ≥ 24 on a 24 GB RTX 3090. Reducing width to 16 makes it fit but destroys model capacity, causing zero-prediction collapse.

### Solution: Factorized Spatiotemporal Architecture

**Core idea**: Never store a dense `[B, width, H, W, T]` tensor.

**Architecture design**:
```
Input [B,H,W,T,C]
  → Spatial encoder (2D Conv/SpectralConv): [B,H,W,T,C] → [B,H,W,T,spatial_dim]
    - Process each time frame independently with 2D ops
    - Memory: O(H×W×spatial_dim) per frame
  → Temporal aggregator (1D Conv/Attention): [B,H×W,T,spatial_dim] → [B,H×W,temporal_dim]
    - Process each spatial point's time series
    - Or use linear attention / state space models
  → Spatial decoder (2D Conv): [B,H,W,temporal_dim] → [B,H,W,T]
    - Expand back to full spatiotemporal
```

**Memory estimate** (width=64):
- Spatial features: 400×400×64 = 10.2M floats = 41 MB per frame × 160 frames = 6.4 GB (process frame-by-frame)
- Temporal features: (400×400) × 64 = 10.2M = 41 MB
- Total: ~6.5 GB (vs 48 GB for dense 3D)

---

## 4. The DeepONet Zero-Prediction Root Cause (Root Cause #4)

### Evidence

All SDR-DeepONet variants converge to val L2 ≈ 1.0 (zero-prediction):
- factorized_temporal: val L2 1.001
- low_rank_temporal: val L2 1.000
- local_spatial: val L2 1.000 (overfit single-sample: 0.917)

### Root Cause: Global Pooling Destroys Spatial Information

The factorized_temporal branch does:
1. Spatial encoding → global average pool → single latent vector per sample
2. This latent vector conditions a temporal basis function
3. The output is `sum_r latent[r] × basis_r(t)` — identical for ALL spatial positions!

This means the model can only predict spatially-uniform wavefields. Since the target wavefield has spatial structure (wavefronts, reflections), the optimal uniform prediction is near-zero (minimizes L2 error when you can't capture spatial patterns).

The local_spatial branch preserves spatial dimensions (outputs `[B,H,W,R]` coefficients), but:
- The temporal basis is shared across all spatial positions
- At 32×32 resolution with small capacity, the model can't fit
- Single-sample overfit shows gradients flow but generalization fails

### Solution

- **Abandon global-pooling branch designs** for this task
- Use spatial-preserving architectures (convolutional, not pooling-based)
- The temporal basis should be spatially-conditioned, not global

---

## 5. The Missing Physics Training (Root Cause #5)

### Evidence

All successful training runs use ONLY supervised relative L2 loss. The physics losses (PDE residual, energy stability, receiver waveform, komega spectral) are implemented but **have never been used in any successful training run**. The current best 64×64 checkpoint uses komega band weighting in the loss but only with `komega_weight=0.12` as a spectral-band regularization — not a physics constraint.

### Root Cause

Physics losses require:
1. **Denormalization**: Convert model output back to physical units
2. **Accurate metadata**: dx, dz, dt_saved, source parameters
3. **Careful masking**: Source region, boundary regions
4. **Loss balancing**: Physics terms can dominate or vanish relative to supervised loss

None of these are fundamentally blocking — they're implemented but untested in training.

### Solution

- **Phase in physics losses gradually** after supervised baseline is stable:
  - Stage 1: Add receiver waveform loss (λ=0.05) — most direct physical constraint
  - Stage 2: Add spectral band loss (λ=0.02) — addresses frequency bias
  - Stage 3: Add PDE residual (λ=1e-4) — enforces wave equation
  - Stage 4: Add energy stability (λ=1e-3) — prevents late-time divergence

---

## 6. Recommended Path Forward

### Immediate (This Week)

1. **Implement Factorized 2D+1D Architecture** (addresses Root Causes #1, #3)
   - 2D spatial processor (FNO or Conv) per time frame
   - 1D temporal processor (attention or state-space) per spatial point
   - Target: 64×64×160 training at width=64 (3× current capacity)

2. **Implement Multi-Resolution FNO** (addresses Root Cause #1)
   - Low-res FNO (32×32, modes=16) for global wave propagation
   - High-res CNN (64×64 or 128×128) for local high-frequency detail
   - Zero-init high-res branch for safe migration from existing checkpoints

3. **Fix komega evaluation** (addresses Root Cause #2)
   - Add resolution-normalized band limits
   - Generate multi-resolution evaluation reports

### Short-term (Next 2 Weeks)

4. **Progressive Mode Expansion Curriculum**
   - 32×32 modes=16 → 64×64 modes=24 → 128×128 modes=32
   - Each stage doubles frequency coverage

5. **Activate Physics Losses**
   - Start with receiver loss on 64×64 baseline
   - Gradually add spectral and PDE losses

### Medium-term

6. **Full 400×400×160 Training**
   - Use factorized architecture at width=64
   - Progressive spatial resolution: 64→128→256→400
   - Target: val L2 < 0.15, komega_high < 2.0

---

## Appendix A: Key Metric Comparison

| Checkpoint | Resolution | val L2 | komega_high (uniform) | komega_high (marmousi) |
|---|---|---|---|---|
| FNO54 best | 400×400×54 | 0.067 | N/A (54-frame) | N/A |
| 32×32 continue | 32×32×160 | 0.488 | 1.29 | 0.82 |
| 64×64 refine3 | 64×64×160 | 0.444 | **5.07** | 1.02 |
| 128 shortval | 128×128×160 | 0.650 | N/A | N/A |
| 400 width16 | 400×400×160 | 1.001 | N/A | N/A |

## Appendix B: Per-Time Error Analysis

At the best 64×64 checkpoint:
- **Early times (t=0-4)**: Extreme relative L2 (26→2.1) due to tiny target norms (~0.002-0.09)
- **Mid times (t=5-30)**: Stable at 0.22-0.30, the model's best regime
- **Late times (t=150-159)**: Growing to 1.6-2.0, phase error accumulation
- **Worst active time**: t=159 (last frame), mean relative L2 = 1.99

The late-time error growth is consistent with:
1. Phase velocity errors in the FNO spectral representation
2. Accumulated dispersion from truncated frequency modes
3. Missing physics constraints (no energy conservation or PDE residual)

## Appendix C: Architecture Comparison

| Architecture | Memory Scaling | Spectral Coverage | Status |
|---|---|---|---|
| Dense 3D FNO | O(H×W×T×W) | modes/(H/2) | Works at ≤64×64 |
| Factorized 2D+1D | O(H×W×W + T×W) | Full temporal, partial spatial | **Recommended** |
| Multi-Resolution | O(H_low×W_low×T×W + H×W×W) | Full at low-res, CNN at high-res | **Recommended** |
| Autoregressive | O(H×W×W) per step | Full spatial | Slow inference |
| DeepONet (global pool) | O(H×W + T×R) | None (destroyed) | **Abandoned** |
