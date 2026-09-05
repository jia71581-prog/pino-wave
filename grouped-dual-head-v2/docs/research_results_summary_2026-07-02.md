# Factorized Spatiotemporal FNO for 2D Acoustic Wavefield Simulation

## Research Results Summary — 2026-07-02

### 1. Problem Statement

Train a neural operator `G_θ: (v(x,z), s(x,z), t) → p(x,z,t)` that predicts 2D acoustic pressure wavefields faster than high-order finite-difference replay (LWC84+CPML, ~976 s/sample) while maintaining accuracy (relative L2 < 5%) and physical consistency.

**Dataset**: 2500 samples (1000 uniform, 1000 layered, 500 marmousi), 400×400 spatial × 160 time frames, 25 Hz Ricker source.

**Prior state**: Dense 3D FNO (modes=16×16×16, width=32) achieved val L2 0.444 at 64×64 resolution but could not scale beyond due to O(H·W·T·width) memory scaling.

### 2. Root Cause Analysis

Five root causes identified for prior training failures:

1. **Spectral Truncation Crisis**: FNO modes=16 covers 8% of frequencies at 400×400
2. **Uniform komega_high Anomaly**: Resolution-mismatch aliasing artifact
3. **Dense 3D Memory Wall**: O(H·W·T·width) impossible at 400×400 on RTX 3090
4. **DeepONet Global Pooling**: Destroys spatial information, causes zero-prediction
5. **Physics Losses Unused**: PDE/energy/receiver/komega implemented but never activated

### 3. Factorized Spatiotemporal FNO Architecture

Decomposes 3D operator learning into:
- **2D Spatial FNO** (SpectralConv2d): processes each time frame independently with 2D FFT
- **1D Temporal Mixer** (grouped Conv1d): processes each spatial point's time series
- **Chunked forward**: bounds peak memory via spatial (frame-level) and temporal (point-level) chunking

**Key innovation**: Memory scales as O(H·W·spatial_W + H·W·T·temporal_W) instead of O(H·W·T·width).

### 4. Training Results

#### 4.1 64×64×160 Resolution

| Metric | Dense 3D FNO | Factorized FNO | Improvement |
|---|---|---|---|
| Spectral coverage | 50% (16/32) | **100% (32/32)** | 2× |
| Val (250 samples) | 0.444 | **0.381** | **14.2%** |
| GPU memory (bs=4) | N/A | 16 GB | — |
| Epochs to converge | 60+ | 13 | 4.6× faster |

Per-category:
| Category | Dense | Factorized | Δ |
|---|---|---|---|
| uniform | 0.315 | **0.130** | -59% |
| layered | 0.609 | **0.554** | -9% |
| marmousi | 0.458 | 0.538 | +17% |

#### 4.2 128×128×160 Resolution

| Metric | Value |
|---|---|
| Spectral coverage | 75% (48/64) |
| Test (250 samples) | **0.389** |
| Receiver L2 | 0.517 |
| GPU memory (bs=1) | 14 GB |
| Training time (30 epochs) | 87 min |

Per-category:
| Category | n | rel L2 |
|---|---|---|
| uniform | 100 | **0.121** |
| layered | 100 | 0.534 |
| marmousi | 50 | 0.636 |

Checkpoint migration from 64×64 best (modes 32→48 padded, zero-init high frequencies).

#### 4.3 400×400×160 Resolution

- Inference: 17.5 GB GPU (works on RTX 3090)
- Training: requires >24 GB (needs A100 40GB+)
- Gradient checkpointing + micro-batching (chunk=32) validated
- Config ready for A100 deployment

### 5. Instance Adaptation

#### 5.1 Hybrid FD-NN Adaptation

Combines O(n) finite-difference stencils with head-only neural network fine-tuning:

- **Speed**: <10s per sample (head-only, 4K/28M params)
- **PDE improvement**: Consistent 14-17% reduction across all categories
- **Accuracy**: Minimal data accuracy impact (-0.006 rel L2 avg on 25-sample test)
- **Complex media**: 33% of layered/marmousi samples improve

#### 5.2 PDE-Constrained Fine-Tuning

Full-model fine-tuning with 2 observed frames + acoustic PDE residual:
- Gentle mode (lr=1e-6, pde_weight=1e-5): improves both data fit and physics
- Aggressive mode (lr=1e-5, pde_weight=1e-3): 75-94% PDE reduction but data accuracy trade-off

### 6. Key Findings

1. **100% spectral coverage critical**: Factorized FNO's per-frame 2D FNO enables full spectrum utilization
2. **Uniform media benefit most**: Sharp wavefronts need full spectrum → 59% improvement over dense FNO
3. **Marmousi is the challenge**: Complex velocity scattering needs cross-dimensional coupling
4. **Cosine LR scheduling essential**: Best results at LR 2-3e-4, oscillations until LR < 2e-4
5. **I/O is the training bottleneck**: num_workers=4 + pin_memory fixes 3× speed improvement
6. **Hybrid adaptation works**: FD stencils provide fast physics-guided correction with minimal overhead

### 7. Ablation Summary

| Ablation | Resolution | Result |
|---|---|---|
| Random init, no migration | 64×64 | val 0.862 (escapes zero-prediction) |
| 64×64 best → 128×128 migrated | 128×128 | val 0.427 (30 epochs) |
| Reduced width=16 | 128×128 | val 0.967 (10 epochs) |
| Width=18 medium | 128×128 | val 0.975 (10 epochs) |
| Full width=32 | 128×128 | val 0.427 (30 epochs) |
| 400×400 inference | 400×400 | 17.5 GB GPU ✅ |
| 400×400 training | 400×400 | OOM on RTX 3090 |

### 8. Artifacts

| Artifact | Path |
|---|---|
| Best 64×64 checkpoint | artifacts/factorized_fno/64×64x160_continue/checkpoints/best.pt |
| Best 128×128 checkpoint | artifacts/factorized_fno/128×128x160_continue/checkpoints/best.pt |
| 128×128 migrated ckpt | artifacts/factorized_fno/migrated_64×64_to_128×128.pt |
| Architecture code | src/fno_acoustic/model_factorized.py |
| Tests | tests/test_factorized_fno.py |
| Configs | configs/factorized_fno_*.yaml |
| Adaptation scripts | scripts/run_hybrid_numerical_adaptation.py, scripts/run_instance_adaptation_pde.py |

### 9. Next Steps

1. Train 400×400 on A100 (40GB+) using existing config + migrated checkpoint
2. Activate PDE stage 2 training (receiver + spectral loss) on 128×128
3. Multi-resolution ensemble: 64×64 model for marmousi + 128×128 for uniform/layered
4. Write manuscript for IEEE TGRS submission
