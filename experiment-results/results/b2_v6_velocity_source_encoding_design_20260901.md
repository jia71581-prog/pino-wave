# B2-v6 velocity and source encoding research design

## Local diagnosis

The full grouped parent already has a strong learned encoding stack:

- velocity: a position-aware, multiscale complex-spectral pyramid plus pooled
  tokens and a rank vector;
- source: a five-parameter MLP, source-map CNN, source-point sample of the
  medium pyramid, and rank projection;
- fusion: coordinate/time Fourier-Gabor features and straight-ray travel-time
  features.

The B2 correction branch bypasses these representations and receives only
seven hand-built maps: normalized velocity, two log-velocity gradients,
source Gaussian, straight local travel estimate, and x/z coordinates. It does
not receive source frequency or onset explicitly and has no multiscale medium
representation.

The train panel spans 8.19--29.91 Hz and approximately 5.28--69.49 stored-grid
points per wavelength. Source `t0` is exactly `1.5/f0`, so it is redundant as
an independent scalar but remains useful as an explicitly normalized onset
coordinate.

## Implemented encoders

### E1: enriched physical conditioning

`enriched_physical_conditioning` extends the base seven channels to fifteen:

1. normalized `f0` and `t0` maps;
2. sine/cosine of the source-arrival phase `2*pi*f0*travel`;
3. log points-per-wavelength `v/(f0*dx)`;
4. 5x5 and 17x17 velocity contrasts;
5. coarse log-velocity interface magnitude.

This option is inexpensive, deterministic, and uses deployment-available
quantities only.

### E2: frozen-parent latent reuse

`ParentLatentConditioner` projects and concatenates:

1. full-resolution level of the frozen parent medium pyramid;
2. frozen source-map CNN field;
3. frozen five-parameter source hidden vector broadcast spatially.

Only the small 1x1/MLP projections are trained offline. The parent encoders
remain frozen, and online adaptation still updates only low-dimensional
correction coefficients.

## One-variable ablation ladder

1. E0: original B2 seven-channel conditioning.
2. E1: E0 plus physical15 features; all other training/adaptation settings
   fixed.
3. E2: E0 plus projected frozen-parent latents; all other settings fixed.
4. Only after E1/E2 evidence: combine the winning encoder with an offline
   residual POD/local neural-element basis.

Primary train-only screening metrics are future relative L2 after online
self-supervised adaptation, nonworse rate, per-family means, late/high-band
error, and total parent+adaptation runtime. A candidate must pass on fresh
group-disjoint train confirmation before any validation access.

## Literature mapping

- Seismic neural operators motivate joint generalization over velocity
  structures and source locations: https://arxiv.org/abs/2108.05421
- Wavelet and multiwavelet operators motivate localized multiscale medium
  features: https://arxiv.org/abs/2205.02191
- Geometry-informed operators motivate explicit local geometric/position
  representations fused with global spectral latents:
  https://arxiv.org/abs/2309.00583
- PINO motivates mixing data and physics representations across resolutions,
  subject here to the disclosed LWC84/Q1 mismatch:
  https://arxiv.org/abs/2111.03794

## Bindings and verification

- implementation SHA256:
  `d39af7783a0888663d875bb2771ba26b13a02986f0e6741187a3a2ecc0c80ab9`
- test SHA256:
  `914acd3ea6a93580c64319d8496caad0c2da68e951b94698ad425de0ef7149cb`
- seven encoder/FE adapter tests passed;
- Target-5 read-only audit remains passed;
- no GPU experiment, validation access, or `test_id` access was introduced by
  this encoding work.
