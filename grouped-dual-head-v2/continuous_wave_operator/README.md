# Continuous Wave Operator

This package learns pressure at arbitrary physical `(x, z, t)` queries from the unified
LWC-84 HDF5 dataset. The medium is encoded once and its cache can be reused for multiple
sources; receiver locations and query times do not need to lie on the saved grid.

## Tensor contract

- velocity: `[B, 1, Z, X]`, metres/second;
- source map: `[B, S, 1, Z, X]`, nonnegative with unit mass for every shot;
- source parameters: `[B, S, 5]` as `(x_s, z_s, f0, t0, amplitude)`;
- queries: `[B, S, Q, 3]` as physical `(x, z, t)`;
- pressure: `[B, S, Q]`.

`ContinuousWaveOperator.encode_medium`, `encode_sources`, and `query` expose the cache
boundary explicitly. The final output is exactly linear in source amplitude and applies
smooth hard gates at `t=0` and the top free surface `z=0`.

## Training

The loader reads complete HDF5 time frames and then gathers spatial queries. Training uses
sample/time/space residual EMA sampling mixed with uniform and medium-family support, plus
clipped inverse-probability correction. Checkpoints atomically store the model, optimizer,
scheduler, normalization binding, AIS state, and all RNG states.

```bash
python -m continuous_wave_operator.scripts.train \
  --config continuous_wave_operator/configs/smoke.yaml --device cuda \
  --output artifacts/continuous_wave_operator/smoke_cuda

python -m continuous_wave_operator.scripts.train \
  --config continuous_wave_operator/configs/production.yaml --device cuda \
  --output artifacts/continuous_wave_operator/production
```

Evaluation and arbitrary receiver queries use `continuous_wave_operator.scripts.evaluate`
and `continuous_wave_operator.scripts.query_receivers`. FWI is intentionally not implemented;
the differentiable medium input and reusable source/query API are prepared for a later FWI loop.

