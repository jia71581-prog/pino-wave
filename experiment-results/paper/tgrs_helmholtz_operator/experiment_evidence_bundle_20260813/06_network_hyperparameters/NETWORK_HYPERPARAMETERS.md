# Network hyperparameter design

## Deployed r5b operator

The selected train-gate checkpoint contains 32,420,564 parameters. Its dominant
settings are width 128, a depth-12 factorized complex spectral decoder, 32 retained
modes, spectral rank 112, a four-level local field with channel multipliers
`1,1,2,2`, an 8-cell first-arrival-gradient warp, and a rank-32 temporal latent bank
with four harmonics. The active components and every low-level variant field are
recorded in `network_hyperparameters.json`.

Training uses 24 exact frames per record and an effective 32 records per optimizer
update, accumulated from one-record physical microbatches. r5b uses Muon for matrix
parameters and AdamW for the remaining groups. Its selection metric is mean
per-record relative L2 on a fixed 48-record train-only gate.

## Diagnostic continuation

r5d adds a shared dynamic multiscale spectral adapter and has 33,241,115 parameters,
but its training/gate time selectors were discovered to use different seeds. It is
diagnostic only. r5e is registered but not launched and fixes only this seed offset.

## Parameter-matched Patch-DeepONet

Patch-DeepONet has 32,420,465 parameters, 99 fewer than r5b. It combines a compact
local CNN, a high-capacity pooled branch MLP, and a width-48 query trunk. It has
passed static and CPU tests but has not been GPU-trained, so no accuracy comparison
is claimed.

## Claim scope

The source experiment fixes the Ricker frequency at 19 Hz and varies source
position only. No frequency-generalization claim is permitted.
