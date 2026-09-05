# Truth-leaking oracle diagnostic -- not a model prediction

This directory is intentionally excluded from model predictions and paper
results. The construction uses the reference wavefield directly:

`oracle = target + (prediction - target) / 5`

It therefore reduces relative error by a factor of five by construction. It is
not evidence of model accuracy, post-processing generalization, fine-tuning, or
physical correction.

Permitted use: oracle visualization and debugging only.

Forbidden uses:

- do not rename this artifact as a prediction;
- do not use it in model selection, metrics, tables, or manuscript figures that
  represent model performance;
- do not replace the sealed r5b prediction manifest or any canonical output;
- do not use it to claim superiority over DeepONet, LWC-84, or another method.

Only frames 80, 240, and 400 from the locked rank-15 Marmousi slice are
materialized. The complete 401-frame oracle is stored only as the virtual formula
and source-file hashes in `oracle_manifest.json`; no complete model output is
created.
