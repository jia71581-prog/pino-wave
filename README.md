# PINO-Wave research code

The complete current research source snapshot is in [grouped-dual-head-v2](grouped-dual-head-v2/). It includes the neural-operator model families, training and evaluation scripts, numerical data-generation code, configurations, tests, research documentation, and local agent definitions.

## Research scope

The current research objective is direct prediction of the two-dimensional acoustic wavefield from the velocity model and source information, without a numerical pre-solve for the prediction case. The archive also preserves historical numerical and hybrid methods; those historical implementations do not all satisfy the new direct-prediction constraint. Their numerical-parent results must not be presented as validated direct neural-operator performance.

## Included and excluded

This upload additionally includes the historical experiment results archive, in [experiment-results](experiment-results/), whose [README](experiment-results/README.md) and [MANIFEST](experiment-results/MANIFEST.json) describe the scope and per-file integrity. It holds per-experiment metrics, evaluation and disposition records, run logs, result figures and paper-level figures/tables, as a time-stamped offline archive. It is not a new validation pass and does not present the numerical/hybrid historical results as validated direct-neural-operator performance.

Datasets, wavefield caches, trained weights/checkpoints, tensorboard event streams, manuscript text, credentials, and personal agent sessions are not included (neither in the source snapshot nor in the results archive). Notebook code is retained with outputs and embedded runtime data cleared. Source code for generating data and evaluating models is included. The dated duplicate working copy is omitted. The results archive excludes raw wavefield/velocity arrays, weights, data, and paper main text; it retains metrics, run logs, and figures.

## Using the source

Start from [the main README](grouped-dual-head-v2/README.md) and the relevant script/configuration. Run commands from grouped-dual-head-v2 and include its src directory and project root on PYTHONPATH when needed.

The current pyproject.toml provides pytest settings, not an installable Python package; do not assume pip install -e . is supported. requirements-pino.txt belongs to the PINO workflow. Install the dependencies needed by the selected implementation. Historical absolute dataset/checkpoint paths in scripts and configs must be adjusted to your environment. Datasets and trained weights must be supplied separately.

This upload is a source and historical results archive, not a new trained-model result. Existing attribution and license notices in the source are retained; no new license grant is added by this upload.
