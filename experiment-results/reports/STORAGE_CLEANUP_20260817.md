# Storage cleanup audit (2026-08-17)

## Scope and safety gate

This cleanup was authorized while preparing the complete research archive. No
file was selected merely because its metric was worse than another run. A
candidate had to be reproducibly classified as regenerable build/cache output,
or as a failed, rejected, or abandoned run that had been superseded and was not
used by the manuscript or its evidence bundle.

Before cleanup, a full archive was created and validated by complete parallel
xz decoding, tar-directory parsing, critical-file presence checks, and
SHA-256 hashing. Its pre-cleanup SHA-256 was
`cece791b8f205788fb8083727e6765d0f16a9726de9e6fbfbfab5749d93c2608`.

## Protected material

The cleanup explicitly preserves:

- every A3, B2-H, and Helmholtz checkpoint and run directory;
- the historical-best update-265 checkpoint and the r34/r35 comparison chain;
- the successful full-data PI-DeepONet training and r10 validation/test outputs;
- the successful CPADC r13/r14 outputs;
- all manuscript sources, final PDFs, figures, evidence-bundle content, and
  files referenced by those records;
- the rejected r33 record because the current r34 chain cites it as its parent;
- source code, experiment configs, and reusable launch/evaluation scripts.

## Removed targets

### Regenerable temporary output

- `tmp/` (rendered PDF-review pages and figure-validation previews)
- `.pytest_cache/`
- all `__pycache__/` directories and `.pyc` bytecode
- LaTeX intermediates `manuscript.aux`, `manuscript.fdb_latexmk`,
  `manuscript.fls`, `manuscript.log`, and `manuscript.out`; the final PDF and
  extracted final text are preserved

### Failed or superseded run residue

- `results/frozen_pi_deeponet_test_id_r9_20260815/`: `worker_failed`, superseded
  by the complete r10 test
- `results/pi_deeponet_lwc84_r6/`: PI entry gate failed, superseded by the
  successful full-data PI run used by r10
- `results/frozen_cpadc_vs_pi_validation_r12_20260815/`: stale `running` receipt
  with no live process or shard output, superseded by complete r13/r14 runs
- `results/patch_deeponet_parameter_matched_r2/` and the r3/r4/r5 Patch-DeepONet
  gate/launch records: failed or rejected abandoned baseline; the active
  comparison is PI-DeepONet
- r5b feature-meta r26, r29, and r31 preregistration/result pairs: terminal
  `completed_rejected`, unreferenced by the paper and current r34 chain
- `results/r5b_full_time_failure_r17_20260816.json` and its preregistration:
  unreferenced failure-diagnosis residue
- the two `train_only_drp_lwc84_confirmation_r1.launch_failed_missing_pythonpath`
  log/PID files; the substantive later experiment records are preserved

## Deletion method

Every target was resolved and inspected before deletion. Directories were
removed with depth-first `find ... -delete` against exact paths; individual
files were removed with exact-path `unlink`. Broad globs and recursive `rm`
were not used.

## Outcome

The cleanup completed successfully and released 36,364,288 allocated bytes.
Post-cleanup checks confirmed that all selected paths were absent, all eight
evidence symlinks remained valid, the final manuscript PDF retained SHA-256
`60bdd9e1a8f1e92244bb50213c1edffb918a6a2491c84210e128c67cfd462b70`,
and the update-265 checkpoint retained SHA-256
`f6efbb81dd1e9baab0eb34b32b125e2cb58cd3b3292ee0db33a29194c84bfea1`.
