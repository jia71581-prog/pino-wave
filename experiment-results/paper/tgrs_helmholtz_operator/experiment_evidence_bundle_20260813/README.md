# TGRS experiment evidence bundle

This folder consolidates the current acoustic-operator experiment evidence without
duplicating large wavefield arrays or changing any running process. Open
`EVIDENCE_STATUS.csv` first: every asset is labelled as historical development,
train-only selection, confirmatory, complete boundary evidence, or pending.

## Directory map

1. `01_complete_wavefields`: full 401-frame truth/coarse/prediction arrays for one
   historical development example per family. The NPZ files are read-only relative
   links to their canonical location.
2. `02_wavefield_snapshots`: prediction/reference/error snapshot panels. Their main
   role is qualitative illustration, not population inference.
3. `03_receiver_waveforms`: receiver traces and common-receiver gathers for the same
   development examples, plus PDE-adaptation diagnostic plots.
4. `04_numerical_dispersion`: analytic phase-velocity curves and numeric CSV for
   FD2, FD4 and LWC-84, plus the learned-dispersion claim gate. The analytic figure
   supports numerical context only; the current claim gate does not establish
   learned dispersion suppression.
5. `05_relative_error`: all 48 per-record errors from the frozen r5b train-only gate,
   plus paired parent/CPADC values for all 480 validation and 480 independent
   `test_id` records. CSV, summary JSON, PNG and vector PDF are provided. The r5b
   panel is selection evidence; the CPADC panel is sealed full-split evidence for a
   relative correction benefit, not an absolute solver-accuracy claim.
6. `06_network_hyperparameters`: proposed r5b, diagnostic r5d/r5e and parameter-
   matched Patch-DeepONet design/configuration files.
7. `07_protocols_and_figure_plan`: frozen fixed-19-Hz, position-only protocol,
   preregistration and figure plan.
8. `08_reproducibility_and_diagnostics`: metric diagnosis, VDS dependency audit and
   architecture audits.
9. `09_pending_position_evaluation`: stable legacy directory name for the completed
   30-slice x 8-position evaluation. It contains the official snapshots, receiver
   gathers, prediction/reference seals, and 240-record score summary.
10. `99_historical_experiment_index`: one-row-per-top-level-result registry. Large
    checkpoints are indexed, not copied.
11. `10_training_runs_read_only`: live read-only links to the selected r5b,
    naturally completed r5c and r5d run directories. They preserve training
    curves, retry histories and checkpoint metadata without copying model weights.
12. `12_phase4b_fixed19_rank15_multisource`: the final true-velocity plus
    eight-source Phase4b composite, prediction seal, complete report, and rendering
    script. It is one locked-slice descriptive case, separate from the 4.51% G3
    Marmousi record.
13. `13_phase4b_rank30_eight_source_superposition`: the input-complexity-maximizing
    Marmousi velocity slice and the physical sum of eight fixed-19-Hz source fields
    at six times. This is qualitative complex-medium evidence, not an accuracy
    promotion.
14. `14_runtime_comparison`: reproducible proposed-method versus LWC-84 latency
    summary. Ours includes its bound parent inference, instance coefficient
    fine-tuning, inverse normalization, and output materialization. The
    approximately twofold reduction is descriptive because the archived runs are
    not case-paired or accuracy-matched.
15. `15_pi_deeponet_train_development_comparison`: same-record accuracy metrics,
    compact snapshot source data, and a hash-bound rendering report for r5b plus
    residual-conditioned instance fine-tuning versus full-data PI-DeepONet. This
    six-record panel is train-only development evidence and cannot support a
    held-out superiority claim.
16. `16_phase4b_update0265_pi_comparison`: historical-best Phase4b update 265
    versus PI-DeepONet on the exact same six records and complete-future windows.
    Mean record errors are 0.105992 and 0.615635. Phase4b includes its external
    sigma-2 LWC-84 background and the result applies no instance adapter.

## Figure logic

- Complete-wavefield figure claim: the model reconstructs propagation structure
  across early, middle and late times on selected development cases.
- Receiver figure claim: time-series diagnostics expose arrival and phase errors
  that field snapshots alone can conceal.
- Dispersion figure claim: LWC-84 has lower analytic phase-velocity error than FD2
  and FD4 at equal points per wavelength; no method is described as dispersion-free.
- Relative-error figure claim: r5b error remains strongly family dependent on the
  fixed train-only gate, with Marmousi as the hardest family.
- Position-boundary figure claim: at one fixed 19-Hz wavelet, r5b retains useful
  early-arrival geometry but loses late scattered content and degrades outside the
  training-position range. It does not pass the source-position-generalization gate.
- PI-DeepONet comparison claim: historical-best SBH and DFO+RCFA have lower
  complete-future error on the six matched train records. The common-time SBH
  snapshots illustrate that direction. SBH includes an external numerical
  background; the RCFA instance step is nearly neutral and fails promotion.

Main-paper candidates are the fixed-19-Hz early/late position panels and the
240-record distribution as boundary evidence. Receiver gathers, analytic FD
dispersion context, historical single-case panels, and adaptation diagnostics
belong in the supplement until comparative gates pass. Legends must state the
split, sample count, source frequency, and whether the result is selection,
boundary, or confirmatory evidence.

## Completed latest-candidate tests (2026-08-13)

The fixed-19-Hz position-only census is complete for 240 records and has mean complete-transient relative L2 `0.581776439647`. The exact same-sample replacement gate against Phase4b failed for all three medium families, so no historical model asset was replaced. See `09_pending_position_evaluation` (stable legacy directory name) and `11_latest_candidate_nonpromotion_test`.

## Phase4b multi-source case study (2026-08-14)

Phase4b update 265 was evaluated on the preregistered rank-15 velocity slice at
all eight fixed-19-Hz source positions. The all-401-frame mean is `0.176883`,
with `0.177279` in range and `0.176224` outside range. The final composite includes
the true velocity model. The separate reused G3 Marmousi result remains `0.045074`
(4.51%). Both results require the external smoothed-background numerical solve.
