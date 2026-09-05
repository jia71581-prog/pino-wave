# PA-CORA P1 supplementary train-only research

## Scope

This diagnostic analyzes the four P1 fit-window caches without opening
validation or `test_id`.  It does not modify the active four-GPU P1 run.

## Window statistics

| Slot | Start regime | Mean target norm | Target q10 | Mean anchor relL2 | Maximum anchor relL2 | Near-zero vs own slot0 | Mean inverse target norm | Spatial-gradient relL2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | causal onset | 81.63 | 31.54 | 0.124 | 0.413 | 0.00% | 0.0171 | 0.175 |
| 1 | early/middle | 81.52 | 20.74 | 0.335 | 1.179 | 0.00% | 0.0218 | 0.416 |
| 2 | middle/late | 72.17 | 9.00 | 0.543 | 1.472 | 0.42% | 0.0424 | 0.637 |
| 3 | fixed 0.65 s start | 63.27 | 0.120 | 1.897 | 48.321 | 21.25% | 7.8465 | 12.852 |

The inverse-target-norm means, which approximate the scale of the relative-L2
gradient with respect to model output, are `[0.0171, 0.0218, 0.0424, 7.8465]`.
Under equal slot sampling, slot 3 therefore represents approximately 98.97% of
the summed mean gradient proxy and is 458 times the slot-0 scale.

## Family and frequency diagnosis

- Slot-3 Uniform has mean anchor relative L2 3.90 and 45% near-zero records.
- Slot-3 Layered has mean anchor relative L2 0.97 and 18.75% near-zero records.
- Slot-3 Marmousi retains substantial energy but remains difficult, with mean
  anchor relative L2 0.82.
- High-frequency sources are increasingly difficult across the windows:
  slot-1/2/3 anchor relative L2 is approximately 0.46/0.72/2.64 for `f0 >= 22 Hz`.
- Only about 0.5--1.1% of sampled target temporal spectral energy lies in the
  registered high band, while high-band anchor relative error rises from 0.41
  at slot 0 to 5.27 at slot 3.

The target mean energy alone hides the failure because energetic Marmousi
records coexist with almost-zero Uniform and Layered late windows.  Record-wise
relative L2 gives the latter extremely large gradients.

## Boundary interpretation

At slot 0, boundary-20 error (0.130) exceeds interior error (0.106), but not by
enough to explain the complete parent floor.  At slots 2 and 3 the top-region
relative values become very large; these ratios are confounded by vanishing
top-region target energy and must not yet be interpreted as proof of a CPML or
free-surface defect.  The queued final-V9 residual diagnostic will report
absolute and relative structure on group-disjoint holdout windows.

## Decision on current P1

The active P1 run is a valid, reproducible test of *equal-weight four-window
replacement*, but it is not a clean test of multitemporal pretraining in
general.  Its early holdout plateau near the anchor and training losses around
1.0--1.4 are consistent with the measured slot-3 loss-scale domination.

Let the registered P1 run finish for complete negative or positive evidence,
but do not promote the equal-weight objective even if one seed fluctuates below
the baseline.  The two-seed frozen gate remains binding.

## P1b recommendation

Change only the data/loss curriculum before testing a new architecture:

1. Remove the fixed 0.65 s slot from the first stable candidate.  Keep onset,
   early/middle, and middle/late windows ending near the current slot-2 regime.
2. Sample slots with probabilities 0.50/0.30/0.20 instead of equal weighting.
3. Use an onset-referenced denominator floor per source,
   `max(||target_window||, 0.25 * ||target_onset||)`, so nearly vanished windows
   cannot dominate the gradient.
4. Use an onset-first curriculum: onset only initially, then introduce slot 1,
   then slot 2 while preserving the registered total record exposure.
5. Balance batches jointly by family and source-frequency bin.
6. Keep the original onset holdout as the primary same-protocol gate, and add
   slot-1 and slot-2 group-disjoint holdouts as mandatory report-only metrics.

The pass rule remains strict improvement with no minimum percentage.  A P1b
candidate must first improve the onset primary metric; multi-phase metrics are
needed to verify that the additional windows did not merely redistribute error.

## Pending evidence

After the active P1 terminal, a detached queue will build three multi-phase
holdout caches and render the final selected V9 residual.  It will report
interface/non-interface, boundary/interior, temporal bands, frequency bands,
spatial gradients, and a plus/minus-two-frame temporal-shift oracle.  This is a
train-only diagnostic and cannot promote a model.
