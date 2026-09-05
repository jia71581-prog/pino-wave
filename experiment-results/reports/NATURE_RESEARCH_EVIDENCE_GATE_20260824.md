# Nature-level research evidence gate

Date: 2026-08-24

## Venue routing

- **Nature Methods is not a current fit.** Its official scope is life-science
  methods and requires strong validation, comparison with available approaches,
  immediate practical relevance, and application to an important biological
  question. The current acoustic-wave operator has no biological application.
- **Nature Biotechnology is not a fit.** The project does not make a
  biotechnology, medical, translational, or community-resource claim.
- **Flagship Nature is not yet supported.** Nature requires outstanding
  scientific importance and a conclusion that interests an interdisciplinary
  readership. A specialist train-only source-overfit improvement is not such a
  conclusion, even when technically correct.

Official sources:

- https://www.nature.com/nature/for-authors/editorial-criteria-and-processes
- https://www.nature.com/nmeth/submission-guidelines/about/aims
- https://www.nature.com/nmeth/content
- https://www.nature.com/authors/editorial_policies/

## Current evidence status

### Strong

- Frozen validation and independent test_id reports already exist for the
  historical fine-grid hybrid candidate, including synchronized runtime.
- Serious operator experiments bind manifests, source files, code, parent
  checkpoints, run identities, and terminal records by digest.
- r20-r25 keep validation and test_id sealed and report complete-time,
  family, temporal, spectral, phase, and displacement diagnostics.

### Insufficient for a broad method claim

- The learned WKB branch still fails same-medium unseen-source generalization.
- r25's analytic onset phase improves calibration by 35.2% but improves the
  independent confirmation aggregate by only 12.6%, below its frozen 20% gate.
- The evidence is confined to one acoustic generator, one grid/protocol, and a
  small source-holdout panel. It does not establish general utility across wave
  equations, domains, discretizations, or resolutions.
- A public code/data/protocol package and DOI-backed release plan are not yet
  bound to the learned-method claims.

## Research gate for the next architecture

Do not advance a new source-conditioning architecture on anchor fit alone. It
must satisfy, in this order:

1. Exact-zero or function-preserving initialization and unit/query-invariance
   tests.
2. The existing same-medium source-holdout train panels, with fit, calibration,
   and confirmation reported separately at all 401 stored times.
3. At least 20% confirmation improvement over r23 with no fit-family regression,
   followed by an absolute source-holdout gate rather than a relative claim.
4. A disjoint-medium train confirmation before opening validation.
5. Only after hyperparameters are frozen: one complete validation report, one
   independent test_id report, synchronized end-to-end runtime, and code/data
   availability records.

## Tested source-relative mechanism

r27 tested an exact-zero `(dx, dz, radius)` projection under the required
source-holdout protocol. It was rejected: calibration improved only 1.37%,
confirmation improved only 2.06%, and fit layered error regressed by 29.28%.
Coordinate availability alone therefore does not establish a reusable
source-conditioned Green-field mapper.

## Revised next falsifiable mechanism

Return to the literature-ranked coarse-propagator residual priority. Use a
stable, low-cost source-aware LWC84 field to provide phase, causality, source
position, and onset, then learn only a multiscale residual closure. The first
evidence tier must be train-only and must compare against both the numerical
parent and the current learned WKB branch on the same source-holdout panels.

Do not advance unless independent source confirmation improves by at least 20%
without fit-family regression and the measured end-to-end runtime lower bound
remains compatible with the final 10x speed target.
