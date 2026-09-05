# Evidence-led figure plan

## Main figures

1. **Architecture:** the existing overview defines the no-background r5b configuration; the text and caption must not imply that it is the Phase4b hybrid.
2. **Accuracy-best Phase4b multi-source case:** one composite contains the true rank-15 Marmousi velocity model, eight fixed-19-Hz source positions, and reference/prediction/error wavefields at 0.6 s. The column headings report the eight measured all-401-frame errors; their mean is 0.17688. The caption must disclose the external smoothed-background solve and one-slice scope. The separate 4.51% value belongs to the reused G3 Marmousi record.
3. **Phase4b G3 accuracy table:** the primary checkpoint-bound Marmousi value is 0.04507 (4.51%) across all 401 frames. Keep it in the G3 table rather than attaching it to the multi-source figure.
4. **Fixed-19-Hz early and late wavefields:** the r5b figures define the no-background position-generalization boundary across eight positions on one locked slice.
5. **Position-only distribution:** the 240-record r5b boxplot anchors the population result with 30 velocity slices as the uncertainty units.
6. **Comparative benchmark:** reserve for the parameter-matched PI-DeepONet and same-accuracy LWC-84 results. Do not create a placeholder figure before those gates finish.

## Supplement

- Shallow receiver-line gathers for the eight fixed-19-Hz sources.
- The 0.6-s multi-source snapshot grid.
- Analytic FD2/FD4/LWC-84 phase-velocity context is supplementary validation of the numerical reference. It does not establish learned dispersion suppression.
- The historical fixed-15-Hz Phase4b receiver gather, receiver traces, and single-source true Marmousi snapshots, clearly labelled as hybrid case evidence rather than r5b results.
- The rank-30 Marmousi eight-source physical superposition at 0.20--1.00 s. This input-complexity-selected panel is the preferred qualitative display of complex-medium scattering, but its 0.28828 summed-field error must not replace the separate 4.51% G3 accuracy result.
- r5c--r5g train-only retry curves and gradient diagnostics.
- CPADC validation and test_id distributions as a separate deployment contribution.

## Legend contract

Every fixed-position legend must state: r5b checkpoint; fixed 19 Hz; 30 unseen velocity slices; eight repeated positions per slice; 30 independent uncertainty units; interpolation versus outside-training-range roles; and whether the panel is qualitative or a complete 240-record statistic.

The Phase4b multi-source legend must state: `update_0265.pt`; fixed 19 Hz; one locked rank-15 velocity slice; eight source positions; all 401 saved times for the column errors; external Gaussian-smoothed-background numerical solve; and no population inference. The 0.04507 (4.51%) Marmousi value belongs to the separate reused G3 development record and must not appear as the caption metric for the multi-source figure.

The Results topic sentence is: “A sealed fixed-frequency test retained useful early-arrival geometry but exposed increasing error outside the training-position range and in late scattered coda.”
