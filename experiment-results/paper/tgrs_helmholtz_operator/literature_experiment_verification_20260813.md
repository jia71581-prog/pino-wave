# Literature-to-experiment verification log

Registered: 2026-08-13 UTC. This log separates what the cited source establishes from what our experiments must still demonstrate. It does not import numerical claims from other papers as evidence for our model.

## Accepted evidence

| Proposed manuscript statement | Primary source and locator | Verdict | Consequence for our experiment |
|---|---|---|---|
| Seismic neural operators have been evaluated across source locations as well as velocity models. | Yang et al., *IEEE TGRS* 2023, doi:10.1109/TGRS.2023.3264210, abstract/method summary. | CONFIRMED for the cited U-shaped neural operator, not for our model. | Report held-out source positions explicitly and compare predictions with the numerical reference on the same medium. |
| Frequency-domain Helmholtz neural operators have been evaluated across source locations. | Zou et al., *GJI* 2024, doi:10.1093/gji/ggae342, abstract and conclusions. | CONFIRMED for the cited method. | Our source-position experiment fixes source frequency, amplitude, and onset so position is the only controlled variable. |
| A neural wave solver can be trained on many sources on one Marmousi model and tested on separate source locations. | Moseley et al., arXiv:2006.11894, Sec. 4.3 and Fig. 2/7: 100 training and 20 test source locations. | CONFIRMED. | Our qualitative panel must show all preregistered sources from one identical Marmousi slice, plus quantitative results across every held-out Marmousi source. |
| A time-domain Koopman neural operator has been evaluated on Marmousi with source frequency fixed at 20 Hz and source positions separated into 900 training and 300 validation sources. | Bi et al., *GJI* 2026, doi:10.1093/gji/ggag196, Sec. 3.3.1 and Fig. 21. | CONFIRMED for its single-Marmousi, future-wavefield setting. | Fixed-frequency position control is the directly comparable design. We additionally separate position interpolation from outside-range positions and score complete 401-frame transients on 30 unseen velocity slices. |
| Small source displacements in laterally heterogeneous Marmousi media can create meaningful wavefield differences. | Alkhalifah, *GJI* 2010, doi:10.1111/j.1365-246X.2010.04800.x, Figs. 8--12. | CONFIRMED. The official article uses a 15-Hz Ricker source and shows a common-scale difference for a 25-m source shift. | Do not infer source generalization from visually similar snapshots. Include common-shot gathers, difference fields, phase/lag metrics, and error against distance to the nearest training source. |
| FNO seismic surrogates can exhibit frequency bias, with larger high-frequency error. | Kong et al., *Seismological Research Letters* 2025, doi:10.1785/0220250085, abstract/results. | CONFIRMED for the evaluated 3-D FNO. | Report spatial spectral bands and receiver phase/lag; a full-field relative norm alone cannot support a dispersion claim. |
| Broad wave-equation benchmarks find strong in-distribution performance but materially weaker out-of-distribution behavior. | Liu et al., *TMLR* 2024, WaveBench, OpenReview:6wpInwnzs8. | CONFIRMED at benchmark scope. | Keep the group-disjoint in-distribution test separate from source-location extrapolation and any future velocity OOD test. |
| Variable-velocity frequency-domain FNOs are a relevant seismic comparison. | Li et al., *IEEE TGRS* 2023, doi:10.1109/TGRS.2023.3333663. | CONFIRMED. | Include a documented frequency-domain/FNO-family comparator when compatible; do not substitute an unverified collapsed pilot. |

## Partial or rejected evidence

| Candidate claim | Source | Verdict | Reason |
|---|---|---|---|
| Published neural-operator speedups directly establish our end-to-end speed advantage. | Yang et al. 2023; Zou et al. 2024. | PARTIAL. | Their timings motivate the comparison but do not bind our preprocessing, background solve, adaptation, frame materialization, hardware, or matched-accuracy threshold. We require a same-device end-to-end benchmark. |
| Neural operators are intrinsically faster than optimized GPU finite differences. | Bi et al., *GJI* 2026, Sec. 4.2. | REJECTED. | That study reports GPU FD becoming faster than KNO for larger tested models. Our runtime claim therefore requires the same GPU, synchronized full materialization, and a matched-accuracy gate; it may fail. |
| MscaleFNO proves that our multiscale adapter controls seismic numerical dispersion. | You et al., arXiv:2412.20183. | REJECTED for this claim. | It supports multiscale spectral-bias motivation in oscillatory function spaces, but it is a preprint and does not validate our 2-D transient architecture or dataset. |
| A visually accurate late snapshot proves long-time stability. | Any single-source snapshot study. | REJECTED. | Stability requires all 401 registered frames, early/middle/late errors, final-third error slope, and receiver phase/lag/coherence. |
| Lower error than a collapsed global DeepONet proves superiority over DeepONet. | Our historical pilot. | REJECTED. | A comparative claim requires a non-collapsed, parameter-matched local Patch-DeepONet that first passes the frozen train-only overfit gate. |
| Small mean relative error proves that learned numerical dispersion has been overcome. | No source supports this implication. | REJECTED. | Use the bounded phrase “suppresses learned high-wavenumber and phase error” only if group-cluster confidence intervals pass the preregistered spectral and receiver-phase gates. |

## Resulting experiment additions

1. **Marmousi source-position generalization.** Select the representative velocity slice using input velocity only, then discard its original variable-frequency records from the evidence panel. Generate the main panel at fixed 19 Hz, fixed amplitude, and fixed onset while varying only source position. Show the velocity/source map, reference/prediction/error at fixed early, middle, and late times, and shallow receiver gathers.
2. **Source-position distance analysis.** Measure Euclidean position distance to the nearest same-family training source and report metres plus domain-diagonal units. Plot record, late-window, high-band, receiver lag, and coherence against position distance only.
3. **Interpolation versus position extrapolation.** Report five positions within the Marmousi training-position range and three positions outside that range separately. Do not introduce a frequency-generalization claim.
4. **Complete-transient stability.** Evaluate every one of 401 frames, report final-third error slope and worst-time error in addition to means, and fail the claim on any non-finite trajectory or late divergence.
5. **Learned-dispersion analysis.** Report low/middle/high complex spatial-spectrum errors, receiver phase/lag/coherence, and a space-time spectral ridge metric if its synthetic recovery test passes. Compare with analytic FD2/FD4/LWC-84 phase-velocity curves as context, not as an assertion that the learned model is dispersion-free.
6. **Strong baseline and fair speed.** Admit DeepONet only after a train-only overfit gate. Bind parameters, frames, optimizer, updates, hardware, preprocessing, and selection budget. Time LWC-84 and the full learned path on one device with synchronization and all 401 frames materialized.

## Citation guard

Only the CONFIRMED scope above may enter the manuscript as literature fact. PARTIAL items may motivate our protocol but cannot be used to claim achieved performance. REJECTED items remain documented to prevent later claim inflation.
