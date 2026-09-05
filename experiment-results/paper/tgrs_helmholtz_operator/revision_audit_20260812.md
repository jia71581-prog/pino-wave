# TGRS manuscript revision audit

Date: 2026-08-12

Authoritative manuscript: `manuscript.tex`. The Markdown manuscript is a historical development snapshot and was intentionally not synchronized.

## Claim-to-evidence map

| Claim | Evidence status | Manuscript treatment |
|---|---|---|
| One time-independent spectral state reconstructs every queried time | Supported by construction | Retained as query invariance; explicitly separated from physical accuracy |
| The truncated representation can reach relative L2 below 0.05 | Development only, three records | Scoped to the zero-training oracle and exact record count |
| The smoothed-velocity background improves prediction | Development only, reused three-record triplet | Reported descriptively; population generalization language removed |
| Render output-rank loss causes the late-coda error | Partial diagnostic evidence | Reframed as the leading, testable architectural hypothesis rather than a proven root cause |
| The method is more accurate than FNO3D, U-NO, factorized FNO, or DeepONet | Unsupported because the pilot baselines collapsed | Numerical comparison table removed; overfit sanity gate required |
| The method generalizes across velocity populations | Not authorized | Removed from title, abstract, Results, Discussion, and Conclusion |
| The method is faster than the numerical solver | Not measured end to end | Withheld; background and adaptation costs remain inside the runtime boundary |

## Changes made

- Replaced the method-heavy solver title with an evidence-bounded representation-and-diagnostics title.
- Rebuilt the abstract around one contribution and one quantitative development result.
- Converted the Introduction contribution list into a single narrative claim hierarchy.
- Defined query invariance as a shared predicted state, not guaranteed physical phase accuracy.
- Marked G3 as withheld from fitting but reused for development choices.
- Removed the unbound `0.053` baseline table, which conflicted with the `0.076` development table and used collapsed baselines.
- Reframed the late-coda analysis from a unique root cause to a ranked hypothesis.
- Preserved the IEEE/TGRS structure, equations, citation keys, figures, and development values that remain traceable.

## Static verification

- Every citation key used by `manuscript.tex` has a matching `\bibitem`.
- Every `\ref` and `\eqref` target has a matching `\label`.
- LaTeX environment counts are balanced by static inspection.
- No submission PDF was built because this host does not provide `pdflatex`, `latexmk`, `tectonic`, or `chktex`.

## Citation audit

The local citation scan found no placeholders or duplicate citation keys. A bounded metadata check corrected four material errors while preserving the existing LaTeX keys:

- `yang2023seismic`: corrected the title to *Rapid Seismic Waveform Modeling and Inversion with Neural Operators* and added IEEE TGRS volume, pages, and DOI.
- `kong2023freqbias`: corrected the title, publication year to 2025, venue to *Seismological Research Letters*, and DOI.
- `mscalefno`: corrected the authors to Zhilin You, Zhenli Xu, and Wei Cai and added arXiv:2412.20183.
- `li2021pino`: corrected the formal publication to *ACM/IMS Journal of Data Science* 1(3), 2024, with DOI 10.1145/3648506.

The recent references that carry central novelty claims still require a final publisher-level claim-to-source audit before submission.

## Evidence required before submission

1. Bind the accepted parent checkpoint, manifest, code, configuration, and metric digests.
2. Freeze adaptation and abstention settings using train-only evidence.
3. Run the complete group-disjoint validation once, followed by `test_id` once only if validation passes.
4. Measure same-device synchronized mean and P95 end-to-end runtime, including every online component.
5. Establish non-collapsed, parameter- and budget-matched baselines that pass an overfit sanity gate.
6. Run a parameter-matched time-query decoder ablation and background, residual, carrier, and full-model component ablations.
7. Test the rank-preserving architecture under a preregistered train-only promotion gate.

## High-value follow-up experiments

- Add WaveBench transfer with official splits and compatible official checkpoints.
- Add separate out-of-distribution axes for frequency, topology, source geometry, and resolution.
- Add a downstream FWI-gradient, inversion-convergence, or reverse-time-migration evaluation.
- Report learning curves by independent velocity group and include failure cases and abstention-risk curves.
