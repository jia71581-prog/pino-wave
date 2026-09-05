# Phase4b fixed-19-Hz rank-15 multi-source case study

This folder binds the final composite figure to Phase4b checkpoint update 265
(`f6efbb81dd1e9baab0eb34b32b125e2cb58cd3b3292ee0db33a29194c84bfea1`).
The true Marmousi velocity model is fixed and only the eight registered source
positions change. Source frequency is fixed at 19 Hz.

The eight all-401-frame relative L2 errors range from 0.15499 to 0.20372 and
average 0.17688. The five in-range positions average 0.17728; the three
outside-range positions average 0.17622. This is one velocity-slice case study,
not a population estimate. It is separate from the reused G3 Marmousi record,
whose checkpoint-bound all-401-frame error is 0.04507 (4.51%).

`prediction_manifest.json` proves that all eight predictions were sealed before
reference wavefields were accessed. `report.json` binds the sealed references,
per-position metrics, background attribution, checkpoint, protocol, and final
figure hashes. The large sealed wavefields remain at the canonical paths listed
in the manifest rather than being duplicated here.
