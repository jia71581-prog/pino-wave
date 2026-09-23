# Group 3: Coarse-grid LWC-84 dispersion comparison — method

## What
The dataset's own LWC-84 + three-sided CFS-CPML solver re-run on coarse grids
for the two group-1 records, interpolated back to 201x201 and scored against
stored truth, to quantify classical numerical dispersion vs the neural
operator's error character.

## Why 51x51, not 50x50
The solver hard-requires odd node-centred grids (`LWC84CPMLSolver.__init__`
raises on even nx/nz). 51x51 nodes at dx = 40 m spans the same 2000 m domain
with coarse nodes exactly on fine nodes (stride 4 of the 201 grid) — the
nearest admissible configuration to the requested 50x50. 101x101 (dx = 20 m,
stride 2) is included as a second point on the resolution trend.

## Medium / source consistency
- Velocity: stored 201x201 velocity nodally decimated (`v[::4, ::4]`,
  `v[::2, ::2]`). No anti-alias smoothing — the coarse error therefore
  includes medium-sampling error on top of pure time-stepping dispersion;
  both are unavoidable consequences of solving at that resolution, and the
  paper text should say "coarse-grid solution error" when citing the totals.
- Source: physical coordinates + Ricker parameters passed to the solver,
  which builds its own bilinear point-source stencil with 1/(dx*dz) delta
  normalisation on the coarse grid (`bilinear_point_source`). No manual
  wavelet or source-map downsampling.
- Time stepping: dt = 2.5e-4 s (native-protocol dt; CFL at dx=40 m is 0.04-0.10,
  comfortably stable), 401 output frames at 2.5 ms, npml = 20 nodes,
  c_ref = 6750 m/s, float32, free-surface top. This mirrors the frozen
  native-grid benchmark protocol (`benchmark_classical_lwc84_runtime_accuracy.py`).
- QC: solver metrics checked per solve (finite, lwc_qmax < 1, free-surface
  row exactly zero).

## Back-interpolation and metric
Coarse (401, n, n) fields are bilinearly upsampled (`align_corners=True`,
exact at shared nodes) to (401, 201, 201) and scored with the paper metric:
relative L2 in float64 over the future window (onset+8..400) and its three
equal bands (`np.array_split`). The neural prediction is scored identically.

## Hardware
CPU only (task constraint). Wall times (12-21 s/solve) are provenance only —
NOT comparable to the GPU runtime table (group 4).

## Key numbers (future relL2, bands early/mid/late)
| record | neural | LWC 51x51 (dx=40 m) | LWC 101x101 (dx=20 m) |
|---|---|---|---|
| marmousi_00076 | 0.279 (0.124/0.245/0.409) | 0.514 (0.470/0.514/0.558) | 0.120 (0.122/0.118/0.119) |
| layered_00299 | 0.196 (0.108/0.311/0.552) | 0.103 (0.066/0.138/0.328) | 0.030 (0.020/0.046/0.065) |

Signature contrast (figures + NUMBERS.json):
- Coarse LWC: phase lag (+2.5..+7.5 ms receiver lags on marmousi at dx=40 m),
  spectral downshift of the receiver trace (centroid 12.97 -> 10.66 Hz on
  marmousi), error roughly flat-in-time (bands nearly equal) — classic
  dispersion, worst where velocity structure is fine (marmousi).
- Neural operator: in-phase early arrivals (0 ms lag at most receivers),
  error grows with time (bands 0.12 -> 0.41), late-coda amplitude deficit
  rather than phase error.

## Limitations
- Two records; trends match the family physics (marmousi has the finest
  structure) but are not a family census.
- dx = 40 m gives ~3.3 points per minimum wavelength at f0~11.5 Hz and
  vmin = 1500 m/s (marmousi); that is below the ~4 ppw eighth-order comfort
  zone — which is the point of the demonstration, but it means the 51x51
  number is dominated by dispersion+medium sampling, not solver failure
  (QC passed, energies bounded).
- The layered record's coarse solves benefit from vmin = 2068 m/s
  (~4.5 ppw at dx=40 m), so coarse LWC beats the neural operator there;
  reported as-is.

## Products
- `SOLVES.json` (all solves, QC, CFL, wall times), `NUMBERS.json` (comparison
  numbers incl. spectra), `dispersion_<sid>.{png,pdf}`,
  `<sid>_coarse{51,101}_{native,on201}.npy`
- Scripts: `../scripts/run_group3_coarse_lwc.py`,
  `../scripts/make_group3_dispersion_figures.py`
