# Elastic VTI Neural-Operator Slides Design

**Date:** 2026-07-11  
**Target deck:** `reports/no_asvgd_group_meeting_20260710/main.tex`

## Objective

Insert a self-contained Elastic VTI neural-operator section immediately after
the frame `背景：传统 FWI 的强项与盲区` and before
`背景：为什么需要贝叶斯化 FWI`. The section must establish that a fast,
differentiable elastic forward surrogate has been trained and evaluated before
the deck introduces probabilistic FWI.

The section covers three media classes: Uniform, Layered, and Marmousi. It must
show the governing equations, data contract, pretraining and fine-tuning losses,
model-space context, source locations, wavefield comparisons, receiver traces,
and a same-hardware runtime comparison against the traditional FD8 solver.

## Section Size and Narrative

The insertion contains 14 frames: five common frames plus three frames for each
of the three model classes.

### Common frames

1. **Elastic wave equation and boundary conditions**
   - Present the 2-D P--SV displacement equation
     \(\rho \partial_{tt}\mathbf u=\nabla\!\cdot\boldsymbol\sigma+\mathbf f\),
     \(\boldsymbol\sigma=\mathbf C:\boldsymbol\varepsilon(\mathbf u)\).
   - State the isotropic VTI-special-case mapping
     \(C_{11}=C_{33}=\rho v_p^2\), \(C_{44}=\rho v_s^2\), and
     \(C_{13}=C_{11}-2C_{44}\).
   - Show the vertical body-force source, stress-release free surface, and
     left/right/bottom CPML.

2. **Dataset and operator contract**
   - Traditional teacher: FD8 staggered-grid elastic solver on
     `400x400`, 5000 time steps, \(\Delta t=10^{-4}\) s.
   - Saved target: `200x200x101x2`; training/evaluation target after
     antialiased downsampling is `128x128x101x2` for Uniform/Layered and
     `160x160x101x2` for Marmousi.
   - Inputs: time, \(v_p\), and \(v_s\); outputs: \(u_x\) and \(u_z\).
   - Source is a 25 Hz vertical body force at the marked physical location.

3. **Pretraining objective**
   - Display the exact implemented objective:
     \[
     \mathcal L_{\rm pre}=\mathcal L_{\rm rel}^{\rm global}
     +10^{-4}\mathcal L_{\rm PDE}
     +10^{-5}\mathcal L_{\rm energy}
     +10^{-3}\mathcal L_{\rm receiver}.
     \]
   - Note that MSE and gradient-loss weights are zero in the accepted configs.
   - Briefly map each term to fidelity, physics consistency, stability, and
     receiver-domain accuracy.

4. **Component-balanced fine-tuning objective**
   - Display the exact implemented fine-tuning objective:
     \[
     \mathcal L_{\rm ft}=\frac12\left(
       \mathcal L_{\rm rel}^{u_x}+\mathcal L_{\rm rel}^{u_z}\right)
     +10^{-4}\mathcal L_{\rm PDE}
     +10^{-5}\mathcal L_{\rm energy}
     +10^{-3}\mathcal L_{\rm receiver}.
     \]
   - Explain that the component-balanced term prevents the lower-energy
     \(u_x\) channel from being hidden by a global vector norm.
   - Record the accepted fine-tuning hyperparameters: AdamW,
     learning rate \(2\times10^{-5}\), weight decay \(10^{-5}\), and gradient
     clipping at 0.5.

5. **Cross-model result overview**
   - Show \(u_x\)/\(u_z\) full-field relative L2 and receiver-line relative L2
     for Uniform, Layered, and Marmousi.
   - Show Marmousi before and after component-balanced fine-tuning.
   - Use only results evaluated on fixed, explicitly recorded validation indices.

### Per-model frames

Each model class receives the same three-frame sequence.

1. **Velocity/source and \(u_x\) wavefield**
   - Left: \(v_p\), \(v_s\), and a clearly visible source marker.
   - Right: Reference, Prediction, and Error at 0.165 s and 0.335 s.
   - Use matched color limits for Reference and Prediction and a separate,
     clearly labelled error scale.

2. **\(u_z\) wavefield**
   - Reference, Prediction, and Error at the same two times.
   - Include a small source/medium context inset so the frame remains
     interpretable on its own.

3. **Receiver traces and runtime**
   - Left/main: \(u_x\) and \(u_z\) receiver gathers plus representative trace
     overlays.
   - Right: FD8 versus PINO runtime and speedup, with workload and device stated.
   - Include one sentence interpreting waveform agreement rather than reporting
     the number alone.

## Result Sources and Acceptance Rules

- Uniform:
  `artifacts/elastic_vti_pino_uniform/evaluation`.
- Layered:
  `artifacts/elastic_vti_pino_layered/evaluation`.
- Marmousi pretraining:
  `artifacts/elastic_vti_pino_marmousi/evaluation`.
- Marmousi fine-tuning fallback:
  `artifacts/elastic_vti_pino_marmousi_component_balanced_short/evaluation_fixed4`.
- Marmousi full fine-tuning may replace the fallback only if training has
  completed and the checkpoint is re-evaluated on the same fixed indices
  `[3, 23, 40, 48]` with provenance saved in its evaluation directory.

No metric is copied from training logs when a fixed-index physical-space
evaluation metric is available. The existing receiver-center \(u_x\) relative
error is not highlighted because symmetry makes the true center trace nearly
zero and the ratio unstable.

## Runtime Benchmark Contract

Runtime numbers must be measured afresh rather than reusing the acoustic
`976 s` baseline.

- Device: the same NVIDIA RTX 3090 for both methods.
- Scope: single-sample forward evaluation; exclude HDF5 loading, plotting, and
  file writing.
- Traditional method: project FD8 staggered-grid elastic solver, physical
  `400x400` grid, 5000 steps, output sampled to the stored target contract.
- Neural operator: batch size 1, `128x128x101` output for Uniform/Layered and
  `160x160x101` output for Marmousi, in both displacement components.
- Synchronize CUDA before and after every timed region.
- PINO timing: five warmups followed by 20 measured repetitions; report median
  and interquartile range.
- FD8 timing: one warmup followed by three measured repetitions; report median
  and range.
- Repeat for Uniform, Layered, and Marmousi samples. If stencil runtime is
  effectively identical across media, retain per-model measurements but state
  that medium complexity does not change the stencil workload materially.
- Every timing frame must state the unequal internal resolutions so the speedup
  is interpreted as end-to-end surrogate acceleration for the accepted output
  task, not equal-grid kernel acceleration.

## Figure Production

Create a focused figure-generation script under
`reports/no_asvgd_group_meeting_20260710/` that reads the recorded evaluation
artifacts and timing report. It will generate compact slide-native figures in
the deck's `figures/` directory. Existing evaluation images remain untouched.

The figures will follow the deck palette (`deepblue`, `accent`, `forest`) and
use large Chinese/Latin labels suitable for projection. Raster figures must be
at least 160 dpi. Source markers must remain visible after embedding.

## Beamer Integration

- Add `\section{弹性波神经算子正演}` at the insertion point.
- Preserve the existing Tsinghua theme, colors, fonts, title metadata, and
  navigation style.
- Use minimal explanatory text and let figures occupy most of each result frame.
- Do not change the later probabilistic-FWI claims or figures.
- Keep all new paths relative through the existing `\graphicspath`.

## Verification

1. Regenerate all elastic slide figures from recorded artifacts.
2. Run the runtime benchmark and preserve its JSON/CSV provenance.
3. Compile with the deck's existing XeLaTeX/latexmk workflow until references
   settle.
4. Fail on missing figures, LaTeX errors, or overfull boxes introduced by the
   new section.
5. Render the complete PDF to slide images and inspect the 14 inserted frames
   for clipping, unreadable labels, source-marker visibility, and consistent
   color scales.
6. Confirm that the original probabilistic-FWI section begins immediately after
   the new elastic section and remains unchanged in content.
