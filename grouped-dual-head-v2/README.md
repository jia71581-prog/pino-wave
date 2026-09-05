# Fourier Neural Operator for Acoustic Wavefield Simulation
<div style="text-align: right; font-family: Helvetica, Arial, sans-serif; color: #555; margin-top: 20px;">
  <strong>Author: Yang Cui</strong><br>
  King Fahd University of Petroleum and Minerals, Saudi Arabia<br>
  Uppsala University, Uppsala, Sweden
</div> <br>

This repository provides **2D and 3D** acoustic wavefield simulation codes based on Fourier Neural Operators (FNO). It includes data preparation, model architecture, training loops, inference scripts, and result comparisons.

The repo is designed for **educational purposes**. You can directly train the models and easily modify them for your own experiments.

## LWC-84 acoustic dataset (401 solver nodes → 201 saved nodes)

The node-centred 2 km LWC-84/CFS-CPML dataset implementation is documented in
[`docs/data/acoustic_lwc84_2km_401x401_to_201_v1.md`](docs/data/acoustic_lwc84_2km_401x401_to_201_v1.md),
with the numerical scheme and CPML equations in
[`docs/numerics/lwc84_cpml.md`](docs/numerics/lwc84_cpml.md). The frozen contract saves
401 frames over 0–1 s on the 201×201 anti-aliased storage grid. Production remains
hard-blocked until the conservative disk-space gate passes; the user-authorized Marmousi
derivative is explicitly marked as a normalized-coordinate interpolation.

The new medium/source-decoupled continuous operator, adaptive query training, and arbitrary
receiver interfaces are documented in
[`continuous_wave_operator/README.md`](continuous_wave_operator/README.md).

## Phase-aligned Complex-FNO Multi-Input DeepONet V3

The experimental V3 operator in [`grouped_ufno_mionet_v3/`](grouped_ufno_mionet_v3/)
combines a learned-complex multiscale U-FNO medium encoder with source, differentiable
travel-time, and periodic-coordinate branches in a multi-input DeepONet/MIONet. It keeps
V2's encode-once medium reuse, local pyramid, source-map features, query residual,
source/time FiLM dense path, and separate true-target supervision for point and complete
wavefield outputs. Each record contains exactly one source; receiver observations are not
model inputs.

The bounded uniform/layered/marmousi nine-record accuracy gate passed on 2026-07-17. The
implementation, commands, checkpoint identities, figures, metrics, and limitations are
recorded in
[`docs/superpowers/reports/2026-07-17-phase-aligned-complex-fno-mionet-v3-results.md`](docs/superpowers/reports/2026-07-17-phase-aligned-complex-fno-mionet-v3-results.md).
This gate verifies exact saved times only. Continuous-time generalization still requires a
separately authorized pilot using interpolated-time supervision; FWI, anomaly media, and a
production launch remain out of scope.

## Exact-time full-support recovery

The registered recovery run in
[`configs/saved_time_v4/full_support_adamw_batch48.yaml`](configs/saved_time_v4/full_support_adamw_batch48.yaml)
uses every 2,240 training record in each epoch, four exact stored frames per appearance,
effective batch 48, and staged unfreezing of the complete V4 operator. The physical
microbatch is 12 because full-model backpropagation at 16 exceeds a 24 GB GPU; whole-forward
checkpointing keeps the measured smoke peak at 20.11 GB. The read-only schedule/time audit is:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python scripts/audit_saved_time_v4_full_support.py \
  --config configs/saved_time_v4/full_support_adamw_batch48.yaml
```

The gated supervisor waits for the fixed-panel three-epoch pilot, launches the 50-epoch run
only when the pilot gate passes, and then streams all 480 held-out records over all 401 exact
stored time indices. Runtime logs and checkpoints live under
`/data/jiayh/saved_time_v5_training_recovery_batch48/`.

## 2D Acoustic Wavefield Simulation using FNO

Yang, Yan, et al. "Seismic wave propagation and inversion with neural operators." *The Seismic Record* 1.3 (2021): 126-134.<br>

This project evaluates the effectiveness of the Fourier Neural Operator (FNO; Li et al., 2020) for simulating acoustic wavefields in the time-spatial domain. We utilize a 2D velocity model extracted from the OpenFWI dataset to generate wavefields. A total of 50 wavefields, each comprising 1,000 snapshots, are split into 42 training and 8 validation datasets. The FNO model was trained for 1,000 epochs, with each epoch taking approximately 6.5 seconds on a workstation equipped with an NVIDIA RTX A4500 GPU. To facilitate ease of use, we have provided pre-trained models, allowing users to test the simulation during the course and experiment with the training code later.
![2D Results](https://github.com/cuiyang512/FNO-Acoustic-Wave-Simulation/blob/main/figs/2d_results_fno.png)

## 3D Acoustic Wavefield Simulation using Modified UFNO

3D wavefield modeling is significantly more computationally demanding than the 2D case. To address this, we adopt the **U-Net enhanced Fourier Neural Operator (UFNO)**, which augments the standard FNO block with a parallel U-Net branch. This improves the model's ability to capture fine-grained temporal and spatial features.

To fit the model on a single NVIDIA RTX 3090 GPU, we modified the original UFNO by reducing the number of dense connections, substantially lowering memory usage while maintaining strong performance.

### Implementation Details

Target wavefield data were generated using the **Deepwave** finite-difference solver on a $96 \times 96 \times 96$ velocity model with **10 m** grid spacing. The acoustic wave equation was solved with:
- Fourth-order spatial discretization
- 20-cell Perfectly Matched Layer (PML) boundaries
- 5 Hz Ricker wavelet source
- 1.0 s simulation time, from which **20 spatiotemporal snapshots** were uniformly sampled

The UFNO surrogate model learns to map the static velocity model and source location directly to the full time-evolving wavefield. It was trained on **30 source locations** and evaluated on **5 unseen test sources**.

As shown in the results below, the data-driven surrogate accurately reproduces the spatiotemporal dynamics of the propagating wavefield. Rather than producing a smoothed kinematic approximation, the model captures high-frequency effects such as **diffraction and multipathing**. Errors remain low throughout the simulation, indicating that the surrogate preserves both phase coherence and amplitude fidelity even after the wavefront interacts with structural heterogeneities.
![3D Results](https://github.com/cuiyang512/FNO-Acoustic-Wave-Simulation/blob/main/figs/ufno_results_comparison_new.png)

# Reference
    We utilized the FNO-torch 1.6.0 in this package:  https://github.com/zongyi-li/fourier_neural_operator/tree/master/FNO-torch.1.6
    
    We also made some modifications using this package: https://github.com/yanyangg/AcousticNeuralOperator
    
    We used Devito to generate the training samples: https://github.com/devitocodes/devito


## Development
    We welcome contributions from the open-source community. To contribute, please contact the development team for guidance.

## Contact
For questions, bug reports, development ideas, or collaboration opportunities, please reach out to:
  - Yang Cui: [yang.cui512@gmail.com](mailto:yang.cui512@gmail.com)
