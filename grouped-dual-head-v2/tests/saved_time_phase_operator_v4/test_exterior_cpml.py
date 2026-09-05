from __future__ import annotations

import numpy as np
import torch
import yaml

from fno_acoustic.data_generation.cpml import build_cfs_cpml_profiles
from fno_acoustic.data_generation.grid import AcousticGrid, BoundaryConfig
from saved_time_phase_operator_v4.exterior_cpml import (
    ExteriorCPMLContract,
    build_saved_exterior_cpml_profiles,
    contract_from_dataset_config,
    crop_physical_domain,
    extend_velocity_to_exterior,
    profile_sampling_report,
)


FROZEN = (
    "/data/jiayh/data/"
    "acoustic_lwc84_2km_401x401_to_201_marmousi1_4m_v2/frozen_config.yaml"
)


def test_registered_contract_has_twenty_exterior_layers_and_free_top():
    config = yaml.safe_load(open(FROZEN, encoding="utf-8"))
    contract = contract_from_dataset_config(config)
    assert contract.cpml_layers == 20
    assert (contract.extended_nz, contract.extended_nx) == (221, 241)
    assert contract.physical_z_slice == slice(0, 201)
    assert contract.physical_x_slice == slice(20, 221)
    assert contract.physical_thickness_m == 200.0
    assert contract.top == "free_surface_dirichlet"
    assert (contract.left, contract.right, contract.bottom) == (
        "cpml",
        "cpml",
        "cpml",
    )


def test_saved_profiles_are_exact_factor_two_samples_of_fine_profiles():
    contract = ExteriorCPMLContract()
    saved = build_saved_exterior_cpml_profiles(
        contract, c_ref_mps=6750.0, dtype=torch.float64
    )
    fine_grid = AcousticGrid(
        nx=401,
        nz=401,
        dx_m=5.0,
        dz_m=5.0,
        lx_m=2000.0,
        lz_m=2000.0,
        centering="node",
    )
    fine = build_cfs_cpml_profiles(
        fine_grid,
        BoundaryConfig(npml=40),
        dt_s=contract.internal_dt_s,
        c_ref_mps=6750.0,
        target_reflection=contract.target_reflection,
        polynomial_order=contract.polynomial_order,
        kappa_max=contract.kappa_max,
        minimum_frequency_hz=contract.minimum_frequency_hz,
        device="cpu",
        dtype=torch.float64,
    )
    report = profile_sampling_report(saved, fine)
    assert max(report.values()) == 0.0
    physical_z, physical_x = contract.physical_z_slice, contract.physical_x_slice
    assert torch.count_nonzero(saved.active_x[physical_z, physical_x]) == 0
    assert torch.count_nonzero(saved.active_z[physical_z, physical_x]) == 0
    assert torch.count_nonzero(saved.active_z[0]) == 0
    assert torch.all(saved.active_x[:, :20])
    assert torch.all(saved.active_x[:, -20:])
    assert torch.all(saved.active_z[-20:])


def test_velocity_extension_and_physical_crop_roundtrip_numpy_and_torch():
    contract = ExteriorCPMLContract(physical_nz=11, physical_nx=13, cpml_layers=2)
    physical = np.arange(11 * 13, dtype=np.float32).reshape(11, 13)
    extended_numpy = extend_velocity_to_exterior(physical, contract)
    assert extended_numpy.shape == (13, 17)
    np.testing.assert_array_equal(crop_physical_domain(extended_numpy, contract), physical)
    np.testing.assert_array_equal(
        extended_numpy[:11, :2], np.repeat(physical[:, :1], 2, axis=1)
    )
    np.testing.assert_array_equal(
        extended_numpy[:11, -2:], np.repeat(physical[:, -1:], 2, axis=1)
    )
    np.testing.assert_array_equal(
        extended_numpy[-2:], extended_numpy[10:11].repeat(2, axis=0)
    )

    physical_torch = torch.from_numpy(physical)[None]
    extended_torch = extend_velocity_to_exterior(physical_torch, contract)
    torch.testing.assert_close(
        crop_physical_domain(extended_torch, contract), physical_torch
    )
