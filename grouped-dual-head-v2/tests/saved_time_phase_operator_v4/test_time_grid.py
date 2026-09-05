import pytest
import torch

from saved_time_phase_operator_v4.time_grid import SavedTimeGrid


def test_saved_time_grid_maps_exact_values_to_indices():
    grid = SavedTimeGrid.from_values(torch.linspace(0.0, 1.0, 401, dtype=torch.float64))
    values = torch.tensor([[0.0, 0.25, 1.0]], dtype=torch.float64)

    assert torch.equal(grid.indices(values), torch.tensor([[0, 100, 400]]))


def test_saved_time_grid_rejects_midpoints():
    grid = SavedTimeGrid.from_values(torch.linspace(0.0, 1.0, 401, dtype=torch.float64))

    with pytest.raises(ValueError, match="stored HDF5 time"):
        grid.indices(torch.tensor([[0.00125]], dtype=torch.float64))


def test_saved_time_grid_rejects_nonuniform_or_nonmonotonic_axes():
    with pytest.raises(ValueError, match="uniform"):
        SavedTimeGrid.from_values(torch.tensor([0.0, 0.1, 0.25]))
    with pytest.raises(ValueError, match="strictly increasing"):
        SavedTimeGrid.from_values(torch.tensor([0.0, 0.1, 0.1]))


def test_saved_time_grid_preserves_requested_device():
    grid = SavedTimeGrid.from_values(torch.linspace(0.0, 1.0, 401, dtype=torch.float64))
    requested = torch.tensor([0.5], device="cuda" if torch.cuda.is_available() else "cpu")

    assert grid.indices(requested).device == requested.device
