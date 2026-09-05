import pytest
import torch

from grouped_ufno_mionet import OperatorConfig, SourceParameters, SingleSourceRecord
from grouped_ufno_mionet.model.mionet import mionet_contract
from grouped_ufno_mionet.model.operator import GroupedSingleSourceUFNOMIONetOperator


def test_default_contract_and_single_source_validation():
    cfg = OperatorConfig()
    assert cfg.model.rank == 64
    assert cfg.model.width == 96
    with pytest.raises(ValueError, match="exactly one source"):
        SingleSourceRecord(torch.ones(1, 9, 9), torch.ones(2, 5))


def test_mionet_rank_product():
    medium = torch.tensor([[1.0, 2.0]])
    source = torch.tensor([[3.0, 4.0]])
    trunk = torch.tensor([[[5.0, 6.0], [7.0, 8.0]]])
    torch.testing.assert_close(mionet_contract(medium, source, trunk), torch.tensor([[63.0, 85.0]]))


@pytest.mark.parametrize("width,rank", [(16, 8)])
def test_query_dense_and_velocity_gradient(width, rank):
    operator = GroupedSingleSourceUFNOMIONetOperator(width=width, rank=rank)
    velocity = torch.full((1, 1, 33, 33), 2000.0, requires_grad=True)
    source = torch.tensor([[1000.0, 500.0, 10.0, 0.1, 1.0]])
    coords = torch.tensor([[[100.0, 200.0, 0.0], [400.0, 800.0, 0.2]]])
    query = operator.query_pressure(velocity, source, coords)
    assert query.shape == (1, 2)
    assert query[0, 0] == 0
    query[:, 1].sum().backward()
    assert torch.isfinite(velocity.grad).all() and velocity.grad.abs().sum() > 0
    dense = operator.predict_wavefield(velocity.detach(), source, torch.linspace(0.0, 0.4, 3))
    assert dense.shape == (1, 3, 33, 33)


def test_shared_medium_source_isolation():
    operator = GroupedSingleSourceUFNOMIONetOperator(width=16, rank=8)
    velocity = torch.full((1, 1, 33, 33), 2000.0)
    source = torch.tensor([[1000.0, 500.0, 10.0, 0.1, 1.0], [900.0, 700.0, 12.0, 0.1, 1.0]])
    coords = torch.tensor([[[100.0, 200.0, 0.3]], [[100.0, 200.0, 0.3]]])
    before = operator.query_pressure(velocity, source, coords)
    changed = source.clone(); changed[1, 0] += 100.0
    after = operator.query_pressure(velocity, changed, coords)
    torch.testing.assert_close(after[0], before[0])
    assert not torch.allclose(after[1], before[1])
