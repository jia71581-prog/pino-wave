from __future__ import annotations

import numpy as np
import pytest

from scripts.audit_source_configuration_distance import source_position_distances


def test_nearest_source_position_distances() -> None:
    train = np.asarray([[0.0, 0.0], [100.0, 100.0]])
    query = np.asarray([[0.0, 0.0], [50.0, 50.0]])
    values = source_position_distances(train, query, domain_x_m=100.0, domain_z_m=100.0)
    assert values["nearest_spatial_distance_domain_diagonal"][0] == 0.0
    assert values["nearest_spatial_distance_domain_diagonal"][1] == pytest.approx(0.5)
    assert values["nearest_spatial_distance_m"][1] == pytest.approx(np.sqrt(5000.0))


def test_distance_rejects_empty_support() -> None:
    with pytest.raises(ValueError):
        source_position_distances(
            np.empty((0, 2)), np.ones((1, 2)), domain_x_m=1.0, domain_z_m=1.0
        )
