"""CPU checks of independently expressed local symbols against production kernels."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/audit_acoustic_dispersion_protocol.py"
SPEC = importlib.util.spec_from_file_location("dispersion_audit", PATH)
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


def test_low_wavenumber_limit_without_cos_cancellation():
    for theta in [1.e-4, 1.e-8, 1.e-12]:
        assert abs(m.symbol_1d(theta) / theta ** 2 - 1.) < 2.e-15
    d = m.dispersion(1.e-8, 1500., 37., 5., .000125)
    assert abs(d["spatial_error"]) < 2.e-15
    assert abs(d["total_error"]) < 2.e-15


@pytest.mark.parametrize("args", [(0,1500,0,5,.001),(1,-1,0,5,.001),(1,1500,91,5,.001),
                                  (1,1500,0,0,.001),(1,1500,0,5,float("nan")),
                                  (float("inf"),1500,0,5,.001)])
def test_invalid_inputs(args):
    with pytest.raises(ValueError):
        m.dispersion(*args)


def test_spectral_stability_is_not_monotone_branch_or_nyquist():
    q = np.array([0., 6., 9., 12., 12.1])
    phases = m.phase_from_q(q)
    assert phases[0] == phases[3] == 0.
    assert phases[1] > phases[2] > phases[3]
    assert np.isnan(phases[4])
    d = m.dispersion(1., 6750., 0., 5., .001)
    assert d["mode_spectral_stable"] and not d["full_spectrum_stable"]
    d = m.dispersion(200., 1500., 0., 5., .000125)
    assert d["grid_alias"] and np.isnan(d["total_error"])
    with pytest.raises(ValueError):
        m.phase_from_q(-1.)


def test_restriction_amplitude_energy_and_alias():
    d = m.restriction_transfer(30., 1500., 0.)
    assert abs(d["amplitude"] - .818135621484342) < 1.e-12
    assert d["energy"] == d["amplitude"] ** 2
    assert not d["output_alias"]
    d = m.restriction_transfer(100., 1500., 0.)
    assert d["output_alias"] and not d["input_alias"]


def test_rectangular_crop_overlap():
    assert m.crop_overlap((0,0),(50,0)) == .975
    assert m.crop_overlap((0,0),(2000,0)) == 0.
    assert m.crop_overlap((0,0),(500,250),1000,500) == .25
    assert m.crop_overlap((0,0),(0,0),1000,500) == 1.
    with pytest.raises(ValueError):
        m.crop_overlap((float("nan"),0),(0,0))


def test_ricker_energy_band():
    f,w,b = m.band(30.)
    assert len(f) == 258 and 30. in f
    assert abs(w.sum() - 1.) < 1.e-15
    assert b[0] < 30. < b[1]
    assert abs(b[1] - 53.7337226484) < 1.e-8


def test_production_laplacian_step_and_restriction():
    assert m.production_check()["passed"]


def test_changed_input_hash_fails(tmp_path):
    path = tmp_path / "metadata.json"
    path.write_text("original")
    expected = {str(path): m.sha(path)}
    m.verify_bindings(expected)
    path.write_text("changed")
    with pytest.raises(RuntimeError, match="Frozen binding changed"):
        m.verify_bindings(expected)


def test_metadata_preserves_manifest_ood_split_label(tmp_path):
    path = tmp_path / "manifest.jsonl"
    rows = [dict(medium_type="marmousi", split=split, crop_x0_m=x, crop_z0_m=0.)
            for split, x in [("train", 0.), ("validation", 50.), ("ood_canonical", 3000.)]]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    summary, _ = m.metadata_audit(path)
    assert summary["unique_crop_counts"] == dict(train=1, validation=1, ood_canonical=1)
    assert summary["validation"]["max_overlap"] == .975
    assert summary["ood_canonical"]["max_overlap"] == 0.
