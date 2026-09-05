from __future__ import annotations

import yaml

from fno_acoustic.config import load_config


def test_config_parse_and_expand(tmp_path, monkeypatch):
    monkeypatch.setenv("PINODATA", str(tmp_path))
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump({"data": {"path": "$PINODATA/file.hdf5"}}), encoding="utf-8")
    cfg = load_config(path)
    assert "$" not in cfg["data"]["path"]
