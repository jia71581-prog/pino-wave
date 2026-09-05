from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .grid import AcousticGrid, BoundaryConfig, OutputTimeGrid


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return resolve_config(config)


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = copy.deepcopy(config)
    paths = resolved.setdefault("paths", {})
    for key, value in list(paths.items()):
        if value is not None:
            paths[key] = str(Path(value).expanduser())
    resolved["config_sha256"] = config_sha256(resolved, include_existing_hash=False)
    return resolved


def config_sha256(config: dict[str, Any], *, include_existing_hash: bool = False) -> str:
    payload = copy.deepcopy(config)
    if not include_existing_hash:
        payload.pop("config_sha256", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def grid_from_config(config: dict[str, Any]) -> AcousticGrid:
    grid = config.get("grid", {})
    return AcousticGrid(
        nx=int(grid.get("nx", 400)),
        nz=int(grid.get("nz", 400)),
        dx_m=float(grid.get("dx_m", 5.0)),
        dz_m=float(grid.get("dz_m", 5.0)),
        lx_m=float(grid.get("lx_m", 2000.0)),
        lz_m=float(grid.get("lz_m", 2000.0)),
        centering=str(grid.get("centering", "cell")),
    )


def boundaries_from_config(config: dict[str, Any]) -> BoundaryConfig:
    b = config.get("boundaries", {})
    return BoundaryConfig(
        top=str(b.get("top", "free_surface_dirichlet")),
        left=str(b.get("left", "cpml")),
        right=str(b.get("right", "cpml")),
        bottom=str(b.get("bottom", "cpml")),
        npml=int(b.get("npml", 40)),
        cpml_target_reflection=float(b.get("cpml_target_reflection", 1.0e-8)),
        cpml_polynomial_order=int(b.get("cpml_polynomial_order", 3)),
        cpml_outside_physical_domain=bool(b.get("cpml_outside_physical_domain", True)),
    )


def time_from_config(config: dict[str, Any]) -> OutputTimeGrid:
    t = config.get("time", {})
    return OutputTimeGrid(nt_out=int(t.get("nt_out", 61)), dt_out_s=float(t.get("dt_out_s", 0.01)))


def artifact_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["artifact_root"])


def shard_root(config: dict[str, Any]) -> Path:
    return Path(config["paths"]["shard_root"])
