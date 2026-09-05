# DCLP-NO Evidence Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the acoustic dataset, two-cycle/four-cycle observation times, near-surface receiver line, figure cases, parent identity, and source provenance before any LoRA or paper experiment.

**Architecture:** A small `tgrs_dclp_no` package owns paper-specific immutable protocols without changing the legacy CLFC contract. A command-line audit binds those protocols to the existing V3 manifest and writes canonical JSON artifacts used by all later stages.

**Tech Stack:** Python 3.13, PyTorch, NumPy, h5py, PyYAML, pytest, SHA-256.

---

## File Map

- Create `tgrs_dclp_no/__init__.py`: public protocol exports.
- Create `tgrs_dclp_no/protocol.py`: observation and receiver geometry.
- Create `tgrs_dclp_no/io.py`: canonical JSON and atomic writes.
- Create `tgrs_dclp_no/provenance.py`: file/tree hashing and immutable audit payload.
- Create `configs/tgrs_dclp_no/protocol.yaml`: frozen paper protocol.
- Create `scripts/audit_tgrs_dclp_no.py`: dataset/checkpoint/source audit CLI.
- Create `tests/tgrs_dclp_no/test_protocol.py`: timing and receiver tests.
- Create `tests/tgrs_dclp_no/test_provenance.py`: deterministic hash and audit tests.

### Task 1: Add the fixed observation and receiver protocol

**Files:**
- Create: `tests/tgrs_dclp_no/test_protocol.py`
- Create: `tgrs_dclp_no/__init__.py`
- Create: `tgrs_dclp_no/protocol.py`

- [ ] **Step 1: Write the failing protocol tests**

```python
from __future__ import annotations

import torch

from tgrs_dclp_no.protocol import (
    future_indices,
    near_surface_receiver_indices,
    two_cycle_observation_indices,
)


def test_two_cycle_observation_indices_use_nearest_saved_time_with_earlier_tie():
    time_s = torch.arange(401, dtype=torch.float64) * 0.0025
    indices = two_cycle_observation_indices(
        time_s,
        t0_s=0.10,
        f0_hz=20.0,
    )
    assert indices == (80, 120)
    tied = two_cycle_observation_indices(
        torch.arange(9, dtype=torch.float64) * 0.1,
        t0_s=0.0,
        f0_hz=8.0,
    )
    assert tied[0] == 2


def test_future_indices_start_strictly_after_second_observation():
    assert future_indices(401, (80, 120)).tolist() == list(range(121, 401))


def test_near_surface_receiver_line_resolves_to_37_registered_nodes():
    x_m = torch.arange(201, dtype=torch.float64) * 10.0
    z_m = torch.arange(201, dtype=torch.float64) * 10.0
    receivers = near_surface_receiver_indices(x_m, z_m)
    assert receivers == tuple((2, x) for x in range(10, 191, 5))
    assert len(receivers) == 37
```

- [ ] **Step 2: Run the tests and verify the module is absent**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_protocol.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'tgrs_dclp_no'`.

- [ ] **Step 3: Implement the protocol**

Create `tgrs_dclp_no/protocol.py` with:

```python
from __future__ import annotations

from typing import Sequence

import torch


def _axis(values: torch.Tensor | Sequence[float], *, name: str) -> torch.Tensor:
    axis = torch.as_tensor(values, dtype=torch.float64).flatten()
    if axis.numel() < 2 or not bool(torch.isfinite(axis).all()):
        raise ValueError(f"{name} must contain at least two finite values")
    if bool((axis[1:] <= axis[:-1]).any()):
        raise ValueError(f"{name} must be strictly increasing")
    return axis


def _nearest_earlier_tie(axis: torch.Tensor, requested: float) -> int:
    value = torch.tensor(float(requested), dtype=axis.dtype)
    if not bool(torch.isfinite(value)):
        raise ValueError("requested time must be finite")
    distance = torch.abs(axis - value)
    return int(torch.nonzero(distance == distance.min(), as_tuple=False)[0, 0])


def two_cycle_observation_indices(
    time_s: torch.Tensor | Sequence[float],
    *,
    t0_s: float,
    f0_hz: float,
) -> tuple[int, int]:
    axis = _axis(time_s, name="time_s")
    frequency = float(f0_hz)
    onset = float(t0_s)
    if frequency <= 0.0 or not torch.isfinite(torch.tensor([frequency, onset])).all():
        raise ValueError("source onset and positive frequency must be finite")
    observed = (
        _nearest_earlier_tie(axis, onset + 2.0 / frequency),
        _nearest_earlier_tie(axis, onset + 4.0 / frequency),
    )
    if not 0 <= observed[0] < observed[1] < axis.numel() - 1:
        raise ValueError("two-cycle observations leave no blind future interval")
    return observed


def future_indices(time_count: int, observed: tuple[int, int]) -> torch.Tensor:
    count = int(time_count)
    first, second = (int(value) for value in observed)
    if not 0 <= first < second < count - 1:
        raise ValueError("observed indices must be ordered and leave future frames")
    return torch.arange(second + 1, count, dtype=torch.long)


def near_surface_receiver_indices(
    x_m: torch.Tensor | Sequence[float],
    z_m: torch.Tensor | Sequence[float],
    *,
    depth_m: float = 20.0,
    x_start_m: float = 100.0,
    x_stop_m: float = 1900.0,
    x_stride_m: float = 50.0,
) -> tuple[tuple[int, int], ...]:
    x_axis = _axis(x_m, name="x_m")
    z_axis = _axis(z_m, name="z_m")
    requested_x = torch.arange(
        float(x_start_m),
        float(x_stop_m) + 0.5 * float(x_stride_m),
        float(x_stride_m),
        dtype=torch.float64,
    )
    if requested_x.numel() != 37:
        raise ValueError("registered TGRS receiver line must contain 37 receivers")
    z_index = _nearest_earlier_tie(z_axis, float(depth_m))
    x_indices = tuple(_nearest_earlier_tie(x_axis, float(value)) for value in requested_x)
    if not torch.allclose(
        x_axis[torch.tensor(x_indices)],
        requested_x,
        atol=1.0e-9,
        rtol=0.0,
    ) or abs(float(z_axis[z_index]) - float(depth_m)) > 1.0e-9:
        raise ValueError("receiver coordinates are not registered grid nodes")
    if z_index == 0:
        raise ValueError("receiver line cannot lie on the pressure-free top row")
    return tuple((z_index, index) for index in x_indices)
```

Create `tgrs_dclp_no/__init__.py` with:

```python
"""Reproducible evidence pipeline for the acoustic DCLP-NO TGRS paper."""

from .protocol import (
    future_indices,
    near_surface_receiver_indices,
    two_cycle_observation_indices,
)

__all__ = [
    "future_indices",
    "near_surface_receiver_indices",
    "two_cycle_observation_indices",
]
```

- [ ] **Step 4: Run the protocol tests**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_protocol.py
```

Expected: `3 passed`.

### Task 2: Add canonical artifact I/O and provenance hashing

**Files:**
- Create: `tests/tgrs_dclp_no/test_provenance.py`
- Create: `tgrs_dclp_no/io.py`
- Create: `tgrs_dclp_no/provenance.py`

- [ ] **Step 1: Write failing deterministic-provenance tests**

```python
from __future__ import annotations

import json

from tgrs_dclp_no.io import write_json_atomic
from tgrs_dclp_no.provenance import sha256_file, source_manifest


def test_atomic_json_is_canonical_and_finite(tmp_path):
    output = tmp_path / "value.json"
    write_json_atomic({"b": 2, "a": 1}, output)
    assert json.loads(output.read_text()) == {"a": 1, "b": 2}
    assert output.read_text().endswith("\n")


def test_source_manifest_is_stable_and_excludes_generated_caches(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "a.pyc").write_bytes(b"noise")
    first = source_manifest(tmp_path, include_suffixes=(".py",))
    second = source_manifest(tmp_path, include_suffixes=(".py",))
    assert first == second
    assert first["files"] == {
        "a.py": sha256_file(tmp_path / "a.py"),
    }
```

- [ ] **Step 2: Verify the tests fail**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_provenance.py
```

Expected: import failure for `tgrs_dclp_no.io`.

- [ ] **Step 3: Implement canonical JSON writing**

Create `tgrs_dclp_no/io.py` with:

```python
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Mapping


def canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    try:
        return (
            json.dumps(
                dict(payload),
                sort_keys=True,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf8")
    except (TypeError, ValueError) as error:
        raise ValueError("artifact payload must be finite JSON") from error


def write_json_atomic(payload: Mapping[str, object], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{output.name}.",
        suffix=".tmp",
        dir=output.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return output
```

- [ ] **Step 4: Implement file and source-tree manifests**

Create `tgrs_dclp_no/provenance.py` with:

```python
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(
    root: str | Path,
    *,
    include_suffixes: Iterable[str] = (".py", ".yaml", ".yml", ".tex", ".bib"),
) -> dict[str, object]:
    base = Path(root).resolve()
    suffixes = tuple(str(value) for value in include_suffixes)
    excluded_parts = {
        ".pytest_cache",
        ".superpowers",
        "__pycache__",
        "artifacts",
        "paper",
    }
    files = {
        str(path.relative_to(base)): sha256_file(path)
        for path in sorted(base.rglob("*"))
        if path.is_file()
        and path.suffix in suffixes
        and not excluded_parts.intersection(path.relative_to(base).parts)
    }
    digest = hashlib.sha256()
    for name, value in sorted(files.items()):
        digest.update(name.encode("utf8"))
        digest.update(value.encode("ascii"))
    return {
        "schema": "dclp_no_source_manifest_v1",
        "root": str(base),
        "files": files,
        "aggregate_sha256": digest.hexdigest(),
    }
```

- [ ] **Step 5: Run the provenance tests**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_provenance.py
```

Expected: `2 passed`.

### Task 3: Freeze the machine-readable paper protocol

**Files:**
- Create: `configs/tgrs_dclp_no/protocol.yaml`
- Create: `scripts/audit_tgrs_dclp_no.py`

- [ ] **Step 1: Create the fixed protocol configuration**

```yaml
schema: dclp_no_tgrs_protocol_v1
seed: 372
paths:
  source_h5: /home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5
  travel_time_h5: /home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5
  parent_checkpoint: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long/run/best.pt
  parent_identity: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long/run/run_identity.json
expected:
  parent_checkpoint_sha256: 0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f
  train_records: 2240
  validation_records: 480
  test_records: 480
  saved_times: 401
  height: 201
  width: 201
observation:
  first_cycles_after_t0: 2.0
  second_cycles_after_t0: 4.0
  tie_break: earlier
  allowed_truth_snapshot_count: 2
  evaluation: strictly_after_second_snapshot
receivers:
  depth_m: 20.0
  x_start_m: 100.0
  x_stop_m: 1900.0
  x_stride_m: 50.0
  count: 37
figure_cases:
  selection_seed: 372
  per_family: 1
  families: [uniform, layered, marmousi]
```

- [ ] **Step 2: Implement the audit CLI**

Create `scripts/audit_tgrs_dclp_no.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grouped_ufno_mionet_v3.data.index import build_manifest, validate_expected_counts
from tgrs_dclp_no.io import write_json_atomic
from tgrs_dclp_no.protocol import near_surface_receiver_indices
from tgrs_dclp_no.provenance import sha256_file, source_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root", default=str(ROOT))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text())
    manifest = build_manifest(config["paths"]["source_h5"])
    expected = config["expected"]
    validate_expected_counts(
        manifest,
        {
            "train": int(expected["train_records"]),
            "validation": int(expected["validation_records"]),
            "test": int(expected["test_records"]),
        },
    )
    checkpoint = Path(config["paths"]["parent_checkpoint"]).resolve()
    checkpoint_sha = sha256_file(checkpoint)
    if checkpoint_sha != str(expected["parent_checkpoint_sha256"]):
        raise ValueError("parent checkpoint SHA-256 mismatch")
    if len(manifest.time_s) != int(expected["saved_times"]):
        raise ValueError("saved-time count mismatch")
    if len(manifest.z_m) != int(expected["height"]) or len(manifest.x_m) != int(expected["width"]):
        raise ValueError("stored-grid geometry mismatch")
    receivers = near_surface_receiver_indices(
        torch.tensor(manifest.x_m),
        torch.tensor(manifest.z_m),
        depth_m=float(config["receivers"]["depth_m"]),
        x_start_m=float(config["receivers"]["x_start_m"]),
        x_stop_m=float(config["receivers"]["x_stop_m"]),
        x_stride_m=float(config["receivers"]["x_stride_m"]),
    )
    if len(receivers) != int(config["receivers"]["count"]):
        raise ValueError("receiver count mismatch")
    payload = {
        "schema": "dclp_no_audit_v1",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_manifest_digest": manifest.digest,
        "dataset_source_sha256": manifest.source_file_sha256,
        "split_counts": manifest.counts_after,
        "families": list(manifest.allowed_medium_types),
        "excluded_families": list(manifest.excluded_medium_types),
        "time_count": len(manifest.time_s),
        "grid_shape": [len(manifest.z_m), len(manifest.x_m)],
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": checkpoint_sha,
        "receivers_zx": [list(value) for value in receivers],
        "receiver_count": len(receivers),
        "source": source_manifest(args.source_root),
    }
    write_json_atomic(payload, args.output)
    print(Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: Compile and audit on CPU**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m py_compile \
  tgrs_dclp_no/__init__.py \
  tgrs_dclp_no/protocol.py \
  tgrs_dclp_no/io.py \
  tgrs_dclp_no/provenance.py \
  scripts/audit_tgrs_dclp_no.py
/home/jiayh/miniconda3/bin/python scripts/audit_tgrs_dclp_no.py \
  --config configs/tgrs_dclp_no/protocol.yaml \
  --source-root . \
  --output artifacts/tgrs_dclp_no/audit/source_manifest.json
```

Expected: both commands exit 0; the second prints the absolute audit path.

### Task 4: Freeze deterministic validation and test manifests

**Files:**
- Create: `tgrs_dclp_no/selection.py`
- Create: `tests/tgrs_dclp_no/test_selection.py`
- Create: `scripts/freeze_tgrs_manifests.py`

- [ ] **Step 1: Test selection before implementation**

```python
from types import SimpleNamespace

from tgrs_dclp_no.selection import select_family_cases


def test_case_selection_is_group_distinct_and_family_balanced():
    rows = tuple(
        SimpleNamespace(
            split="validation",
            medium_type=family,
            group_id=f"{family}-{index}",
            sample_id=f"{family}:{index}",
            source_index=index,
        )
        for family in ("uniform", "layered", "marmousi")
        for index in range(4)
    )
    selected = select_family_cases(rows, split="validation", per_family=1, seed=372)
    assert [row.medium_type for row in selected] == ["uniform", "layered", "marmousi"]
    assert len({row.group_id for row in selected}) == 3
```

- [ ] **Step 2: Verify failure**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_selection.py
```

Expected: import failure for `tgrs_dclp_no.selection`.

- [ ] **Step 3: Implement deterministic case selection**

Create `tgrs_dclp_no/selection.py` with:

```python
from __future__ import annotations

import numpy as np


def select_family_cases(records, *, split: str, per_family: int, seed: int):
    if int(per_family) <= 0:
        raise ValueError("per_family must be positive")
    rng = np.random.default_rng(int(seed))
    selected = []
    for family in ("uniform", "layered", "marmousi"):
        candidates = [
            row for row in records
            if row.split == str(split) and row.medium_type == family
        ]
        order = rng.permutation(len(candidates))
        seen = set()
        family_rows = []
        for position in order:
            row = candidates[int(position)]
            if row.group_id in seen:
                continue
            seen.add(row.group_id)
            family_rows.append(row)
            if len(family_rows) == int(per_family):
                break
        if len(family_rows) != int(per_family):
            raise ValueError(f"insufficient distinct {family} groups")
        selected.extend(family_rows)
    return tuple(selected)
```

- [ ] **Step 4: Implement manifest freezing**

Create `scripts/freeze_tgrs_manifests.py` with a CLI that:

```python
manifest = build_manifest(config["paths"]["source_h5"])
figure_rows = select_family_cases(
    manifest.records,
    split="validation",
    per_family=int(config["figure_cases"]["per_family"]),
    seed=int(config["figure_cases"]["selection_seed"]),
)
test_split = str(config["expected"]["test_split"])
test_rows = tuple(row for row in manifest.records if row.split == test_split)
write_json_atomic(
    {
        "schema": "dclp_no_figure_cases_v1",
        "selection_seen_metrics": False,
        "records": [row.__dict__ for row in figure_rows],
    },
    output / "figure_cases.json",
)
write_json_atomic(
    {
        "schema": "dclp_no_test_manifest_v1",
        "records": [row.__dict__ for row in test_rows],
        "record_count": len(test_rows),
    },
    output / "test_records.json",
)
```

The CLI arguments are:

```python
parser.add_argument("--config", required=True)
parser.add_argument("--output-dir", required=True)
```

- [ ] **Step 5: Run selection tests and freeze manifests**

Run:

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q tests/tgrs_dclp_no/test_selection.py
/home/jiayh/miniconda3/bin/python scripts/freeze_tgrs_manifests.py \
  --config configs/tgrs_dclp_no/protocol.yaml \
  --output-dir artifacts/tgrs_dclp_no/manifests
```

Expected: test passes; `figure_cases.json` contains three distinct groups and
`test_records.json` contains 480 records.

### Task 5: Run the evidence-contract verification suite

**Files:**
- Update generated audit only: `artifacts/tgrs_dclp_no/audit/source_manifest.json`

- [ ] **Step 1: Run focused tests**

```bash
/home/jiayh/miniconda3/bin/python -m pytest -q \
  tests/tgrs_dclp_no/test_protocol.py \
  tests/tgrs_dclp_no/test_provenance.py \
  tests/tgrs_dclp_no/test_selection.py
```

Expected: all tests pass.

- [ ] **Step 2: Verify no protected artifact changed**

Run:

```bash
sha256sum \
  ../pretraining/gate4_long/run/best.pt \
  ../pretraining/gate4_long/run/run_identity.json
```

Expected: `best.pt` starts with
`0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f`.

- [ ] **Step 3: Refresh the source checkpoint**

```bash
/home/jiayh/miniconda3/bin/python scripts/audit_tgrs_dclp_no.py \
  --config configs/tgrs_dclp_no/protocol.yaml \
  --source-root . \
  --output artifacts/tgrs_dclp_no/audit/source_manifest.json
```

Expected: exit code 0 and deterministic audit output.
