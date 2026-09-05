# Parameter-Matched Patch-DeepONet Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, validate, and launch an independent 32.29-million-parameter Patch-DeepONet on the exact gate-4 dataset and produce fair raw-parent/DeepONet and two-frame-CLFC/DeepONet comparisons.

**Architecture:** Add an isolated `patch_deeponet_baseline` package. It reuses the existing content-bound V3 manifest, exact `appearance16` time policy, physical normalizer, and registered Eikonal cache, while owning its model, query sampler, losses, checkpoints, training gates, and evaluator. The model combines a four-channel local CNN branch with a source/arrival-aware Fourier trunk and query-local residual head; training uses deterministic 65,536-query record batches split into 16,384-query backward chunks.

**Tech Stack:** Python 3.12, PyTorch 2.8/CUDA 12.8, h5py, NumPy, PyYAML, pytest, Matplotlib, tmux.

---

## Execution Context

Run every command from:

```bash
cd /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/project
```

Use this interpreter:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python
```

The editable workspace has no `.git` directory. Do not initialize a new repository or modify the historical repository. Where this plan normally calls for a commit, run the source-snapshot command introduced in Task 1. If Git metadata is restored later, replace each snapshot with a normal focused commit.

The approved design is:

`docs/superpowers/specs/2026-07-24-parammatched-patch-deeponet-baseline-design.md`

Design SHA-256 at plan time:

`2806059fe06bf7cda356766ec224aefe6918b755ddad0e207260735da04e0cb9`

## File Map

### New package

- `patch_deeponet_baseline/__init__.py` — public baseline interfaces.
- `patch_deeponet_baseline/config.py` — strict YAML schema and configuration digest.
- `patch_deeponet_baseline/audit.py` — provenance, split/leakage checks, and source snapshots.
- `patch_deeponet_baseline/features.py` — target-free static and query descriptor construction.
- `patch_deeponet_baseline/model.py` — parameter-matched local Patch-DeepONet.
- `patch_deeponet_baseline/sampling.py` — deterministic mixture query sampler and pair construction.
- `patch_deeponet_baseline/data.py` — exact-frame record materialization.
- `patch_deeponet_baseline/losses.py` — weighted Huber, hard causality, and paired spatial-gradient loss.
- `patch_deeponet_baseline/checkpoint.py` — identity-bound update-boundary checkpoints and exact resume.
- `patch_deeponet_baseline/training.py` — update loop, validation, optimizer schedule, and JSONL logging.
- `patch_deeponet_baseline/gates.py` — overfit/pilot/main entry decisions.
- `patch_deeponet_baseline/evaluation.py` — dense prediction, metrics, receiver traces, and figures.

### New commands and configuration

- `configs/patch_deeponet/parammatched_seed372.yaml` — frozen production configuration.
- `scripts/audit_patch_deeponet_baseline.py` — read-only audit and schedule census.
- `scripts/train_patch_deeponet_baseline.py` — dry-run, overfit, pilot, and main training CLI.
- `scripts/run_patch_deeponet_pipeline.py` — gated pilot selection and main-run supervisor.
- `scripts/evaluate_patch_deeponet_baseline.py` — fixed three-instance comparison CLI.
- `scripts/check_patch_deeponet_run.py` — live PID/GPU/log/checkpoint verification.
- `run_patch_deeponet_baseline.sh` — durable tmux launcher.

### New tests

- `tests/patch_deeponet_baseline/test_config_and_audit.py`
- `tests/patch_deeponet_baseline/test_features.py`
- `tests/patch_deeponet_baseline/test_model.py`
- `tests/patch_deeponet_baseline/test_sampling.py`
- `tests/patch_deeponet_baseline/test_data.py`
- `tests/patch_deeponet_baseline/test_losses.py`
- `tests/patch_deeponet_baseline/test_checkpoint_and_training.py`
- `tests/patch_deeponet_baseline/test_gates_and_cli.py`
- `tests/patch_deeponet_baseline/test_evaluation.py`

Do not modify the existing parent, CLFC, pristine-project, or historical DeepONet artifacts.

## Task 1: Strict Configuration, Provenance Audit, and Snapshot Fallback

**Files:**

- Create: `patch_deeponet_baseline/__init__.py`
- Create: `patch_deeponet_baseline/config.py`
- Create: `patch_deeponet_baseline/audit.py`
- Create: `configs/patch_deeponet/parammatched_seed372.yaml`
- Create: `scripts/audit_patch_deeponet_baseline.py`
- Create: `tests/patch_deeponet_baseline/test_config_and_audit.py`

- [ ] **Step 1: Write failing strict-config and split-audit tests**

```python
from dataclasses import replace
from pathlib import Path

import pytest

from patch_deeponet_baseline.audit import audit_experiment
from patch_deeponet_baseline.config import BaselineConfig


CONFIG = Path("configs/patch_deeponet/parammatched_seed372.yaml")


def test_production_config_fixes_the_fairness_contract():
    cfg = BaselineConfig.from_yaml(CONFIG)
    assert cfg.train.epochs == 40
    assert cfg.train.macro_records * cfg.train.macros_per_update == 32
    assert cfg.train.seed == 372
    assert cfg.train.appearance_offset == 15
    assert cfg.sampling.frames_per_record == 16
    assert cfg.sampling.queries_per_record == 65_536
    assert cfg.sampling.query_chunk_size == 16_384
    assert cfg.model.target_parameters == 32_294_258
    assert cfg.loss.delta == 0.5
    assert cfg.loss.energy_floor_fraction == 0.01
    assert cfg.loss.spatial_gradient_weight == 0.1


def test_config_rejects_unknown_keys(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("unknown: true\n", encoding="utf8")
    with pytest.raises(ValueError, match="unknown"):
        BaselineConfig.from_yaml(path)


def test_production_audit_binds_counts_hash_and_three_instances(tmp_path: Path):
    cfg = BaselineConfig.from_yaml(CONFIG)
    result = audit_experiment(cfg, output_dir=tmp_path)
    assert result.counts == {"train": 2240, "validation": 480, "test_id": 480}
    assert result.parent_sha256 == (
        "0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f"
    )
    assert set(result.comparison_sample_ids) == {
        "validation:uniform:00003",
        "validation:layered:00082",
        "validation:marmousi:00087",
    }
    assert (tmp_path / "provenance.json").is_file()
    assert (tmp_path / "split_manifest.json").is_file()


def test_audit_fails_closed_on_wrong_parent_hash(tmp_path: Path):
    cfg = BaselineConfig.from_yaml(CONFIG)
    bad = replace(cfg, expected_parent_sha256="0" * 64)
    with pytest.raises(ValueError, match="parent checkpoint"):
        audit_experiment(bad, output_dir=tmp_path)
```

- [ ] **Step 2: Run the tests and verify import/config failures**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline/test_config_and_audit.py -v
```

Expected: FAIL because `patch_deeponet_baseline` does not exist.

- [ ] **Step 3: Implement strict dataclasses and digest**

Use frozen dataclasses with these public fields and reject unknown YAML keys:

```python
@dataclass(frozen=True)
class PathsConfig:
    source_h5: str
    travel_time_h5: str
    normalization_json: str
    parent_checkpoint: str
    parent_eval_dir: str
    clfc_eval_dir: str
    artifact_dir: str


@dataclass(frozen=True)
class ModelConfig:
    static_channels: int = 4
    branch_width: int = 48
    latent_dim: int = 1984
    trunk_hidden: int = 2749
    trunk_layers: int = 5
    fourier_bands: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    target_parameters: int = 32_294_258
    parameter_tolerance_fraction: float = 0.001


@dataclass(frozen=True)
class SamplingConfig:
    frames_per_record: int = 16
    spatial_queries_per_frame: int = 4096
    anchors_per_frame: int = 2048
    queries_per_record: int = 65_536
    query_chunk_size: int = 16_384
    uniform_fraction: float = 0.50
    energy_fraction: float = 0.25
    arrival_fraction: float = 0.25
    importance_cap_median: float = 20.0


@dataclass(frozen=True)
class LossConfig:
    delta: float = 0.5
    energy_floor_fraction: float = 0.01
    hard_causality: bool = True
    hard_causality_lead_cycles: float = 1.0
    spatial_gradient_weight: float = 0.1
    spectrum_weight: float = 0.0
    receiver_weight: float = 0.0
    teacher_weight: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    macro_records: int = 8
    macros_per_update: int = 4
    microbatch_records: int = 1
    seed: int = 372
    appearance_offset: int = 15
    learning_rate_candidates: tuple[float, ...] = (1e-4, 3e-4, 1e-3)
    learning_rate: float | None = None
    pilot_updates: int = 200
    warmup_epochs: int = 2
    minimum_lr_factor: float = 0.05
    weight_decay: float = 1e-6
    gradient_clip: float = 1.0
    adamw_implementation: str = "fused"
    maximum_peak_cuda_gib: float = 23.5


@dataclass(frozen=True)
class ValidationConfig:
    panel_records: int = 48
    frames_per_record: int = 32
    spatial_queries_per_frame: int = 4096
    final_frames_per_record: int = 401
    comparison_sample_ids: tuple[str, ...] = (
        "validation:uniform:00003",
        "validation:layered:00082",
        "validation:marmousi:00087",
    )


@dataclass(frozen=True)
class BaselineConfig:
    paths: PathsConfig
    model: ModelConfig
    sampling: SamplingConfig
    loss: LossConfig
    train: TrainConfig
    validation: ValidationConfig
    expected_parent_sha256: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BaselineConfig":
        with Path(path).open(encoding="utf8") as handle:
            payload = yaml.safe_load(handle) or {}
        allowed = {
            "paths", "model", "sampling", "loss", "train", "validation",
            "expected_parent_sha256",
        }
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown BaselineConfig keys: {unknown}")

        def strict_make(dataclass_type, values):
            values = dict(values or {})
            names = {item.name for item in fields(dataclass_type)}
            extra = sorted(set(values) - names)
            if extra:
                raise ValueError(
                    f"unknown {dataclass_type.__name__} keys: {extra}"
                )
            return dataclass_type(**values)

        model_values = dict(payload.get("model") or {})
        if "fourier_bands" in model_values:
            model_values["fourier_bands"] = tuple(
                float(value) for value in model_values["fourier_bands"]
            )
        train_values = dict(payload.get("train") or {})
        if "learning_rate_candidates" in train_values:
            train_values["learning_rate_candidates"] = tuple(
                float(value) for value in train_values["learning_rate_candidates"]
            )
        validation_values = dict(payload.get("validation") or {})
        if "comparison_sample_ids" in validation_values:
            validation_values["comparison_sample_ids"] = tuple(
                str(value) for value in validation_values["comparison_sample_ids"]
            )
        result = cls(
            paths=strict_make(PathsConfig, payload.get("paths")),
            model=strict_make(ModelConfig, model_values),
            sampling=strict_make(SamplingConfig, payload.get("sampling")),
            loss=strict_make(LossConfig, payload.get("loss")),
            train=strict_make(TrainConfig, train_values),
            validation=strict_make(ValidationConfig, validation_values),
            expected_parent_sha256=str(payload["expected_parent_sha256"]),
        )
        result.validate()
        return result

    def digest(self) -> str:
        encoded = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf8")
        return hashlib.sha256(encoded).hexdigest()

    def validate(self) -> None:
        if (
            self.train.epochs != 40
            or self.train.seed != 372
            or self.train.appearance_offset != 15
        ):
            raise ValueError("production epochs, seed, and appearance offset are fixed")
        effective_records = self.train.macro_records * self.train.macros_per_update
        if effective_records != 32:
            raise ValueError("effective record batch must be 32")
        if self.train.epochs * (2240 // effective_records) != 2800:
            raise ValueError("production schedule must resolve to 2800 updates")
        if (
            self.sampling.frames_per_record
            * self.sampling.spatial_queries_per_frame
            != self.sampling.queries_per_record
        ):
            raise ValueError("query budget is internally inconsistent")
        fractions = (
            self.sampling.uniform_fraction,
            self.sampling.energy_fraction,
            self.sampling.arrival_fraction,
        )
        if abs(sum(fractions) - 1.0) > 1e-12 or any(value <= 0.0 for value in fractions):
            raise ValueError("sampling fractions must be positive and sum to one")
        if self.sampling.queries_per_record % self.sampling.query_chunk_size:
            raise ValueError("query chunk size must divide the record query budget")
        if self.model.target_parameters != 32_294_258:
            raise ValueError("parent parameter target changed")
        if (
            self.loss.delta != 0.5
            or self.loss.energy_floor_fraction != 0.01
            or self.loss.spatial_gradient_weight != 0.1
            or any(
                value != 0.0
                for value in (
                    self.loss.spectrum_weight,
                    self.loss.receiver_weight,
                    self.loss.teacher_weight,
                )
            )
        ):
            raise ValueError("primary baseline loss contract changed")
        if self.train.adamw_implementation != "fused":
            raise ValueError("production AdamW implementation must be fused")
        if self.train.maximum_peak_cuda_gib != 23.5:
            raise ValueError("production CUDA memory gate must remain 23.5 GiB")
        if self.train.learning_rate is not None and self.train.learning_rate not in self.train.learning_rate_candidates:
            raise ValueError("resolved learning rate must come from the approved pilot candidates")
```

The production YAML must contain the approved absolute data, travel, normalization, parent, parent-evaluation, CLFC-evaluation, and artifact paths. The `validate()` method above is the single source of truth for schedule and query-budget checks.

```yaml
paths:
  source_h5: /home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5
  travel_time_h5: /home/jiayh/Data/data/processed/hybrid_travel_layered_eikonal_ray12_v1.h5
  normalization_json: /home/jiayh/Data/data/processed/grouped_v3_normalization.json
  parent_checkpoint: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/pretraining/gate4_long/run/best.pt
  parent_eval_dir: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/parent_epoch40_smoke3/run_parent
  clfc_eval_dir: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/clfc_two_frame_epoch40/smoke3
  artifact_dir: /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/deeponet_same_vds_parammatched_seed372
model:
  static_channels: 4
  branch_width: 48
  latent_dim: 1984
  trunk_hidden: 2749
  trunk_layers: 5
  fourier_bands: [1.0, 2.0, 4.0, 8.0]
  target_parameters: 32294258
  parameter_tolerance_fraction: 0.001
sampling:
  frames_per_record: 16
  spatial_queries_per_frame: 4096
  anchors_per_frame: 2048
  queries_per_record: 65536
  query_chunk_size: 16384
  uniform_fraction: 0.50
  energy_fraction: 0.25
  arrival_fraction: 0.25
  importance_cap_median: 20.0
loss:
  delta: 0.5
  energy_floor_fraction: 0.01
  hard_causality: true
  hard_causality_lead_cycles: 1.0
  spatial_gradient_weight: 0.1
  spectrum_weight: 0.0
  receiver_weight: 0.0
  teacher_weight: 0.0
train:
  epochs: 40
  macro_records: 8
  macros_per_update: 4
  microbatch_records: 1
  seed: 372
  appearance_offset: 15
  learning_rate_candidates: [0.0001, 0.0003, 0.001]
  learning_rate: null
  pilot_updates: 200
  warmup_epochs: 2
  minimum_lr_factor: 0.05
  weight_decay: 0.000001
  gradient_clip: 1.0
  adamw_implementation: fused
  maximum_peak_cuda_gib: 23.5
validation:
  panel_records: 48
  frames_per_record: 32
  spatial_queries_per_frame: 4096
  final_frames_per_record: 401
  comparison_sample_ids:
    - validation:uniform:00003
    - validation:layered:00082
    - validation:marmousi:00087
expected_parent_sha256: 0005aa6a154ec303205f8b50224edae5a35328453678db3d85e39f016f22469f
```

- [ ] **Step 4: Implement fail-closed provenance and leakage audit**

Reuse `build_manifest`, `validate_expected_counts`, `sha256_file`, and `EikonalTravelCache`. The central audit must follow this interface:

```python
@dataclass(frozen=True)
class AuditResult:
    counts: dict[str, int]
    manifest_digest: str
    split_digest: str
    parent_sha256: str
    normalization_sha256: str
    travel_time_sha256: str
    comparison_sample_ids: tuple[str, ...]


def audit_experiment(config: BaselineConfig, *, output_dir: str | Path) -> AuditResult:
    manifest = build_manifest(config.paths.source_h5)
    validate_expected_counts(
        manifest, {"train": 2240, "validation": 480, "test_id": 480}
    )
    split_sets = {name: set(manifest.indices_by_split[name]) for name in ("train", "validation", "test_id")}
    if any(
        split_sets[left] & split_sets[right]
        for left, right in (("train", "validation"), ("train", "test_id"), ("validation", "test_id"))
    ):
        raise ValueError("dataset splits overlap")
    parent_sha = sha256_file(config.paths.parent_checkpoint)
    if parent_sha != config.expected_parent_sha256:
        raise ValueError("parent checkpoint SHA-256 mismatch")
    EikonalTravelCache(config.paths.travel_time_h5, source_h5=config.paths.source_h5)
    records = {record.sample_id: record for record in manifest.records}
    requested = tuple(config.validation.comparison_sample_ids)
    missing = tuple(sample for sample in requested if sample not in records)
    if missing:
        raise ValueError(f"comparison sample IDs are missing: {missing}")
    if any(records[sample].split != "validation" for sample in requested):
        raise ValueError("comparison sample IDs must all belong to validation")
    for sample in requested:
        directory = sample.replace(":", "_")
        for root in (config.paths.parent_eval_dir, config.paths.clfc_eval_dir):
            if not (Path(root) / directory / "fields.pt").is_file():
                raise FileNotFoundError(Path(root) / directory / "fields.pt")
            if not (Path(root) / directory / "evaluation.json").is_file():
                raise FileNotFoundError(Path(root) / directory / "evaluation.json")
    with (Path(config.paths.clfc_eval_dir) / "summary.json").open(encoding="utf8") as handle:
        clfc_summary = json.load(handle)
    if clfc_summary["parent_checkpoint_sha256_after"] != parent_sha:
        raise ValueError("CLFC comparison artifact uses a different parent checkpoint")
    split_payload = {
        name: list(manifest.indices_by_split[name])
        for name in ("train", "validation", "test_id")
    }
    split_digest = hashlib.sha256(
        json.dumps(split_payload, sort_keys=True, separators=(",", ":")).encode("utf8")
    ).hexdigest()
    result = AuditResult(
        counts={name: len(values) for name, values in split_sets.items()},
        manifest_digest=manifest.digest,
        split_digest=split_digest,
        parent_sha256=parent_sha,
        normalization_sha256=sha256_file(config.paths.normalization_json),
        travel_time_sha256=sha256_file(config.paths.travel_time_h5),
        comparison_sample_ids=requested,
    )
    write_audit_files(result, manifest, output_dir)
    return result
```

Define the referenced atomic writer explicitly:

```python
def write_json_atomic(payload, path):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    try:
        with partial.open("x", encoding="utf8") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)


def write_audit_files(result, manifest, output_dir):
    root = Path(output_dir)
    write_json_atomic(asdict(result), root / "provenance.json")
    write_json_atomic(
        {
            "manifest_digest": manifest.digest,
            "split_digest": result.split_digest,
            "indices_by_split": manifest.indices_by_split,
        },
        root / "split_manifest.json",
    )
    write_json_atomic(
        {
            "passed": True,
            "split_overlap_count": 0,
            "comparison_sample_ids": result.comparison_sample_ids,
            "comparison_split": "validation",
            "teacher_inputs_allowed": False,
            "receiver_inputs_allowed": False,
        },
        root / "leakage_audit.json",
    )
```

Add `write_source_snapshot(label, roots, output_dir)` that hashes only source/config/test files and writes `source_snapshot_<label>.json`.

- [ ] **Step 5: Run focused tests and the production audit**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline/test_config_and_audit.py -v
/home/jiayh/miniforge3/envs/PINO/bin/python scripts/audit_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml
```

Expected: tests PASS; audit prints `train=2240 validation=480 test_id=480`, parent hash match, split overlap `0`, and writes the three audit JSON files.

- [ ] **Step 6: Record the no-Git task snapshot**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python scripts/audit_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --snapshot-label task01_config_audit
```

Expected: `audit/source_snapshot_task01_config_audit.json` exists with SHA-256 entries and no `.git` mutation.

## Task 2: Target-Free Static and Query Features

**Files:**

- Create: `patch_deeponet_baseline/features.py`
- Create: `tests/patch_deeponet_baseline/test_features.py`
- Modify: `patch_deeponet_baseline/__init__.py`

- [ ] **Step 1: Write failing shape, finiteness, and target-independence tests**

```python
import torch

from patch_deeponet_baseline.features import build_query_features, build_static_fields


def test_static_fields_are_four_channel_target_free():
    velocity = torch.full((1, 5, 7), 2000.0)
    source_map = torch.zeros(1, 5, 7)
    source_map[0, 2, 3] = 1.0
    travel = torch.linspace(0.0, 0.4, 35).reshape(5, 7)
    fields = build_static_fields(
        velocity, source_map, travel,
        velocity_center_mps=2500.0,
        velocity_scale_mps=1000.0,
    )
    assert fields.shape == (4, 5, 7)
    assert torch.isfinite(fields).all()


def test_query_features_have_fixed_26_and_11_dimensions():
    coords = torch.tensor([[[100.0, 200.0, 0.3], [500.0, 800.0, 0.7]]])
    source = torch.tensor([[100.0, 50.0, 10.0, 0.15, 1.0]])
    travel = torch.tensor([[0.05, 0.20]])
    static_local = torch.zeros(1, 2, 4)
    trunk, local = build_query_features(coords, source, travel, static_local)
    assert trunk.shape == (1, 2, 26)
    assert local.shape == (1, 2, 11)
    assert torch.isfinite(trunk).all() and torch.isfinite(local).all()
```

- [ ] **Step 2: Run and confirm failure**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline/test_features.py -v
```

Expected: FAIL because the feature module is absent.

- [ ] **Step 3: Implement four static channels and the exact feature dimensions**

`build_static_fields` must concatenate normalized velocity, source map, travel time divided by 1 s, and standardized slowness contrast. `build_query_features` must use:

```python
def build_query_features(coords_xyz_t, source_parameters, travel_time_s, static_local):
    coords = torch.as_tensor(coords_xyz_t)
    source = torch.as_tensor(source_parameters, dtype=coords.dtype, device=coords.device)
    travel = torch.as_tensor(travel_time_s, dtype=coords.dtype, device=coords.device)
    source_scale = coords.new_tensor((2000.0, 2000.0, 50.0, 1.2, 1.0))
    source_normalized = source / source_scale
    time = coords[..., 2]
    f0 = source[:, None, 2]
    t0 = source[:, None, 3]
    tau = time - t0 - travel
    phase_cycles = tau * f0
    base = torch.cat(
        (
            coords / coords.new_tensor((2000.0, 2000.0, 1.0)),
            source_normalized[:, None].expand(-1, coords.shape[1], -1),
            travel[..., None],
            phase_cycles[..., None],
        ),
        dim=-1,
    )
    bands = coords.new_tensor((1.0, 2.0, 4.0, 8.0))
    angles = 2.0 * torch.pi * torch.stack((time, phase_cycles), dim=-1).unsqueeze(-1) * bands
    periodic = torch.cat((torch.sin(angles).flatten(-2), torch.cos(angles).flatten(-2)), dim=-1)
    trunk = torch.cat((base, periodic), dim=-1)  # 10 + 16 = 26
    local = torch.cat(
        (static_local, source_normalized[:, None].expand(-1, coords.shape[1], -1),
         travel[..., None], phase_cycles[..., None]),
        dim=-1,
    )  # 4 + 5 + 1 + 1 = 11
    return trunk, local
```

Validate all ranks, shapes, finite values, positive source frequency, and coordinate bounds.

- [ ] **Step 4: Run tests and snapshot**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline/test_features.py -v
/home/jiayh/miniforge3/envs/PINO/bin/python scripts/audit_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --snapshot-label task02_features
```

Expected: feature tests PASS and the task-02 snapshot is written.

## Task 3: Parameter-Matched Local Patch-DeepONet

**Files:**

- Create: `patch_deeponet_baseline/model.py`
- Create: `tests/patch_deeponet_baseline/test_model.py`
- Modify: `patch_deeponet_baseline/__init__.py`

- [ ] **Step 1: Write failing model-contract tests**

```python
import torch

from patch_deeponet_baseline.model import LocalPatchDeepONet, count_trainable_parameters


def small_model():
    return LocalPatchDeepONet(
        static_channels=4, branch_width=8, latent_dim=16,
        trunk_hidden=24, trunk_layers=3, trunk_input_dim=26, query_input_dim=11,
    )


def test_query_forward_and_chunked_dense_are_consistent():
    torch.manual_seed(7)
    model = small_model().eval()
    static = torch.randn(1, 4, 9, 11)
    source = torch.tensor([[100.0, 50.0, 10.0, 0.15, 1.0]])
    coords = torch.rand(1, 117, 3) * torch.tensor((2000.0, 2000.0, 1.0))
    travel = torch.rand(1, 117) * 0.4
    direct = model(static, source, coords, travel)
    chunked = model.predict_queries(static, source, coords, travel, chunk_size=19)
    torch.testing.assert_close(chunked, direct, atol=1e-6, rtol=1e-5)


def test_production_model_matches_parent_capacity():
    model = LocalPatchDeepONet.production()
    count = count_trainable_parameters(model)
    assert count == 32_293_282
    assert abs(count - 32_294_258) / 32_294_258 < 0.001


def test_decoder_is_not_zero_initialized_and_all_groups_receive_gradients():
    torch.manual_seed(9)
    model = small_model()
    static = torch.randn(1, 4, 7, 7)
    source = torch.tensor([[100.0, 50.0, 10.0, 0.15, 1.0]])
    coords = torch.rand(1, 32, 3) * torch.tensor((2000.0, 2000.0, 1.0))
    travel = torch.rand(1, 32) * 0.2
    model(static, source, coords, travel).square().mean().backward()
    assert torch.count_nonzero(model.decoder.weight).item() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_model_output_is_hard_zero_before_physical_source_start():
    model = small_model().eval()
    static = torch.randn(1, 4, 7, 7)
    source = torch.tensor([[100.0, 50.0, 10.0, 0.15, 1.0]])
    coords = torch.tensor([[[100.0, 100.0, 0.0], [100.0, 100.0, 0.2]]])
    travel = torch.zeros(1, 2)
    output = model(static, source, coords, travel)
    assert output[0, 0].item() == 0.0
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline/test_model.py -v
```

Expected: FAIL because `LocalPatchDeepONet` is absent.

- [ ] **Step 3: Implement the model with explicit branch/trunk/query boundaries**

The production constructor is fixed:

```python
@classmethod
def production(cls) -> "LocalPatchDeepONet":
    return cls(
        static_channels=4,
        branch_width=48,
        latent_dim=1984,
        trunk_hidden=2749,
        trunk_layers=5,
        trunk_input_dim=26,
        query_input_dim=11,
    )
```

The modules and fusion must be:

```python
self.branch = nn.Sequential(
    nn.Conv2d(static_channels, branch_width, 3, padding=1),
    nn.GELU(),
    nn.Conv2d(branch_width, branch_width, 3, padding=1),
    nn.GELU(),
    nn.Conv2d(branch_width, latent_dim, 1),
)

trunk = []
current = trunk_input_dim
for _ in range(trunk_layers - 1):
    trunk.extend((nn.Linear(current, trunk_hidden), nn.GELU()))
    current = trunk_hidden
trunk.append(nn.Linear(current, latent_dim))
self.trunk = nn.Sequential(*trunk)
self.query_net = nn.Sequential(
    nn.Linear(query_input_dim, latent_dim),
    nn.GELU(),
    nn.Linear(latent_dim, latent_dim),
)
self.decoder = nn.Linear(latent_dim, 1)
```

Use `torch.nn.functional.grid_sample(branch_latent, grid, mode="bilinear", padding_mode="border", align_corners=True)` to sample branch latents, and use the same explicit call for static local features. Fuse:

```python
joint = (local_branch * self.trunk(trunk_features) + self.query_net(query_features))
joint = joint / math.sqrt(float(self.latent_dim))
residual = torch.nn.functional.linear(
    joint, self.decoder.weight, bias=None
).squeeze(-1)
phase_cycles = (
    query_coords_xyz_t[..., 2]
    - source_parameters[:, None, 3]
    - query_travel_time_s
) * source_parameters[:, None, 2]
ricker = (
    1.0 - 2.0 * torch.pi**2 * phase_cycles.square()
) * torch.exp(-torch.pi**2 * phase_cycles.square())
output = residual + self.decoder.bias[0] * ricker
cutoff = source_parameters[:, None, 3] - 1.0 / source_parameters[:, None, 2]
active = query_coords_xyz_t[..., 2] >= cutoff
return output * active.to(output.dtype)
```

Do not zero-initialize the decoder. Initialize its learned residual weight at
0.1 times the framework default and initialize its scalar Ricker-prior
amplitude to 0.1. The bias is not a constant output offset. This target-free
gate-driven correction preserves the parameter count and prevents the observed
zero-predictor collapse. Validate inputs before allocating large tensors.
`predict_queries` must encode and evaluate chunks under `torch.no_grad()` while
preserving query order.

- [ ] **Step 4: Run model tests and record the exact count**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline/test_model.py -v
/home/jiayh/miniforge3/envs/PINO/bin/python - <<'PY'
from patch_deeponet_baseline.model import LocalPatchDeepONet, count_trainable_parameters
print(count_trainable_parameters(LocalPatchDeepONet.production()))
PY
```

Expected: tests PASS and output is exactly `32293282`.

- [ ] **Step 5: Snapshot**

Run the audit CLI with `--snapshot-label task03_model`.

## Task 4: Deterministic 65,536-Query Mixture Sampler

**Files:**

- Create: `patch_deeponet_baseline/sampling.py`
- Create: `tests/patch_deeponet_baseline/test_sampling.py`

- [ ] **Step 1: Write failing deterministic/count/probability tests**

```python
import torch

from patch_deeponet_baseline.sampling import sample_record_queries


def test_sampler_returns_exact_budget_pairs_and_finite_weights():
    target = torch.randn(16, 21, 21)
    travel = torch.linspace(0.0, 0.5, 21 * 21).reshape(21, 21)
    times = torch.linspace(0.1, 0.8, 16)
    source = torch.tensor([100.0, 50.0, 10.0, 0.15, 1.0])
    first = sample_record_queries(
        target, travel, times, source,
        x_m=torch.linspace(0.0, 2000.0, 21),
        z_m=torch.linspace(0.0, 2000.0, 21),
        anchors_per_frame=32, seed=372,
    )
    repeated = sample_record_queries(
        target, travel, times, source,
        x_m=torch.linspace(0.0, 2000.0, 21),
        z_m=torch.linspace(0.0, 2000.0, 21),
        anchors_per_frame=32, seed=372,
    )
    assert first.coords_xyz_t.shape == (16 * 64, 3)
    assert first.pairs.shape == (16 * 32, 2)
    assert torch.equal(first.flat_indices, repeated.flat_indices)
    assert torch.isfinite(first.importance_weight).all()
    assert torch.all(first.mixture_probability > 0)
    assert torch.isfinite(first.energy_denominator).all()
    assert torch.all(first.energy_denominator > 0)
    assert first.importance_weight.max() <= 20.0 * first.importance_weight.median() + 1e-6


def test_production_query_arithmetic_is_exact():
    assert 16 * (2048 + 2048) == 65_536
    assert 65_536 % 16_384 == 0
```

- [ ] **Step 2: Run and verify failure**

Run the sampling test file; expect import failure.

- [ ] **Step 3: Implement anchor mixture, adjacent partners, and corrected weights**

Define:

```python
@dataclass(frozen=True)
class QueryBatch:
    coords_xyz_t: torch.Tensor
    target: torch.Tensor
    travel_time_s: torch.Tensor
    mixture_probability: torch.Tensor
    importance_weight: torch.Tensor
    energy_denominator: torch.Tensor
    flat_indices: torch.Tensor
    pairs: torch.Tensor
```

For each frame, select 2048 anchors as exactly 1024 uniform, 512 amplitude-energy, and 512 arrival-band draws. Use the supplied `torch.Generator`, never global RNG. Construct one valid right/down/left/up neighbor per anchor, yielding 4096 points per frame. Compute:

```python
def valid_four_neighbour_count(height, width):
    degree = torch.full((height, width), 4.0)
    degree[0] -= 1.0
    degree[-1] -= 1.0
    degree[:, 0] -= 1.0
    degree[:, -1] -= 1.0
    return degree.flatten()


def four_neighbour_sum(value):
    result = torch.zeros_like(value)
    result[1:, :] += value[:-1, :]
    result[:-1, :] += value[1:, :]
    result[:, 1:] += value[:, :-1]
    result[:, :-1] += value[:, 1:]
    return result


uniform_p = torch.full((height * width,), 1.0 / (height * width))
energy_p = (frame.abs().flatten() + 1e-8)
energy_p = energy_p / energy_p.sum()
arrival_score = torch.exp(
    -0.5 * (((time_s - source_t0_s - travel.flatten()) * source_f0_hz) / 1.5).square()
) + 1e-8
arrival_p = arrival_score / arrival_score.sum()
mixture_p = 0.50 * uniform_p + 0.25 * energy_p + 0.25 * arrival_p
degree = valid_four_neighbour_count(height, width)
neighbour_p = four_neighbour_sum((mixture_p / degree).reshape(height, width)).flatten()
selected_probability = torch.cat(
    (mixture_p[selected_anchor], neighbour_p[selected_neighbour])
)
weight = selected_probability.reciprocal()
weight = weight / weight.mean()
weight = torch.minimum(weight, 20.0 * weight.median())
weight = weight / weight.mean()
```

For the approved `energy_floor_fraction=0.01`, compute each frame RMS and the maximum frame RMS for the record:

```python
frame_rms = target.float().square().mean(dim=(-2, -1)).sqrt()
frame_denominator = torch.maximum(
    frame_rms,
    0.01 * frame_rms.max(),
).clamp_min(1e-8)
```

Choose each anchor's partner uniformly from its valid four-neighbour cells using the same local generator. `valid_four_neighbour_count` returns 2, 3, or 4 by grid location; `four_neighbour_sum` accumulates incoming anchor probability from the four directions. This makes `neighbour_p` the exact marginal sampling probability for partner points rather than incorrectly reusing the anchor mixture. Assign the corresponding frame denominator to every selected query. Store anchor/neighbor pair indices after concatenation. Duplicates are permitted and remain represented in the selected probability.

- [ ] **Step 4: Run tests and snapshot**

Expected: deterministic tests PASS; production arithmetic is 65,536 queries and four chunks.

## Task 5: Exact `appearance16` Record Data Adapter

**Files:**

- Create: `patch_deeponet_baseline/data.py`
- Create: `tests/patch_deeponet_baseline/test_data.py`

- [ ] **Step 1: Write failing real-data contract tests**

```python
from pathlib import Path

import pytest
import torch

from patch_deeponet_baseline.config import BaselineConfig
from patch_deeponet_baseline.data import DeepONetDataModule


CONFIG = Path("configs/patch_deeponet/parammatched_seed372.yaml")


def test_train_metadata_read_does_not_materialize_wavefield():
    module = DeepONetDataModule.from_config(BaselineConfig.from_yaml(CONFIG))
    record = module.train_records[0]
    assert record.split == "train"
    assert not hasattr(record, "wavefield")


@pytest.mark.skipif(not CONFIG.is_file(), reason="production paths unavailable")
def test_materialized_record_is_exact_appearance16_and_query_bound():
    module = DeepONetDataModule.from_config(BaselineConfig.from_yaml(CONFIG))
    batch = module.materialize_train_record(0, appearance=0, query_seed=372)
    assert batch.static_fields.shape == (4, 201, 201)
    assert batch.frame_indices.shape == (16,)
    assert batch.query.coords_xyz_t.shape == (65_536, 3)
    assert batch.query.target.shape == (65_536,)
    assert torch.isfinite(batch.query.target).all()
    assert batch.sample_id == module.train_records[0].sample_id
```

- [ ] **Step 2: Run and verify failure**

Run the data test file; expect missing module failure.

- [ ] **Step 3: Implement a read-only adapter over existing data contracts**

Define:

```python
@dataclass(frozen=True)
class TrainRecordBatch:
    sample_id: str
    medium_type: str
    static_fields: torch.Tensor
    source_parameters: torch.Tensor
    frame_indices: torch.Tensor
    frame_times_s: torch.Tensor
    dense_target_normalized: torch.Tensor
    dense_travel_time_s: torch.Tensor
    query: QueryBatch
```

`DeepONetDataModule.from_config` must build one manifest, one `PhysicalNormalizer`, split-specific `V3WavefieldDataset` views, and one `EikonalTravelCache`. `materialize_train_record` must:

1. call `appearance_time_indices(record.time_s.numpy(), source_t0_s=float(source[3]), source_f0_hz=float(source[2]), sample_id=record.sample_id, appearance=appearance, seed=372, count=16)`;
2. call `read_wavefield` only on `self.train_records`;
3. require `target.exact.all()` and equal left/right indices;
4. encode pressure using source amplitude;
5. read travel by sample ID;
6. build four static fields;
7. call the query sampler.

Add a separate `materialize_validation_record` that uses deterministic `validation_time_indices`; it must not call energy-weighted sampling.

- [ ] **Step 4: Run focused tests and a one-record materialization probe**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline/test_data.py -v
/home/jiayh/miniforge3/envs/PINO/bin/python - <<'PY'
from patch_deeponet_baseline.config import BaselineConfig
from patch_deeponet_baseline.data import DeepONetDataModule
c = BaselineConfig.from_yaml("configs/patch_deeponet/parammatched_seed372.yaml")
b = DeepONetDataModule.from_config(c).materialize_train_record(0, appearance=0, query_seed=372)
print(b.sample_id, tuple(b.frame_indices.shape), tuple(b.query.target.shape))
PY
```

Expected: PASS and output ends with `(16,) (65536,)`.

- [ ] **Step 5: Snapshot**

Run the task snapshot as `task05_data`.

## Task 6: Weighted Robust Loss, Hard Causality, and Paired Gradients

**Files:**

- Create: `patch_deeponet_baseline/losses.py`
- Create: `tests/patch_deeponet_baseline/test_losses.py`

- [ ] **Step 1: Write failing numerical tests**

```python
import torch

from patch_deeponet_baseline.losses import patch_deeponet_loss


def test_perfect_prediction_has_zero_components():
    target = torch.tensor([0.0, 1.0, 2.0, 3.0])
    pairs = torch.tensor([[0, 1], [2, 3]])
    report = patch_deeponet_loss(
        target.clone(), target, torch.ones_like(target), torch.ones_like(target), pairs,
        query_time_s=torch.tensor([0.0, 0.2, 0.4, 0.6]),
        source_f0_hz=10.0, source_t0_s=0.15,
        delta=0.5, gradient_weight=0.1, hard_causality_lead_cycles=1.0,
    )
    assert report.total.item() == 0.0
    assert report.data.item() == 0.0
    assert report.spatial_gradient.item() == 0.0


def test_importance_weights_change_data_term_and_causality_masks_output():
    prediction = torch.tensor([9.0, 1.0, 0.0], requires_grad=True)
    target = torch.tensor([0.0, 0.0, 1.0])
    report = patch_deeponet_loss(
        prediction, target, torch.tensor([1.0, 1.0, 4.0]),
        torch.ones_like(target), torch.tensor([[1, 2]]),
        query_time_s=torch.tensor([0.0, 0.1, 0.3]),
        source_f0_hz=10.0, source_t0_s=0.15,
        delta=0.5, gradient_weight=0.1, hard_causality_lead_cycles=1.0,
    )
    assert torch.isfinite(report.total)
    report.total.backward()
    assert prediction.grad is not None
```

- [ ] **Step 2: Run and verify failure**

Run the losses test file; expect import failure.

- [ ] **Step 3: Implement additive chunk-safe loss**

Define:

```python
@dataclass(frozen=True)
class QueryLoss:
    total: torch.Tensor
    data: torch.Tensor
    spatial_gradient: torch.Tensor
    active_fraction: torch.Tensor


def patch_deeponet_loss(
    prediction,
    target,
    importance_weight,
    energy_denominator,
    pairs,
    *,
    query_time_s,
    source_f0_hz,
    source_t0_s,
    delta,
    gradient_weight,
    hard_causality_lead_cycles,
) -> QueryLoss:
    predicted = torch.as_tensor(prediction)
    reference = torch.as_tensor(target, device=predicted.device, dtype=predicted.dtype)
    weight = torch.as_tensor(
        importance_weight, device=predicted.device, dtype=predicted.dtype
    )
    denominator = torch.as_tensor(
        energy_denominator, device=predicted.device, dtype=predicted.dtype
    )
    times = torch.as_tensor(
        query_time_s, device=predicted.device, dtype=predicted.dtype
    )
    if not (
        predicted.ndim == 1
        and predicted.shape == reference.shape == weight.shape == denominator.shape == times.shape
    ):
        raise ValueError("query loss vectors must have matching one-dimensional shapes")
    if (
        not torch.isfinite(predicted).all()
        or not torch.isfinite(reference).all()
        or not torch.isfinite(weight).all()
        or not torch.isfinite(denominator).all()
        or torch.any(weight <= 0)
        or torch.any(denominator <= 0)
    ):
        raise ValueError("query loss inputs must be finite with positive weights and denominators")
    if float(source_f0_hz) <= 0.0 or float(delta) <= 0.0 or float(gradient_weight) < 0.0:
        raise ValueError("source frequency and delta must be positive and gradient weight nonnegative")
    cutoff = float(source_t0_s) - float(hard_causality_lead_cycles) / float(source_f0_hz)
    active = times >= cutoff
    masked_prediction = predicted * active.to(predicted.dtype)
    masked_target = reference * active.to(reference.dtype)
    normalized_prediction = masked_prediction / denominator
    normalized_target = masked_target / denominator
    point = torch.nn.functional.huber_loss(
        normalized_prediction,
        normalized_target,
        reduction="none",
        delta=float(delta),
    )
    data = (weight * point).sum() / weight.sum().clamp_min(1e-8)
    pair_index = torch.as_tensor(pairs, device=predicted.device, dtype=torch.long)
    if pair_index.ndim != 2 or pair_index.shape[1] != 2:
        raise ValueError("spatial pairs must have shape [pair,2]")
    if pair_index.numel() and (
        int(pair_index.min()) < 0 or int(pair_index.max()) >= predicted.numel()
    ):
        raise ValueError("spatial pair index is outside the query vector")
    left, right = pair_index.unbind(dim=1)
    pair_denominator = 0.5 * (denominator[left] + denominator[right])
    predicted_gradient = (
        masked_prediction[right] - masked_prediction[left]
    ) / pair_denominator
    target_gradient = (
        masked_target[right] - masked_target[left]
    ) / pair_denominator
    pair_point = torch.nn.functional.huber_loss(
        predicted_gradient,
        target_gradient,
        reduction="none",
        delta=float(delta),
    )
    pair_weight = 0.5 * (weight[left] + weight[right])
    spatial_gradient = (
        (pair_weight * pair_point).sum() / pair_weight.sum().clamp_min(1e-8)
        if pair_index.shape[0]
        else predicted.new_zeros(())
    )
    total = data + float(gradient_weight) * spatial_gradient
    return QueryLoss(
        total=total,
        data=data,
        spatial_gradient=spatial_gradient,
        active_fraction=active.float().mean(),
    )
```

The positional tensor contract is `(prediction, target, importance_weight, energy_denominator, pairs)`. Apply hard causality at `t >= t0 - lead_cycles/f0`, normalize both masked prediction and target by `energy_denominator`, then use `torch.nn.functional.huber_loss(normalized_prediction, normalized_target, reduction="none", delta=0.5)`. The weighted data term is `(weight * point_loss).sum() / weight.sum()`. Pair loss compares normalized `prediction[j] - prediction[i]` with normalized `target[j] - target[i]` and uses the same robust delta. Validate positive finite denominators, finite weights, and indices.

Expose a `scale_for_accumulation(loss, *, record_count, query_fraction)` helper so every 16,384-query chunk contributes `chunk_queries / 65_536 / 32` to the update gradient.

- [ ] **Step 4: Run tests and snapshot**

Expected: all loss tests PASS, perfect prediction is exactly zero, and nonfinite input is rejected.

## Task 7: Deterministic Schedule, Update-Boundary Checkpoint, and Trainer

**Files:**

- Create: `patch_deeponet_baseline/checkpoint.py`
- Create: `patch_deeponet_baseline/training.py`
- Create: `tests/patch_deeponet_baseline/test_checkpoint_and_training.py`

- [ ] **Step 1: Write failing schedule/checkpoint/resume tests**

```python
from pathlib import Path

import torch

from patch_deeponet_baseline.checkpoint import load_checkpoint, save_checkpoint
from patch_deeponet_baseline.training import build_main_schedule


def test_main_schedule_is_exactly_40_epochs_2800_updates():
    schedule = build_main_schedule(record_count=2240, epochs=40, seed=372)
    assert len(schedule) == 40 * 280
    assert len(schedule) // 4 == 2800
    assert all(len(spec.record_indices) == 8 for spec in schedule)
    assert min(schedule[0].appearance_indices) == 15


def test_checkpoint_round_trip_restores_next_update(tmp_path: Path):
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = save_checkpoint(
        tmp_path / "latest.pt", model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, epoch=2, global_step=140,
        next_macro_index=560, manifest_digest="m", split_digest="s",
        config_digest="c", model_digest="d", metrics={"val_relative_l2": 0.9},
    )
    restored = torch.nn.Linear(3, 1)
    metadata = load_checkpoint(
        path, model=restored, optimizer=None, scheduler=None, scaler=None,
        expected_manifest_digest="m", expected_split_digest="s",
        expected_config_digest="c", expected_model_digest="d",
    )
    assert metadata.global_step == 140
    assert metadata.next_macro_index == 560
```

- [ ] **Step 2: Run and verify failure**

Run the checkpoint/training test file; expect import failure.

- [ ] **Step 3: Implement update-boundary checkpoint format**

Checkpoint payload must include:

```python
CHECKPOINT_FORMAT = "parammatched_patch_deeponet_v1"


@dataclass(frozen=True)
class CheckpointMetadata:
    epoch: int
    global_step: int
    next_macro_index: int
    metrics: dict[str, float]


def capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_checkpoint(
    path, *, model, optimizer, scheduler, scaler, epoch, global_step,
    next_macro_index, manifest_digest, split_digest, config_digest,
    model_digest, metrics,
):
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("checkpoints are allowed only at clean update boundaries")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.partial.{os.getpid()}")
    payload = {
        "format": CHECKPOINT_FORMAT,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "next_macro_index": int(next_macro_index),
        "query_cursor": 0,
        "gradient_accumulation_cursor": 0,
        "manifest_digest": str(manifest_digest),
        "split_digest": str(split_digest),
        "config_digest": str(config_digest),
        "model_digest": str(model_digest),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
        "rng_state": capture_rng_state(),
        "metrics": {str(key): float(value) for key, value in metrics.items()},
    }
    try:
        with partial.open("xb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def load_checkpoint(
    path, *, model, optimizer, scheduler, scaler,
    expected_manifest_digest, expected_split_digest,
    expected_config_digest, expected_model_digest,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    expected = {
        "format": CHECKPOINT_FORMAT,
        "manifest_digest": expected_manifest_digest,
        "split_digest": expected_split_digest,
        "config_digest": expected_config_digest,
        "model_digest": expected_model_digest,
        "query_cursor": 0,
        "gradient_accumulation_cursor": 0,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch")
    model.load_state_dict(payload["model_state"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if scaler is not None:
        scaler.load_state_dict(payload["scaler_state"])
    random.setstate(payload["rng_state"]["python"])
    np.random.set_state(payload["rng_state"]["numpy"])
    torch.set_rng_state(payload["rng_state"]["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in payload["rng_state"]:
        torch.cuda.set_rng_state_all(payload["rng_state"]["torch_cuda"])
    return CheckpointMetadata(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        next_macro_index=int(payload["next_macro_index"]),
        metrics=dict(payload["metrics"]),
    )
```

Save only after `optimizer.step()` and `optimizer.zero_grad(set_to_none=True)`. Assert every parameter gradient is `None` before writing so the zero cursors are truthful. Use an `xb` partial file, `fsync`, and `os.replace`. Load must reject any identity mismatch before applying model state.

- [ ] **Step 4: Implement schedule, optimizer, streamed update, and validation**

Reuse:

```python
def build_main_schedule(*, record_count, epochs, seed, appearance_offset=15):
    return build_full_support_schedule(
        record_count,
        epochs=epochs,
        macro_records=8,
        macros_per_update=4,
        seed=seed,
        appearance_offset=appearance_offset,
    )
```

For one optimizer update:

1. consume four macro specs, totaling 32 records;
2. materialize one record at a time with `appearance=spec.appearance_indices[record_offset]` and `query_seed=372 + spec.step * 1009 + record_offset * 9176`;
3. split its 65,536 queries into four 16,384-query chunks;
4. recompute the branch per chunk to free each autograd graph immediately;
5. scale each chunk loss by `chunk_size / 65_536 / 32`;
6. use CUDA autocast and GradScaler;
7. unscale, reject nonfinite gradients, clip norm to 1.0, step once;
8. append one `updates.jsonl` record atomically.

Use two-epoch linear warm-up and cosine decay to factor 0.05. Reproduce the parent record panel exactly with `np.random.default_rng(372).permutation(480)[:48]`. Before training, freeze 32 `validation_time_indices` frames per record with seed `372 + 7919` and panel offset zero, plus 4096 deterministic spatial grid points per frame, in `audit/validation_panel.json`. Reuse these exact IDs for every pilot and main validation event. Checkpoint selection uses only this fixed panel metric.

Construct AdamW with `fused=True` for CUDA production parameters and `fused=False` only for CPU unit tests. Record the resolved backend in `run_identity.json`; a CUDA production run that silently falls back from fused AdamW fails the audit.

- [ ] **Step 5: Add tiny injected-model trainer tests**

Test one update with a small model and synthetic data. Assert exactly one optimizer step after four macro accumulations, finite component logs, correct `global_step`, and identical next loss after save/resume with restored RNG.

- [ ] **Step 6: Run tests and snapshot**

Expected: schedule resolves to 2800 updates, checkpoint identity mismatches fail closed, and exact-resume test PASS.

## Task 8: CLI Modes, Learning-Rate Pilots, and Long-Run Gate

**Files:**

- Create: `patch_deeponet_baseline/gates.py`
- Create: `scripts/train_patch_deeponet_baseline.py`
- Create: `scripts/run_patch_deeponet_pipeline.py`
- Create: `tests/patch_deeponet_baseline/test_gates_and_cli.py`

- [ ] **Step 1: Write failing gate and CLI tests**

```python
import torch

from patch_deeponet_baseline.gates import (
    evaluate_pilot_gate,
    resolve_query_chunk_size,
    select_learning_rate,
)
from scripts.train_patch_deeponet_baseline import main


def test_gate_selects_lowest_finite_validation_and_requires_descent():
    reports = {
        1e-4: {"initial": 1.04, "final": 0.97, "prediction_rms_ratio": 0.2, "finite": True},
        3e-4: {"initial": 1.03, "final": 0.88, "prediction_rms_ratio": 0.4, "finite": True},
        1e-3: {"initial": 1.20, "final": 1.50, "prediction_rms_ratio": 2.0, "finite": True},
    }
    assert select_learning_rate(reports) == 3e-4
    assert evaluate_pilot_gate(reports[3e-4]).passed


def test_gate_rejects_zero_collapse():
    decision = evaluate_pilot_gate(
        {"initial": 1.0, "final": 1.0, "prediction_rms_ratio": 0.0, "finite": True}
    )
    assert not decision.passed
    assert "collapse" in decision.reasons


def test_cli_audit_mode(capsys):
    assert main([
        "--config", "configs/patch_deeponet/parammatched_seed372.yaml",
        "--mode", "audit",
    ]) == 0
    assert '"updates": 2800' in capsys.readouterr().out


def test_oom_chunk_fallback_preserves_query_budget():
    attempted = []
    def probe(chunk_size):
        attempted.append(chunk_size)
        if chunk_size > 4096:
            raise torch.OutOfMemoryError("synthetic")
        return chunk_size
    resolved = resolve_query_chunk_size(
        total_queries=65_536,
        candidates=(16_384, 8192, 4096, 2048, 1024),
        probe=probe,
    )
    assert resolved == 4096
    assert attempted == [16_384, 8192, 4096]
    assert 65_536 % resolved == 0
```

- [ ] **Step 2: Run and verify failure**

Run the gate/CLI test file; expect missing modules.

- [ ] **Step 3: Implement explicit gate decisions**

```python
@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reasons: tuple[str, ...]


def evaluate_pilot_gate(report):
    reasons = []
    if not report["finite"]:
        reasons.append("nonfinite")
    if not report["final"] < report["initial"]:
        reasons.append("no_descent")
    if not report["final"] < 1.0:
        reasons.append("zero_baseline_not_beaten")
    if not 0.05 <= report["prediction_rms_ratio"] <= 20.0:
        reasons.append("collapse")
    return GateDecision(not reasons, tuple(reasons))


def resolve_query_chunk_size(*, total_queries, candidates, probe):
    for chunk_size in candidates:
        if total_queries % int(chunk_size):
            continue
        try:
            probe(int(chunk_size))
            return int(chunk_size)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
    raise RuntimeError("no approved query chunk size fits the active GPU")


def select_learning_rate(reports):
    accepted = [
        (float(report["final"]), float(report.get("gradient_norm_variance", float("inf"))), float(rate))
        for rate, report in reports.items()
        if evaluate_pilot_gate(report).passed
    ]
    if not accepted:
        raise RuntimeError("no learning-rate pilot passed the descent gate")
    return min(accepted)[2]
```

This considers only passed finite pilots, minimizes final fixed-panel relative L2, and breaks ties by lower gradient-norm variance.

- [ ] **Step 4: Implement training CLI modes**

Supported modes:

- `audit` — no CUDA allocation; print manifest, parameter, and schedule census.
- `dry-run` — one real record with the full 65,536 selected query IDs, one candidate-sized query-chunk forward/backward, no checkpoint.
- `overfit` — one fixed train record from each family with a reduced query count; write `overfit_gate.json`.
- `pilot` — exactly 200 updates for one required `--learning-rate`.
- `main` — fresh seed-372 40-epoch run or identity-checked `--resume`.

Every mode writes `terminal.json` with `status`, `mode`, `exit_code`, identities, and failure reason.

- [ ] **Step 5: Implement gated pipeline supervisor**

The supervisor must:

1. run audit;
2. run dry-run, trying query chunks `16384`, `8192`, `4096`, `2048`, then `1024` only when the preceding attempt raises `torch.OutOfMemoryError`;
3. preserve 65,536 total query IDs and write the first passing chunk size to the resolved configuration;
4. run a two-update exact-resume probe in `pilot/resume_probe`, comparing an uninterrupted two-update run with a one-update checkpoint plus one resumed update;
5. run the three-record overfit gate;
6. run pilots for `1e-4`, `3e-4`, and `1e-3`;
7. select and verify the winning pilot;
8. write `run/resolved_config.yaml` with the selected learning rate and query chunk size;
9. launch main training in the same durable process only when all gates pass;
10. otherwise write a failed `terminal.json` and exit nonzero.

Use `subprocess.run(command, env=environment, check=True)` with an explicit interpreter and `PYTHONPATH`. Do not use shell interpolation.

`run_identity.json` must record both the immutable base-config digest and the resolved-config digest. The resume probe passes only when the second update has identical record IDs, query-index digest, learning rate, loss within `1e-6` relative tolerance, and model-state SHA-256 in both paths. An OOM fallback must clear CUDA cache and recreate the model/optimizer before retrying; it must never silently reduce query count or effective record batch.

Evaluate pilots at update 0 and every 25 updates. Define `initial` as the mean of the first three fixed-panel measurements and `final` as the mean of the last three; compute `prediction_rms_ratio` against target RMS on the same panel and gradient-norm variance over the final 50 updates.

- [ ] **Step 6: Run audit/CLI tests and a CUDA dry-run**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline/test_gates_and_cli.py -v
CUDA_VISIBLE_DEVICES=0 /home/jiayh/miniforge3/envs/PINO/bin/python \
  scripts/train_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --mode dry-run
```

Expected: tests PASS; dry-run reports finite loss, finite gradients, peak CUDA below 23.5 GiB, and parameter gap below 0.1%.

- [ ] **Step 7: Snapshot**

Record `task08_cli_gates`.

## Task 9: Dense Three-Instance Evaluator and Comparison Figures

**Files:**

- Create: `patch_deeponet_baseline/evaluation.py`
- Create: `scripts/evaluate_patch_deeponet_baseline.py`
- Create: `tests/patch_deeponet_baseline/test_evaluation.py`

- [ ] **Step 1: Write failing metric and fixed-instance tests**

```python
import torch

from patch_deeponet_baseline.evaluation import (
    comparison_metrics, fixed_comparison_sample_ids, receiver_indices,
)


def test_fixed_instances_match_approved_panel():
    assert fixed_comparison_sample_ids() == (
        "validation:uniform:00003",
        "validation:layered:00082",
        "validation:marmousi:00087",
    )


def test_perfect_field_metrics_are_zero_error():
    target = torch.randn(1, 12, 21, 21)
    metrics = comparison_metrics(
        target, target, source_t0_s=0.15, source_f0_hz=10.0,
        time_s=torch.linspace(0, 1, 12),
    )
    assert metrics["fullfield_relative_l2"] == 0.0
    assert metrics["frame_relative_l2_p50"] == 0.0
    assert metrics["receiver"]["relative_l2"] == 0.0


def test_receiver_layout_is_deterministic_and_in_bounds():
    indices = receiver_indices(height=201, width=201)
    assert len(indices) >= 8
    assert all(0 <= z < 201 and 0 <= x < 201 for z, x in indices)
```

- [ ] **Step 2: Run and verify failure**

Run the evaluator tests; expect missing module.

- [ ] **Step 3: Implement chunked dense prediction and metrics**

For each fixed validation record:

1. construct all 401 exact stored times;
2. predict one time block at a time and query the `201 x 201` grid in chunks;
3. decode normalized pressure;
4. read truth only after checkpoint selection;
5. save CPU tensors in `fields.pt`.

`comparison_metrics` must return:

```python
FIXED_COMPARISON_SAMPLE_IDS = (
    "validation:uniform:00003",
    "validation:layered:00082",
    "validation:marmousi:00087",
)


def fixed_comparison_sample_ids():
    return FIXED_COMPARISON_SAMPLE_IDS


def snapshot_indices(time_s, source_t0_s, count=6):
    axis = np.asarray(time_s, dtype=np.float64)
    onset = int(np.searchsorted(axis, float(source_t0_s), side="left"))
    onset = min(max(onset, 0), len(axis) - 1)
    end_time = min(float(axis[-1]), float(source_t0_s) + 0.60)
    end = int(np.searchsorted(axis, end_time, side="right")) - 1
    end = min(max(end, onset), len(axis) - 1)
    return np.linspace(onset, end, int(count)).round().astype(np.int64)


def receiver_indices(height, width):
    x_fraction = (0.15, 0.50, 0.85)
    z_fraction = (0.10, 0.325, 0.55)
    return tuple(
        (
            int(round(z_value * (int(height) - 1))),
            int(round(x_value * (int(width) - 1))),
        )
        for z_value in z_fraction
        for x_value in x_fraction
    )


def maximum_cross_correlation_lag(prediction, target):
    lags = []
    for receiver in range(prediction.shape[-1]):
        left = prediction[..., receiver].reshape(-1)
        right = target[..., receiver].reshape(-1)
        correlation = torch.nn.functional.conv1d(
            left.view(1, 1, -1),
            right.flip(0).view(1, 1, -1),
            padding=right.numel() - 1,
        ).reshape(-1)
        lags.append(int(correlation.argmax()) - (right.numel() - 1))
    return max(lags, key=abs)


def phase_correlation(prediction, target):
    predicted = prediction.float().flatten(start_dim=2)
    reference = target.float().flatten(start_dim=2)
    numerator = (predicted * reference).sum(dim=-1)
    denominator = (
        predicted.norm(dim=-1) * reference.norm(dim=-1)
    ).clamp_min(1e-8)
    informative = reference.norm(dim=-1) > 0.01 * reference.norm(dim=-1).max()
    if not bool(informative.any()):
        return prediction.new_tensor(1.0)
    return (numerator / denominator)[informative].mean()


def spectrum_metrics(prediction, target):
    predicted = torch.fft.rfft2(prediction.float(), norm="ortho")
    reference = torch.fft.rfft2(target.float(), norm="ortho")
    height, width = target.shape[-2:]
    kz = torch.fft.fftfreq(height, device=target.device).abs()
    kx = torch.fft.rfftfreq(width, device=target.device).abs()
    radius = torch.sqrt(kz[:, None].square() + kx[None, :].square())
    radius = radius / radius.max().clamp_min(1e-8)
    masks = {
        "low": radius <= 1.0 / 3.0,
        "middle": (radius > 1.0 / 3.0) & (radius <= 2.0 / 3.0),
        "high": radius > 2.0 / 3.0,
    }
    return {
        name: float(
            (predicted - reference)[..., mask].norm()
            / reference[..., mask].norm().clamp_min(1e-8)
        )
        for name, mask in masks.items()
    }


def receiver_metrics(prediction, target):
    locations = receiver_indices(target.shape[-2], target.shape[-1])
    predicted = torch.stack([prediction[..., z, x] for z, x in locations], dim=-1)
    reference = torch.stack([target[..., z, x] for z, x in locations], dim=-1)
    error = predicted - reference
    relative = error.norm() / reference.norm().clamp_min(1e-8)
    normalized_rmse = error.square().mean().sqrt() / reference.square().mean().sqrt().clamp_min(1e-8)
    lag = maximum_cross_correlation_lag(predicted, reference)
    return {
        "relative_l2": float(relative),
        "normalized_rmse": float(normalized_rmse),
        "max_lag_frames": int(lag),
    }


def comparison_metrics(
    prediction,
    target,
    *,
    source_t0_s,
    source_f0_hz,
    time_s,
    evaluation_start_index=None,
    latency_seconds=0.0,
    peak_cuda_bytes=0,
):
    predicted = torch.as_tensor(prediction)
    reference = torch.as_tensor(target, device=predicted.device)
    times = torch.as_tensor(time_s, device=predicted.device)
    if predicted.shape != reference.shape or predicted.ndim != 4:
        raise ValueError("comparison fields must match [record,time,z,x]")
    if times.ndim != 1 or times.numel() != predicted.shape[1]:
        raise ValueError("comparison time axis does not match the fields")
    error = (predicted.float() - reference.float()).flatten(start_dim=2).norm(dim=-1)
    target_norm = reference.float().flatten(start_dim=2).norm(dim=-1)
    frame_relative = error / target_norm.clamp_min(1e-8)
    if float(source_f0_hz) <= 0.0:
        raise ValueError("source frequency must be positive")
    physical_start_s = float(source_t0_s) - 1.0 / float(source_f0_hz)
    time_mask = times >= physical_start_s
    if evaluation_start_index is not None:
        time_mask = time_mask & (
            torch.arange(times.numel(), device=times.device)
            >= int(evaluation_start_index)
        )
    if not bool(time_mask.any()):
        raise ValueError("comparison metric window is empty")
    active = time_mask[None] & (
        target_norm > 0.01 * target_norm.max()
    )
    active_relative = (
        frame_relative[active].mean()
        if bool(active.any())
        else frame_relative.new_tensor(0.0)
    )
    selected_prediction = predicted[:, time_mask]
    selected_target = reference[:, time_mask]
    selected_frame_relative = frame_relative[:, time_mask]
    fullfield_relative = (
        (predicted.float() - reference.float()).norm()
        / reference.float().norm().clamp_min(1e-8)
    )
    return {
        "fullfield_relative_l2": float(fullfield_relative),
        "active_energy_relative_l2": float(active_relative),
        "frame_relative_l2_p50": float(selected_frame_relative.median()),
        "frame_relative_l2_p95": float(torch.quantile(selected_frame_relative, 0.95)),
        "phase_correlation": float(phase_correlation(selected_prediction, selected_target)),
        "spectrum_relative_l2": spectrum_metrics(selected_prediction, selected_target),
        "receiver": receiver_metrics(selected_prediction, selected_target),
        "latency_seconds": float(latency_seconds),
        "peak_cuda_bytes": int(peak_cuda_bytes),
    }
```

Use the same receiver coordinates, snapshot indices, color limits, and time axes for truth and every method.

- [ ] **Step 4: Implement the two explicit comparison panels**

Load:

- raw parent fields from `artifacts/parent_epoch40_smoke3/run_parent`;
- CLFC adapted fields from `artifacts/clfc_two_frame_epoch40/smoke3`;
- DeepONet fields from the new evaluation directory.

Write:

- `comparison_summary.json`;
- `comparison_table.csv`;
- per-family `wavefield_comparison.png/.pdf`;
- per-family `receiver_waveforms.png/.pdf`.

The table must contain separate rows `parent_raw`, `clfc_two_frame`, and `deeponet_raw`, plus checkpoint SHA, parameters, adaptation flag, and observation-frame count.

Recompute every table metric from the saved tensors instead of mixing pre-existing summary definitions. For the architecture panel, start post-onset metrics at the physical Ricker start. For the deployment panel, read CLFC's two audited `observed_indices` and start all CLFC/DeepONet future metrics at `max(observed_indices) + 1`; DeepONet still receives neither observed value as input.

- [ ] **Step 5: Run evaluator unit tests and a randomly initialized one-frame smoke**

Expected: tests PASS; smoke produces finite fields and figures without reading CLFC observation frames as DeepONet inputs.

- [ ] **Step 6: Snapshot**

Record `task09_evaluation`.

## Task 10: Durable Launcher and Live Run Verification

**Files:**

- Create: `run_patch_deeponet_baseline.sh`
- Create: `scripts/check_patch_deeponet_run.py`
- Modify: `tests/patch_deeponet_baseline/test_gates_and_cli.py`

- [ ] **Step 1: Add failing launcher dry-run tests**

Assert the launcher:

- rejects an existing live tmux session with the same name;
- supports `--dry-run` without mutation;
- prints the exact interpreter, config, artifact root, session name, and log path;
- never references the historical 400-by-400 dataset or teacher checkpoints.

- [ ] **Step 2: Implement a tmux launcher**

The shell entry must resolve fixed paths, create only the new artifact/log directories, and start:

```bash
tmux new-session -d -s deeponet_pm_seed372 \
  "cd /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/project && \
   exec env PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \
   /home/jiayh/miniforge3/envs/PINO/bin/python \
   scripts/run_patch_deeponet_pipeline.py \
   --config configs/patch_deeponet/parammatched_seed372.yaml \
   >> /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/deeponet_same_vds_parammatched_seed372/logs/train.log 2>&1"
```

Before starting, verify the GPU has at least 22 GiB free and no existing session/process owns the artifact directory. Write launch identity atomically from Python, not with shell redirection.

- [ ] **Step 3: Implement live verification**

`check_patch_deeponet_run.py` must inspect:

- tmux session existence;
- Python PID and command line;
- GPU PID, utilization, and allocated memory;
- latest `updates.jsonl` row;
- latest checkpoint metadata;
- pilot/main terminal status;
- age of the last progress record.

Return exit code 0 only when the expected current phase is live or terminal status is complete. Return nonzero for stale logs, identity mismatches, dead sessions without terminal JSON, or nonfinite metrics.

- [ ] **Step 4: Run all focused tests before launch**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python -m pytest \
  tests/patch_deeponet_baseline -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run production audit and launch**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python scripts/audit_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml
bash run_patch_deeponet_baseline.sh
```

Expected: audit PASS and tmux session `deeponet_pm_seed372` starts.

- [ ] **Step 6: Verify the live job**

Run:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python scripts/check_patch_deeponet_run.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml
nvidia-smi
tmux capture-pane -pt deeponet_pm_seed372:0 -S -80
```

Expected during early execution: live Python PID, GPU memory/utilization above idle, audit/dry-run/overfit/pilot phase identified, and a recent log timestamp. Expected after main entry: `updates.jsonl` advances and `checkpoints/latest.pt` appears.

- [ ] **Step 7: Record final implementation snapshot**

Run the source snapshot as `task10_launched`.

## Task 11: Final Training and Comparison Verification

**Files:**

- Modify only if a verified defect is found: focused files from Tasks 1–10
- Create after results: `docs/superpowers/reports/2026-07-24-parammatched-patch-deeponet-baseline-results.md`

- [ ] **Step 1: Verify the pilot gate before calling the run “long training”**

Inspect all three pilot `terminal.json` files and the selected-gate JSON. Require:

- exactly 200 updates for every candidate;
- finite losses/gradients;
- selected final validation relative L2 below its initial window and below 1.0;
- prediction RMS ratio in `[0.05, 20]`;
- selected learning rate recorded in `run/resolved_config.yaml`;
- main checkpoint initialized from random seed 372, not a pilot checkpoint.

- [ ] **Step 2: Verify main-run schedule and resume safety**

Check `run_identity.json`, `updates.jsonl`, and `latest.pt`:

```bash
/home/jiayh/miniforge3/envs/PINO/bin/python scripts/check_patch_deeponet_run.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml --require-main
```

Expected: parameter count `32293282`, target `32294258`, 40 epochs, 2800 planned updates, effective record batch 32, frames 16, queries 65,536, and checkpoint identities matching audit digests.

- [ ] **Step 3: Run a controlled resume proof**

Inspect `pilot/resume_probe/comparison.json` and require identical next update number, sampled record IDs, query digest, learning rate, model-state digest, and loss within `1e-6` relative tolerance. Do not interrupt the live main run solely to repeat this already-isolated proof.

- [ ] **Step 4: Run final three-instance evaluation after `best.pt` is available**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 /home/jiayh/miniforge3/envs/PINO/bin/python \
  scripts/evaluate_patch_deeponet_baseline.py \
  --config configs/patch_deeponet/parammatched_seed372.yaml \
  --checkpoint /home/jiayh/Data/FNO-Acoustic-Wave-Simulation-gate4-localfield-20260724/artifacts/deeponet_same_vds_parammatched_seed372/run/checkpoints/best.pt
```

Expected: all three fixed validation directories, finite fields, four figure files per family as wavefield and receiver PNG/PDF pairs, comparison JSON/CSV, and no modification of parent/CLFC artifacts.

- [ ] **Step 5: Write the results report**

Record:

- exact dataset, split, config, model, checkpoint, parent, and CLFC hashes;
- pilot selection evidence;
- training/validation trends and best epoch/update;
- parameter count, wall time, peak memory, and inference latency;
- raw parent versus raw DeepONet table;
- strict two-frame CLFC versus raw DeepONet table;
- links to wavefield and receiver figures;
- limitations, including whether the zero-collapse gate passed and whether uniform full-space relative L2 remained denominator-sensitive.

- [ ] **Step 6: Completion check**

Do not claim completion solely because the process exited. Completion requires:

- `terminal.json` status `complete`;
- 2800 optimizer updates;
- finite `best.pt` and `latest.pt`;
- stable-descent evidence;
- exact three-instance evaluation;
- live artifact hashes;
- focused tests still passing.

If any condition fails, report the failed gate and the latest recoverable checkpoint instead of silently extending training.
