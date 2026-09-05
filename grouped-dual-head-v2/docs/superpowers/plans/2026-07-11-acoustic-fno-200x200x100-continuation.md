# Acoustic FNO 200x200x100 Continuation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从现有验证最优 Acoustic FNO 权重稳定续训 `200 x 200 x 100` 模型，并在 GPU smoke 通过后以 `nohup` 提交正式任务。

**Architecture:** 保持原始四层 AcousticFNO3D 和 checkpoint 参数兼容，只改变采样网格与优化配置。生产与 smoke 配置共享 split、归一化、初始化权重和双层重要性采样；smoke 使用独立目录验证两次反向传播、验证、检查点读写和显存峰值。

**Tech Stack:** Python 3.10、PyTorch/CUDA、PyYAML、pytest、HDF5、nohup。

---

## File Map

- Create `configs/pino_hdf5_200x100_adaptive_continue.yaml`: 正式续训配置。
- Create `configs/pino_hdf5_200x100_adaptive_smoke.yaml`: 两步 GPU smoke 配置。
- Create `tests/test_pino_200x100_configs.py`: 配置契约测试。
- Generate `artifacts/acoustic_pino_200x100_adaptive_smoke/`: smoke 产物，不提交。
- Generate `artifacts/acoustic_pino_200x100_adaptive_continue/`: 正式训练产物，不提交。

### Task 1: Add Config Contract Tests

**Files:**
- Create: `tests/test_pino_200x100_configs.py`

- [ ] **Step 1: Write the failing tests**

```python
from pathlib import Path

from fno_acoustic.config import load_config


ROOT = Path(__file__).resolve().parents[1]
BEST = "artifacts/hybrid_drp_pino/fno_400x54_spatial_residual_importance/checkpoints/best.pt"
SPLIT = "artifacts/hybrid_drp_pino/fno_continue_pretrain_optimizer_safe/splits.json"
STATS = "artifacts/pino_adaptation/normalization_stats.json"


def _assert_shared_contract(config_path: str) -> dict:
    cfg = load_config(ROOT / config_path)
    assert cfg["sampling"]["target_height"] == 200
    assert cfg["sampling"]["target_width"] == 200
    assert cfg["sampling"]["max_time_steps"] == 100
    assert cfg["train"]["batch_size"] == 1
    assert cfg["train"]["init_checkpoint"] == BEST
    assert cfg["train"]["freeze_batchnorm_stats"] is True
    assert cfg["model"]["activation_checkpointing"] is True
    assert cfg["data"]["split_manifest"] == SPLIT
    assert cfg["normalization"]["stats_path"] == STATS
    assert cfg["normalization"]["reuse_stats"] is True
    adaptive = cfg["train"]["adaptive_importance_sampling"]
    spatial = cfg["train"]["spatial_importance_sampling"]
    assert adaptive["enabled"] is True
    assert adaptive["uniform_fraction"] == 0.25
    assert spatial["enabled"] is True
    assert spatial["pixels_per_sample"] == 8192
    assert spatial["uniform_fraction"] == 0.25
    assert spatial["reweight_loss"] is True
    return cfg


def test_production_200x100_contract() -> None:
    cfg = _assert_shared_contract("configs/pino_hdf5_200x100_adaptive_continue.yaml")
    assert cfg["train"]["epochs"] == 30
    assert cfg["train"]["adaptive_importance_sampling"]["samples_per_epoch"] == 256
    assert cfg["train"]["checkpoint_dir"] == "artifacts/acoustic_pino_200x100_adaptive_continue/checkpoints"


def test_smoke_200x100_contract() -> None:
    cfg = _assert_shared_contract("configs/pino_hdf5_200x100_adaptive_smoke.yaml")
    assert cfg["train"]["epochs"] == 1
    assert cfg["train"]["max_train_batches"] == 2
    assert cfg["train"]["max_val_batches"] == 3
    assert cfg["train"]["checkpoint_dir"] == "artifacts/acoustic_pino_200x100_adaptive_smoke/checkpoints"
```

- [ ] **Step 2: Verify the tests fail before configs exist**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_pino_200x100_configs.py`

Expected: two `FileNotFoundError` failures for the new YAML files.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_pino_200x100_configs.py
git commit -m "test: define 200x100 acoustic continuation contract"
```

### Task 2: Add Production And Smoke Configs

**Files:**
- Create: `configs/pino_hdf5_200x100_adaptive_continue.yaml`
- Create: `configs/pino_hdf5_200x100_adaptive_smoke.yaml`

- [ ] **Step 1: Create the production config**

Copy the complete HDF5 schema from `configs/pino_hdf5_400_residual_importance.yaml` and apply these exact overrides:

```yaml
data:
  split_manifest: artifacts/hybrid_drp_pino/fno_continue_pretrain_optimizer_safe/splits.json
  split_output_path: artifacts/acoustic_pino_200x100_adaptive_continue/splits.json
sampling:
  target_height: 200
  target_width: 200
  max_time_steps: 100
normalization:
  stats_path: artifacts/pino_adaptation/normalization_stats.json
  reuse_stats: true
model:
  activation_checkpointing: true
train:
  device: cuda
  init_checkpoint: artifacts/hybrid_drp_pino/fno_400x54_spatial_residual_importance/checkpoints/best.pt
  freeze_batchnorm_stats: true
  epochs: 30
  batch_size: 1
  learning_rate: 1.0e-6
  weight_decay: 0.0
  grad_clip: 0.5
  max_train_batches: null
  max_val_batches: 12
  checkpoint_every_epochs: 1
  checkpoint_dir: artifacts/acoustic_pino_200x100_adaptive_continue/checkpoints
  log_dir: artifacts/acoustic_pino_200x100_adaptive_continue/logs
  adaptive_importance_sampling:
    enabled: true
    samples_per_epoch: 256
    uniform_fraction: 0.25
    min_weight: 1.0e-6
    residual_power: 1.0
    default_residual: 1.0
    ema_momentum: 0.25
  spatial_importance_sampling:
    enabled: true
    pixels_per_sample: 8192
    uniform_fraction: 0.25
    min_weight: 1.0e-6
    residual_power: 1.0
    default_residual: 1.0
    ema_momentum: 0.25
    reweight_loss: true
loss:
  relative_l2_weight: 1.0
  mse_weight: 0.05
  eps: 1.0e-8
evaluation:
  split: test
  max_samples: 9
  output_dir: artifacts/acoustic_pino_200x100_adaptive_continue/evaluation
```

- [ ] **Step 2: Create the smoke config**

Copy production config and change only:

```yaml
data:
  split_output_path: artifacts/acoustic_pino_200x100_adaptive_smoke/splits.json
train:
  epochs: 1
  max_train_batches: 2
  max_val_batches: 3
  checkpoint_dir: artifacts/acoustic_pino_200x100_adaptive_smoke/checkpoints
  log_dir: artifacts/acoustic_pino_200x100_adaptive_smoke/logs
  adaptive_importance_sampling:
    samples_per_epoch: 2
evaluation:
  max_samples: 3
  output_dir: artifacts/acoustic_pino_200x100_adaptive_smoke/evaluation
```

- [ ] **Step 3: Run config tests**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_pino_200x100_configs.py`

Expected: `2 passed`.

- [ ] **Step 4: Verify checkpoint compatibility on CPU**

```bash
PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH python - <<'PY'
from fno_acoustic.checkpoint import load_checkpoint
from fno_acoustic.config import load_config
from fno_acoustic.train import build_training_model

cfg = load_config("configs/pino_hdf5_200x100_adaptive_smoke.yaml")
ckpt = load_checkpoint(cfg["train"]["init_checkpoint"], map_location="cpu")
model, _ = build_training_model(cfg)
missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
assert not missing and not unexpected, (missing, unexpected)
print("checkpoint-compatible", ckpt["epoch"], ckpt["best_val_metric"])
PY
```

Expected: `checkpoint-compatible 59 0.0665220208466053`.

- [ ] **Step 5: Commit configs and tests**

```bash
git add configs/pino_hdf5_200x100_adaptive_continue.yaml configs/pino_hdf5_200x100_adaptive_smoke.yaml tests/test_pino_200x100_configs.py
git commit -m "feat: configure 200x100 adaptive acoustic continuation"
```

### Task 3: Run And Validate GPU Smoke

**Files:**
- Generate: `artifacts/acoustic_pino_200x100_adaptive_smoke/`

- [ ] **Step 1: Require at least 12 GiB free VRAM**

Run: `nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free,utilization.gpu --format=csv`

Expected: GPU 0 has at least `12288 MiB` free. Otherwise wait for or pause a competing task before smoke.

- [ ] **Step 2: Run two training steps in the foreground**

```bash
rm -rf artifacts/acoustic_pino_200x100_adaptive_smoke
mkdir -p artifacts/acoustic_pino_200x100_adaptive_smoke
set -o pipefail
PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
python scripts/train_pino.py \
  --config configs/pino_hdf5_200x100_adaptive_smoke.yaml \
  --device cuda \
  --output-json artifacts/acoustic_pino_200x100_adaptive_smoke/train_metrics.json \
  2>&1 | tee artifacts/acoustic_pino_200x100_adaptive_smoke/smoke.log
```

Expected: exit 0, input `[1,200,200,100,3]`, output `[1,200,200,100]`, finite prediction, and no OOM.

- [ ] **Step 3: Verify metrics and checkpoint reload**

```bash
PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH python - <<'PY'
import json
from pathlib import Path
from fno_acoustic.checkpoint import load_checkpoint

root = Path("artifacts/acoustic_pino_200x100_adaptive_smoke")
metrics = json.loads((root / "train_metrics.json").read_text())
assert metrics["input_shape"] == [1, 200, 200, 100, 3]
assert metrics["output_shape"] == [1, 200, 200, 100]
assert metrics["prediction_finite_ratio"] == 1.0
assert metrics["peak_reserved_mb"] < 24000
assert metrics["validation_relative_l2"] == metrics["validation_relative_l2"]
checkpoint = load_checkpoint(root / "checkpoints/last.pt", map_location="cpu")
assert checkpoint["epoch"] == 0
print(metrics["validation_relative_l2"], metrics["peak_allocated_mb"], metrics["peak_reserved_mb"])
PY
```

Expected: finite validation metric, reserved memory below 24 GB, and successful checkpoint reload.

### Task 4: Submit Production With Nohup

**Files:**
- Generate: `artifacts/acoustic_pino_200x100_adaptive_continue/nohup_train.log`
- Generate: `artifacts/acoustic_pino_200x100_adaptive_continue/train.pid`

- [ ] **Step 1: Confirm no duplicate process exists**

Run: `pgrep -af 'train_pino.py.*pino_hdf5_200x100_adaptive_continue.yaml' || true`

Expected: no matching process.

- [ ] **Step 2: Submit production training**

```bash
mkdir -p artifacts/acoustic_pino_200x100_adaptive_continue
nohup env PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
python scripts/train_pino.py \
  --config configs/pino_hdf5_200x100_adaptive_continue.yaml \
  --device cuda \
  --output-json artifacts/acoustic_pino_200x100_adaptive_continue/train_metrics.json \
  > artifacts/acoustic_pino_200x100_adaptive_continue/nohup_train.log 2>&1 \
  < /dev/null &
echo $! > artifacts/acoustic_pino_200x100_adaptive_continue/train.pid
```

Expected: immediate shell return and a numeric PID file.

- [ ] **Step 3: Perform startup-only verification**

```bash
sleep 10
pid=$(cat artifacts/acoustic_pino_200x100_adaptive_continue/train.pid)
ps -o pid,stat,etimes,cmd -p "$pid"
tail -n 30 artifacts/acoustic_pino_200x100_adaptive_continue/nohup_train.log
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader
```

Expected: PID exists, no OOM/NaN traceback, and initialization or first training step appears. Do not monitor long-term after this check.

### Task 5: Final Verification And Handoff

**Files:**
- Verify: `configs/pino_hdf5_200x100_adaptive_continue.yaml`
- Verify: `artifacts/acoustic_pino_200x100_adaptive_continue/train.pid`
- Verify: `artifacts/acoustic_pino_200x100_adaptive_continue/nohup_train.log`

- [ ] **Step 1: Run focused verification**

```bash
PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_pino_200x100_configs.py
git diff --check
```

Expected: `2 passed` and no whitespace errors.

- [ ] **Step 2: Report runtime paths**

Report PID, config, initialization checkpoint, log path, checkpoint directory, smoke validation metric, peak allocated/reserved memory, and competing GPU jobs. State that startup passed but long-run convergence has not yet been evaluated.

