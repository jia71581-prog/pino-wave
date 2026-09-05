# Elastic VTI Marmousi Optimizer Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从现有 Marmousi 验证最优权重启动三个受控 AdamW 学习率候选，选择达到严格验证门槛的参数并提交独立正式精调任务。

**Architecture:** 三个候选共享模型、数据、split、归一化和损失，只改变初始学习率；所有候选用 `init_checkpoint` 创建新 optimizer/scheduler，禁止 `--resume`。候选结果由独立选择脚本按验证 relative L2 和预定义 tie-break 规则封存，随后才运行组件评估和正式精调。

**Tech Stack:** Python 3.10、PyTorch/CUDA、PyYAML、pytest、CSV/JSON、nohup。

---

## File Map

- Create `configs/pino_elastic_vti_marmousi_opt_lr2e5.yaml`: `2e-5` 短程候选。
- Create `configs/pino_elastic_vti_marmousi_opt_lr5e5.yaml`: `5e-5` 短程候选。
- Create `configs/pino_elastic_vti_marmousi_opt_lr1e4.yaml`: `1e-4` 短程候选。
- Create `scripts/select_marmousi_optimizer_candidate.py`: 确定性读取候选指标并封存选择。
- Create `tests/test_marmousi_optimizer_refinement.py`: 配置契约和候选选择测试。
- Generate `configs/pino_elastic_vti_marmousi_opt_selected_finetune.yaml`: 胜出后生成的正式配置。
- Generate `artifacts/elastic_vti_pino_marmousi_optimizer_sweep/`: 候选与选择产物。
- Generate `artifacts/elastic_vti_pino_marmousi_optimizer_finetune/`: 正式精调产物。

### Task 1: Define Config And Selection Contracts

**Files:**
- Create: `tests/test_marmousi_optimizer_refinement.py`

- [ ] **Step 1: Write failing config tests**

```python
from pathlib import Path

from fno_acoustic.config import load_config


ROOT = Path(__file__).resolve().parents[1]
BASELINE = "artifacts/elastic_vti_pino_marmousi/checkpoints/best.pt"
SPLIT = "artifacts/elastic_vti_pino_marmousi/splits.json"
STATS = "artifacts/elastic_vti_pino_marmousi/normalization_stats.json"


def test_optimizer_candidate_configs_control_all_non_lr_variables() -> None:
    paths = {
        "lr2e5": 2.0e-5,
        "lr5e5": 5.0e-5,
        "lr1e4": 1.0e-4,
    }
    signatures = []
    for tag, lr in paths.items():
        cfg = load_config(ROOT / f"configs/pino_elastic_vti_marmousi_opt_{tag}.yaml")
        assert cfg["train"]["init_checkpoint"] == BASELINE
        assert cfg["data"]["split_manifest"] == SPLIT
        assert cfg["normalization"]["stats_path"] == STATS
        assert cfg["normalization"]["reuse_stats"] is True
        assert cfg["train"]["learning_rate"] == lr
        assert cfg["train"]["weight_decay"] == 1.0e-5
        assert cfg["train"]["grad_clip"] == 0.5
        assert cfg["train"]["epochs"] == 3
        assert cfg["train"]["max_train_batches"] == 64
        assert cfg["train"]["max_val_batches"] == 8
        assert cfg["train"]["batch_size"] == 1
        assert cfg["train"]["scheduler"] == "cosine"
        signatures.append((cfg["model"], cfg["loss"], cfg["sampling"], cfg["data"]["path_glob"]))
    assert signatures[1:] == signatures[:-1]
```

- [ ] **Step 2: Write failing selector tests**

```python
import json

from select_marmousi_optimizer_candidate import select_candidate


def test_selector_rejects_all_candidates_above_threshold(tmp_path) -> None:
    rows = [{"tag": "lr2e5", "learning_rate": 2e-5, "validation_relative_l2": 0.105}]
    result = select_candidate(rows, threshold=0.1045, tie_tolerance=1e-4)
    assert result["accepted"] is False
    assert result["selected"] is None


def test_selector_uses_lower_lr_inside_tie_tolerance() -> None:
    rows = [
        {"tag": "lr2e5", "learning_rate": 2e-5, "validation_relative_l2": 0.10442},
        {"tag": "lr5e5", "learning_rate": 5e-5, "validation_relative_l2": 0.10435},
        {"tag": "lr1e4", "learning_rate": 1e-4, "validation_relative_l2": 0.10410},
    ]
    result = select_candidate(rows, threshold=0.1045, tie_tolerance=1e-4)
    assert result["accepted"] is True
    assert result["selected"]["tag"] == "lr1e4"


def test_selector_requires_finite_metrics() -> None:
    rows = [{"tag": "lr2e5", "learning_rate": 2e-5, "validation_relative_l2": float("nan")}]
    result = select_candidate(rows, threshold=0.1045, tie_tolerance=1e-4)
    assert result["accepted"] is False
```

- [ ] **Step 3: Run tests and verify missing files fail**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_marmousi_optimizer_refinement.py`

Expected: failures for missing configs and selector module.

### Task 2: Implement Deterministic Candidate Selection

**Files:**
- Create: `scripts/select_marmousi_optimizer_candidate.py`
- Modify: `tests/test_marmousi_optimizer_refinement.py`

- [ ] **Step 1: Implement selection logic and CLI**

```python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def select_candidate(rows: list[dict[str, Any]], threshold: float, tie_tolerance: float) -> dict[str, Any]:
    finite = [row for row in rows if math.isfinite(float(row["validation_relative_l2"]))]
    eligible = [row for row in finite if float(row["validation_relative_l2"]) < threshold]
    if not eligible:
        return {"accepted": False, "selected": None, "candidates": rows, "threshold": threshold}
    best_metric = min(float(row["validation_relative_l2"]) for row in eligible)
    tied = [row for row in eligible if float(row["validation_relative_l2"]) <= best_metric + tie_tolerance]
    selected = min(tied, key=lambda row: float(row["learning_rate"]))
    return {"accepted": True, "selected": selected, "candidates": rows, "threshold": threshold}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", action="append", nargs=3, metavar=("TAG", "LR", "METRICS_JSON"), required=True)
    parser.add_argument("--threshold", type=float, default=0.1045)
    parser.add_argument("--tie-tolerance", type=float, default=1e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for tag, lr, metrics_path in args.candidate:
        payload = json.loads(Path(metrics_path).read_text())
        rows.append({"tag": tag, "learning_rate": float(lr), "validation_relative_l2": float(payload["validation_relative_l2"]), "metrics_path": metrics_path})
    result = select_candidate(rows, args.threshold, args.tie_tolerance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make selector importable in tests**

Add to the test before importing the selector:

```python
import sys
sys.path.insert(0, str(ROOT / "scripts"))
```

- [ ] **Step 3: Run selector unit tests**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_marmousi_optimizer_refinement.py -k selector`

Expected: `3 passed`.

### Task 3: Add Three Controlled Candidate Configs

**Files:**
- Create: `configs/pino_elastic_vti_marmousi_opt_lr2e5.yaml`
- Create: `configs/pino_elastic_vti_marmousi_opt_lr5e5.yaml`
- Create: `configs/pino_elastic_vti_marmousi_opt_lr1e4.yaml`

- [ ] **Step 1: Copy the complete Marmousi base config three times**

Use `configs/pino_elastic_vti_marmousi_gpu.yaml` as the source so schema, model and loss remain identical.

- [ ] **Step 2: Apply shared candidate overrides**

```yaml
data:
  split_manifest: artifacts/elastic_vti_pino_marmousi/splits.json
normalization:
  stats_path: artifacts/elastic_vti_pino_marmousi/normalization_stats.json
  reuse_stats: true
train:
  init_checkpoint: artifacts/elastic_vti_pino_marmousi/checkpoints/best.pt
  epochs: 3
  batch_size: 1
  weight_decay: 1.0e-5
  scheduler: cosine
  grad_clip: 0.5
  early_stopping_patience: null
  max_train_batches: 64
  max_val_batches: 8
```

Set `learning_rate` to `2e-5`, `5e-5`, and `1e-4`. Set each `split_output_path`, `checkpoint_dir`, `log_dir`, and evaluation output to `artifacts/elastic_vti_pino_marmousi_optimizer_sweep/<tag>/...`.

- [ ] **Step 3: Run all focused tests**

Run: `PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_marmousi_optimizer_refinement.py`

Expected: `4 passed`.

- [ ] **Step 4: Commit implementation files**

```bash
git add scripts/select_marmousi_optimizer_candidate.py tests/test_marmousi_optimizer_refinement.py configs/pino_elastic_vti_marmousi_opt_lr2e5.yaml configs/pino_elastic_vti_marmousi_opt_lr5e5.yaml configs/pino_elastic_vti_marmousi_opt_lr1e4.yaml
git commit -m "feat: add Marmousi optimizer refinement sweep"
```

### Task 4: Stop Baseline And Run Sequential Sweep

**Files:**
- Generate: `artifacts/elastic_vti_pino_marmousi_optimizer_sweep/baseline_state.json`
- Generate: candidate logs, metrics and checkpoints.

- [ ] **Step 1: Record baseline checkpoint and process state**

Record current PID, epoch, global step, best metric, checkpoint SHA256 and GPU memory in `baseline_state.json`. Verify `best.pt` loads before stopping the process.

- [ ] **Step 2: Stop only the current Marmousi elastic training process**

Send `SIGTERM`, wait up to 20 seconds, then use `SIGKILL` only if it remains alive. Do not stop unrelated Uniform or FWI processes.

- [ ] **Step 3: Run candidates sequentially in the foreground**

For each tag `lr2e5`, `lr5e5`, `lr1e4`, run:

```bash
PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH CUDA_VISIBLE_DEVICES=0 \
python scripts/train_elastic_vti_pino.py \
  --config configs/pino_elastic_vti_marmousi_opt_TAG.yaml \
  --device cuda \
  --output-json artifacts/elastic_vti_pino_marmousi_optimizer_sweep/TAG/train_metrics.json \
  > artifacts/elastic_vti_pino_marmousi_optimizer_sweep/TAG/train.log 2>&1
```

Expected for every run: exit 0, finite ratio 1.0, exactly 3 metric rows, and independent best/last checkpoints.

- [ ] **Step 4: Select candidate without test access**

Run the selector with all three `train_metrics.json` paths and write `artifacts/elastic_vti_pino_marmousi_optimizer_sweep/selection.json`.

Expected: either one accepted candidate below `0.1045`, or an explicit rejected result with `selected: null`.

### Task 5: Evaluate Accepted Candidate And Submit Finetune

**Files:**
- Generate: accepted candidate evaluation.
- Generate: `configs/pino_elastic_vti_marmousi_opt_selected_finetune.yaml`.
- Generate: `artifacts/elastic_vti_pino_marmousi_optimizer_finetune/`.

- [ ] **Step 1: Stop if no candidate passes**

If `selection.json` has `accepted: false`, do not create a production config and report that optimizer tuning did not beat the baseline.

- [ ] **Step 2: Run sealed four-sample component evaluation**

Use `scripts/evaluate_current_training_visuals.py` with the accepted candidate `best.pt`, the candidate config, split `val`, sample indices `3,23,40,48`, and an independent evaluation directory.

Expected: `u_z` relative L2 no greater than `0.084756944 * 1.01 = 0.085604513`.

- [ ] **Step 3: Create production config from the accepted optimizer**

Copy the accepted candidate config, reset `init_checkpoint` to the original baseline `best.pt`, and set:

```yaml
train:
  epochs: 40
  max_train_batches: null
  max_val_batches: 8
  early_stopping_patience: 10
  early_stopping_min_delta: 1.0e-4
  checkpoint_every_epochs: 1
  checkpoint_dir: artifacts/elastic_vti_pino_marmousi_optimizer_finetune/checkpoints
  log_dir: artifacts/elastic_vti_pino_marmousi_optimizer_finetune/logs
evaluation:
  output_dir: artifacts/elastic_vti_pino_marmousi_optimizer_finetune/evaluation
```

- [ ] **Step 4: Submit production with nohup and verify startup**

Start `scripts/train_elastic_vti_pino.py` with the selected config, write PID and `nohup_train.log`, wait 10 seconds, and verify the PID is alive with no OOM/NaN traceback. Stop monitoring after startup verification.

### Task 6: Final Verification And Handoff

**Files:**
- Verify all configs, tests, `selection.json`, evaluation and production PID/log.

- [ ] **Step 1: Run focused tests and whitespace validation**

```bash
PATH=/home/jiayh/miniforge3/envs/qwen/bin:$PATH pytest -q tests/test_marmousi_optimizer_refinement.py
git diff --check
```

Expected: all focused tests pass and no whitespace errors.

- [ ] **Step 2: Report outcome without overstating accuracy**

Report baseline, all three candidate metrics, accepted/rejected decision, component metrics, selected learning rate, production PID and paths. If rejected, state that no optimizer candidate met the predefined accuracy gate and preserve the old best checkpoint.

