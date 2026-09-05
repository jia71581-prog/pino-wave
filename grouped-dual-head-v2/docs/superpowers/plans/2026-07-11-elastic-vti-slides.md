# Elastic VTI Neural-Operator Slides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a verified 14-frame Elastic VTI neural-operator section to the group-meeting Beamer deck before the probabilistic-FWI section.

**Architecture:** A focused Python asset builder will read existing evaluation artifacts, load fixed validation samples, render slide-native wavefield/receiver/result figures, and benchmark PINO inference against the FD8 teacher with explicit provenance. The Beamer source will consume only generated figures and concise formulas, leaving all existing probabilistic-FWI content unchanged.

**Tech Stack:** Python 3.10, PyTorch, HDF5/h5py, NumPy, Matplotlib, LaTeX Beamer/XeLaTeX, pytest.

---

## File Structure

- Create `reports/no_asvgd_group_meeting_20260710/make_elastic_vti_section.py`: load checkpoints and fixed-index data, generate compact figures, and write a metrics manifest.
- Create `scripts/benchmark_elastic_vti_pino.py`: same-device FD8/PINO timing with CUDA synchronization and machine-readable output.
- Create `tests/test_elastic_vti_slide_assets.py`: test loss-label data, frame selection, timing summaries, and artifact manifest validation without requiring a full GPU benchmark.
- Modify `reports/no_asvgd_group_meeting_20260710/main.tex`: insert the 14-frame section at the approved location.
- Generate `reports/no_asvgd_group_meeting_20260710/figures/elastic_*.png`: slide-native visual assets.
- Generate `reports/no_asvgd_group_meeting_20260710/elastic_vti_runtime.json`: benchmark provenance and timing summary.
- Generate `reports/no_asvgd_group_meeting_20260710/elastic_vti_slide_metrics.json`: exact metrics and source paths used by the deck.

### Task 1: Test the slide-asset data contract

**Files:**
- Create: `tests/test_elastic_vti_slide_assets.py`
- Create: `reports/no_asvgd_group_meeting_20260710/make_elastic_vti_section.py`

- [ ] **Step 1: Write failing tests for frame selection and metrics extraction**

```python
from pathlib import Path

from reports.no_asvgd_group_meeting_20260710.make_elastic_vti_section import (
    select_nearest_frames,
    summarize_evaluation_metrics,
)


def test_select_nearest_frames_uses_requested_times() -> None:
    times = [0.0, 0.165, 0.335, 0.5]
    assert select_nearest_frames(times, [0.165, 0.335]) == [1, 2]


def test_summarize_evaluation_metrics_reads_both_components(tmp_path: Path) -> None:
    metrics = {
        "per_component_mean": {
            "component_0": {"relative_l2": 0.1, "receiver_line_relative_l2": 0.2},
            "component_1": {"relative_l2": 0.3, "receiver_line_relative_l2": 0.4},
        }
    }
    path = tmp_path / "metrics.json"
    path.write_text(__import__("json").dumps(metrics), encoding="utf-8")
    assert summarize_evaluation_metrics(path) == {
        "ux_relative_l2": 0.1,
        "uz_relative_l2": 0.3,
        "ux_receiver_relative_l2": 0.2,
        "uz_receiver_relative_l2": 0.4,
    }
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_elastic_vti_slide_assets.py
```

Expected: collection fails because `make_elastic_vti_section.py` does not exist.

- [ ] **Step 3: Implement the minimal pure-data helpers**

```python
def select_nearest_frames(times_s, requested_s):
    times = np.asarray(times_s, dtype=np.float64)
    return [int(np.argmin(np.abs(times - float(value)))) for value in requested_s]


def summarize_evaluation_metrics(path: Path) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    ux = payload["per_component_mean"]["component_0"]
    uz = payload["per_component_mean"]["component_1"]
    return {
        "ux_relative_l2": float(ux["relative_l2"]),
        "uz_relative_l2": float(uz["relative_l2"]),
        "ux_receiver_relative_l2": float(ux["receiver_line_relative_l2"]),
        "uz_receiver_relative_l2": float(uz["receiver_line_relative_l2"]),
    }
```

- [ ] **Step 4: Run the tests and verify pass**

Run the same pytest command. Expected: `2 passed`.

- [ ] **Step 5: Commit the data-contract helpers**

```bash
git add tests/test_elastic_vti_slide_assets.py reports/no_asvgd_group_meeting_20260710/make_elastic_vti_section.py
git commit -m "test: define elastic slide asset contract"
```

### Task 2: Implement reproducible FD8/PINO timing

**Files:**
- Create: `scripts/benchmark_elastic_vti_pino.py`
- Modify: `tests/test_elastic_vti_slide_assets.py`

- [ ] **Step 1: Add failing timing-summary tests**

```python
from scripts.benchmark_elastic_vti_pino import summarize_seconds


def test_summarize_seconds_reports_median_iqr_and_range() -> None:
    result = summarize_seconds([1.0, 2.0, 3.0, 4.0])
    assert result["median_s"] == 2.5
    assert result["min_s"] == 1.0
    assert result["max_s"] == 4.0
    assert result["q25_s"] == 1.75
    assert result["q75_s"] == 3.25
```

- [ ] **Step 2: Run the targeted test and verify failure**

Run:

```bash
PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_elastic_vti_slide_assets.py::test_summarize_seconds_reports_median_iqr_and_range
```

Expected: import failure because the benchmark script does not exist.

- [ ] **Step 3: Implement timing helpers and CLI**

The script must expose:

```python
def cuda_timed(callable_, repetitions: int, warmups: int) -> list[float]:
    for _ in range(warmups):
        callable_()
    torch.cuda.synchronize()
    values = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
    return values


def summarize_seconds(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "median_s": float(np.median(array)),
        "q25_s": float(np.percentile(array, 25)),
        "q75_s": float(np.percentile(array, 75)),
        "min_s": float(array.min()),
        "max_s": float(array.max()),
    }
```

The CLI must accept `--model {uniform,layered,marmousi}`, `--fd-repetitions`,
`--pino-repetitions`, and `--output`. It must load the corresponding fixed
sample and checkpoint, time only solver/model calls, and write JSON containing
device, shapes, repetitions, raw timings, summaries, and
`speedup=fd_median/pino_median`.

- [ ] **Step 4: Run unit tests**

Expected: all slide-asset tests pass.

- [ ] **Step 5: Run three same-device benchmarks**

```bash
for model in uniform layered marmousi; do
  PATH=/home/jiayh/miniconda3/bin:$PATH PYTHONPATH=src \
  python scripts/benchmark_elastic_vti_pino.py \
    --model "$model" --device cuda \
    --fd-warmups 1 --fd-repetitions 3 \
    --pino-warmups 5 --pino-repetitions 20 \
    --output "reports/no_asvgd_group_meeting_20260710/runtime_${model}.json"
done
```

Expected: each JSON has finite positive timing values and speedup greater than zero.

- [ ] **Step 6: Merge benchmark provenance**

Run the script's `--merge` mode to write
`reports/no_asvgd_group_meeting_20260710/elastic_vti_runtime.json` and verify it
contains all three model keys.

- [ ] **Step 7: Commit timing implementation and reports**

```bash
git add scripts/benchmark_elastic_vti_pino.py tests/test_elastic_vti_slide_assets.py reports/no_asvgd_group_meeting_20260710/runtime_*.json reports/no_asvgd_group_meeting_20260710/elastic_vti_runtime.json
git commit -m "feat: benchmark elastic PINO against FD8"
```

### Task 3: Generate slide-native elastic figures

**Files:**
- Modify: `reports/no_asvgd_group_meeting_20260710/make_elastic_vti_section.py`
- Modify: `tests/test_elastic_vti_slide_assets.py`
- Generate: `reports/no_asvgd_group_meeting_20260710/figures/elastic_*.png`
- Generate: `reports/no_asvgd_group_meeting_20260710/elastic_vti_slide_metrics.json`

- [ ] **Step 1: Add a failing manifest validation test**

```python
from reports.no_asvgd_group_meeting_20260710.make_elastic_vti_section import validate_manifest


def test_validate_manifest_requires_all_models_and_figures() -> None:
    manifest = {
        "models": {
            name: {"figures": {kind: f"{name}_{kind}.png" for kind in ("ux", "uz", "receivers")}}
            for name in ("uniform", "layered", "marmousi")
        }
    }
    validate_manifest(manifest)
```

- [ ] **Step 2: Implement fixed-index loading and compact plotting**

Implement six public functions with these exact contracts:

- `load_case(model_name: str, sample_index: int) -> dict[str, Any]` loads the
  configured checkpoint and fixed sample, decodes physical-space target and
  prediction, and returns velocity, source map, time, target, prediction,
  sample index, checkpoint path, and evaluation metrics.
- `plot_ux_context(case: dict[str, Any], output: Path) -> None` writes the
  velocity/source plus two-time \(u_x\) comparison.
- `plot_uz_comparison(case: dict[str, Any], output: Path) -> None` writes the
  two-time \(u_z\) comparison and medium inset.
- `plot_receiver_runtime(case: dict[str, Any], runtime: dict[str, Any], output: Path) -> None`
  writes both receiver components, trace overlays, and runtime bars.
- `plot_cross_model_overview(metrics: dict[str, Any], output: Path) -> None`
  writes full-field and receiver relative-L2 bars, including Marmousi
  pretraining and fine-tuning.
- `validate_manifest(manifest: dict[str, Any]) -> None` raises `ValueError`
  unless all three model keys exist, each has `ux`, `uz`, and `receivers`
  figure keys, and every metrics value is finite.

`plot_ux_context` must use a left column for \(v_p\), \(v_s\), and source
marker and a right 3x2 grid for Reference/Prediction/Error at 0.165 s and
0.335 s. `plot_uz_comparison` uses the same 3x2 comparison with a compact
medium/source inset. `plot_receiver_runtime` contains both component gathers,
representative trace overlays, and FD8/PINO runtime bars. All field comparisons
must share Reference/Prediction limits and use a separate residual limit.

- [ ] **Step 3: Generate all figures and manifest**

```bash
PATH=/home/jiayh/miniconda3/bin:$PATH PYTHONPATH=src \
python reports/no_asvgd_group_meeting_20260710/make_elastic_vti_section.py \
  --runtime reports/no_asvgd_group_meeting_20260710/elastic_vti_runtime.json \
  --output-dir reports/no_asvgd_group_meeting_20260710/figures \
  --manifest reports/no_asvgd_group_meeting_20260710/elastic_vti_slide_metrics.json
```

Expected figures:

```text
elastic_cross_model_overview.png
elastic_uniform_ux.png
elastic_uniform_uz.png
elastic_uniform_receivers_runtime.png
elastic_layered_ux.png
elastic_layered_uz.png
elastic_layered_receivers_runtime.png
elastic_marmousi_ux.png
elastic_marmousi_uz.png
elastic_marmousi_receivers_runtime.png
```

- [ ] **Step 4: Validate images and tests**

Run tests, then use `identify` to verify every figure is non-empty and at least
1600 pixels wide. Expected: tests pass and no figure fails the size gate.

- [ ] **Step 5: Commit generated assets**

```bash
git add reports/no_asvgd_group_meeting_20260710/make_elastic_vti_section.py reports/no_asvgd_group_meeting_20260710/figures/elastic_*.png reports/no_asvgd_group_meeting_20260710/elastic_vti_slide_metrics.json tests/test_elastic_vti_slide_assets.py
git commit -m "feat: generate elastic VTI presentation figures"
```

### Task 4: Insert the 14-frame Beamer section

**Files:**
- Modify: `reports/no_asvgd_group_meeting_20260710/main.tex`

- [ ] **Step 1: Record the insertion boundary**

Verify `背景：传统 FWI 的强项与盲区` is immediately followed by
`背景：为什么需要贝叶斯化 FWI` before editing.

- [ ] **Step 2: Add the common five frames**

Add `\section{弹性波神经算子正演}` and five frames implementing the equations,
data flow, exact pretraining objective, exact component-balanced fine-tuning
objective, and cross-model overview. Use the existing `tagbox`, `deepblue`,
`accent`, and `forest` definitions.

- [ ] **Step 3: Add the nine model frames**

For each model, add three frames and embed the exact generated filenames from
Task 3. Captions must identify fixed sample index, component, times, checkpoint
stage, and timing scope.

- [ ] **Step 4: Check TeX structure**

```bash
rg -n '\\section\{弹性波神经算子正演\}|elastic_(uniform|layered|marmousi)' reports/no_asvgd_group_meeting_20260710/main.tex
```

Expected: one section marker and exactly nine per-model figure references.

- [ ] **Step 5: Commit Beamer source**

```bash
git add reports/no_asvgd_group_meeting_20260710/main.tex
git commit -m "feat: add elastic VTI section to group meeting deck"
```

### Task 5: Compile and visually verify the deck

**Files:**
- Update: `reports/no_asvgd_group_meeting_20260710/main.pdf`
- Update: `reports/no_asvgd_group_meeting_20260710/group_meeting_no_asvgd_bfwi_20260710.pdf`
- Generate: `reports/no_asvgd_group_meeting_20260710/review_elastic/`

- [ ] **Step 1: Compile with XeLaTeX**

```bash
cd reports/no_asvgd_group_meeting_20260710
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
cp main.pdf group_meeting_no_asvgd_bfwi_20260710.pdf
```

Expected: exit code 0 and no missing-file errors.

- [ ] **Step 2: Enforce warning gates**

```bash
rg -n 'Overfull|LaTeX Error|File .* not found' main.log
```

Expected: no newly introduced overfull boxes, errors, or missing figures in the
14 inserted frames.

- [ ] **Step 3: Render the inserted frame range**

Use `pdftoppm -png -r 130 main.pdf review_elastic/slide` and identify the 14
new pages by their titles. Generate a contact sheet for rapid review.

- [ ] **Step 4: Inspect visual quality**

Check every inserted page for source-marker visibility, readable equations,
matched field color scales, uncropped labels, receiver trace legibility, and
runtime-scope disclosure. Correct any issue and repeat compile/render.

- [ ] **Step 5: Run regression verification**

```bash
PATH=/home/jiayh/miniconda3/bin:$PATH pytest -q tests/test_elastic_vti_slide_assets.py
git diff --check
```

Expected: tests pass and the diff check reports no whitespace errors in files
owned by this task.

- [ ] **Step 6: Commit final compiled deck**

```bash
git add reports/no_asvgd_group_meeting_20260710/main.pdf reports/no_asvgd_group_meeting_20260710/group_meeting_no_asvgd_bfwi_20260710.pdf reports/no_asvgd_group_meeting_20260710/review_elastic
git commit -m "docs: compile and verify elastic VTI presentation"
```
