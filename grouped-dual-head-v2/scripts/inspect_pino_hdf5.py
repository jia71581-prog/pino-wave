from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fno_acoustic.schema import dataset_limited_samples, finite_stats, inspect_hdf5, monotonic_stats, semantic_candidates
from fno_acoustic.utils import ensure_dir, write_json


def _dataset_markdown(schema: dict, sample_stats: dict, candidates: dict) -> str:
    lines = ["# pino.hdf5 schema", "", f"Path: `{schema['path']}`", "", "## File attributes", ""]
    for key, value in schema.get("attrs", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines += ["", "## Datasets", ""]
    for path, item in schema["datasets"].items():
        lines.append(f"### `{path}`")
        lines.append(f"- shape: `{item['shape']}`")
        lines.append(f"- rank: `{item['rank']}`")
        lines.append(f"- dtype: `{item['dtype']}`")
        lines.append(f"- chunks: `{item['chunks']}`")
        lines.append(f"- compression: `{item['compression']}`")
        lines.append(f"- attrs: `{item['attrs']}`")
        if path in sample_stats:
            lines.append(f"- sample_stats: `{sample_stats[path]}`")
        lines.append("")
    lines += ["## Semantic candidates", ""]
    for key, values in candidates.items():
        lines.append(f"- {key}: {values}")
    lines += [
        "",
        "## Selected canonical mapping",
        "",
        "- velocity: `/nu`, shape `[2500, 400, 400]`, axes `[sample, x, z]`",
        "- wavefield: `/tensor`, shape `[2500, 160, 400, 400]`, axes `[sample, time, x, z]`, canonical `[N, H, W, T]`",
        "- source map: `/source_mask`, shape `[2500, 400, 400]`, axes `[sample, x, z]`",
        "- source indices: `/source_x_idx`, `/source_z_idx`, units grid index",
        "- time: `/t-coordinate`, shape `[160]`, monotonic physical seconds",
        "- x/z coordinates: `/x-coordinate`, `/y-coordinate`; dx dataset `/dx` reports 5 m",
        "- source frequency/amplitude: `/source_frequency_hz`, `/source_amplitude`; inspected samples are constant 25 Hz and 1.0 amplitude",
    ]
    return "\n".join(lines) + "\n"


def _save_image(path: Path, array: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
    im = ax.imshow(array)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def inspect(path: Path, json_path: Path, markdown_path: Path, sample_dir: Path | None = None) -> dict:
    schema = inspect_hdf5(path)
    sample_stats = {}
    with h5py.File(path, "r") as h5:
        for dset_path in schema["datasets"]:
            dset = h5[dset_path]
            sample_stats[dset_path] = dataset_limited_samples(dset)
            lname = dset_path.lower()
            if "time" in lname or dset_path in {"/t-coordinate"}:
                sample_stats[dset_path]["monotonic"] = monotonic_stats(np.asarray(dset[()]))
        if sample_dir:
            ensure_dir(sample_dir)
            if "nu" in h5:
                _save_image(sample_dir / "velocity_sample_000.png", h5["nu"][0], "nu sample 0")
            if "source_mask" in h5:
                _save_image(sample_dir / "source_mask_sample_000.png", h5["source_mask"][0], "source_mask sample 0")
            if "tensor" in h5:
                tensor = h5["tensor"][0]
                for idx in [0, tensor.shape[0] // 2, tensor.shape[0] - 1]:
                    _save_image(sample_dir / f"wavefield_sample_000_t{idx:04d}.png", tensor[idx], f"tensor sample 0 t={idx}")
    candidates = semantic_candidates(schema)
    payload = {"schema": schema, "sample_stats": sample_stats, "semantic_candidates": candidates}
    write_json(json_path, payload)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_dataset_markdown(schema, sample_stats, candidates), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, default=None)
    args = parser.parse_args()
    payload = inspect(args.path, args.json, args.markdown, args.sample_dir)
    print(json.dumps({"datasets": len(payload["schema"]["datasets"]), "json": str(args.json)}, indent=2))


if __name__ == "__main__":
    main()
