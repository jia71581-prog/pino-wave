#!/usr/bin/env python3
"""Benchmark Elastic VTI FD8 forward modeling against PINO inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import h5py
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from elastic_vti.config import load_config as load_solver_config
from elastic_vti.medium_sampling import MediumSample
from elastic_vti.solver_torch import simulate_torch
from elastic_vti.units import stiffness_from_vp_vs_isotropic
from fno_acoustic.checkpoint import load_checkpoint
from fno_acoustic.config import load_config
from fno_acoustic.data_elastic_vti import PinoHDF5Dataset
from fno_acoustic.model_elastic_vti import ElasticVTIFNO3D


MODEL_SPECS = {
    "uniform": {
        "config": ROOT / "configs/pino_elastic_vti_uniform_gpu.yaml",
        "checkpoint": ROOT / "artifacts/elastic_vti_pino_uniform/checkpoints/best.pt",
        "solver_config": Path("/home/jiayh/Data/.atan/outputs/elastic_vti_uniform_train_gpu/resolved_config.yaml"),
        "sample_index": 6,
    },
    "layered": {
        "config": ROOT / "configs/pino_elastic_vti_layered_gpu.yaml",
        "checkpoint": ROOT / "artifacts/elastic_vti_pino_layered/checkpoints/best.pt",
        "solver_config": Path(
            "/home/jiayh/Data/.atan/outputs/layered_two_strict_elastic_vti_train_gpu_batch256/resolved_config.yaml"
        ),
        "sample_index": 3,
    },
    "marmousi": {
        "config": ROOT / "configs/pino_elastic_vti_marmousi_gpu.yaml",
        "checkpoint": ROOT / "artifacts/elastic_vti_pino_marmousi/checkpoints/best.pt",
        "solver_config": Path(
            "/home/jiayh/Data/.atan/outputs/real_marmousi_wavefields_500_gpu_20260711_000540/resolved_config.yaml"
        ),
        "sample_index": 3,
    },
}


def summarize_seconds(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError("timings must be a non-empty sequence of finite positive seconds")
    return {
        "median_s": float(np.median(array)),
        "q25_s": float(np.percentile(array, 25)),
        "q75_s": float(np.percentile(array, 75)),
        "min_s": float(array.min()),
        "max_s": float(array.max()),
    }


def cuda_timed(callable_: Callable[[], object], repetitions: int, warmups: int) -> list[float]:
    if repetitions < 1 or warmups < 0:
        raise ValueError("repetitions must be >= 1 and warmups must be >= 0")
    for _ in range(int(warmups)):
        callable_()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(int(repetitions)):
        torch.cuda.synchronize()
        started = time.perf_counter()
        callable_()
        torch.cuda.synchronize()
        values.append(time.perf_counter() - started)
    return values


def _resize_field(array: np.ndarray, nz: int, nx: int) -> np.ndarray:
    tensor = torch.as_tensor(array, dtype=torch.float32)[None, None]
    return F.interpolate(tensor, size=(nz, nx), mode="bilinear", align_corners=False)[0, 0].numpy()


def _fd_medium(dataset: PinoHDF5Dataset, global_index: int, solver_config: dict) -> MediumSample:
    shard_idx, local_index = dataset._locate_sample(global_index)
    with h5py.File(dataset.paths[shard_idx], "r") as h5:
        vp_saved = np.asarray(h5["medium/vp_m_s"][local_index], dtype=np.float32)
        vs_saved = np.asarray(h5["medium/vs_m_s"][local_index], dtype=np.float32)
        model_type = h5["meta/model_type"][local_index]
        if isinstance(model_type, bytes):
            model_type = model_type.decode("utf-8")
    nz, nx = int(solver_config["domain"]["nz"]), int(solver_config["domain"]["nx"])
    vp = _resize_field(vp_saved, nz, nx)
    vs = _resize_field(vs_saved, nz, nx)
    rho = float(solver_config["medium"]["rho_kg_m3"])
    c11, c13, c33, c44 = stiffness_from_vp_vs_isotropic(vp, vs, rho)
    return MediumSample(
        vp_m_s=vp.astype(np.float32),
        vs_m_s=vs.astype(np.float32),
        c11_gpa=(np.asarray(c11) / 1.0e9).astype(np.float32),
        c13_gpa=(np.asarray(c13) / 1.0e9).astype(np.float32),
        c33_gpa=(np.asarray(c33) / 1.0e9).astype(np.float32),
        c44_gpa=(np.asarray(c44) / 1.0e9).astype(np.float32),
        model_type=str(model_type),
        split="benchmark",
        family_id=f"slide_benchmark_{global_index}",
        sample_seed=0,
        metadata={"source": dataset.paths[shard_idx], "local_index": int(local_index)},
    )


def benchmark_model(
    model_name: str,
    *,
    device: str,
    fd_warmups: int,
    fd_repetitions: int,
    pino_warmups: int,
    pino_repetitions: int,
) -> dict:
    if model_name not in MODEL_SPECS:
        raise ValueError(f"unknown model {model_name!r}")
    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the accepted benchmark contract requires an available CUDA device")
    spec = MODEL_SPECS[model_name]
    config = load_config(spec["config"])
    checkpoint = load_checkpoint(spec["checkpoint"], map_location="cpu")
    sample_index = int(spec["sample_index"])
    dataset = PinoHDF5Dataset(config, [sample_index], normalization_stats=checkpoint["normalization_stats"], return_normalized=True)
    sample = dataset[0]
    input_tensor = sample["input"].unsqueeze(0).to(device)
    model_config = {key: value for key, value in checkpoint["model_config"].items() if key != "name"}
    model = ElasticVTIFNO3D(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    @torch.inference_mode()
    def run_pino() -> torch.Tensor:
        return model(input_tensor)

    solver_config = load_solver_config(spec["solver_config"])
    medium = _fd_medium(dataset, sample_index, solver_config)

    def run_fd() -> object:
        return simulate_torch(medium, solver_config, device=device)

    pino_raw = cuda_timed(run_pino, pino_repetitions, pino_warmups)
    fd_raw = cuda_timed(run_fd, fd_repetitions, fd_warmups)
    pino_summary = summarize_seconds(pino_raw)
    fd_summary = summarize_seconds(fd_raw)
    return {
        "model": model_name,
        "device": torch.cuda.get_device_name(torch.device(device)),
        "sample_index": sample_index,
        "checkpoint": str(Path(spec["checkpoint"]).resolve()),
        "pino_config": str(Path(spec["config"]).resolve()),
        "solver_config": str(Path(spec["solver_config"]).resolve()),
        "pino_output_shape": list(run_pino().shape),
        "fd_physical_shape": [int(solver_config["domain"]["nz"]), int(solver_config["domain"]["nx"])],
        "fd_time_steps": int(solver_config["time"]["nt"]),
        "pino": {"warmups": pino_warmups, "repetitions": pino_repetitions, "raw_s": pino_raw, **pino_summary},
        "fd8": {"warmups": fd_warmups, "repetitions": fd_repetitions, "raw_s": fd_raw, **fd_summary},
        "speedup": float(fd_summary["median_s"] / pino_summary["median_s"]),
        "timing_scope": "CUDA compute plus output materialization; excludes HDF5 loading, plotting, and file writing",
    }


def merge_reports(paths: Sequence[Path]) -> dict:
    reports = [json.loads(Path(path).read_text(encoding="utf-8")) for path in paths]
    merged = {report["model"]: report for report in reports}
    if set(merged) != set(MODEL_SPECS):
        raise ValueError(f"merge requires {sorted(MODEL_SPECS)}, got {sorted(merged)}")
    return {"models": merged, "comparison_scope": "same RTX 3090, batch=1, per-sample forward task"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(MODEL_SPECS))
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--fd-warmups", type=int, default=1)
    parser.add_argument("--fd-repetitions", type=int, default=3)
    parser.add_argument("--pino-warmups", type=int, default=5)
    parser.add_argument("--pino-repetitions", type=int, default=20)
    parser.add_argument("--merge", nargs="*", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.merge is not None:
        payload = merge_reports(args.merge)
    elif args.model:
        payload = benchmark_model(
            args.model,
            device=args.device,
            fd_warmups=args.fd_warmups,
            fd_repetitions=args.fd_repetitions,
            pino_warmups=args.pino_warmups,
            pino_repetitions=args.pino_repetitions,
        )
    else:
        raise SystemExit("provide --model or --merge")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
