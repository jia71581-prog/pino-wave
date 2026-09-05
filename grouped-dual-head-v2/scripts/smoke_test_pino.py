from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.evaluate import run_evaluation
from fno_acoustic.train import run_training
from fno_acoustic.utils import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/pino_adaptation/smoke_metrics.json"))
    args = parser.parse_args()
    result = run_training(args.config, device_override=args.device, output_metrics=args.output_json)
    evaluation = run_evaluation(args.config, result["checkpoint_best"], split="val")
    result["evaluation"] = {
        "metrics": "artifacts/pino_adaptation/evaluation/metrics.json",
        "per_time": "artifacts/pino_adaptation/evaluation/per_time_metrics.csv",
        "figures": evaluation.get("figures", []),
    }
    write_json(args.output_json, result)
    print(result)


if __name__ == "__main__":
    main()
