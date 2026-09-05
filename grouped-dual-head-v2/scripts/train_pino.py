from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    result = run_training(args.config, device_override=args.device, output_metrics=args.output_json)
    print(result)


if __name__ == "__main__":
    main()
