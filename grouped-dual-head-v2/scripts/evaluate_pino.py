from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fno_acoustic.evaluate import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val")
    args = parser.parse_args()
    result = run_evaluation(args.config, args.checkpoint, args.split)
    print(result)


if __name__ == "__main__":
    main()
