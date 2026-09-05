from __future__ import annotations

import argparse
import json
from pathlib import Path


def export_code(notebook: Path, output: Path) -> int:
    with notebook.open("r", encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    code_cells = [cell for cell in cells if cell.get("cell_type") == "code"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        f.write(f"# Exported from {notebook}\n")
        f.write(f"# Code cells: {len(code_cells)}\n\n")
        for idx, cell in enumerate(code_cells):
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            f.write(f"\n# %% cell {idx}\n")
            f.write(source)
            if not source.endswith("\n"):
                f.write("\n")
    return len(code_cells)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=Path, default=Path("Fourier_Acoustic_train.ipynb"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pino_adaptation/original_notebook_code.py"))
    args = parser.parse_args()
    count = export_code(args.notebook, args.output)
    print(f"exported {count} code cells to {args.output}")


if __name__ == "__main__":
    main()
