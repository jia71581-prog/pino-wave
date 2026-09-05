from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OpenFWIAdapter:
    root: str | Path | None
    enabled: bool = False
    families: tuple[str, ...] = ("curvevel-b", "flatfault-b", "curvefault-b")
    use_velocity_only: bool = True

    def list_velocity_models(self) -> list[Path]:
        if not self.enabled or self.root is None:
            return []
        root = Path(self.root)
        if not root.exists():
            return []
        suffixes = {".npy", ".npz", ".h5", ".hdf5"}
        return sorted(path for path in root.rglob("*") if path.suffix.lower() in suffixes)
