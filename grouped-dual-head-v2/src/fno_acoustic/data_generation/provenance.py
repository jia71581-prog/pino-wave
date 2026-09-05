from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_commit(root: str | Path) -> str:
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(root), text=True, capture_output=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"
