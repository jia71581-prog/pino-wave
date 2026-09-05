from __future__ import annotations

import os
from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, *, state=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "state": state or {}, "extra": extra or {}}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    with open(tmp, "rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_checkpoint(path, model, optimizer=None, *, map_location="cpu"):
    payload = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
