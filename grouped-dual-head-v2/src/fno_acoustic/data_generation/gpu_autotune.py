from __future__ import annotations

from typing import Iterable


def choose_batch_size(
    *,
    candidates: Iterable[int],
    free_vram_gib: float,
    per_sample_gib: float,
    fixed_overhead_gib: float,
    headroom_fraction: float,
    headroom_min_gib: float,
) -> dict[str, float | int | list[int]]:
    reserved = max(float(headroom_min_gib), float(free_vram_gib) * float(headroom_fraction))
    usable = max(0.0, float(free_vram_gib) - reserved - float(fixed_overhead_gib))
    fitting = [int(c) for c in candidates if int(c) * float(per_sample_gib) <= usable + 1.0e-12]
    batch = max(fitting) if fitting else 0
    if batch < 1:
        raise RuntimeError("no batch size candidate fits available VRAM with required headroom")
    return {
        "batch_size": int(batch),
        "fitting_candidates": fitting,
        "free_vram_gib": float(free_vram_gib),
        "reserved_headroom_gib": float(reserved),
        "usable_vram_gib": float(usable),
    }
