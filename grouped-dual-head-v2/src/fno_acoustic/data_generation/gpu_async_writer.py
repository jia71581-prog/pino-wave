from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RingSlot:
    index: int
    array: np.ndarray


class PinnedRingBuffer:
    def __init__(self, *, slot_count: int, shape: tuple[int, ...], dtype, pinned: bool = False) -> None:
        if slot_count < 2:
            raise ValueError("slot_count must be at least 2")
        self.pinned = bool(pinned)
        self._slots = [RingSlot(i, np.zeros(shape, dtype=dtype)) for i in range(int(slot_count))]
        self._free = list(range(int(slot_count)))
        self._ready: list[int] = []

    def acquire_for_write(self) -> RingSlot:
        if not self._free:
            raise RuntimeError("no free ring buffer slots")
        return self._slots[self._free.pop(0)]

    def mark_ready(self, slot: RingSlot) -> None:
        if slot.index in self._ready:
            raise RuntimeError("slot is already ready")
        self._ready.append(slot.index)

    def pop_ready(self) -> RingSlot:
        if not self._ready:
            raise RuntimeError("no ready ring buffer slots")
        index = self._ready.pop(0)
        slot = self._slots[index]
        self._free.append(index)
        return slot
