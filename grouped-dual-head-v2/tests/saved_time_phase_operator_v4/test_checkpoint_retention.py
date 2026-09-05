"""Checkpoint-retention (keep-last-K) guard against per-epoch disk hoarding.

A 40-epoch run keeps 40 x ~0.32 GiB of epoch_NNNN.pt snapshots; with the warp run
and B1 both writing, the ~25 GiB free data disk would fill mid-training and break
checkpoint writes. prune_epoch_checkpoints keeps only the most recent K plus the
current best epoch. best.pt/latest.pt are hardlinks INTO checkpoints/, so pruning a
named epoch file never loses the best data (verified here by hardlinking).
"""
from __future__ import annotations

import os
from pathlib import Path

from scripts.train_saved_time_v4_full_support import prune_epoch_checkpoints


def _touch_epochs(d: Path, epochs, nbytes=16):
    d.mkdir(parents=True, exist_ok=True)
    for e in epochs:
        (d / f"epoch_{e:04d}.pt").write_bytes(b"x" * nbytes)


def _present(d: Path):
    return sorted(int(p.stem.split("_", 1)[-1]) for p in d.glob("epoch_*.pt"))


def test_none_keeps_all_epochs(tmp_path):
    d = tmp_path / "checkpoints"
    _touch_epochs(d, range(1, 8))
    removed = prune_epoch_checkpoints(d, keep_last=None, best_epoch=3)
    assert removed == []
    assert _present(d) == list(range(1, 8))


def test_zero_or_negative_disables(tmp_path):
    d = tmp_path / "checkpoints"
    _touch_epochs(d, range(1, 6))
    assert prune_epoch_checkpoints(d, keep_last=0, best_epoch=None) == []
    assert prune_epoch_checkpoints(d, keep_last=-3, best_epoch=None) == []
    assert _present(d) == list(range(1, 6))


def test_keeps_last_k_plus_best(tmp_path):
    d = tmp_path / "checkpoints"
    _touch_epochs(d, range(1, 11))          # epochs 1..10
    removed = prune_epoch_checkpoints(d, keep_last=3, best_epoch=2)
    # keep last 3 (8,9,10) plus best (2); remove 1,3,4,5,6,7
    assert _present(d) == [2, 8, 9, 10]
    assert removed == [1, 3, 4, 5, 6, 7]


def test_best_inside_last_k_is_not_double_counted(tmp_path):
    d = tmp_path / "checkpoints"
    _touch_epochs(d, range(1, 6))           # 1..5
    removed = prune_epoch_checkpoints(d, keep_last=2, best_epoch=5)
    assert _present(d) == [4, 5]            # best 5 already in last-2
    assert removed == [1, 2, 3]


def test_best_hardlink_data_survives_prune(tmp_path):
    """best.pt hardlinks into checkpoints/; pruning the named epoch file must not lose data."""
    root = tmp_path
    d = root / "checkpoints"
    _touch_epochs(d, range(1, 6))
    best_src = d / "epoch_0002.pt"
    best_link = root / "best.pt"
    os.link(best_src, best_link)            # hardlink, as the trainer does
    # prune with best_epoch=None so epoch_0002 is a deletion candidate (last-1 keeps only 5)
    removed = prune_epoch_checkpoints(d, keep_last=1, best_epoch=None)
    assert 2 in removed and not best_src.exists()
    # the hardlink still holds the bytes -> best.pt is intact
    assert best_link.exists() and best_link.read_bytes() == b"x" * 16


def test_ignores_non_epoch_files(tmp_path):
    d = tmp_path / "checkpoints"
    _touch_epochs(d, [1, 2, 3])
    (d / "latest.pt").write_bytes(b"y" * 8)
    (d / "notes.txt").write_bytes(b"z")
    prune_epoch_checkpoints(d, keep_last=1, best_epoch=None)
    assert _present(d) == [3]
    assert (d / "latest.pt").exists() and (d / "notes.txt").exists()
