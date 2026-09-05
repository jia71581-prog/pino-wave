"""Merge GPU-sharded smoothed-background P_bg caches into one provider-readable file.

The full/large-pool P_bg cache is built by running build_smoothed_background_cache.py
as N disjoint per-GPU processes (each writes a shard over a record subset). This script
concatenates the shards into a single lwc84_multifidelity_teacher_v1 cache with strictly
increasing source_index (the NumericalTeacherCache read() contract), which
BackgroundFieldProvider then loads unchanged.

All shards must share sigma, schema, time axis, and be record-disjoint.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import h5py


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="shard h5 paths to merge")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shards = [Path(s) for s in args.shards]
    for s in shards:
        if not s.exists():
            raise FileNotFoundError(f"missing shard {s}")

    sigma = None
    time_s = None
    time_indices = None
    rows = []  # (source_index, sample_id, shard_path, row_in_shard)
    for sp in shards:
        with h5py.File(sp, "r") as h:
            if str(h.attrs.get("status")) != "complete":
                raise ValueError(f"shard {sp} not complete")
            sg = float(h.attrs["background_sigma_saved_cells"])
            sigma = sg if sigma is None else sigma
            if abs(sg - sigma) > 1e-9:
                raise ValueError(f"shard {sp} sigma {sg} != {sigma}")
            ts = np.asarray(h["time_s"][:])
            ti = np.asarray(h["time_indices"][:])
            time_s = ts if time_s is None else time_s
            time_indices = ti if time_indices is None else time_indices
            if not np.array_equal(ts, time_s) or not np.array_equal(ti, time_indices):
                raise ValueError(f"shard {sp} time axis differs")
            sids = [s.decode() if isinstance(s, bytes) else str(s) for s in h["sample_id"][:]]
            srcs = np.asarray(h["source_index"][:])
            for row, (src, sid) in enumerate(zip(srcs, sids)):
                rows.append((int(src), sid, str(sp), row))

    # Strictly increasing source_index; reject duplicates (record-disjoint contract).
    rows.sort(key=lambda r: r[0])
    src_seen = set()
    for src, sid, _, _ in rows:
        if src in src_seen:
            raise ValueError(f"duplicate source_index {src} across shards (not disjoint)")
        src_seen.add(src)

    nt, nz, nx = len(time_s), 201, 201
    out_path = Path(args.out)
    text_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(out_path, "w") as out:
        out.attrs["schema"] = "lwc84_multifidelity_teacher_v1"
        out.attrs["status"] = "complete"
        out.attrs["source_manifest_sha256"] = ""
        out.attrs["background_sigma_saved_cells"] = float(sigma)
        out.attrs["background_kind"] = "gaussian_smoothed_velocity_lwc84_solve"
        out.create_dataset("sample_id", data=np.asarray([r[1] for r in rows], dtype=object), dtype=text_dtype)
        out.create_dataset("source_index", data=np.asarray([r[0] for r in rows], dtype=np.int64))
        out.create_dataset("time_indices", data=time_indices)
        out.create_dataset("time_s", data=time_s)
        wf = out.create_dataset(
            "wavefield", shape=(len(rows), nt, nz, nx), dtype=np.float32,
            chunks=(1, 1, nz, nx), compression="gzip", compression_opts=1,
        )
        # Stream each row from its shard to bound memory.
        open_shards: dict[str, h5py.File] = {}
        try:
            for out_row, (_, _, sp, in_row) in enumerate(rows):
                if sp not in open_shards:
                    open_shards[sp] = h5py.File(sp, "r")
                wf[out_row] = open_shards[sp]["wavefield"][in_row]
        finally:
            for h in open_shards.values():
                h.close()
    print(f"merged {len(shards)} shards -> {out_path}  records={len(rows)} "
          f"sigma={sigma} size={out_path.stat().st_size/1e6:.1f}MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
