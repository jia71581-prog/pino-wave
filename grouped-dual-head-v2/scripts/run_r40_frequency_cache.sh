#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2

MANIFEST="results/r40_frequency_manifest_20260828.json"
OUTPUT="results/r40_frequency_cache_v2_20260828"
PYTHON="/root/miniconda3/bin/python"

if [[ -e "${OUTPUT}" ]]; then
  echo "refusing to overwrite existing ${OUTPUT}" >&2
  exit 2
fi
mkdir -p "${OUTPUT}"

run_subset() {
  local subset="$1"
  local -a pids=()
  local failed=0
  for gpu in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="${gpu}" PYTHONPATH=scripts "${PYTHON}" \
      scripts/build_r40_frequency_cache.py \
      --manifest "${MANIFEST}" \
      --subset "${subset}" \
      --shard-index "${gpu}" \
      --shard-count 4 \
      --output "${OUTPUT}/${subset}_shard${gpu}.h5" \
      --device cuda:0 \
      --solver-batch-size 4 \
      --model-batch-size 16 \
      --amp \
      >"${OUTPUT}/${subset}_shard${gpu}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "${subset} cache generation failed" >&2
    exit 1
  fi
}

run_subset fit
run_subset holdout

"${PYTHON}" - <<'PY'
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

root = Path("results/r40_frequency_cache_v2_20260828")
summaries = []
for subset in ("fit", "holdout"):
    for shard in range(4):
        path = root / f"{subset}_shard{shard}.h5.summary.json"
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
selection = {item["selection_sha256"] for item in summaries}
if len(selection) != 1:
    raise RuntimeError("R40 cache selection hashes differ")
payload = {
    "schema": "r40_frequency_cache_bundle_v2",
    "status": "complete",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "selection_sha256": next(iter(selection)),
    "fit_record_count": sum(x["record_count"] for x in summaries if x["subset"] == "fit"),
    "holdout_record_count": sum(x["record_count"] for x in summaries if x["subset"] == "holdout"),
    "validation_opened": False,
    "test_id_opened": False,
    "shards": summaries,
}
canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
payload["bundle_sha256"] = hashlib.sha256(canonical).hexdigest()
(root / "bundle_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(payload, indent=2, sort_keys=True))
PY
