#!/usr/bin/env bash
# Resumable, collision-free parallel push of the new Marmousi VDS shards.
# Run this on the data server; each source HDF5 and its checksum sidecar are
# assigned to exactly one bucket, so several rsync processes can safely append
# different partial files at the destination.
set -uo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: $0 SOURCE_DIR DESTINATION SSH_KEY SSH_PORT BUCKETS LOG_DIR COMPLETE_MARKER RUN_LABEL" >&2
  exit 64
fi

source_dir=$1
destination=$2
ssh_key=$3
ssh_port=$4
bucket_count=$5
log_dir=$6
complete_marker=$7
run_label=$8

if [[ ! -d "$source_dir/shards" || ! -f "$source_dir/dataset_v1.h5" ]]; then
  echo "invalid source dataset: $source_dir" >&2
  exit 65
fi
if [[ ! -f "$ssh_key" || ! "$bucket_count" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid SSH key or bucket count" >&2
  exit 65
fi

mkdir -p "$log_dir"
bucket_dir="$log_dir/${run_label}_buckets"
mkdir "$bucket_dir"

for ((bucket = 0; bucket < bucket_count; bucket++)); do
  : > "$bucket_dir/bucket_$(printf '%02d' "$bucket").txt"
done

# Largest-first round robin keeps the buckets approximately balanced.  The
# sidecar follows its HDF5 so a bucket is independently verifiable/resumable.
bucket=0
while read -r relative_path size_bytes; do
  list="$bucket_dir/bucket_$(printf '%02d' "$bucket").txt"
  printf '%s\n%s.sha256\n' "$relative_path" "$relative_path" >> "$list"
  printf '%s %s\n' "$relative_path" "$size_bytes" >> "$bucket_dir/assignment.tsv"
  bucket=$(((bucket + 1) % bucket_count))
done < <(
  find "$source_dir/shards" -mindepth 2 -maxdepth 2 -type f -name '*.h5' \
    -printf '%P %s\n' | sort -k2,2nr
)

h5_count=$(wc -l < "$bucket_dir/assignment.tsv")
if [[ "$h5_count" -ne 127 ]]; then
  echo "unexpected new-shard HDF5 count: $h5_count (expected 127)" >&2
  exit 66
fi

rsh="ssh -i $ssh_key -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o StrictHostKeyChecking=yes -p $ssh_port"

echo "$(date --iso-8601=seconds) syncing VDS metadata"
rsync -a --exclude='/shards/***' -e "$rsh" "$source_dir/" "$destination/" || exit $?

sync_bucket() {
  local bucket_id=$1
  local list=$2
  local attempt=0
  while true; do
    attempt=$((attempt + 1))
    echo "$(date --iso-8601=seconds) bucket=$bucket_id attempt=$attempt start"
    rsync -a --partial --append-verify --human-readable --stats --timeout=180 \
      --files-from="$list" -e "$rsh" \
      "$source_dir/shards/" "$destination/shards/"
    rc=$?
    echo "$(date --iso-8601=seconds) bucket=$bucket_id attempt=$attempt rc=$rc"
    [[ "$rc" -eq 0 ]] && return 0
    sleep 20
  done
}

pids=()
for ((bucket = 0; bucket < bucket_count; bucket++)); do
  bucket_id=$(printf '%02d' "$bucket")
  sync_bucket "$bucket_id" "$bucket_dir/bucket_${bucket_id}.txt" &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "$pid"
done

printf 'completed_at=%s\nrun_label=%s\nbuckets=%s\nh5_files=%s\n' \
  "$(date --iso-8601=seconds)" "$run_label" "$bucket_count" "$h5_count" \
  > "$complete_marker"
echo "$(date --iso-8601=seconds) all buckets complete"
