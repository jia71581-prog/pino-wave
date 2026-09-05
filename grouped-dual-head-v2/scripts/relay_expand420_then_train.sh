#!/bin/bash
# Auto-relay daemon: wait for 4 P_bg solve shards -> merge into expanded sigma2 cache
# (existing N192 + new 228/family = 420/family, 1152 train records) -> verify coverage
# -> launch the 40-epoch large-sampling SOAP continue-pretraining (32 frames), all in
# background so cc can exit. Logs to $LOG.
set -u
cd /root/autodl-tmp/work/FNO-Acoustic-Wave-Simulation/grouped-dual-head-v2
export HDF5_USE_FILE_LOCKING=FALSE
PY=/root/miniconda3/bin/python
EXP=/dev/shm/g3cache/expand420
OLD=/dev/shm/g3cache/background_pbg_sigma2_g3pool_N192.h5
MERGED=/dev/shm/g3cache/background_pbg_sigma2_g3pool_N420.h5
NORM=/root/autodl-tmp/home/jiayh/Data/data/processed/grouped_v3_normalization.before_tgrs_ablation_identity_20260726T1050.json
CKPT=results/helmholtz_g3_aplus1_r8_sigma2_N192_ddp4_soap_ep80/latest.pt
OUT=results/helmholtz_g3_aplus1_r8_sigma2_N420_ddp4_soap_frames32_ep40
LOG=$EXP/relay.log

echo "[relay] started $(date)" > $LOG

# 1. Wait for all 4 solve processes to finish (shard h5 written + status=complete).
while true; do
  done=0
  for g in 0 1 2 3; do
    if [ -f $EXP/shard_$g.h5 ]; then
      st=$($PY -c "import h5py;print(h5py.File('$EXP/shard_$g.h5','r').attrs.get('status'))" 2>/dev/null)
      [ "$st" = "complete" ] && done=$((done+1))
    fi
  done
  echo "[relay] $(date +%H:%M:%S) shards complete: $done/4" >> $LOG
  [ $done -eq 4 ] && break
  # abort if all build procs died but shards not complete (crash guard)
  alive=$(ps aux | grep build_smoothed_background_cache | grep -v grep | wc -l)
  if [ $alive -eq 0 ] && [ $done -lt 4 ]; then
    echo "[relay] ERROR: build procs gone but only $done/4 shards complete; aborting" >> $LOG
    exit 1
  fi
  sleep 60
done

# 2. Merge new shards + existing N192 cache -> N420 (strictly increasing source_index).
echo "[relay] merging $(date)" >> $LOG
$PY scripts/merge_background_shards.py \
  --shards $OLD $EXP/shard_0.h5 $EXP/shard_1.h5 $EXP/shard_2.h5 $EXP/shard_3.h5 \
  --out $MERGED >> $LOG 2>&1
if [ ! -f $MERGED ]; then echo "[relay] ERROR merge failed" >> $LOG; exit 1; fi

# 3. Verify coverage: expect 420/family train + 3 validation.
$PY - <<PYEOF >> $LOG 2>&1
import h5py, collections
f=h5py.File("$MERGED","r"); ids=[s.decode() for s in f["sample_id"][()]]
tr=collections.Counter(i.split('_')[1] for i in ids if i.startswith('train'))
val=[i for i in ids if i.startswith('validation')]
print("[relay] merged total", len(ids), "train per-family", dict(tr), "validation", len(val))
assert all(v>=420 for v in tr.values()), "coverage < 420/family"
assert len(val)==3, "validation triplet missing"
f.close()
print("[relay] coverage OK")
PYEOF
if [ $? -ne 0 ]; then echo "[relay] ERROR coverage check failed" >> $LOG; exit 1; fi

# 3b. Coverage verified -> free shm: drop the 4 shards (now merged in) so the
# subsequent training has headroom. Keep OLD N192 as a fallback until training starts.
rm -f $EXP/shard_0.h5 $EXP/shard_1.h5 $EXP/shard_2.h5 $EXP/shard_3.h5
echo "[relay] removed shards after merge $(df -h /dev/shm | tail -1)" >> $LOG

# 4. Launch the 40-epoch large-sampling SOAP continue-pretraining, 4-GPU DDP.
echo "[relay] launching 40ep large-sampling training $(date)" >> $LOG
mkdir -p $OUT
torchrun --nproc_per_node=4 --master_port=29541 scripts/diagnose_helmholtz_g3_heldout.py \
  --warmstart-helmholtz $CKPT --background-cache $MERGED --normalization-json $NORM \
  --records-per-family 420 --helmholtz-rank 8 --helmholtz-frequencies 64 --optimizer soap \
  --dense-learning-rate 1e-4 --backbone-learning-rate 5e-5 --local-field-learning-rate 5e-4 \
  --epochs 40 --macro-records 6 --macros-per-update 4 --evaluate-every 48 \
  --training-frames 32 \
  --artifact-dir $OUT >> $OUT/train.stdout.log 2>&1
echo "[relay] training exited $(date) rc=$?" >> $LOG
