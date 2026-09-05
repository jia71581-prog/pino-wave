# Current dataset missing-file diagnosis (2026-08-13)

## Conclusion

There is no missing backing data in the dataset used by r5d or by the registered
fixed-19-Hz position experiment. The current `dataset_v1.h5` is deliberately a
small HDF5 virtual-dataset index (VDS), not a monolithic copy of all wavefields.
Its `velocity_mps` and `wavefield` arrays each resolve through 501 backing shard
files. A metadata-only audit found zero absent and zero empty sources.

## Why historical logs reported missing files

1. One obsolete dataset-planning command targeted
   `/autodl-pub/data/jiayh/...`, whose parent filesystem is read-only. Directory
   creation failed there before any dataset could be written.
2. One obsolete Helmholtz smoke command requested the retired top-level index
   `/root/autodl-tmp/home/jiayh/Data/data/acoustic_lwc84_2km_401x401_to_201_v1/dataset_v1.h5`.
   That index is absent. The underlying v1 shards remain present and are valid
   backing sources of the current composite v2 VDS.
3. A separate historical rendering failure was a normalization-manifest mismatch,
   not a missing file. Failing closed there prevented incompatible statistics from
   being silently reused.

## Current verified dependencies

- Current VDS resolves from `/data/jiayh/...` through the `/data/jiayh` symlink to
  `/root/autodl-tmp/data/jiayh`.
- It exposes 4,003 records, 401 saved times, `201 x 201` velocity maps, and
  `401 x 201 x 201` complete wavefields.
- The required 16 top-level datasets are present.
- HDF5 binding attributes match the registered reference protocol: canonical
  config digest `7afe...13d`, dataset manifest `281f...57d`, generator commit
  `02749b1...94`, and Marmousi source digest `f430...9dd`.
- The prepared Marmousi array exists and its SHA-256 is
  `004086...c0deb`, exactly as registered.

The frozen YAML file's byte hash (`77fbd1...cbc6`) is intentionally different
from the HDF5 `config_sha256`: the latter is the generator's canonicalized config
digest, not a hash of the serialized YAML bytes.

## Safe correction

`scripts/audit_current_vds_dependencies.py` now performs a bounded, metadata-only,
fail-closed check of the active VDS and all virtual sources. It never indexes the
velocity or wavefield arrays, so it does not add material I/O pressure to active
training. The verified report is
`results/current_acoustic_vds_dependency_audit_20260813.json`.

Do not copy shards, rebuild the VDS, recreate the retired v1 index, or rewrite
absolute paths while r5d is active. The current symlink and VDS resolution are
working and all registered dependencies are present.
