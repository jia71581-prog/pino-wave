# AIS dataset content binding v2

## Scope and safety

This contract binds the logical values consumed by formal AIS training and
evaluation before any GPU work begins. Development and tests use tiny HDF5
fixtures only. Generating the production manifest for the 2,500-sample,
241.399 GiB dataset is an explicit later operation and is not part of
implementation verification.

The sole implementation of canonical encoding, sample/global hashing, Merkle
construction, content-root construction, and manifest validation lives in
`src/fno_acoustic/ais_dataset_binding.py`. The offline generator and runtime
store call that module; they must not carry independent hashing logic.

## Primitive encoding

All unsigned integers use big-endian byte order.

- `uint32(n)`: four-byte unsigned big-endian integer.
- `uint64(n)`: eight-byte unsigned big-endian integer.
- `blob(value)`: `uint64(len(value)) || value`.
- `text(value)`: UTF-8 encode the string, then encode it as `blob`.

Canonical arrays are encoded in this exact order:

1. `text("array-v1")`
2. field name as `text`
3. canonical dtype as `text`
4. rank as `uint32`
5. each shape extent as `uint64`, in axis order
6. byte count as `uint64`
7. C-order data bytes

Sample numeric arrays are normalized to little-endian `<f4`; global coordinates
are normalized to little-endian `<f8`. String values are decoded as UTF-8 and
encoded with `text`; invalid UTF-8 is fatal. Hashing must use views,
`memoryview`, and bounded chunks where possible so that reading one formal
sample does not create an additional roughly 100 MiB copy.

## Sample leaves and Merkle tree

For sample ID `i`, the sample payload is the following concatenation in exact
order:

1. `text("AIS-DENSE-SAMPLE-v1")`
2. `uint64(i)`
3. canonical array framing for `tensor[i]` as `<f4`
4. canonical array framing for `nu[i]` as `<f4`
5. canonical array framing for `source_mask[i]` as `<f4`
6. `model_type[i]`, decoded as UTF-8 and encoded with `text`

`leaf_digest = SHA256(sample_payload)`. Manifest sample digests are lowercase
hex strings in a sample-ID-ordered array.

Merkle nodes use domain separation:

- leaf node: `SHA256(0x00 || leaf_digest_bytes)`
- parent: `SHA256(0x01 || left_node || right_node)`

At every odd-width level, duplicate the final node before pairing. The final
node is `sample_merkle_root`, represented as lowercase hex. An empty sample set
is invalid.

## Global digest

The global payload starts with `text("AIS-DENSE-GLOBAL-v1")`, followed in exact
order by:

1. `t-coordinate` as canonical `<f8`
2. `x-coordinate` converted from kilometres to metres, then canonical `<f8`
3. `y-coordinate` converted from kilometres to metres, then canonical `<f8`
4. the required dataset schema
5. root attributes

The required dataset schema uses this exact byte framing:

1. `text("dataset-schema-v1")`
2. entry count as `uint32`
3. for each entry: `text(name)`, `text(dtype)`, rank as `uint32`, each full
   shape extent as `uint64`, axis count as `uint32`, then each axis name as
   `text`

There are exactly four entries in this fixed order and with these constants:

| name | dtype | axis contract |
| --- | --- | --- |
| `tensor` | `<f4` | `sample,time,x,y` |
| `nu` | `<f4` | `sample,x,y` |
| `source_mask` | `<f4` | `sample,x,y` |
| `model_type` | `utf8-decoded` | `sample` |

The coordinate arrays bind their own names, `<f8` dtype, and shapes through the
array framing that precedes the schema. The schema binds logical shape and axis
meaning without binding HDF5 chunking or compression. Required numeric datasets
whose logical dtype is not the listed dtype are invalid; fixed- and
variable-width HDF5 strings are both canonicalized as `utf8-decoded`.

Root attributes use this exact framing:

1. `text("root-attrs-v1")`
2. attribute count as `uint32`
3. for each lexicographically sorted key: `text(key)`, one-byte type tag, then
   the tag-specific payload

The tags and payloads are:

| tag | value | payload |
| --- | --- | --- |
| `00` | `None` | empty |
| `01` | false | empty |
| `02` | true | empty |
| `03` | signed int64 | eight-byte signed big-endian integer |
| `04` | float64 | IEEE-754 eight-byte big-endian float |
| `05` | text | `text(value)` |
| `06` | bytes | `blob(value)` |
| `07` | ndarray | ndarray framing below |

The ndarray payload is `text("ndarray-v1")`, `text(canonical_dtype)`, rank as
`uint32`, each extent as `uint64`, byte count as `uint64`, then C-order bytes.
It deliberately has no field name. Supported ndarray dtypes are exactly `|b1`,
`<i8`, `<u8`, `<f4`, and `<f8`; numeric data are converted to the named
little-endian representation. This covers the production `save_indices` int64
attribute. Unsupported scalar/array types, out-of-range integers, invalid UTF-8,
and object arrays are fatal.

`global_sha256 = SHA256(global_payload)`.

## Dataset content root

The logical content root is:

```text
SHA256(0x02 || global_digest_bytes || merkle_digest_bytes || uint64(sample_count))
```

The logical root deliberately excludes the dataset path, filesystem stat
metadata, HDF5 chunk layout, and compression settings. Equivalent logical HDF5
files therefore have the same content root.

## Manifest v2

The final manifest is canonical JSON encoded as UTF-8 with
`sort_keys=True`, `separators=(",", ":")`, `ensure_ascii=False`, followed by
one newline. Its top-level key set is exact; extra or missing keys are fatal:

```json
{
  "schema": "ais_dataset_content_manifest",
  "version": 2,
  "hash_contract": "ais_dense_logical_v1",
  "hash_algorithm": "sha256",
  "dataset_path": "/canonical/absolute/regular-file.h5",
  "dataset_stat": {
    "device": 0,
    "inode": 0,
    "size_bytes": 0,
    "mtime_ns": 0
  },
  "sample_count": 2500,
  "required_datasets": [
    {"name":"tensor","shape":[2500,160,400,400],"dtype":"<f4","axis_contract":["sample","time","x","y"]},
    {"name":"nu","shape":[2500,400,400],"dtype":"<f4","axis_contract":["sample","x","y"]},
    {"name":"source_mask","shape":[2500,400,400],"dtype":"<f4","axis_contract":["sample","x","y"]},
    {"name":"model_type","shape":[2500],"dtype":"utf8-decoded","axis_contract":["sample"]}
  ],
  "global_sha256": "lowercase-64-hex",
  "sample_digests": ["digest-for-ID-0", "digest-for-ID-1"],
  "sample_merkle_root": "lowercase-64-hex",
  "dataset_content_root": "lowercase-64-hex"
}
```

All JSON integers are nonnegative integers (not booleans). `sample_digests` has
exactly `sample_count` lowercase 64-hex entries; array index is the sample ID,
so no separate digest ID is encoded. Every required-dataset object has exactly
the keys `name`, `shape`, `dtype`, and `axis_contract`. `dataset_stat` has
exactly `device`, `inode`, `size_bytes`, and `mtime_ns`, captured with `fstat`
from the opened regular file. `dataset_path` is its canonical absolute path.

Formal paths accept version 2 only. Legacy version 1 manifests fail closed.
The loader requires the raw file bytes to equal `canonical_json_bytes(parsed)`
and therefore rejects noncanonical whitespace, alternate escaping, and extra
keys. Manifest validation validates every hex
digest and list count, recomputes the Merkle root and dataset content root, and
checks global/schema/stat bindings before a GPU can be selected.

## Offline generator and recovery

`scripts/generate_ais_dataset_content_manifest.py`:

- refuses symlink dataset, output, lock, and checkpoint paths;
- takes an exclusive `flock` for the output namespace;
- opens a regular dataset file and binds its `fstat`; any stat change during
  generation is fatal;
- uses `<output>.lock` and `<output>.partial`; both paths are derived exactly
  from the final output path;
- publishes a canonical partial checkpoint atomically after every eight newly
  completed leaves and once at the end;
- resumes only when dataset identity, stat, global digest, contract, and the
  contiguous sample-digest prefix match;
- produces byte-identical final output after interruption and resume;
- treats an identical existing final manifest as idempotent success;
- refuses to replace a different existing final manifest;
- atomically renames a fully fsynced temporary final file.

The partial checkpoint has this exact canonical-JSON structure:

```json
{
  "schema": "ais_dataset_content_manifest_partial",
  "version": 1,
  "header": {
    "schema": "ais_dataset_content_manifest",
    "version": 2,
    "hash_contract": "ais_dense_logical_v1",
    "hash_algorithm": "sha256",
    "dataset_path": "/canonical/absolute/regular-file.h5",
    "dataset_stat": {"device":0,"inode":0,"size_bytes":0,"mtime_ns":0},
    "sample_count": 2500,
    "required_datasets": [
      {"name":"tensor","shape":[2500,160,400,400],"dtype":"<f4","axis_contract":["sample","time","x","y"]},
      {"name":"nu","shape":[2500,400,400],"dtype":"<f4","axis_contract":["sample","x","y"]},
      {"name":"source_mask","shape":[2500,400,400],"dtype":"<f4","axis_contract":["sample","x","y"]},
      {"name":"model_type","shape":[2500],"dtype":"utf8-decoded","axis_contract":["sample"]}
    ],
    "global_sha256": "lowercase-64-hex"
  },
  "committed_count": 8,
  "sample_digests": ["digest-for-ID-0-through-7"]
}
```

The top-level and header key sets are exact. `committed_count` equals the digest
list length and the list is the contiguous ID prefix starting at zero. On
`--resume`, the generator must re-read HDF5 samples `0..committed_count-1`,
recompute each digest, and compare the whole prefix before continuing. The
header `required_datasets` must equal the complete four-entry schema freshly
recomputed from the current HDF5 file, with exact objects, fields, ordering,
shapes, dtypes, and axes; resume compares it field-for-field with no coercion.
The generator also freshly checks every other header field. The
partial file is not authority and there is no trust-partial mode. Resume thus
does not save prefix scan I/O; checkpointing every eight leaves only limits
loss of serialized progress and detects intervening mutation. A partial file is
never accepted by runtime validation. The resume CLI prints this full-prefix
rehash cost before scanning so that checkpointing is not mistaken for an I/O
optimization.

The 241.399 GiB production scan is intrinsically expensive and must be launched
explicitly outside this implementation task. Each logical sample is 103.68 MB
(98.877 MiB): the production tensor's natural HDF5 chunk is
`[1,160,400,400]`, approximately 102.4 MB, and `nu`/`source_mask` are 0.64 MB
each. The default generator processes one whole sample at a time in increasing
ID order and is expected to use roughly 105--120 MB RSS. Small arrays use their
natural HDF5 read. Time-block mode is accepted only when the requested block is
at least the HDF5 tensor chunk's time extent and is an integer multiple of that
extent. A smaller or misaligned request fails with a recommendation to use the
whole-sample default. In particular, the production chunk time extent is 160,
so `--time-block 16` is rejected: it would decompress/read the same natural
chunk about ten times and is not a low-memory mode for this file. On a separately
chunked compatible fixture, streamed framing and the digest must be byte-identical
to whole-sample mode. No parallel or reordered hash stream is allowed. Runtime
hashes the already-read sample NumPy arrays via `memoryview`, avoiding another
approximately 100 MiB copy.

Before reading a manifest or partial file, the implementation opens without
following the leaf symlink, `fstat`s it, and rejects a size greater than 2 MiB.
The generic schema ceiling is 1,000,000 samples; the registered formal AIS
manifest separately requires exactly 2,500. Required dataset objects, shapes,
dtypes, and axes are exact, `model_type` decoded UTF-8 is at most 4,096 bytes,
and the total canonical encoded root-attribute payload is at most 16 MiB.
`RecursionError` and `MemoryError` are converted to deterministic validation
failure. Every existing parent path component for the dataset, manifest,
partial, lock, and output is rejected if it is a symlink.

Ignoring metadata overhead, ideal sequential throughput of 0.2, 0.5, 1, 2, and
3 GB/s gives approximately 21.6, 8.64, 4.32, 2.16, and 1.44 minutes. Expected
wall time is typically 3--10 minutes on NVMe and 20--30 minutes on HDD, but must
be measured when the production scan is separately authorized.

## Runtime verification

`DenseCPUQueryStore` accepts an explicit dataset-content binding. `None` remains
available only to historical unit fixtures and non-formal callers. Construction
of a bound store validates the manifest v2 contract, canonical dataset path,
regular-file stat audit fields, required global coordinate/schema/root-attribute
digest, sample count, Merkle root, and content root.

On every `read_scene(i)`, the store hashes the already-read NumPy values for
`tensor[i]`, `nu[i]`, `source_mask[i]`, and `model_type[i]` through the sole
canonical implementation and compares the leaf digest. It also cheaply
recomputes the global coordinate/schema/root-attribute digest on every read.
The verified-ID set is evidence only and never suppresses either rehash. The ID
is added only after the post-read `fstat` succeeds; leaf/global/stat failure
leaves the evidence set unchanged. This defeats content mutation followed by
restoration of the manifest mtime. Cache entries are never used as authority
across reads or processes. The leaf is verified before its NumPy values are
converted to Torch or returned for model consumption.

The verified-ID summary hash is exactly
`SHA256(text("AIS-VERIFIED-IDS-v1") || uint64(count) || each sorted unique ID as
uint64)`. Summaries record that digest, `verified_sample_count`, and
`verified_sample_scope="process_unique"`. The count and IDs describe only the
current child process cache.

Formal schema-5 checkpoints additionally persist exactly
`cumulative_verified_sample_ids` (sorted unique integer list),
`cumulative_verified_sample_count`, `cumulative_verified_sample_ids_sha256`, and
`dataset_content_root`. These fields are mutually checked. Resume inherits the
checkpoint list and unions it with IDs freshly verified by the current store;
the inherited list never skips per-read verification. Finalize-only inherits
the checkpoint cumulative set unchanged. Summaries retain the process-unique
fields and additionally publish the exact cumulative list/count/hash with
`run_cumulative_verified_sample_scope="run_cumulative"`.

The runner computes the deterministic expected cumulative set. Gate O is the
single registered overfit ID. H1/H2/H3 are the cyclic registered training-order
prefix through the exact target update, union the complete registered validation
set because validation cadence is no greater than every target boundary. Parent
checkpoint cumulative IDs carry phase/resume history. A zero-ID formal
checkpoint or summary is invalid.
Runner decisions use only the exact run-cumulative set; process-unique fields
remain diagnostic and may be empty only for finalize-only execution.

Formal training, evaluation, B1 baseline preparation, and the serial runner
require a valid v2 binding. A missing manifest, v1 manifest, noncanonical JSON,
invalid schema, malformed digest, internally inconsistent Merkle/content root,
wrong path, wrong global digest, or initial stat drift fails in the CPU
preflight before a GPU child is selected. A self-consistent forged manifest with
an incorrect leaf cannot be disproved by metadata alone; it fails lazily when
that sample is first read, before conversion to Torch or model consumption.
The existing manifest-file SHA-256 remains the checkpoint
`dataset_binding_sha256`. Training/evaluation summaries additionally record the
logical `dataset_content_root`, verified sample count, and a canonical hash of
the sorted verified sample IDs; the runner validates these fields before
publishing a row.

## Required tiny-fixture tests

Tests mutate each consumed sample field, each shared coordinate, dataset schema,
and each supported root-attribute type and must observe binding failure.
Mutating an unconsumed `wavelet` dataset must preserve the content root.
Additional tests cover canonical/format/internal-root tampering at preflight,
self-consistent fake leaves at every scene read, restored-mtime mutation,
post-stat evidence-set atomicity, zero-dimensional NumPy attributes, v1
rejection, interrupted resume with byte-identical output, final
idempotence/difference refusal, size/schema/text/attribute resource ceilings,
parent-component symlinks, locking, chunk-compatible and rejected time-blocks,
stat drift, cumulative checkpoint/resume/finalize-only IDs, deterministic runner
stage sets, and formal rejection of a fake manifest. No test or verification
command scans the production dataset.
