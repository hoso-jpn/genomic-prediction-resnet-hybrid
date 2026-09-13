# Bounded GS panel loading (#26)

`load_gs_panel(..., max_memory_bytes=512 * 1024**2)` retains float64,
NaN missing calls and opaque IDs. It allocates one C-contiguous sample-by-marker
array and parses marker rows directly into its columns. It does no imputation,
MAF filtering or feature selection.

Producer schema v1/v2 currently has no shape field. Before reading matrix bytes
(including the checksum pass), the loader checks metadata file sizes and counts
metadata rows in a streaming pass. An optional `matrix_shape: [markers, samples]`
is reconciled with those counts. Every actual matrix row, header and metadata ID
is then checked; an understated count cannot overrun the allocation.

The planning estimate is `8 * samples * markers + 8 * metadata_file_bytes +
1024 * (samples + markers)` bytes. It includes allowances for pandas text,
ID arrays, duplicate sets and a parsed row. It is a conservative admission policy,
not an OS RSS guarantee; interpreter/library baseline, allocator behavior and
pathological text lengths can differ. Metadata larger than the budget is refused
before constructing full metadata objects. Raising the budget is explicit.

This implementation deliberately supports **bounded dense loading only**. There
is no disk cache, memmap, temporary matrix or partial cache reuse; temporary disk
use is zero. Disk-full errors on a cache therefore do not apply. Over-budget
panels are refused before matrix reading. A future disk-backed implementation
would need its own storage-space and atomic-publication contract; NumPy memmap
alone does not guarantee downstream training memory bounds.

The baselines still make fold slices, imputed arrays, PCA/relationship matrices
and model tensors. Loader success is not a claim of end-to-end large-panel
support. A 300 by 10M float64 array alone needs 24 GB (decimal), so it is refused
by the default budget.

## Measured scope (2026-09-13, Linux, CPU, Python 3.11.16)

`uv run --frozen python scripts/benchmark_panel_loader.py --samples S --markers M`
generates synthetic input in the parent and measures each load in a fresh child.
Peak RSS includes interpreter, NumPy, pandas, metadata and duplicate checks.
Generation is excluded from wall time/RSS. Checksums are enabled.

| Samples | Markers | Wall seconds | Peak RSS KiB | Temporary disk bytes |
| ---: | ---: | ---: | ---: | ---: |
| 50 | 1,000 | 0.0174 | 74,092 | 0 |
| 500 | 1,000 | 0.1247 | 78,192 | 0 |
| 50 | 10,000 | 0.1614 | 80,932 | 0 |

These three points vary sample and marker counts independently. They establish
only small synthetic-load behavior, not performance extrapolations to 10M markers.

Validation covers budget refusal before gzip/checksums, shape overrun, damaged
gzip and fresh retry, checksum errors, v1/v2, empty panels, ID/order preservation,
float64/NaN and owning contiguous output. No caller files are written by loading.

Reference: [NumPy memmap](https://numpy.org/doc/stable/reference/generated/numpy.memmap.html).
