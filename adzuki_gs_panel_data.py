"""Loader for adzuki-snp-pipeline's Genomic Selection (GS) panel output.

The producer (`hoso-jpn/adzuki-snp-pipeline`, `docs/gs_panel_data_contract.md`)
writes four files per cohort:

```text
<cohort_id>.gs_panel.genotype_matrix.tsv.gz   variant rows x sample columns
<cohort_id>.gs_panel.sample_metadata.tsv      one row per sample
<cohort_id>.gs_panel.variant_metadata.tsv     one row per variant
<cohort_id>.gs_panel.manifest.json            schema_version, encoding, checksums
```

The dosage encoding (`-1` hom-ref, `0` het, `+1` hom-alt, `nan` missing)
is the same additive scale as ``soynam_data.GENOTYPE_ENCODING``, and the
matrix has the same on-disk orientation as ``soynam_data`` genotype files
(marker/variant rows, sample columns), so this loader writes each marker directly into the final sample-major array.

Interpretation is taken from the manifest, not assumed: the manifest
embeds the encoding contract, and anything this loader was not written
against (a different ``schema_version``, encoding schema, orientation,
missing token, or ploidy constraint) fails explicitly instead of being
read under the wrong assumptions.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]

MANIFEST_SUFFIX = ".gs_panel.manifest.json"
MATRIX_SUFFIX = ".gs_panel.genotype_matrix.tsv.gz"
SAMPLE_METADATA_SUFFIX = ".gs_panel.sample_metadata.tsv"
VARIANT_METADATA_SUFFIX = ".gs_panel.variant_metadata.tsv"

SUPPORTED_SCHEMA_VERSIONS = (1, 2)
SUPPORTED_ENCODING_SCHEMA = "diploid_additive_dosage_v1"
EXPECTED_ORIENTATION = "variant_rows_by_sample_columns"
EXPECTED_MISSING_TOKEN = "nan"
EXPECTED_PLOIDY = "diploid_only"
EXPECTED_DOSAGES = {"0/0": -1.0, "0/1_or_1/0": 0.0, "1/1": 1.0}
ALLOWED_DOSAGES = (-1.0, 0.0, 1.0)
DEFAULT_MAX_MEMORY_BYTES = 512 * 1024**2

VARIANT_KEY_COLUMN = "variant_key"
SAMPLE_ID_COLUMN = "sample_id"
SAMPLE_INDEX_COLUMN = "sample_index"
VARIANT_INDEX_COLUMN = "variant_index"


@dataclass(frozen=True)
class AdzukiGsPanel:
    """One cohort's GS panel, in the same in-memory shape as SoyNAM data.

    ``genotypes`` is sample rows by variant columns (the on-disk file is
    transposed on load), with ``nan`` for missing calls and no imputation
    applied. Phenotypes are not part of this panel: the producer emits
    genotypes and provenance only, so a caller that needs phenotypes must
    join them by ``sample_ids`` itself.
    """

    genotypes: FloatArray
    sample_ids: StringArray
    variant_keys: StringArray
    cohort_id: str
    manifest: dict[str, Any]
    sample_metadata: pd.DataFrame
    variant_metadata: pd.DataFrame

    @property
    def panel_status(self) -> str | None:
        """The producer's machine-readable status, e.g. ``"empty"``."""
        status = self.manifest.get("panel_status")
        return str(status) if status is not None else None

    @property
    def is_empty(self) -> bool:
        """True when the panel has samples but zero GS-eligible variants."""
        return self.genotypes.shape[1] == 0


def _resolve_cohort_id(panel_dir: Path, cohort_id: str | None) -> str:
    if cohort_id is not None:
        manifest_path = panel_dir / f"{cohort_id}{MANIFEST_SUFFIX}"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"no GS panel manifest at {manifest_path}")
        return cohort_id

    manifests = sorted(panel_dir.glob(f"*{MANIFEST_SUFFIX}"))
    if not manifests:
        raise FileNotFoundError(f"no '*{MANIFEST_SUFFIX}' file found in {panel_dir}")
    if len(manifests) > 1:
        found = sorted(path.name for path in manifests)
        raise ValueError(f"multiple cohorts found; pass cohort_id explicitly: {found}")
    return manifests[0].name.removesuffix(MANIFEST_SUFFIX)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_manifest(manifest: dict[str, Any], *, cohort_id: str) -> None:
    """Reject any manifest this loader was not written against.

    The producer records the encoding contract inside the manifest
    precisely so a reader does not have to infer it; a mismatch here means
    the file on disk is not the schema this loader parses, so it is an
    error rather than a warning.
    """
    _require(isinstance(manifest, dict), "GS panel manifest must be a JSON object")
    schema_version = manifest.get("schema_version")
    _require(
        type(schema_version) is int and schema_version in SUPPORTED_SCHEMA_VERSIONS,
        f"unsupported GS panel schema_version: {schema_version!r} "
        f"(expected one of {SUPPORTED_SCHEMA_VERSIONS})",
    )
    # v2 changes container provenance keys; the dosage/matrix contract is v1
    # in both manifest versions. Preserve containers without reinterpreting it.
    parameters = manifest.get("parameters")
    _require(
        isinstance(parameters, dict)
        and type(parameters.get("sample_ploidy")) is int
        and parameters["sample_ploidy"] == 2,
        "manifest parameters.sample_ploidy must be 2",
    )
    manifest_cohort = manifest.get("cohort_id")
    _require(
        manifest_cohort == cohort_id,
        f"manifest cohort_id {manifest_cohort!r} does not match the file "
        f"prefix {cohort_id!r}",
    )

    encoding = manifest.get("genotype_encoding")
    _require(
        isinstance(encoding, dict),
        "manifest is missing the genotype_encoding block",
    )
    assert isinstance(encoding, dict)  # narrowed by the check above
    _require(
        encoding.get("schema") == SUPPORTED_ENCODING_SCHEMA,
        f"unsupported genotype encoding schema: {encoding.get('schema')!r} "
        f"(expected {SUPPORTED_ENCODING_SCHEMA!r})",
    )
    _require(
        encoding.get("matrix_orientation") == EXPECTED_ORIENTATION,
        f"unexpected matrix orientation: {encoding.get('matrix_orientation')!r} "
        f"(expected {EXPECTED_ORIENTATION!r})",
    )
    _require(
        encoding.get("missing_token") == EXPECTED_MISSING_TOKEN,
        f"unexpected missing token: {encoding.get('missing_token')!r} "
        f"(expected {EXPECTED_MISSING_TOKEN!r})",
    )
    _require(
        encoding.get("ploidy") == EXPECTED_PLOIDY,
        f"unexpected ploidy constraint: {encoding.get('ploidy')!r} "
        f"(expected {EXPECTED_PLOIDY!r}); schema v1 is diploid-only",
    )

    dosages = encoding.get("dosage_by_genotype")
    _require(
        isinstance(dosages, dict)
        and all(type(value) in (int, float) for value in dosages.values())
        and dosages == EXPECTED_DOSAGES,
        f"unexpected dosage table: {dosages!r} (expected {EXPECTED_DOSAGES})",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checksums(
    panel_dir: Path, manifest: dict[str, Any], filenames: list[str]
) -> None:
    """Check the given panel files against the manifest's own checksums.

    Only the files this loader reads are checked; the manifest also
    records inputs (VCFs, reference FASTA) that are not distributed with
    the panel, and a missing entry for one of those is not this loader's
    concern.
    """
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict):
        # A malformed manifest is bad data, not a caller type error.
        raise ValueError("manifest is missing the checksums block")
    for filename in filenames:
        expected = checksums.get(filename)
        if expected is None:
            raise ValueError(f"manifest records no checksum for '{filename}'")
        actual = f"sha256:{_sha256(panel_dir / filename)}"
        if actual != expected:
            raise ValueError(
                f"checksum mismatch for '{filename}': the file does not match "
                "the manifest it was distributed with"
            )


@dataclass(frozen=True)
class PanelLoadEstimate:
    """Conservative planning estimate, not a guaranteed RSS upper bound."""

    samples: int
    markers: int
    array_bytes: int
    metadata_bytes: int
    estimated_memory_bytes: int
    temporary_disk_bytes: int = 0


def estimate_panel_load(
    sample_path: Path, variant_path: Path, *, max_memory_bytes: int
) -> PanelLoadEstimate:
    if type(max_memory_bytes) is not int or max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be a positive integer")
    metadata_bytes = sample_path.stat().st_size + variant_path.stat().st_size
    # pandas strings, ID arrays, duplicate sets and one parsed matrix row.
    # Reject oversized metadata before building Python objects or opening gzip.
    if 8 * metadata_bytes > max_memory_bytes:
        raise ValueError("GS panel metadata exceeds memory budget before matrix read")
    counts = []
    for path in (sample_path, variant_path):
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            next(reader, None)
            counts.append(sum(bool(row) for row in reader))
    samples, markers = counts
    array_bytes = samples * markers * np.dtype(np.float64).itemsize
    estimated = array_bytes + 8 * metadata_bytes + 1024 * (samples + markers)
    if estimated > max_memory_bytes:
        raise ValueError(
            f"GS panel estimated memory {estimated} bytes exceeds memory budget "
            f"{max_memory_bytes} bytes (shape={samples}x{markers}); "
            "use a smaller panel or explicitly raise max_memory_bytes"
        )
    return PanelLoadEstimate(samples, markers, array_bytes, metadata_bytes, estimated)


def _read_matrix(
    path: Path, *, expected_samples: int, expected_markers: int
) -> tuple[list[str], list[str], FloatArray]:
    """Parse one row at a time directly into the final C-contiguous array."""
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        if not header or header[0] != VARIANT_KEY_COLUMN:
            raise ValueError(f"matrix header must start with '{VARIANT_KEY_COLUMN}'")
        sample_ids = header[1:]
        if not sample_ids:
            raise ValueError("matrix header lists no samples")
        if len(sample_ids) != expected_samples:
            raise ValueError("sample metadata does not match the matrix shape")
        matrix = np.empty((expected_samples, expected_markers), dtype=np.float64)
        variant_keys: list[str] = []
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != expected_samples + 1:
                raise ValueError(
                    f"matrix line {line_number} has {len(fields) - 1} dosage "
                    f"cells, expected {expected_samples}"
                )
            marker_index = len(variant_keys)
            if marker_index >= expected_markers:
                raise ValueError("variant metadata does not match the matrix shape")
            variant_keys.append(fields[0])
            for sample_index, token in enumerate(fields[1:]):
                matrix[sample_index, marker_index] = _parse_dosage(
                    token, variant_key=fields[0], sample_id=sample_ids[sample_index]
                )
        if len(variant_keys) != expected_markers:
            raise ValueError("variant metadata does not match the matrix shape")
    return sample_ids, variant_keys, matrix


def _parse_dosage(token: str, *, variant_key: str, sample_id: str) -> float:
    """Convert one dosage cell, rejecting anything outside the contract."""
    stripped = token.strip()
    if stripped == EXPECTED_MISSING_TOKEN:
        return float("nan")
    try:
        value = float(stripped)
    except ValueError as error:
        raise ValueError(
            f"unparsable dosage {token!r} at variant '{variant_key}', "
            f"sample '{sample_id}'"
        ) from error
    if value not in ALLOWED_DOSAGES:
        raise ValueError(
            f"dosage {token!r} at variant '{variant_key}', sample "
            f"'{sample_id}' is outside the contract {ALLOWED_DOSAGES} plus "
            f"'{EXPECTED_MISSING_TOKEN}'"
        )
    return value


def _check_duplicates(values: list[str], *, kind: str) -> None:
    if any(not value.strip() for value in values):
        raise ValueError(f"empty {kind} in the GS panel")
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {kind} in the GS panel: {value!r}")
        seen.add(value)


def load_gs_panel(
    panel_dir: str | Path,
    *,
    cohort_id: str | None = None,
    verify_file_checksums: bool = True,
    max_memory_bytes: int = DEFAULT_MAX_MEMORY_BYTES,
) -> AdzukiGsPanel:
    """Load one cohort's GS panel into a sample-rows-by-variant-columns array.

    ``cohort_id`` may be omitted when the directory holds exactly one
    panel. Set ``verify_file_checksums=False`` only when the manifest's
    checksums are known not to apply (e.g. a deliberately edited fixture);
    by default the three panel files are verified against the manifest
    they came with.
    """
    panel_dir = Path(panel_dir)
    resolved_cohort = _resolve_cohort_id(panel_dir, cohort_id)

    manifest_path = panel_dir / f"{resolved_cohort}{MANIFEST_SUFFIX}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, cohort_id=resolved_cohort)

    matrix_name = f"{resolved_cohort}{MATRIX_SUFFIX}"
    sample_metadata_name = f"{resolved_cohort}{SAMPLE_METADATA_SUFFIX}"
    variant_metadata_name = f"{resolved_cohort}{VARIANT_METADATA_SUFFIX}"
    for name in (matrix_name, sample_metadata_name, variant_metadata_name):
        if not (panel_dir / name).is_file():
            raise FileNotFoundError(f"GS panel file not found: {panel_dir / name}")

    estimate = estimate_panel_load(
        panel_dir / sample_metadata_name,
        panel_dir / variant_metadata_name,
        max_memory_bytes=max_memory_bytes,
    )
    # Future manifests may declare shape. Never trust it without reconciliation.
    if "matrix_shape" in manifest:
        declared = manifest["matrix_shape"]
        if (
            not isinstance(declared, list)
            or any(type(value) is not int for value in declared)
            or declared != [estimate.markers, estimate.samples]
        ):
            raise ValueError("manifest matrix_shape does not match metadata")

    if verify_file_checksums:
        verify_checksums(
            panel_dir,
            manifest,
            [matrix_name, sample_metadata_name, variant_metadata_name],
        )

    sample_ids, variant_keys, matrix = _read_matrix(
        panel_dir / matrix_name,
        expected_samples=estimate.samples,
        expected_markers=estimate.markers,
    )
    _check_duplicates(sample_ids, kind="sample IDs")
    _check_duplicates(variant_keys, kind="variant keys")

    # IDs are opaque strings: pandas inference corrupts '001' and NA-like IDs.
    # Preserve all metadata text; numeric consumers can convert selected fields.
    sample_metadata = pd.read_table(
        panel_dir / sample_metadata_name, dtype=str, keep_default_na=False
    )
    variant_metadata = pd.read_table(
        panel_dir / variant_metadata_name, dtype=str, keep_default_na=False
    )
    _check_metadata_alignment(
        sample_metadata,
        expected=sample_ids,
        id_column=SAMPLE_ID_COLUMN,
        index_column=SAMPLE_INDEX_COLUMN,
        kind="sample",
    )
    _check_metadata_alignment(
        variant_metadata,
        expected=variant_keys,
        id_column=VARIANT_KEY_COLUMN,
        index_column=VARIANT_INDEX_COLUMN,
        kind="variant",
    )

    # Matrix is already sample-major: no full transpose copy is needed.
    return AdzukiGsPanel(
        genotypes=matrix,
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        variant_keys=np.asarray(variant_keys, dtype=np.str_),
        cohort_id=resolved_cohort,
        manifest=manifest,
        sample_metadata=sample_metadata,
        variant_metadata=variant_metadata,
    )


def _check_metadata_alignment(
    frame: pd.DataFrame,
    *,
    expected: list[str],
    id_column: str,
    index_column: str,
    kind: str,
) -> None:
    """Require metadata rows to describe the matrix in the same order.

    The producer writes both files from one pass over the same data, so a
    mismatch means the files were not produced together (or were edited);
    silently re-aligning them would attach the wrong metadata to a row.
    """
    for column in (id_column, index_column):
        if column not in frame.columns:
            raise ValueError(
                f"{kind} metadata is missing the '{column}' column; "
                f"found {list(frame.columns)}"
            )
    actual = [str(value) for value in frame[id_column].tolist()]
    if actual != expected:
        raise ValueError(
            f"{kind} metadata does not match the matrix: "
            f"{len(actual)} row(s) vs {len(expected)} in the matrix, "
            "or a different order"
        )
    positions = frame[index_column].tolist()
    if positions != [str(index) for index in range(len(expected))]:
        raise ValueError(
            f"{kind} metadata '{index_column}' must be 0-indexed positions "
            "matching the matrix order"
        )
