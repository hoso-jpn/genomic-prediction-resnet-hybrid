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
(marker/variant rows, sample columns), so this loader transposes after
reading exactly as ``_load_genotype_frame`` does.

Interpretation is taken from the manifest, not assumed: the manifest
embeds the encoding contract, and anything this loader was not written
against (a different ``schema_version``, encoding schema, orientation,
missing token, or ploidy constraint) fails explicitly instead of being
read under the wrong assumptions.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
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

SUPPORTED_SCHEMA_VERSION = 1
SUPPORTED_ENCODING_SCHEMA = "diploid_additive_dosage_v1"
EXPECTED_ORIENTATION = "variant_rows_by_sample_columns"
EXPECTED_MISSING_TOKEN = "nan"
EXPECTED_PLOIDY = "diploid_only"
EXPECTED_DOSAGES = {"0/0": -1.0, "0/1_or_1/0": 0.0, "1/1": 1.0}
ALLOWED_DOSAGES = (-1.0, 0.0, 1.0)

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
    schema_version = manifest.get("schema_version")
    _require(
        schema_version == SUPPORTED_SCHEMA_VERSION,
        f"unsupported GS panel schema_version: {schema_version!r} "
        f"(expected {SUPPORTED_SCHEMA_VERSION})",
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
        and {key: float(value) for key, value in dosages.items()} == EXPECTED_DOSAGES,
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
        raise ValueError(  # noqa: TRY004
            "manifest is missing the checksums block"
        )
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


def _read_matrix(path: Path) -> tuple[list[str], list[str], FloatArray]:
    """Read the variant-rows-by-sample-columns matrix as written on disk."""
    with gzip.open(path, mode="rt", newline="") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if not header or header[0] != VARIANT_KEY_COLUMN:
            raise ValueError(
                f"matrix header must start with '{VARIANT_KEY_COLUMN}', "
                f"found {header[:1]}"
            )
        sample_ids = header[1:]
        if not sample_ids:
            # A zero-variant panel is a normal outcome; a zero-sample
            # header is not, and the producer treats it as a hard error too.
            raise ValueError("matrix header lists no samples")

        variant_keys: list[str] = []
        rows: list[list[float]] = []
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(sample_ids) + 1:
                raise ValueError(
                    f"matrix line {line_number} has {len(fields) - 1} dosage "
                    f"cells, expected {len(sample_ids)}"
                )
            variant_keys.append(fields[0])
            rows.append(
                [
                    _parse_dosage(token, variant_key=fields[0], sample_id=sample_id)
                    for token, sample_id in zip(fields[1:], sample_ids)
                ]
            )

    matrix = (
        np.asarray(rows, dtype=np.float64)
        if rows
        else np.empty((0, len(sample_ids)), dtype=np.float64)
    )
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
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate {kind} in the GS panel: {duplicates}")


def load_gs_panel(
    panel_dir: str | Path,
    *,
    cohort_id: str | None = None,
    verify_file_checksums: bool = True,
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
    manifest = json.loads(manifest_path.read_text())
    validate_manifest(manifest, cohort_id=resolved_cohort)

    matrix_name = f"{resolved_cohort}{MATRIX_SUFFIX}"
    sample_metadata_name = f"{resolved_cohort}{SAMPLE_METADATA_SUFFIX}"
    variant_metadata_name = f"{resolved_cohort}{VARIANT_METADATA_SUFFIX}"
    for name in (matrix_name, sample_metadata_name, variant_metadata_name):
        if not (panel_dir / name).is_file():
            raise FileNotFoundError(f"GS panel file not found: {panel_dir / name}")

    if verify_file_checksums:
        verify_checksums(
            panel_dir,
            manifest,
            [matrix_name, sample_metadata_name, variant_metadata_name],
        )

    sample_ids, variant_keys, matrix = _read_matrix(panel_dir / matrix_name)
    _check_duplicates(sample_ids, kind="sample IDs")
    _check_duplicates(variant_keys, kind="variant keys")

    sample_metadata = pd.read_table(panel_dir / sample_metadata_name)
    variant_metadata = pd.read_table(panel_dir / variant_metadata_name)
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

    # On disk the matrix is variant rows by sample columns; the in-memory
    # convention (as in SoynamDataset) is sample rows by marker columns.
    return AdzukiGsPanel(
        genotypes=np.ascontiguousarray(matrix.T),
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
    positions = [int(value) for value in frame[index_column].tolist()]
    if positions != list(range(len(expected))):
        raise ValueError(
            f"{kind} metadata '{index_column}' must be 0-indexed positions "
            "matching the matrix order"
        )
