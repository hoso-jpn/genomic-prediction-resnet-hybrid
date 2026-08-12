"""Shared SoyNAM raw dataset loading utilities."""

import csv
import gzip
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

PHENOTYPE_SUFFIX = "_phenotype_data.tsv.gz"
GENOTYPE_SUFFIX = "_SNP_genotype_Wm82.a1.tsv.gz"
PHENOTYPE_COLUMN = "Yld (kg/ha)"
SAMPLE_COLUMN = "Corrected Strain"
GENOTYPE_ENCODING = {
    "A": -1.0,
    "A/A": -1.0,
    "H": 0.0,
    "A/B": 0.0,
    "B": 1.0,
    "B/B": 1.0,
}


@dataclass(frozen=True)
class SoynamDataset:
    """Raw aligned SoyNAM genotype and phenotype records."""

    genotypes: FloatArray
    phenotypes: FloatArray
    family_ids: NDArray[np.str_]
    sample_names: NDArray[np.str_]
    marker_names: NDArray[np.str_]


def _family_id_from_phenotype(path: Path) -> str:
    if not path.name.endswith(PHENOTYPE_SUFFIX):
        raise ValueError(f"unexpected phenotype filename: {path.name}")
    return path.name.removesuffix(PHENOTYPE_SUFFIX)


def _family_id_from_genotype(path: Path) -> str:
    if not path.name.endswith(GENOTYPE_SUFFIX):
        raise ValueError(f"unexpected genotype filename: {path.name}")
    return path.name.removesuffix(GENOTYPE_SUFFIX).removesuffix("_4312")


def _founder_parent_name(family_id: str) -> str:
    return family_id.split("_NAM", maxsplit=1)[0]


def _map_family_files(
    paths: list[Path], family_id_fn: Callable[[Path], str], file_kind: str
) -> dict[str, Path]:
    """Map each family ID to its file, rejecting collisions instead of overwriting."""
    family_files: dict[str, Path] = {}
    for path in sorted(paths):
        family_id = family_id_fn(path)
        if family_id in family_files:
            conflicting = sorted([family_files[family_id].name, path.name])
            raise ValueError(
                f"multiple {file_kind} files map to family '{family_id}': {conflicting}"
            )
        family_files[family_id] = path
    return family_files


def _pair_family_files(data_dir: Path) -> list[tuple[str, Path, Path]]:
    phenotype_files = _map_family_files(
        list(data_dir.glob(f"*{PHENOTYPE_SUFFIX}")),
        _family_id_from_phenotype,
        "phenotype",
    )
    genotype_files = _map_family_files(
        list(data_dir.glob(f"*{GENOTYPE_SUFFIX}")),
        _family_id_from_genotype,
        "genotype",
    )

    if not phenotype_files:
        raise FileNotFoundError(f"no phenotype files found in {data_dir}")

    phenotype_families = set(phenotype_files)
    genotype_families = set(genotype_files)
    if phenotype_families != genotype_families:
        missing_genotypes = sorted(phenotype_families - genotype_families)
        missing_phenotypes = sorted(genotype_families - phenotype_families)
        raise ValueError(
            "phenotype/genotype family pairing is invalid: "
            f"missing_genotypes={missing_genotypes}, "
            f"missing_phenotypes={missing_phenotypes}"
        )

    return [
        (family_id, phenotype_files[family_id], genotype_files[family_id])
        for family_id in sorted(phenotype_families)
    ]


def list_family_files(data_dir: str | Path = "data") -> list[tuple[str, Path, Path]]:
    """List the validated (family_id, phenotype_path, genotype_path) triples.

    Reuses the same file-pairing and collision checks as
    ``load_soynam_dataset`` so callers (e.g. run manifest builders) never
    re-derive which files were actually read from a second, divergent code
    path.
    """
    return _pair_family_files(Path(data_dir))


def _strip_identifiers(values: list[object]) -> list[str]:
    """Strip identifiers, mapping missing values to an empty string."""
    stripped: list[str] = []
    for value in values:
        if pd.isna(value):
            stripped.append("")
        else:
            stripped.append(str(value).strip())
    return stripped


def _reject_missing_or_empty(
    values: list[str],
    *,
    family_id: str,
    filename: str,
    id_kind: str,
    position_label: str,
) -> None:
    blank_positions = [position for position, value in enumerate(values) if value == ""]
    if blank_positions:
        raise ValueError(
            f"missing or empty {id_kind} in family '{family_id}' file "
            f"'{filename}': {position_label} {blank_positions}"
        )


def _reject_duplicates(
    values: list[str], *, family_id: str, filename: str, id_kind: str
) -> None:
    counts = Counter(values)
    duplicates = sorted(value for value, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(
            f"duplicate {id_kind} in family '{family_id}' file "
            f"'{filename}': {duplicates}"
        )


def _validate_identifiers(
    raw_values: list[object],
    *,
    family_id: str,
    filename: str,
    id_kind: str,
    position_label: str,
) -> list[str]:
    """Strip identifiers, then reject missing, empty, or duplicate values."""
    stripped = _strip_identifiers(raw_values)
    _reject_missing_or_empty(
        stripped,
        family_id=family_id,
        filename=filename,
        id_kind=id_kind,
        position_label=position_label,
    )
    _reject_duplicates(
        stripped, family_id=family_id, filename=filename, id_kind=id_kind
    )
    return stripped


def _load_phenotype_frame(path: Path, family_id: str) -> pd.DataFrame:
    """Load a phenotype file indexed by validated sample ID, values left raw."""
    frame = pd.read_table(path, compression="gzip")
    required_columns = {PHENOTYPE_COLUMN, SAMPLE_COLUMN}
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"missing phenotype columns in {path.name}: {sorted(missing_columns)}"
        )

    sample_ids = _validate_identifiers(
        frame[SAMPLE_COLUMN].tolist(),
        family_id=family_id,
        filename=path.name,
        id_kind="phenotype sample ID",
        position_label="data row position(s)",
    )
    return frame.assign(**{SAMPLE_COLUMN: sample_ids}).set_index(SAMPLE_COLUMN)


def _read_genotype_header(path: Path) -> list[str]:
    """Read the raw genotype header before pandas can mangle duplicate columns."""
    with gzip.open(path, mode="rt", newline="") as handle:
        return next(csv.reader(handle, delimiter="\t"))


def _load_genotype_frame(path: Path, family_id: str) -> pd.DataFrame:
    """Load a genotype file as sample rows by marker columns, both validated."""
    header = _read_genotype_header(path)
    if len(header) < 2:
        raise ValueError(
            f"genotype file has no sample ID columns in family '{family_id}' "
            f"file '{path.name}'"
        )
    sample_ids = _validate_identifiers(
        header[1:],
        family_id=family_id,
        filename=path.name,
        id_kind="genotype sample ID",
        position_label="header sample position(s)",
    )

    frame = pd.read_table(path, compression="gzip", index_col=0, dtype=str)
    marker_ids = _validate_identifiers(
        frame.index.tolist(),
        family_id=family_id,
        filename=path.name,
        id_kind="marker ID",
        position_label="data row position(s)",
    )
    frame.index = pd.Index(marker_ids)
    frame.columns = pd.Index(sample_ids)
    return frame.T


def _check_marker_consistency(
    expected_markers: pd.Index | None,
    current_markers: pd.Index,
    *,
    family_id: str,
    genotype_filename: str,
) -> pd.Index:
    """Validate a family's marker index against the reference family's markers."""
    if expected_markers is None:
        return current_markers

    if set(expected_markers) != set(current_markers):
        reference_only = sorted(set(expected_markers) - set(current_markers))
        current_only = sorted(set(current_markers) - set(expected_markers))
        raise ValueError(
            f"marker set differs in family '{family_id}' file "
            f"'{genotype_filename}': reference_only={reference_only}, "
            f"current_only={current_only}"
        )

    if not expected_markers.equals(current_markers):
        raise ValueError(
            f"marker order differs in family '{family_id}' file '{genotype_filename}'"
        )

    return expected_markers


def _match_ril_samples(
    phenotype_samples: list[str],
    genotype_samples: list[str],
    *,
    founder_parent: str,
    family_id: str,
    phenotype_filename: str,
    genotype_filename: str,
) -> list[str]:
    """Match RIL sample sets between phenotype and genotype, excluding the founder."""
    phenotype_ril = [sample for sample in phenotype_samples if sample != founder_parent]
    genotype_ril_set = {
        sample for sample in genotype_samples if sample != founder_parent
    }
    phenotype_ril_set = set(phenotype_ril)

    if phenotype_ril_set != genotype_ril_set:
        phenotype_only = sorted(phenotype_ril_set - genotype_ril_set)
        genotype_only = sorted(genotype_ril_set - phenotype_ril_set)
        raise ValueError(
            f"RIL sample sets differ in family '{family_id}' "
            f"(phenotype file '{phenotype_filename}', genotype file "
            f"'{genotype_filename}'): phenotype_only={phenotype_only}, "
            f"genotype_only={genotype_only}"
        )

    return phenotype_ril


def _is_blank_phenotype_value(value: object) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _convert_phenotype_values(
    raw_values: list[object],
    sample_ids: list[str],
    *,
    family_id: str,
    filename: str,
) -> tuple[FloatArray, BoolArray]:
    """Convert non-blank phenotype values to float, keeping a keep-mask."""
    keep_mask = np.array(
        [not _is_blank_phenotype_value(value) for value in raw_values], dtype=bool
    )
    numeric_values = np.full(len(raw_values), np.nan, dtype=np.float64)
    for position, (value, keep) in enumerate(zip(raw_values, keep_mask)):
        if not keep:
            continue
        try:
            numeric_values[position] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"non-numeric phenotype value in family '{family_id}' file "
                f"'{filename}' for sample '{sample_ids[position]}': {value!r}"
            ) from error
    return numeric_values, keep_mask


def _encode_genotypes(frame: pd.DataFrame, family_id: str) -> FloatArray:
    normalized = frame.astype("string").apply(lambda column: column.str.strip())
    missing = normalized.isna() | normalized.eq("-").fillna(False)
    known = missing | normalized.isin(list(GENOTYPE_ENCODING))
    unknown_mask = ~known.to_numpy(dtype=bool)

    if unknown_mask.any():
        raw_values = normalized.to_numpy(dtype=object)
        unknown = sorted(
            {str(value) for value in raw_values[unknown_mask] if pd.notna(value)}
        )
        raise ValueError(f"unknown genotype symbols in {family_id}: {unknown}")

    encoded = np.full(normalized.shape, np.nan, dtype=np.float64)
    for symbol, value in GENOTYPE_ENCODING.items():
        symbol_mask = normalized.eq(symbol).fillna(False).to_numpy(dtype=bool)
        encoded[symbol_mask] = value
    return encoded


def load_soynam_dataset(
    data_dir: str | Path = "data",
    *,
    family_files: list[tuple[str, Path, Path]] | None = None,
) -> SoynamDataset:
    """Load aligned RIL phenotypes and genotypes, excluding founder parents.

    If ``family_files`` is given (typically a prior ``list_family_files``
    result), it is used as-is instead of re-resolving ``data_dir``. This
    lets a caller fix the exact file list once and pass that same list to
    both this loader and a run manifest, rather than risking a directory
    change between two separate resolutions of the same path.
    """
    data_path = Path(data_dir)
    genotype_blocks: list[FloatArray] = []
    phenotype_blocks: list[FloatArray] = []
    family_labels: list[str] = []
    sample_labels: list[str] = []
    expected_markers: pd.Index | None = None

    resolved_family_files = (
        family_files if family_files is not None else _pair_family_files(data_path)
    )
    for family_id, phenotype_path, genotype_path in resolved_family_files:
        phenotype_frame = _load_phenotype_frame(phenotype_path, family_id)
        genotype_frame = _load_genotype_frame(genotype_path, family_id)

        expected_markers = _check_marker_consistency(
            expected_markers,
            genotype_frame.columns,
            family_id=family_id,
            genotype_filename=genotype_path.name,
        )

        founder_parent = _founder_parent_name(family_id)
        ril_samples = _match_ril_samples(
            phenotype_frame.index.tolist(),
            genotype_frame.index.tolist(),
            founder_parent=founder_parent,
            family_id=family_id,
            phenotype_filename=phenotype_path.name,
            genotype_filename=genotype_path.name,
        )

        phenotype_values, keep_mask = _convert_phenotype_values(
            phenotype_frame.loc[ril_samples, PHENOTYPE_COLUMN].tolist(),
            ril_samples,
            family_id=family_id,
            filename=phenotype_path.name,
        )
        final_samples = [sample for sample, keep in zip(ril_samples, keep_mask) if keep]
        if not final_samples:
            raise ValueError(
                f"no RIL samples remain in family '{family_id}' after "
                "excluding missing phenotypes"
            )

        genotype_block = _encode_genotypes(genotype_frame.loc[final_samples], family_id)
        phenotype_block = phenotype_values[keep_mask]

        genotype_blocks.append(genotype_block)
        phenotype_blocks.append(phenotype_block)
        family_labels.extend([family_id] * len(final_samples))
        sample_labels.extend(final_samples)

    if expected_markers is None:
        raise RuntimeError("marker metadata was not initialized")

    genotypes = np.concatenate(genotype_blocks, axis=0)
    phenotypes = np.concatenate(phenotype_blocks)
    family_ids = np.asarray(family_labels, dtype=str)
    sample_names = np.asarray(sample_labels, dtype=str)
    marker_names = expected_markers.astype(str).to_numpy()

    if genotypes.shape != (phenotypes.size, marker_names.size):
        raise RuntimeError("aligned dataset dimensions are inconsistent")
    if not np.isfinite(phenotypes).all():
        raise ValueError("phenotypes must contain only finite values")

    return SoynamDataset(
        genotypes=genotypes,
        phenotypes=phenotypes,
        family_ids=family_ids,
        sample_names=sample_names,
        marker_names=marker_names,
    )
