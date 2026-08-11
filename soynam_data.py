"""Shared SoyNAM raw dataset loading utilities."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]

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


def _pair_family_files(data_dir: Path) -> list[tuple[str, Path, Path]]:
    phenotype_files = {
        _family_id_from_phenotype(path): path
        for path in sorted(data_dir.glob(f"*{PHENOTYPE_SUFFIX}"))
    }
    genotype_files = {
        _family_id_from_genotype(path): path
        for path in sorted(data_dir.glob(f"*{GENOTYPE_SUFFIX}"))
    }

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


def load_soynam_dataset(data_dir: str | Path = "data") -> SoynamDataset:
    """Load aligned RIL phenotypes and genotypes, excluding founder parents."""
    data_path = Path(data_dir)
    genotype_blocks: list[FloatArray] = []
    phenotype_blocks: list[FloatArray] = []
    family_labels: list[str] = []
    sample_labels: list[str] = []
    expected_markers: pd.Index | None = None

    for family_id, phenotype_path, genotype_path in _pair_family_files(data_path):
        phenotype = pd.read_table(phenotype_path, compression="gzip")
        required_columns = {PHENOTYPE_COLUMN, SAMPLE_COLUMN}
        missing_columns = required_columns - set(phenotype.columns)
        if missing_columns:
            raise ValueError(
                f"missing phenotype columns in {phenotype_path.name}: "
                f"{sorted(missing_columns)}"
            )

        phenotype[PHENOTYPE_COLUMN] = pd.to_numeric(
            phenotype[PHENOTYPE_COLUMN], errors="coerce"
        )
        phenotype = (
            phenotype.dropna(subset=[PHENOTYPE_COLUMN, SAMPLE_COLUMN])
            .drop_duplicates(subset=SAMPLE_COLUMN)
            .set_index(SAMPLE_COLUMN)
        )

        genotype = pd.read_table(genotype_path, compression="gzip", index_col=0).T
        genotype = genotype[~genotype.index.duplicated(keep="first")]

        if expected_markers is None:
            expected_markers = genotype.columns.copy()
        elif not expected_markers.equals(genotype.columns):
            raise ValueError(f"marker order differs in {family_id}")

        aligned_samples = phenotype.index[phenotype.index.isin(genotype.index)]
        parent_name = family_id.split("_NAM", maxsplit=1)[0]
        aligned_samples = aligned_samples[aligned_samples.astype(str) != parent_name]
        if aligned_samples.empty:
            raise ValueError(f"no RIL samples remain in {family_id}")

        genotype_block = _encode_genotypes(genotype.loc[aligned_samples], family_id)
        phenotype_block = phenotype.loc[aligned_samples, PHENOTYPE_COLUMN].to_numpy(
            dtype=np.float64
        )

        genotype_blocks.append(genotype_block)
        phenotype_blocks.append(phenotype_block)
        family_labels.extend([family_id] * aligned_samples.size)
        sample_labels.extend(aligned_samples.astype(str).tolist())

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
