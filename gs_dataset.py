"""Versioned individual-genotype/long-phenotype input for continuous GS traits."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import adzuki_gs_panel_data as gs_panel
import run_manifest
from soynam_data import SoynamDataset

KNOWN_ASSEMBLIES = frozenset({"synthetic-v1", "GCF_016808095.1"})
OBSERVATION_COLUMNS = (
    "observation_id",
    "sample_id",
    "trait",
    "unit",
    "year",
    "site",
    "replicate",
    "value",
)


@dataclass(frozen=True)
class GsDataset:
    """Every row is an observation; repeated genotypes remain explicitly linked."""

    baseline: SoynamDataset
    observations: pd.DataFrame
    panel: gs_panel.AdzukiGsPanel
    manifest: dict[str, Any]
    provenance: dict[str, Any]
    source_paths: tuple[Path, ...]
    excluded_samples: tuple[str, ...]
    excluded_observations: tuple[str, ...]


def read_table(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        header = next(reader, [])
        if len(header) != len(set(header)) or set(required) != set(header):
            raise ValueError(f"invalid columns in {path.name}; required {required}")
        records = []
        for row in reader:
            if len(row) != len(header):
                raise ValueError(f"ragged row in {path.name}")
            records.append(row)
    return pd.DataFrame(records, columns=header)


def require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def relative_file(root: Path, value: Any) -> Path:
    value = require_text(value, "file path")
    candidate = (root / value).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        raise ValueError(
            f"input must be an existing file within dataset directory: {value}"
        )
    return candidate


def validate_reference(panel, reference, *, expected_assembly=None):
    if not isinstance(reference, dict):
        raise ValueError("reference is required")
    assembly = require_text(reference.get("assembly_id"), "assembly_id")
    allowed = KNOWN_ASSEMBLIES if expected_assembly is None else {expected_assembly}
    if assembly not in allowed:
        raise ValueError("unknown assembly; explicitly declare expected_assembly")
    reference_file = require_text(reference.get("fasta_file"), "fasta_file")
    checksum = reference.get("fasta_sha256")
    if not isinstance(checksum, str) or not re.fullmatch(
        r"sha256:[0-9a-f]{64}", checksum
    ):
        raise ValueError("reference fasta_sha256 must be a SHA256 checksum")
    if panel.manifest.get("checksums", {}).get(reference_file) != checksum:
        raise ValueError("reference checksum does not match the panel manifest")
    for row in panel.variant_metadata.to_dict("records"):
        if not all(key in row for key in ("chrom", "pos", "ref", "alt")):
            raise ValueError("variant metadata must carry chrom/pos/ref/alt")
        if (
            row["ref"] not in "ACGT"
            or row["alt"] not in "ACGT"
            or len(row["ref"]) != 1
            or len(row["alt"]) != 1
            or row["ref"] == row["alt"]
        ):
            raise ValueError("only biallelic SNP alleles are supported")
        if not re.fullmatch(r"[1-9][0-9]*", row["pos"]):
            raise ValueError("variant position must be a positive integer")
        key = f"{row['chrom']}:{row['pos']}:{row['ref']}:{row['alt']}"
        if key != row["variant_key"]:
            raise ValueError("variant key/allele metadata mismatch")


def verify_unchanged(dataset: GsDataset) -> None:
    current = {p.name: run_manifest.sha256_file(p) for p in dataset.source_paths}
    if current != dataset.provenance["file_checksums"]:
        raise RuntimeError("dataset inputs changed during execution")


def load_dataset(
    path: Path,
    *,
    expected_assembly=None,
    max_memory_bytes=gs_panel.DEFAULT_MAX_MEMORY_BYTES,
) -> GsDataset:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 1
        or manifest.get("kind") != "gs_individual_dataset"
    ):
        raise ValueError(
            "unsupported individual dataset schema (summary statistics are not input)"
        )
    require_text(manifest.get("species"), "species")
    cohort = require_text(manifest.get("cohort_id"), "cohort_id")
    trait = manifest.get("trait", {})
    for field in ("name", "unit", "method", "scale"):
        require_text(trait.get(field), f"trait.{field}")
    if trait.get("type") != "continuous" or trait.get("direction") not in (
        "higher",
        "lower",
    ):
        raise ValueError(
            "trait must declare continuous type and higher/lower direction"
        )
    if manifest.get("effect_allele") != "ALT":
        raise ValueError("effect_allele must be ALT for this dosage encoding")
    if manifest.get("aggregation") != {"method": "none", "measurement_type": "raw"}:
        raise ValueError(
            "only raw observations without aggregation are supported; BLUE/BLUP require fold-aware correction"
        )
    panel_manifest = relative_file(path.parent, manifest.get("panel_manifest"))
    if panel_manifest.name != cohort + gs_panel.MANIFEST_SUFFIX:
        raise ValueError("panel manifest name/cohort mismatch")
    metadata_path = relative_file(path.parent, manifest.get("sample_metadata"))
    phenotype_path = relative_file(path.parent, manifest.get("phenotypes"))
    panel_paths = [
        panel_manifest.parent / (cohort + suffix)
        for suffix in (
            gs_panel.MATRIX_SUFFIX,
            gs_panel.SAMPLE_METADATA_SUFFIX,
            gs_panel.VARIANT_METADATA_SUFFIX,
        )
    ]
    sources = (path, panel_manifest, metadata_path, phenotype_path, *panel_paths)
    if len({p.name for p in sources}) != len(sources):
        raise ValueError("input basenames must be unique for checksum provenance")
    gs_panel.estimate_panel_load(
        panel_paths[1], panel_paths[2], max_memory_bytes=max_memory_bytes
    )
    if (
        8 * (metadata_path.stat().st_size + phenotype_path.stat().st_size)
        > max_memory_bytes
    ):
        raise ValueError("phenotype/metadata exceeds memory budget")
    before = {p.name: run_manifest.sha256_file(p) for p in sources}
    checksums = manifest.get("checksums", {})
    for p in (panel_manifest, metadata_path, phenotype_path):
        if checksums.get(p.name) != "sha256:" + before[p.name]:
            raise ValueError(f"dataset checksum mismatch for {p.name}")
    if (
        8 * (metadata_path.stat().st_size + phenotype_path.stat().st_size)
        > max_memory_bytes
    ):
        raise ValueError("phenotype/metadata exceeds memory budget")
    panel = gs_panel.load_gs_panel(
        panel_manifest.parent, cohort_id=cohort, max_memory_bytes=max_memory_bytes
    )
    if panel.is_empty:
        raise ValueError("empty genotype panel cannot train a model")
    validate_reference(
        panel, manifest.get("reference"), expected_assembly=expected_assembly
    )
    metadata = read_table(metadata_path, ("sample_id", "line_id", "family_id"))
    for column in ("sample_id", "line_id", "family_id"):
        if any(not value.strip() for value in metadata[column]):
            raise ValueError(f"empty {column}")
    if metadata.sample_id.duplicated().any() or set(metadata.sample_id) != set(
        panel.sample_ids
    ):
        raise ValueError("sample metadata IDs must match genotype IDs exactly")
    if (metadata.groupby("line_id").family_id.nunique() > 1).any():
        raise ValueError("a line cannot belong to multiple families")
    observations = read_table(phenotype_path, OBSERVATION_COLUMNS)
    for column in OBSERVATION_COLUMNS[:-1]:
        if any(not value.strip() for value in observations[column]):
            raise ValueError(f"empty observation {column}")
    if not set(observations.sample_id) <= set(panel.sample_ids):
        raise ValueError("phenotype sample ID is absent from genotype panel")
    if observations.observation_id.duplicated().any():
        raise ValueError("duplicate observation_id")
    if observations.duplicated(
        ["sample_id", "trait", "year", "site", "replicate"]
    ).any():
        raise ValueError("duplicate observation key")
    if set(observations.trait) != {trait["name"]} or set(observations.unit) != {
        trait["unit"]
    }:
        raise ValueError("trait or unit mismatch; implicit conversion is not supported")
    if not observations.year.str.fullmatch(r"[0-9]{4}").all():
        raise ValueError("year must be a four-digit year")
    observations["year"] = observations.year.astype(int)
    observation_lines = observations.merge(
        metadata[["sample_id", "line_id"]],
        on="sample_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    # The biological observation key must be unique before missing values are
    # excluded. Otherwise two tissue samples for one line/plot/replicate could
    # be silently accepted merely because one of the duplicate values is NA.
    if observation_lines.duplicated(
        ["line_id", "trait", "year", "site", "replicate"]
    ).any():
        raise ValueError("duplicate line observation key")
    missing = observations.value.isin(["", "NA", "nan"])
    values = pd.to_numeric(observations.value.mask(missing), errors="raise").to_numpy(
        dtype=float
    )
    if not np.isfinite(values[~missing]).all() or not np.isfinite(values).any():
        raise ValueError("trait is all missing or contains infinite values")
    excluded_observations = tuple(observations.loc[missing, "observation_id"])
    observations = observations.loc[~missing].copy()
    observations["value"] = values[~missing]
    observations = observations.merge(
        metadata, on="sample_id", how="left", validate="many_to_one", sort=False
    )
    if observations.shape[0] * panel.genotypes.shape[1] * 8 * 4 > max_memory_bytes:
        raise ValueError("expanded observation matrix exceeds training input budget")
    positions = {sample: i for i, sample in enumerate(panel.sample_ids)}
    baseline = SoynamDataset(
        genotypes=panel.genotypes[[positions[s] for s in observations.sample_id]],
        phenotypes=observations.value.to_numpy(dtype=float),
        family_ids=observations.family_id.to_numpy(dtype=str),
        sample_names=observations.observation_id.to_numpy(dtype=str),
        marker_names=panel.variant_keys,
    )
    provenance = {
        "file_checksums": before,
        "aggregation_hash": run_manifest.canonical_json_hash(manifest["aggregation"]),
        "trait_hash": run_manifest.canonical_json_hash(trait),
    }
    provenance["dataset_hash"] = run_manifest.canonical_json_hash(provenance)
    result = GsDataset(
        baseline,
        observations,
        panel,
        manifest,
        provenance,
        sources,
        tuple(s for s in panel.sample_ids if s not in set(observations.sample_id)),
        excluded_observations,
    )
    verify_unchanged(result)
    return result
