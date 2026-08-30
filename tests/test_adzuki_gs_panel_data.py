"""Tests for the adzuki-snp-pipeline GS panel loader.

Every fixture here is synthetic and written by the test itself, in the
exact file layout documented by the producer's data contract
(`hoso-jpn/adzuki-snp-pipeline`, `docs/gs_panel_data_contract.md`). No
real cohort data is used or distributed.
"""

import gzip
import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import adzuki_gs_panel_data as gs_panel
from soynam_data import GENOTYPE_ENCODING

COHORT = "cohort"
SAMPLES = ["S1", "S2", "S3"]
VARIANTS = ["Chr1:100:A:T", "Chr1:200:G:C"]
DOSAGE_ROWS = [["-1", "0", "1"], ["nan", "1", "-1"]]


def _sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _write_panel(
    panel_dir: Path,
    *,
    cohort_id: str = COHORT,
    samples: list[str] | None = None,
    variants: list[str] | None = None,
    rows: list[list[str]] | None = None,
    manifest_overrides: dict | None = None,
    sample_metadata_rows: list[dict] | None = None,
    variant_metadata_rows: list[dict] | None = None,
) -> Path:
    """Write one synthetic GS panel in the producer's on-disk layout."""
    samples = SAMPLES if samples is None else samples
    variants = VARIANTS if variants is None else variants
    rows = DOSAGE_ROWS if rows is None else rows
    panel_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = panel_dir / f"{cohort_id}{gs_panel.MATRIX_SUFFIX}"
    with gzip.open(matrix_path, mode="wt", newline="") as handle:
        handle.write("\t".join(["variant_key", *samples]) + "\n")
        for variant_key, dosages in zip(variants, rows):
            handle.write("\t".join([variant_key, *dosages]) + "\n")

    if sample_metadata_rows is None:
        sample_metadata_rows = [
            {
                "cohort_id": cohort_id,
                "sample_index": index,
                "sample_id": sample_id,
                "missing_genotype_count": 0,
                "missing_genotype_rate": 0.0 if variants else "NA",
                "non_standard_genotype_count": 0,
            }
            for index, sample_id in enumerate(samples)
        ]
    if variant_metadata_rows is None:
        variant_metadata_rows = [
            {
                "cohort_id": cohort_id,
                "variant_index": index,
                "variant_key": variant_key,
                "chrom": variant_key.split(":")[0],
                "pos": int(variant_key.split(":")[1]),
                "ref": variant_key.split(":")[2],
                "alt": variant_key.split(":")[3],
                "qual": 100.0,
                "missing_genotype_count": 0,
                "missing_genotype_rate": 0.0,
            }
            for index, variant_key in enumerate(variants)
        ]

    sample_path = panel_dir / f"{cohort_id}{gs_panel.SAMPLE_METADATA_SUFFIX}"
    variant_path = panel_dir / f"{cohort_id}{gs_panel.VARIANT_METADATA_SUFFIX}"
    _write_tsv(sample_path, sample_metadata_rows, SAMPLE_METADATA_COLUMNS)
    _write_tsv(variant_path, variant_metadata_rows, VARIANT_METADATA_COLUMNS)

    manifest = {
        "schema_version": 1,
        "run_id": "20260814T074225Z-c22d6e72",
        "generated_at": "2026-08-14T07:42:25Z",
        "cohort_id": cohort_id,
        "pipeline_version": "0.2.0-dev",
        "git_commit": None,
        "parameters": {"sample_ploidy": 2},
        "genotype_encoding": {
            "schema": "diploid_additive_dosage_v1",
            "dosage_by_genotype": {"0/0": -1, "0/1_or_1/0": 0, "1/1": 1},
            "phasing": "ignored for dosage; 0|1 encodes identically to 0/1",
            "missing_token": "nan",
            "matrix_orientation": "variant_rows_by_sample_columns",
            "ploidy": "diploid_only",
        },
        "panel_status": "empty" if not variants else "populated",
        "checksums": {
            matrix_path.name: _sha256(matrix_path),
            sample_path.name: _sha256(sample_path),
            variant_path.name: _sha256(variant_path),
        },
    }
    if manifest_overrides:
        manifest = _deep_update(manifest, manifest_overrides)
    manifest_path = panel_dir / f"{cohort_id}{gs_panel.MANIFEST_SUFFIX}"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return panel_dir


SAMPLE_METADATA_COLUMNS = [
    "cohort_id",
    "sample_index",
    "sample_id",
    "missing_genotype_count",
    "missing_genotype_rate",
    "non_standard_genotype_count",
]
VARIANT_METADATA_COLUMNS = [
    "cohort_id",
    "variant_index",
    "variant_key",
    "chrom",
    "pos",
    "ref",
    "alt",
    "qual",
    "missing_genotype_count",
    "missing_genotype_rate",
]


def _write_tsv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """Write a metadata TSV, header included even with zero rows.

    The producer always writes the header (an empty panel still lists its
    columns), so the fixture must too.
    """
    header = list(rows[0]) if rows else columns
    lines = ["\t".join(header)]
    lines.extend("\t".join(str(row[column]) for column in header) for row in rows)
    path.write_text("\n".join(lines) + "\n")


def _deep_update(base: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


class LoadGsPanelTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.panel_dir = Path(self._directory.name) / "gs_panel"

    def test_matrix_is_transposed_to_sample_rows(self) -> None:
        _write_panel(self.panel_dir)
        panel = gs_panel.load_gs_panel(self.panel_dir)

        # On disk: variant rows x sample columns. In memory: the SoyNAM
        # convention, sample rows x variant columns.
        self.assertEqual(panel.genotypes.shape, (3, 2))
        self.assertEqual(list(panel.sample_ids), SAMPLES)
        self.assertEqual(list(panel.variant_keys), VARIANTS)
        np.testing.assert_array_equal(panel.genotypes[:, 0], np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_array_equal(panel.genotypes[1:, 1], np.array([1.0, -1.0]))
        self.assertTrue(np.isnan(panel.genotypes[0, 1]))
        self.assertEqual(panel.cohort_id, COHORT)
        self.assertFalse(panel.is_empty)
        self.assertEqual(panel.panel_status, "populated")

    def test_dosages_use_the_same_scale_as_the_soynam_loader(self) -> None:
        _write_panel(self.panel_dir)
        panel = gs_panel.load_gs_panel(self.panel_dir)

        observed = {value for value in panel.genotypes.ravel() if not np.isnan(value)}
        self.assertTrue(observed <= set(GENOTYPE_ENCODING.values()))
        self.assertEqual(gs_panel.EXPECTED_DOSAGES["0/0"], GENOTYPE_ENCODING["A/A"])
        self.assertEqual(
            gs_panel.EXPECTED_DOSAGES["0/1_or_1/0"], GENOTYPE_ENCODING["A/B"]
        )
        self.assertEqual(gs_panel.EXPECTED_DOSAGES["1/1"], GENOTYPE_ENCODING["B/B"])

    def test_empty_panel_keeps_the_sample_list(self) -> None:
        _write_panel(self.panel_dir, variants=[], rows=[])
        panel = gs_panel.load_gs_panel(self.panel_dir)

        # Zero GS-eligible variants is a normal outcome, not an error.
        self.assertEqual(panel.genotypes.shape, (3, 0))
        self.assertEqual(list(panel.sample_ids), SAMPLES)
        self.assertTrue(panel.is_empty)
        self.assertEqual(panel.panel_status, "empty")

    def test_zero_sample_header_is_an_error(self) -> None:
        _write_panel(self.panel_dir, samples=[], rows=[[], []])
        with self.assertRaisesRegex(ValueError, "no samples"):
            gs_panel.load_gs_panel(self.panel_dir)

    def test_cohort_id_is_derived_or_required(self) -> None:
        _write_panel(self.panel_dir, cohort_id="alpha")
        _write_panel(self.panel_dir, cohort_id="beta")

        with self.assertRaisesRegex(ValueError, "multiple cohorts"):
            gs_panel.load_gs_panel(self.panel_dir)

        panel = gs_panel.load_gs_panel(self.panel_dir, cohort_id="beta")
        self.assertEqual(panel.cohort_id, "beta")

        with self.assertRaises(FileNotFoundError):
            gs_panel.load_gs_panel(self.panel_dir, cohort_id="gamma")

    def test_missing_panel_directory_is_reported(self) -> None:
        (self.panel_dir).mkdir(parents=True)
        with self.assertRaisesRegex(FileNotFoundError, "manifest"):
            gs_panel.load_gs_panel(self.panel_dir)


class ManifestValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.panel_dir = Path(self._directory.name) / "gs_panel"

    def _load_with(self, overrides: dict) -> None:
        _write_panel(self.panel_dir, manifest_overrides=overrides)
        gs_panel.load_gs_panel(self.panel_dir)

    def test_unsupported_schema_version_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "schema_version"):
            self._load_with({"schema_version": 2})

    def test_unsupported_encoding_schema_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "encoding schema"):
            self._load_with({"genotype_encoding": {"schema": "tetraploid_dosage_v2"}})

    def test_unexpected_orientation_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "orientation"):
            self._load_with(
                {
                    "genotype_encoding": {
                        "matrix_orientation": "sample_rows_by_variant_columns"
                    }
                }
            )

    def test_unexpected_missing_token_and_ploidy_fail(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing token"):
            self._load_with({"genotype_encoding": {"missing_token": "NA"}})

        with self.assertRaisesRegex(ValueError, "ploidy"):
            self._load_with({"genotype_encoding": {"ploidy": "any"}})

    def test_unexpected_dosage_table_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "dosage table"):
            self._load_with(
                {
                    "genotype_encoding": {
                        "dosage_by_genotype": {"0/0": 0, "0/1_or_1/0": 1, "1/1": 2}
                    }
                }
            )

    def test_cohort_id_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "cohort_id"):
            self._load_with({"cohort_id": "other"})


class IntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = TemporaryDirectory()
        self.addCleanup(self._directory.cleanup)
        self.panel_dir = Path(self._directory.name) / "gs_panel"

    def test_checksum_mismatch_is_rejected(self) -> None:
        _write_panel(self.panel_dir)
        sample_path = self.panel_dir / f"{COHORT}{gs_panel.SAMPLE_METADATA_SUFFIX}"
        # Edit the file after the manifest recorded its checksum: one
        # sample renamed, which the manifest can no longer vouch for.
        sample_path.write_text(sample_path.read_text().replace("S3", "S9"))

        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            gs_panel.load_gs_panel(self.panel_dir)

        # The check can be waived explicitly, and then the edit shows up
        # as the metadata/matrix disagreement it actually is.
        with self.assertRaisesRegex(ValueError, "sample metadata"):
            gs_panel.load_gs_panel(self.panel_dir, verify_file_checksums=False)

    def test_out_of_contract_dosage_is_rejected(self) -> None:
        _write_panel(self.panel_dir, rows=[["-1", "0", "2"], ["nan", "1", "-1"]])
        with self.assertRaisesRegex(ValueError, "outside the contract"):
            gs_panel.load_gs_panel(self.panel_dir)

    def test_unparsable_dosage_is_rejected(self) -> None:
        _write_panel(self.panel_dir, rows=[["-1", "0", "NA"], ["nan", "1", "-1"]])
        with self.assertRaisesRegex(ValueError, "unparsable dosage"):
            gs_panel.load_gs_panel(self.panel_dir)

    def test_short_matrix_row_is_rejected(self) -> None:
        _write_panel(self.panel_dir, rows=[["-1", "0"], ["nan", "1", "-1"]])
        with self.assertRaisesRegex(ValueError, "dosage cells"):
            gs_panel.load_gs_panel(self.panel_dir)

    def test_duplicate_sample_ids_are_rejected(self) -> None:
        _write_panel(self.panel_dir, samples=["S1", "S1", "S3"])
        with self.assertRaisesRegex(ValueError, "duplicate sample IDs"):
            gs_panel.load_gs_panel(self.panel_dir)

    def test_metadata_order_mismatch_is_rejected(self) -> None:
        rows = [
            {
                "cohort_id": COHORT,
                "sample_index": index,
                "sample_id": sample_id,
                "missing_genotype_count": 0,
                "missing_genotype_rate": 0.0,
                "non_standard_genotype_count": 0,
            }
            for index, sample_id in enumerate(reversed(SAMPLES))
        ]
        _write_panel(self.panel_dir, sample_metadata_rows=rows)
        with self.assertRaisesRegex(ValueError, "sample metadata"):
            gs_panel.load_gs_panel(self.panel_dir)

    def test_metadata_missing_column_is_rejected(self) -> None:
        rows = [{"cohort_id": COHORT, "sample_id": sample_id} for sample_id in SAMPLES]
        _write_panel(self.panel_dir, sample_metadata_rows=rows)
        with self.assertRaisesRegex(ValueError, "sample_index"):
            gs_panel.load_gs_panel(self.panel_dir)


if __name__ == "__main__":
    unittest.main()
