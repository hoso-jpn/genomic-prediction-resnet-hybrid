import csv
import gzip
import tempfile
import unittest
from pathlib import Path

import numpy as np

import soynam_data


class SoynamDataTest(unittest.TestCase):
    def _make_data_dir(self) -> Path:
        return Path(self.enterContext(tempfile.TemporaryDirectory()))

    @staticmethod
    def _write_gzip_tsv(path: Path, rows: list[list[str]]) -> None:
        with gzip.open(path, mode="wt", newline="") as handle:
            csv.writer(handle, delimiter="\t").writerows(rows)

    @staticmethod
    def _phenotype_path(data_dir: Path, family_id: str) -> Path:
        return data_dir / f"{family_id}{soynam_data.PHENOTYPE_SUFFIX}"

    @staticmethod
    def _genotype_path(data_dir: Path, family_id: str) -> Path:
        return data_dir / f"{family_id}_4312{soynam_data.GENOTYPE_SUFFIX}"

    def _write_phenotype_file(
        self, data_dir: Path, family_id: str, samples: list[tuple[str, str]]
    ) -> Path:
        path = self._phenotype_path(data_dir, family_id)
        rows = [
            [soynam_data.SAMPLE_COLUMN, soynam_data.PHENOTYPE_COLUMN],
            *[[sample_id, value] for sample_id, value in samples],
        ]
        self._write_gzip_tsv(path, rows)
        return path

    def _write_genotype_file(
        self,
        data_dir: Path,
        family_id: str,
        header: list[str],
        marker_rows: list[list[str]],
    ) -> Path:
        path = self._genotype_path(data_dir, family_id)
        self._write_gzip_tsv(path, [header, *marker_rows])
        return path

    def _write_standard_family(
        self,
        data_dir: Path,
        family_id: str,
        ril_samples: list[str],
        marker_ids: list[str],
    ) -> None:
        parent = family_id.split("_NAM", maxsplit=1)[0]
        self._write_phenotype_file(
            data_dir,
            family_id,
            [(parent, "400.0"), *[(sample, "500.0") for sample in ril_samples]],
        )
        header = ["marker", parent, *ril_samples]
        marker_rows = [
            [marker_id, *["A"] * (len(header) - 1)] for marker_id in marker_ids
        ]
        self._write_genotype_file(data_dir, family_id, header, marker_rows)

    def test_raw_loader_excludes_parent_and_preserves_missing(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir,
            family_id,
            [("Parent-A", "400.0"), ("RIL-1", "500.0"), ("RIL-2", "600.0")],
        )
        self._write_genotype_file(
            data_dir,
            family_id,
            ["marker", "Parent-A", "RIL-1", "RIL-2"],
            [["m1", "A", "-", "B"], ["m2", "B", "A/B", "A"]],
        )

        dataset = soynam_data.load_soynam_dataset(data_dir)

        np.testing.assert_array_equal(
            dataset.sample_names, np.array(["RIL-1", "RIL-2"])
        )
        np.testing.assert_allclose(dataset.phenotypes, [500.0, 600.0])
        self.assertTrue(np.isnan(dataset.genotypes[0, 0]))
        np.testing.assert_allclose(dataset.genotypes[0, 1], 0.0)
        np.testing.assert_allclose(dataset.genotypes[1], [1.0, -1.0])

    def test_pairing_mismatch_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        self._write_phenotype_file(data_dir, "A_NAM01", [("RIL-1", "500.0")])
        with self.assertRaisesRegex(ValueError, "pairing is invalid"):
            soynam_data.load_soynam_dataset(data_dir)

    def test_load_with_fixed_family_files_ignores_later_directory_changes(
        self,
    ) -> None:
        data_dir = self._make_data_dir()
        self._write_standard_family(
            data_dir, "Parent-A_NAM01", ["RIL-1", "RIL-2"], ["m1", "m2"]
        )

        fixed_family_files = soynam_data.list_family_files(data_dir)

        # A family added after the file list was fixed must not appear when
        # that fixed list is passed back in: the caller's earlier resolution
        # (and any checksums computed from it) stays authoritative.
        self._write_standard_family(data_dir, "Parent-B_NAM02", ["RIL-3"], ["m1", "m2"])

        dataset = soynam_data.load_soynam_dataset(
            data_dir, family_files=fixed_family_files
        )
        self.assertEqual(dataset.sample_names.tolist(), ["RIL-1", "RIL-2"])
        self.assertEqual(sorted(set(dataset.family_ids.tolist())), ["Parent-A_NAM01"])

        # Without a fixed list, the loader re-resolves the directory and
        # picks up the family that was added afterward.
        rescanned = soynam_data.load_soynam_dataset(data_dir)
        self.assertEqual(
            sorted(set(rescanned.family_ids.tolist())),
            ["Parent-A_NAM01", "Parent-B_NAM02"],
        )

    def test_loads_multiple_families(self) -> None:
        data_dir = self._make_data_dir()
        self._write_standard_family(
            data_dir, "Parent-A_NAM01", ["RIL-1", "RIL-2"], ["m1", "m2"]
        )
        self._write_standard_family(data_dir, "Parent-B_NAM02", ["RIL-3"], ["m1", "m2"])

        dataset = soynam_data.load_soynam_dataset(data_dir)

        self.assertEqual(dataset.sample_names.tolist(), ["RIL-1", "RIL-2", "RIL-3"])
        self.assertEqual(
            dataset.family_ids.tolist(),
            ["Parent-A_NAM01", "Parent-A_NAM01", "Parent-B_NAM02"],
        )
        self.assertEqual(dataset.marker_names.tolist(), ["m1", "m2"])
        self.assertEqual(dataset.genotypes.shape, (3, 2))
        np.testing.assert_allclose(dataset.phenotypes, [500.0, 500.0, 500.0])

    def test_phenotype_duplicate_sample_id_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir,
            family_id,
            [("Parent-A", "400.0"), ("RIL-1", "500.0"), ("RIL-1", "550.0")],
        )
        self._write_genotype_file(
            data_dir, family_id, ["marker", "Parent-A", "RIL-1"], [["m1", "A", "B"]]
        )

        with self.assertRaisesRegex(
            ValueError, "duplicate phenotype sample ID"
        ) as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(self._phenotype_path(data_dir, family_id).name, message)
        self.assertIn("RIL-1", message)

    def test_phenotype_missing_sample_id_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir, family_id, [("Parent-A", "400.0"), ("", "500.0")]
        )
        self._write_genotype_file(
            data_dir, family_id, ["marker", "Parent-A", "RIL-1"], [["m1", "A", "B"]]
        )

        with self.assertRaisesRegex(
            ValueError, "missing or empty phenotype sample ID"
        ) as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(self._phenotype_path(data_dir, family_id).name, message)

    def test_phenotype_empty_sample_id_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir, family_id, [("Parent-A", "400.0"), ("   ", "500.0")]
        )
        self._write_genotype_file(
            data_dir, family_id, ["marker", "Parent-A", "RIL-1"], [["m1", "A", "B"]]
        )

        with self.assertRaisesRegex(
            ValueError, "missing or empty phenotype sample ID"
        ) as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(self._phenotype_path(data_dir, family_id).name, message)

    def test_genotype_duplicate_sample_header_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir,
            family_id,
            [("Parent-A", "400.0"), ("RIL-1", "500.0")],
        )
        self._write_genotype_file(
            data_dir,
            family_id,
            ["marker", "Parent-A", "RIL-1", "RIL-1"],
            [["m1", "A", "B", "B"]],
        )

        with self.assertRaisesRegex(
            ValueError, "duplicate genotype sample ID"
        ) as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(self._genotype_path(data_dir, family_id).name, message)
        self.assertIn("RIL-1", message)

    def test_genotype_empty_sample_header_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir,
            family_id,
            [("Parent-A", "400.0"), ("RIL-2", "500.0")],
        )
        self._write_genotype_file(
            data_dir,
            family_id,
            ["marker", "Parent-A", "", "RIL-2"],
            [["m1", "A", "B", "A"]],
        )

        with self.assertRaisesRegex(
            ValueError, "missing or empty genotype sample ID"
        ) as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(self._genotype_path(data_dir, family_id).name, message)
        self.assertIn("header sample position(s)", message)

    def test_genotype_header_without_sample_columns_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir, family_id, [("Parent-A", "400.0"), ("RIL-1", "500.0")]
        )
        self._write_genotype_file(data_dir, family_id, ["marker"], [["m1"], ["m2"]])

        with self.assertRaisesRegex(
            ValueError, "genotype file has no sample ID columns"
        ) as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(self._genotype_path(data_dir, family_id).name, message)

    def test_marker_id_duplicate_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir, family_id, [("Parent-A", "400.0"), ("RIL-1", "500.0")]
        )
        self._write_genotype_file(
            data_dir,
            family_id,
            ["marker", "Parent-A", "RIL-1"],
            [["m1", "A", "B"], ["m1", "A", "B"]],
        )

        with self.assertRaisesRegex(ValueError, "duplicate marker ID") as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(self._genotype_path(data_dir, family_id).name, message)
        self.assertIn("m1", message)

    def test_marker_id_missing_or_empty_is_rejected(self) -> None:
        missing_data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            missing_data_dir, family_id, [("Parent-A", "400.0"), ("RIL-1", "500.0")]
        )
        self._write_genotype_file(
            missing_data_dir,
            family_id,
            ["marker", "Parent-A", "RIL-1"],
            [["", "A", "B"], ["m2", "B", "A"]],
        )
        with self.assertRaisesRegex(ValueError, "missing or empty marker ID"):
            soynam_data.load_soynam_dataset(missing_data_dir)

        empty_data_dir = self._make_data_dir()
        self._write_phenotype_file(
            empty_data_dir, family_id, [("Parent-A", "400.0"), ("RIL-1", "500.0")]
        )
        self._write_genotype_file(
            empty_data_dir,
            family_id,
            ["marker", "Parent-A", "RIL-1"],
            [["  ", "A", "B"], ["m2", "B", "A"]],
        )
        with self.assertRaisesRegex(ValueError, "missing or empty marker ID"):
            soynam_data.load_soynam_dataset(empty_data_dir)

    def test_phenotype_only_sample_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir,
            family_id,
            [("Parent-A", "400.0"), ("RIL-1", "500.0"), ("RIL-2", "600.0")],
        )
        self._write_genotype_file(
            data_dir,
            family_id,
            ["marker", "Parent-A", "RIL-1"],
            [["m1", "A", "B"]],
        )

        with self.assertRaisesRegex(ValueError, "RIL sample sets differ") as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn("phenotype_only=['RIL-2']", message)
        self.assertIn("genotype_only=[]", message)

    def test_genotype_only_sample_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir, family_id, [("Parent-A", "400.0"), ("RIL-1", "500.0")]
        )
        self._write_genotype_file(
            data_dir,
            family_id,
            ["marker", "Parent-A", "RIL-1", "RIL-3"],
            [["m1", "A", "B", "A"]],
        )

        with self.assertRaisesRegex(ValueError, "RIL sample sets differ") as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn("phenotype_only=[]", message)
        self.assertIn("genotype_only=['RIL-3']", message)

    def test_founder_parent_only_on_one_side_is_ignored(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir, family_id, [("Parent-A", "400.0"), ("RIL-1", "500.0")]
        )
        self._write_genotype_file(
            data_dir,
            family_id,
            ["marker", "RIL-1"],
            [["m1", "A"], ["m2", "B"]],
        )

        dataset = soynam_data.load_soynam_dataset(data_dir)

        self.assertEqual(dataset.sample_names.tolist(), ["RIL-1"])
        np.testing.assert_allclose(dataset.phenotypes, [500.0])
        self.assertEqual(dataset.genotypes.shape, (1, 2))

    def test_marker_set_mismatch_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        self._write_standard_family(data_dir, "Parent-A_NAM01", ["RIL-1"], ["m1", "m2"])
        self._write_standard_family(data_dir, "Parent-B_NAM02", ["RIL-2"], ["m1", "m3"])

        with self.assertRaisesRegex(ValueError, "marker set differs") as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn("Parent-B_NAM02", message)
        self.assertIn(self._genotype_path(data_dir, "Parent-B_NAM02").name, message)
        self.assertIn("reference_only=['m2']", message)
        self.assertIn("current_only=['m3']", message)

    def test_marker_order_mismatch_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        self._write_standard_family(data_dir, "Parent-A_NAM01", ["RIL-1"], ["m1", "m2"])
        self._write_standard_family(data_dir, "Parent-B_NAM02", ["RIL-2"], ["m2", "m1"])

        with self.assertRaisesRegex(ValueError, "marker order differs") as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn("Parent-B_NAM02", message)
        self.assertIn(self._genotype_path(data_dir, "Parent-B_NAM02").name, message)

    def test_missing_phenotype_excluded_after_matching(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir,
            family_id,
            [("Parent-A", "400.0"), ("RIL-1", ""), ("RIL-2", "500.0")],
        )
        self._write_genotype_file(
            data_dir,
            family_id,
            ["marker", "Parent-A", "RIL-1", "RIL-2"],
            [["m1", "A", "A", "B"]],
        )

        dataset = soynam_data.load_soynam_dataset(data_dir)

        self.assertEqual(dataset.sample_names.tolist(), ["RIL-2"])
        np.testing.assert_allclose(dataset.phenotypes, [500.0])
        self.assertEqual(dataset.genotypes.shape, (1, 1))

    def test_non_numeric_phenotype_value_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir, family_id, [("Parent-A", "400.0"), ("RIL-1", "abc")]
        )
        self._write_genotype_file(
            data_dir, family_id, ["marker", "Parent-A", "RIL-1"], [["m1", "A", "B"]]
        )

        with self.assertRaisesRegex(
            ValueError, "non-numeric phenotype value"
        ) as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(self._phenotype_path(data_dir, family_id).name, message)
        self.assertIn("RIL-1", message)

    def test_missing_phenotype_column_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_gzip_tsv(
            self._phenotype_path(data_dir, family_id),
            [[soynam_data.SAMPLE_COLUMN], ["Parent-A"], ["RIL-1"]],
        )
        self._write_genotype_file(
            data_dir, family_id, ["marker", "Parent-A", "RIL-1"], [["m1", "A", "B"]]
        )

        with self.assertRaisesRegex(ValueError, "missing phenotype columns") as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(soynam_data.PHENOTYPE_COLUMN, message)
        self.assertIn(self._phenotype_path(data_dir, family_id).name, message)

    def test_genotype_family_id_collision_is_rejected(self) -> None:
        data_dir = self._make_data_dir()
        family_id = "Parent-A_NAM01"
        self._write_phenotype_file(
            data_dir, family_id, [("Parent-A", "400.0"), ("RIL-1", "500.0")]
        )
        genotype_with_suffix = self._genotype_path(data_dir, family_id)
        genotype_without_suffix = data_dir / f"{family_id}{soynam_data.GENOTYPE_SUFFIX}"
        rows = [["marker", "Parent-A", "RIL-1"], ["m1", "A", "B"]]
        self._write_gzip_tsv(genotype_with_suffix, rows)
        self._write_gzip_tsv(genotype_without_suffix, rows)

        with self.assertRaisesRegex(
            ValueError, "multiple genotype files map to family"
        ) as context:
            soynam_data.load_soynam_dataset(data_dir)

        message = str(context.exception)
        self.assertIn(family_id, message)
        self.assertIn(genotype_with_suffix.name, message)
        self.assertIn(genotype_without_suffix.name, message)


if __name__ == "__main__":
    unittest.main()
