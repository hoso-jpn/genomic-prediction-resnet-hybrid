import json
import os
import tempfile
import unittest
from datetime import UTC
from pathlib import Path
from typing import Any, ClassVar
from unittest import mock

import numpy as np
import pandas as pd

import resnet_baseline
import run_manifest


class RunIdAndTimestampTest(unittest.TestCase):
    def test_run_id_matches_expected_format(self) -> None:
        run_id = run_manifest.new_run_id()
        timestamp, _, suffix = run_id.partition("-")
        self.assertRegex(timestamp, r"^\d{8}T\d{6}Z$")
        self.assertRegex(suffix, r"^[0-9a-f]{8}$")

    def test_run_id_accepts_explicit_injection(self) -> None:
        from datetime import datetime

        moment = datetime(2026, 8, 12, 12, 34, 56, tzinfo=UTC)
        run_id = run_manifest.new_run_id(now=moment, suffix="a1b2c3d4")
        self.assertEqual(run_id, "20260812T123456Z-a1b2c3d4")

    def test_utc_now_iso_format(self) -> None:
        from datetime import datetime

        moment = datetime(2026, 8, 12, 12, 34, 56, tzinfo=UTC)
        self.assertEqual(run_manifest.utc_now_iso(moment), "2026-08-12T12:34:56Z")


class ChecksumTest(unittest.TestCase):
    def test_sha256_file_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.bin"
            path.write_bytes(b"deterministic content")
            self.assertEqual(
                run_manifest.sha256_file(path), run_manifest.sha256_file(path)
            )

    def test_sha256_file_changes_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample.bin"
            path.write_bytes(b"first")
            first = run_manifest.sha256_file(path)
            path.write_bytes(b"second")
            second = run_manifest.sha256_file(path)
            self.assertNotEqual(first, second)

    def test_source_file_checksums_keyed_by_filename_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "module.py"
            path.write_text("x = 1\n")
            checksums = run_manifest.source_file_checksums([path])
            self.assertEqual(list(checksums), ["module.py"])
            self.assertEqual(checksums["module.py"], run_manifest.sha256_file(path))

    def test_describe_input_files_uses_names_not_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory)
            phenotype_path = data_dir / "A_NAM01_phenotype_data.tsv.gz"
            genotype_path = data_dir / "A_NAM01_4312_SNP_genotype_Wm82.a1.tsv.gz"
            phenotype_path.write_bytes(b"phenotype")
            genotype_path.write_bytes(b"genotype")

            described = run_manifest.describe_input_files(
                [("A_NAM01", phenotype_path, genotype_path)]
            )

            self.assertEqual(len(described), 1)
            entry = described[0]
            self.assertEqual(entry["family_id"], "A_NAM01")
            self.assertEqual(entry["phenotype_file"], phenotype_path.name)
            self.assertEqual(entry["genotype_file"], genotype_path.name)
            self.assertEqual(
                entry["phenotype_sha256"], run_manifest.sha256_file(phenotype_path)
            )
            self.assertEqual(
                entry["genotype_sha256"], run_manifest.sha256_file(genotype_path)
            )
            serialized = json.dumps(entry)
            self.assertNotIn(str(data_dir), serialized)


class GitCommitTest(unittest.TestCase):
    def test_environment_variable_takes_priority(self) -> None:
        with (
            mock.patch.dict(os.environ, {"GIT_COMMIT_SHA": "deadbeef"}, clear=False),
            mock.patch("run_manifest.subprocess.run") as run,
        ):
            self.assertEqual(run_manifest.git_commit_sha(), "deadbeef")
            run.assert_not_called()

    def test_returns_none_when_git_is_unavailable(self) -> None:
        environment = dict(os.environ)
        environment.pop("GIT_COMMIT_SHA", None)
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("run_manifest.subprocess.run", side_effect=OSError("no git")),
        ):
            self.assertIsNone(run_manifest.git_commit_sha())

    def test_returns_none_when_git_command_fails(self) -> None:
        environment = dict(os.environ)
        environment.pop("GIT_COMMIT_SHA", None)
        completed = mock.Mock(returncode=128, stdout="")
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch("run_manifest.subprocess.run", return_value=completed),
        ):
            self.assertIsNone(run_manifest.git_commit_sha())


class LibraryVersionsTest(unittest.TestCase):
    def test_known_and_unknown_packages(self) -> None:
        versions = run_manifest.library_versions(["numpy", "not-a-real-package-xyz"])
        self.assertIsInstance(versions["numpy"], str)
        self.assertIsNone(versions["not-a-real-package-xyz"])


class SanitizeCommandTest(unittest.TestCase):
    def test_absolute_data_dir_with_space_separated_value_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            command = run_manifest.sanitize_command(
                "resnet_baseline.py", ["--data-dir", str(data_dir), "--seed", "42"]
            )
            self.assertEqual(
                command["arguments"], ["--data-dir", "data", "--seed", "42"]
            )
            serialized = json.dumps(command)
            self.assertNotIn(str(data_dir), serialized)
            self.assertNotIn(temporary_directory, serialized)

    def test_absolute_data_dir_with_equals_form_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            command = run_manifest.sanitize_command(
                "resnet_baseline.py", [f"--data-dir={data_dir}", "--seed", "42"]
            )
            self.assertEqual(command["arguments"], ["--data-dir=data", "--seed", "42"])
            serialized = json.dumps(command)
            self.assertNotIn(str(data_dir), serialized)
            self.assertNotIn(temporary_directory, serialized)

    def test_absolute_output_dir_with_space_separated_value_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "results"
            command = run_manifest.sanitize_command(
                "resnet_baseline.py", ["--output-dir", str(output_dir)]
            )
            self.assertEqual(command["arguments"], ["--output-dir", "results"])
            serialized = json.dumps(command)
            self.assertNotIn(str(output_dir), serialized)
            self.assertNotIn(temporary_directory, serialized)

    def test_absolute_output_dir_with_equals_form_is_redacted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "results"
            command = run_manifest.sanitize_command(
                "resnet_baseline.py", [f"--output-dir={output_dir}"]
            )
            self.assertEqual(command["arguments"], ["--output-dir=results"])
            serialized = json.dumps(command)
            self.assertNotIn(str(output_dir), serialized)
            self.assertNotIn(temporary_directory, serialized)

    def test_relative_dotted_data_dir_is_reduced_to_basename(self) -> None:
        command = run_manifest.sanitize_command(
            "resnet_baseline.py", ["--data-dir", "./data"]
        )
        self.assertEqual(command["arguments"], ["--data-dir", "data"])

    def test_executable_and_ordinary_options_are_preserved(self) -> None:
        command = run_manifest.sanitize_command(
            "resnet_baseline.py",
            ["--device", "cpu", "--seed", "42", "--max-epochs", "1"],
        )
        self.assertEqual(command["executable"], "resnet_baseline.py")
        self.assertEqual(
            command["arguments"],
            ["--device", "cpu", "--seed", "42", "--max-epochs", "1"],
        )

    def test_arguments_remain_a_list_not_a_shell_string(self) -> None:
        command = run_manifest.sanitize_command(
            "gblup_baseline.py", ["--data-dir", "data"]
        )
        self.assertIsInstance(command["arguments"], list)


def _iter_strings(value: object) -> Any:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


class MetadataPathLeakTest(unittest.TestCase):
    def test_metadata_never_contains_the_temporary_directory_absolute_path(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            data_dir = Path(temporary_directory) / "data"
            data_dir.mkdir()
            phenotype_path = data_dir / "A_NAM01_phenotype_data.tsv.gz"
            genotype_path = data_dir / "A_NAM01_4312_SNP_genotype_Wm82.a1.tsv.gz"
            phenotype_path.write_bytes(b"phenotype")
            genotype_path.write_bytes(b"genotype")
            source_path = data_dir / "module.py"
            source_path.write_text("x = 1\n")

            metadata = {
                "schema_version": run_manifest.SCHEMA_VERSION,
                "run_id": run_manifest.new_run_id(),
                "created_at": run_manifest.utc_now_iso(),
                "git_commit": run_manifest.git_commit_sha(),
                "source_file_checksums": run_manifest.source_file_checksums(
                    [source_path]
                ),
                "command": run_manifest.sanitize_command(
                    "resnet_baseline.py",
                    [
                        "--data-dir",
                        str(data_dir),
                        f"--output-dir={temporary_directory}/results",
                    ],
                ),
                "input_files": run_manifest.describe_input_files(
                    [("A_NAM01", phenotype_path, genotype_path)]
                ),
            }

            leaked = [
                value
                for value in _iter_strings(metadata)
                if temporary_directory in value
            ]
            self.assertEqual(leaked, [], leaked)


class OuterSplitTest(unittest.TestCase):
    sample_ids: ClassVar[list[str]] = ["RIL-1", "RIL-2", "RIL-3", "RIL-4"]
    family_ids: ClassVar[list[str]] = ["NAM01", "NAM01", "NAM02", "NAM03"]

    def test_same_input_yields_same_hash(self) -> None:
        first = run_manifest.build_outer_split(self.sample_ids, self.family_ids)
        second = run_manifest.build_outer_split(self.sample_ids, self.family_ids)
        self.assertEqual(first["outer_split_hash"], second["outer_split_hash"])

    def test_order_change_yields_different_hash(self) -> None:
        original = run_manifest.build_outer_split(self.sample_ids, self.family_ids)
        reordered_samples = list(reversed(self.sample_ids))
        reordered_families = list(reversed(self.family_ids))
        reordered = run_manifest.build_outer_split(
            reordered_samples, reordered_families
        )
        self.assertNotEqual(original["outer_split_hash"], reordered["outer_split_hash"])

    def test_hash_is_unaffected_by_global_random_state(self) -> None:
        np.random.seed(1)
        first = run_manifest.build_outer_split(self.sample_ids, self.family_ids)
        np.random.seed(999)
        second = run_manifest.build_outer_split(self.sample_ids, self.family_ids)
        self.assertEqual(first["outer_split_hash"], second["outer_split_hash"])

    def test_train_and_test_families_never_overlap(self) -> None:
        split = run_manifest.build_outer_split(self.sample_ids, self.family_ids)
        for fold in split["folds"]:
            self.assertFalse(
                set(fold["train_family_ids"]) & set(fold["test_family_ids"])
            )

    def test_each_family_is_held_out_exactly_once(self) -> None:
        split = run_manifest.build_outer_split(self.sample_ids, self.family_ids)
        held_out = [fold["held_out_family"] for fold in split["folds"]]
        self.assertEqual(sorted(held_out), sorted(set(self.family_ids)))
        self.assertEqual(len(held_out), len(set(held_out)))

    def test_ordered_samples_preserve_input_order(self) -> None:
        split = run_manifest.build_outer_split(self.sample_ids, self.family_ids)
        recovered = [entry["sample_id"] for entry in split["ordered_samples"]]
        self.assertEqual(recovered, self.sample_ids)

    def test_rejects_mismatched_lengths(self) -> None:
        with self.assertRaises(ValueError):
            run_manifest.build_outer_split(["only-one"], self.family_ids)

    def test_rejects_fewer_than_two_families(self) -> None:
        with self.assertRaises(ValueError):
            run_manifest.build_outer_split(["RIL-1", "RIL-2"], ["NAM01", "NAM01"])


class ResnetInnerSplitTest(unittest.TestCase):
    def test_same_seed_gives_same_validation_assignment(self) -> None:
        families = np.asarray(["A", "B", "C"])
        first = resnet_baseline.select_validation_family(families, 0, 42)
        second = resnet_baseline.select_validation_family(families, 0, 42)
        self.assertEqual(first, second)

    def test_different_fold_index_can_change_assignment(self) -> None:
        families = np.asarray(["A", "B", "C"])
        first = resnet_baseline.select_validation_family(families, 0, 42)
        second = resnet_baseline.select_validation_family(families, 1, 42)
        self.assertNotEqual(first, second)

    def test_outer_split_hash_is_independent_of_inner_seed(self) -> None:
        sample_ids = ["RIL-1", "RIL-2", "RIL-3", "RIL-4"]
        family_ids = ["NAM01", "NAM01", "NAM02", "NAM03"]
        outer_low_seed = run_manifest.build_outer_split(sample_ids, family_ids)
        # The inner validation-family seed used elsewhere never participates in
        # building the outer split, so varying it must never move this hash.
        resnet_baseline.select_validation_family(np.asarray(family_ids), 0, seed=1)
        outer_high_seed = run_manifest.build_outer_split(sample_ids, family_ids)
        resnet_baseline.select_validation_family(np.asarray(family_ids), 0, seed=999)
        self.assertEqual(
            outer_low_seed["outer_split_hash"], outer_high_seed["outer_split_hash"]
        )


class JsonAndNpzRoundTripTest(unittest.TestCase):
    def test_json_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "doc.json"
            payload = {"schema_version": 1, "values": [1, 2, 3], "nested": {"a": True}}
            run_manifest.write_json(path, payload)
            self.assertEqual(json.loads(path.read_text()), payload)

    def test_json_write_accepts_numpy_scalars_and_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "doc.json"
            payload = {
                "count": np.int64(3),
                "value": np.float64(1.5),
                "flag": np.bool_(True),
                "array": np.asarray([1.0, 2.0]),
            }
            run_manifest.write_json(path, payload)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded["count"], 3)
            self.assertEqual(loaded["value"], 1.5)
            self.assertEqual(loaded["flag"], True)
            self.assertEqual(loaded["array"], [1.0, 2.0])

    def test_npz_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "arrays.npz"
            arrays = {
                "mask": np.asarray([True, False, True]),
                "means": np.asarray([0.1, 0.2, 0.3]),
            }
            run_manifest.write_npz(path, arrays)
            with np.load(path) as loaded:
                np.testing.assert_array_equal(loaded["mask"], arrays["mask"])
                np.testing.assert_allclose(loaded["means"], arrays["means"])


class WriteRunArtifactsTest(unittest.TestCase):
    def _predictions(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "family_id": ["A", "B"],
                "sample_name": ["s1", "s2"],
                "observed_yield_kg_ha": [1.0, 2.0],
                "predicted_yield_kg_ha": [1.1, 2.1],
            }
        )

    def _write(self, output_dir: Path, run_id: str) -> Path:
        return run_manifest.write_run_artifacts(
            output_dir=output_dir,
            run_id=run_id,
            metadata={"schema_version": 1, "run_id": run_id},
            split={"schema_version": 1},
            preprocessing={"schema_version": 1},
            preprocessing_arrays={"mask": np.asarray([True, False])},
            metrics={"schema_version": 1},
            predictions=self._predictions(),
        )

    def test_writes_all_six_files_and_compat_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            run_dir = self._write(output_dir, "20260812T000000Z-aaaaaaaa")

            for name in (
                "metadata.json",
                "split.json",
                "preprocessing.json",
                "preprocessing_arrays.npz",
                "metrics.json",
                "predictions.csv",
            ):
                self.assertTrue((run_dir / name).is_file(), name)

            self.assertTrue((output_dir / "oof_predictions.csv").is_file())
            compat = pd.read_csv(output_dir / "oof_predictions.csv")
            self.assertEqual(
                compat.columns.tolist(),
                [
                    "family_id",
                    "sample_name",
                    "observed_yield_kg_ha",
                    "predicted_yield_kg_ha",
                ],
            )

    def test_rejects_reuse_of_the_same_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            self._write(output_dir, "20260812T000000Z-bbbbbbbb")
            with self.assertRaises(FileExistsError):
                self._write(output_dir, "20260812T000000Z-bbbbbbbb")

    def test_failed_write_leaves_no_final_or_temp_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            run_id = "20260812T000000Z-cccccccc"
            with (
                mock.patch(
                    "run_manifest.write_npz", side_effect=RuntimeError("disk full")
                ),
                self.assertRaises(RuntimeError),
            ):
                self._write(output_dir, run_id)

            artifacts_dir = output_dir / "artifacts"
            self.assertFalse((artifacts_dir / run_id).exists())
            self.assertFalse((artifacts_dir / f".tmp-{run_id}").exists())
            self.assertFalse((output_dir / "oof_predictions.csv").exists())

            # A subsequent attempt with the same run_id must still succeed.
            run_dir = self._write(output_dir, run_id)
            self.assertTrue(run_dir.is_dir())

    def test_second_run_atomically_replaces_the_compat_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            first_predictions = pd.DataFrame(
                {
                    "family_id": ["A"],
                    "sample_name": ["s1"],
                    "observed_yield_kg_ha": [1.0],
                    "predicted_yield_kg_ha": [1.1],
                }
            )
            second_predictions = pd.DataFrame(
                {
                    "family_id": ["B"],
                    "sample_name": ["s2"],
                    "observed_yield_kg_ha": [2.0],
                    "predicted_yield_kg_ha": [2.2],
                }
            )

            run_manifest.write_run_artifacts(
                output_dir=output_dir,
                run_id="20260812T000000Z-dddddddd",
                metadata={"schema_version": 1},
                split={"schema_version": 1},
                preprocessing={"schema_version": 1},
                preprocessing_arrays={},
                metrics={"schema_version": 1},
                predictions=first_predictions,
            )
            run_manifest.write_run_artifacts(
                output_dir=output_dir,
                run_id="20260812T000000Z-eeeeeeee",
                metadata={"schema_version": 1},
                split={"schema_version": 1},
                preprocessing={"schema_version": 1},
                preprocessing_arrays={},
                metrics={"schema_version": 1},
                predictions=second_predictions,
            )

            compat = pd.read_csv(output_dir / "oof_predictions.csv")
            self.assertEqual(compat["sample_name"].tolist(), ["s2"])
            self.assertEqual(list(output_dir.glob(".oof_predictions.csv.tmp")), [])


if __name__ == "__main__":
    unittest.main()
