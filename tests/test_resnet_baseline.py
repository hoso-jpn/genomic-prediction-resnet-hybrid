import os
import subprocess
import unittest
from unittest import mock

import numpy as np
import torch

from resnet_baseline import (
    ResNetConfig,
    _cuda_driver_api_version,
    _device_environment_info,
    _nvidia_driver_version,
    build_fold_preprocessing_entry,
    build_transform_record,
    fit_feature_transform,
    make_oof_frame,
    predict_resnet_fold,
    select_validation_family,
    transform_features,
)
from soynam_data import SoynamDataset


class _FakeDeviceProperties:
    """Stand-in for torch.cuda.get_device_properties on a CPU-only machine."""

    name = "Fake GPU"
    major = 8
    minor = 6


class ResNetBaselineTest(unittest.TestCase):
    def test_feature_statistics_are_fitted_on_training_rows_only(self) -> None:
        train = np.asarray(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [1.0, 0.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, -1.0],
                [1.0, np.nan, 1.0, -1.0],
            ]
        )
        test = np.asarray([[100.0, np.nan, -100.0, 100.0]])
        config = ResNetConfig(
            pca_components=2,
            min_observed_rate=0.5,
            maf_threshold=0.0,
        )
        transform = fit_feature_transform(train, config, seed=7)
        transformed, principal_components = transform_features(test, transform)

        np.testing.assert_array_equal(
            transform.retained_markers,
            np.asarray([True, True, True, False]),
        )
        np.testing.assert_allclose(transform.marker_means, [0.0, 0.0, 0.0])
        self.assertTrue(np.isfinite(transformed).all())
        self.assertTrue(np.isfinite(principal_components).all())

    def test_validation_family_selection_is_deterministic(self) -> None:
        families = np.asarray(["B", "A", "C", "A"])
        self.assertEqual(select_validation_family(families, 0, 42), "A")
        self.assertEqual(select_validation_family(families, 1, 42), "B")

    def test_outer_test_phenotypes_do_not_affect_predictions(self) -> None:
        family_ids = np.repeat(np.asarray(["A", "B", "C"]), 4)
        base = np.asarray(
            [
                [-1, -1, -1, -1, 1, 1, 1, 1],
                [-1, 0, -1, 0, 1, 0, 1, 0],
                [1, 0, 1, 0, -1, 0, -1, 0],
                [1, 1, 1, 1, -1, -1, -1, -1],
            ],
            dtype=np.float64,
        )
        genotypes = np.tile(base, (3, 1))
        phenotypes = np.asarray([1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6], dtype=float)
        test_indices = np.flatnonzero(family_ids == "A")
        train_indices = np.flatnonzero(family_ids != "A")
        config = ResNetConfig(
            hidden_dim=4,
            num_blocks=1,
            dropout_rate=0.0,
            pca_components=2,
            batch_size=4,
            max_epochs=1,
            patience=1,
            min_observed_rate=1.0,
            maf_threshold=0.0,
        )
        first, _ = predict_resnet_fold(
            genotypes,
            phenotypes,
            family_ids,
            train_indices,
            test_indices,
            0,
            config,
            torch.device("cpu"),
        )
        changed = phenotypes.copy()
        changed[test_indices] += 10_000.0
        second, _ = predict_resnet_fold(
            genotypes,
            changed,
            family_ids,
            train_indices,
            test_indices,
            0,
            config,
            torch.device("cpu"),
        )
        np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)

    def test_oof_columns_match_gblup_contract(self) -> None:
        dataset = SoynamDataset(
            genotypes=np.zeros((2, 1)),
            phenotypes=np.asarray([1.0, 2.0]),
            family_ids=np.asarray(["A", "B"]),
            sample_names=np.asarray(["a1", "b1"]),
            marker_names=np.asarray(["m1"]),
        )
        frame = make_oof_frame(dataset, np.asarray([1.5, 2.5]))
        self.assertEqual(
            frame.columns.tolist(),
            [
                "family_id",
                "sample_name",
                "observed_yield_kg_ha",
                "predicted_yield_kg_ha",
            ],
        )

    def test_build_transform_record_matches_fitted_transform(self) -> None:
        train = np.asarray(
            [
                [-1.0, -1.0, -1.0, -1.0],
                [1.0, 0.0, 1.0, -1.0],
                [-1.0, 1.0, -1.0, -1.0],
                [1.0, np.nan, 1.0, -1.0],
            ]
        )
        config = ResNetConfig(
            pca_components=2, min_observed_rate=0.5, maf_threshold=0.0
        )
        transform = fit_feature_transform(train, config, seed=7)

        record, arrays = build_transform_record(
            "fold_000_final", transform, target_mean=3300.0, target_scale=125.0
        )

        self.assertEqual(
            record["input_feature_count"], int(transform.retained_markers.size)
        )
        self.assertEqual(
            record["retained_marker_count"], int(transform.retained_markers.sum())
        )
        self.assertEqual(record["output_feature_count"], transform.pca.n_components_)
        self.assertEqual(record["target_mean"], 3300.0)
        self.assertEqual(record["target_scale"], 125.0)
        np.testing.assert_array_equal(
            arrays["fold_000_final_marker_mask"], transform.retained_markers
        )
        np.testing.assert_allclose(
            arrays["fold_000_final_imputation_mean"], transform.marker_means
        )
        np.testing.assert_allclose(
            arrays["fold_000_final_standardization_mean"], transform.marker_means
        )
        np.testing.assert_allclose(
            arrays["fold_000_final_standardization_scale"], transform.marker_scales
        )
        np.testing.assert_allclose(
            arrays["fold_000_final_pca_mean"], transform.pca.mean_
        )
        np.testing.assert_allclose(
            arrays["fold_000_final_pca_components"], transform.pca.components_
        )
        np.testing.assert_allclose(
            arrays["fold_000_final_pca_explained_variance_ratio"],
            transform.pca.explained_variance_ratio_,
        )

    def test_fold_preprocessing_entry_distinguishes_selection_and_final_transforms(
        self,
    ) -> None:
        family_ids = np.asarray(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
        genotypes = np.asarray(
            [[0.0, 0.0]] * 4
            + [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]
            + [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]],
            dtype=np.float64,
        )
        phenotypes = np.arange(12, dtype=np.float64)
        test_indices = np.flatnonzero(family_ids == "A")
        train_indices = np.flatnonzero(family_ids != "A")
        config = ResNetConfig(
            hidden_dim=4,
            num_blocks=1,
            dropout_rate=0.0,
            pca_components=1,
            batch_size=4,
            max_epochs=1,
            patience=1,
            min_observed_rate=1.0,
            maf_threshold=0.0,
            seed=42,
        )

        _, fold_record = predict_resnet_fold(
            genotypes,
            phenotypes,
            family_ids,
            train_indices,
            test_indices,
            0,
            config,
            torch.device("cpu"),
        )

        # "B" and "C" hold distinct genotype means, so excluding the
        # validation family from the selection-stage fit must yield a
        # different imputation mean than the full outer-training refit.
        self.assertNotEqual(fold_record.validation_family, fold_record.held_out_family)
        self.assertFalse(
            np.array_equal(
                fold_record.selection_transform.marker_means,
                fold_record.final_transform.marker_means,
            )
        )

        entry, arrays = build_fold_preprocessing_entry(fold_record)
        selection_refs = set(entry["selection_transform"]["arrays"].values())
        final_refs = set(entry["final_transform"]["arrays"].values())
        self.assertTrue(selection_refs.isdisjoint(final_refs))
        self.assertEqual(set(arrays), selection_refs | final_refs)

        np.testing.assert_allclose(
            arrays[entry["selection_transform"]["arrays"]["imputation_mean_ref"]],
            fold_record.selection_transform.marker_means,
        )
        np.testing.assert_allclose(
            arrays[entry["final_transform"]["arrays"]["imputation_mean_ref"]],
            fold_record.final_transform.marker_means,
        )

    def test_fold_record_captures_target_standardization_per_stage(self) -> None:
        family_ids = np.asarray(["A"] * 4 + ["B"] * 4 + ["C"] * 4)
        genotypes = np.asarray(
            [[0.0, 0.0]] * 4
            + [[-1.0, -1.0], [-1.0, -1.0], [-1.0, -1.0], [1.0, 1.0]]
            + [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]],
            dtype=np.float64,
        )
        phenotypes = np.arange(12, dtype=np.float64)
        test_indices = np.flatnonzero(family_ids == "A")
        train_indices = np.flatnonzero(family_ids != "A")
        config = ResNetConfig(
            hidden_dim=4,
            num_blocks=1,
            dropout_rate=0.0,
            pca_components=1,
            batch_size=4,
            max_epochs=1,
            patience=1,
            min_observed_rate=1.0,
            maf_threshold=0.0,
            seed=42,
        )

        _, fold_record = predict_resnet_fold(
            genotypes,
            phenotypes,
            family_ids,
            train_indices,
            test_indices,
            0,
            config,
            torch.device("cpu"),
        )

        # Independently reconstruct which samples fed each stage's target
        # standardization, so the assertions below check real agreement
        # with the computation, not a value copied from the implementation.
        validation_mask = family_ids[train_indices] == fold_record.validation_family
        fit_indices = train_indices[~validation_mask]
        expected_selection_mean = float(np.mean(phenotypes[fit_indices]))
        expected_selection_scale = float(np.std(phenotypes[fit_indices]))
        expected_final_mean = float(np.mean(phenotypes[train_indices]))
        expected_final_scale = float(np.std(phenotypes[train_indices]))

        self.assertAlmostEqual(
            fold_record.selection_target_mean, expected_selection_mean
        )
        self.assertAlmostEqual(
            fold_record.selection_target_scale, expected_selection_scale
        )
        self.assertAlmostEqual(fold_record.final_target_mean, expected_final_mean)
        self.assertAlmostEqual(fold_record.final_target_scale, expected_final_scale)
        self.assertNotEqual(
            fold_record.selection_target_mean, fold_record.final_target_mean
        )

        entry, _ = build_fold_preprocessing_entry(fold_record)
        self.assertEqual(
            entry["selection_transform"]["target_mean"],
            fold_record.selection_target_mean,
        )
        self.assertEqual(
            entry["selection_transform"]["target_scale"],
            fold_record.selection_target_scale,
        )
        self.assertEqual(
            entry["final_transform"]["target_mean"], fold_record.final_target_mean
        )
        self.assertEqual(
            entry["final_transform"]["target_scale"], fold_record.final_target_scale
        )

    def test_device_environment_info_for_cpu(self) -> None:
        info = _device_environment_info("cpu", torch.device("cpu"))
        self.assertEqual(info["device_requested"], "cpu")
        self.assertEqual(info["device_resolved"], "cpu")
        self.assertIsNone(info["cuda_version"])
        self.assertIsNone(info["cudnn_version"])
        # GPU-only fields stay absent rather than being filled in with
        # values borrowed from a device that did not run this job.
        self.assertIsNone(info["gpu_name"])
        self.assertIsNone(info["gpu_compute_capability"])
        self.assertIsNone(info["nvidia_driver_version"])

    def test_device_environment_info_reads_the_environment_label(self) -> None:
        with mock.patch.dict(os.environ, {"GPRH_ENVIRONMENT": "cuda-12.1"}):
            info = _device_environment_info("cpu", torch.device("cpu"))
        self.assertEqual(info["environment_label"], "cuda-12.1")

        with mock.patch.dict(os.environ, {"GPRH_ENVIRONMENT": ""}):
            info = _device_environment_info("cpu", torch.device("cpu"))
        self.assertIsNone(info["environment_label"])

    def test_device_environment_info_for_cuda_without_cudnn(self) -> None:
        with (
            mock.patch("torch.backends.cudnn.is_available", return_value=False),
            mock.patch(
                "torch.cuda.get_device_properties",
                return_value=_FakeDeviceProperties(),
            ),
        ):
            info = _device_environment_info("auto", torch.device("cuda"))
        self.assertEqual(info["device_requested"], "auto")
        self.assertEqual(info["device_resolved"], "cuda")
        self.assertIsNone(info["cudnn_version"])
        self.assertEqual(info["gpu_name"], "Fake GPU")
        self.assertEqual(info["gpu_compute_capability"], "8.6")

    def test_device_environment_info_for_cuda_with_cudnn(self) -> None:
        with (
            mock.patch("torch.backends.cudnn.is_available", return_value=True),
            mock.patch("torch.backends.cudnn.version", return_value=8900),
            mock.patch(
                "torch.cuda.get_device_properties",
                return_value=_FakeDeviceProperties(),
            ),
        ):
            info = _device_environment_info("cuda", torch.device("cuda"))
        self.assertEqual(info["cudnn_version"], 8900)

    def test_cuda_driver_api_version_is_formatted_or_none(self) -> None:
        # ``_cuda_getDriverVersion`` is a private helper that CPU-only
        # builds do not ship, which is why the reader is best-effort.
        def patched(value: object) -> mock._patch:
            return mock.patch.object(
                torch._C, "_cuda_getDriverVersion", value, create=True
            )

        with patched(lambda: 12010):
            self.assertEqual(_cuda_driver_api_version(), "12.1")

        with patched(lambda: 0):
            self.assertIsNone(_cuda_driver_api_version())

        def raise_runtime_error() -> int:
            raise RuntimeError("no driver")

        with patched(raise_runtime_error):
            self.assertIsNone(_cuda_driver_api_version())

        with patched(None):
            self.assertIsNone(_cuda_driver_api_version())

    def test_nvidia_driver_version_reads_nvidia_smi(self) -> None:
        completed = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=0, stdout="535.161.08\n", stderr=""
        )
        with mock.patch("subprocess.run", return_value=completed):
            self.assertEqual(_nvidia_driver_version(), "535.161.08")

    def test_nvidia_driver_version_is_none_without_nvidia_smi(self) -> None:
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            self.assertIsNone(_nvidia_driver_version())

        failed = subprocess.CompletedProcess(
            args=["nvidia-smi"], returncode=9, stdout="", stderr="not found"
        )
        with mock.patch("subprocess.run", return_value=failed):
            self.assertIsNone(_nvidia_driver_version())


if __name__ == "__main__":
    unittest.main()
