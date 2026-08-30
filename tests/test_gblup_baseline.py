import contextlib
import io
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import gblup_baseline as gblup


def _fake_wandb_module() -> types.ModuleType:
    """Build a stand-in W&B module that records calls and contacts nothing.

    Every test below injects this into ``sys.modules``; no test imports the
    real client or opens a network connection, so an ``online`` selection is
    only ever observed as the arguments the CLI would have passed.
    """
    module = types.ModuleType("wandb")
    module.calls = []
    module.init = lambda **kwargs: module.calls.append(("init", kwargs))
    module.log = lambda payload: module.calls.append(("log", payload))
    module.finish = lambda: module.calls.append(("finish", None))
    return module


def _make_predictive_dataset() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    rng = np.random.default_rng(20260810)
    training_sample_count = 120
    test_sample_count = 60
    marker_count = 24
    sample_count = training_sample_count + test_sample_count
    allele_frequencies = rng.uniform(0.15, 0.45, size=marker_count)
    dosages = rng.binomial(
        2, allele_frequencies, size=(sample_count, marker_count)
    ).astype(np.float64)
    genotypes = dosages - 1.0
    genotypes[rng.random(genotypes.shape) < 0.05] = np.nan
    complete_for_signal = np.where(
        np.isnan(genotypes), np.nanmean(genotypes, axis=0), genotypes
    )
    marker_effects = rng.normal(size=marker_count)
    genetic_signal = complete_for_signal @ marker_effects
    noise_scale = 0.10 * float(np.std(genetic_signal))
    phenotypes = (
        500.0 + genetic_signal + rng.normal(scale=noise_scale, size=sample_count)
    )
    train_indices = np.arange(training_sample_count)
    test_indices = np.arange(training_sample_count, sample_count)
    return genotypes, phenotypes, train_indices, test_indices


class GblupBaselineTest(unittest.TestCase):
    def test_vanraden_relationship_uses_training_statistics(self) -> None:
        train = np.array(
            [
                [-1.0, -1.0, np.nan],
                [0.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        test = np.array([[np.nan, 1.0, 1.0]])
        prepared = gblup.prepare_fold_relationships(
            train, test, min_observed_rate=0.25, maf_threshold=0.0
        )

        means = np.nanmean(train, axis=0)
        imputed_train = np.where(np.isnan(train), means, train)
        imputed_test = np.where(np.isnan(test), means, test)
        frequencies = (means + 1.0) / 2.0
        denominator = 2.0 * np.sum(frequencies * (1.0 - frequencies))
        expected_train = (
            (imputed_train - means) @ (imputed_train - means).T / denominator
        )
        expected_train.flat[:: expected_train.shape[0] + 1] += gblup.DIAGONAL_JITTER
        expected_cross = (
            (imputed_test - means) @ (imputed_train - means).T / denominator
        )

        np.testing.assert_allclose(prepared.marker_means, means)
        np.testing.assert_allclose(prepared.relationship_train, expected_train)
        np.testing.assert_allclose(prepared.relationship_test_train, expected_cross)

    def test_test_genotypes_do_not_change_training_preprocessing(self) -> None:
        rng = np.random.default_rng(7)
        train = rng.choice([-1.0, 0.0, 1.0], size=(40, 10)).astype(float)
        train[rng.random(train.shape) < 0.1] = np.nan
        test = rng.choice([-1.0, 0.0, 1.0], size=(8, 10)).astype(float)
        changed_test = np.full_like(test, np.nan)
        changed_test[:, ::2] = 1.0

        original = gblup.prepare_fold_relationships(train, test)
        changed = gblup.prepare_fold_relationships(train, changed_test)

        np.testing.assert_array_equal(
            original.retained_markers, changed.retained_markers
        )
        np.testing.assert_allclose(original.marker_means, changed.marker_means)
        np.testing.assert_allclose(
            original.relationship_train, changed.relationship_train
        )
        self.assertAlmostEqual(original.denominator, changed.denominator)

        phenotypes_train = rng.normal(size=train.shape[0])
        original_fit = gblup.fit_gblup_reml(
            original.relationship_train, phenotypes_train
        )
        changed_fit = gblup.fit_gblup_reml(changed.relationship_train, phenotypes_train)
        self.assertAlmostEqual(original_fit.lambda_ratio, changed_fit.lambda_ratio)
        self.assertAlmostEqual(
            original_fit.genetic_variance,
            changed_fit.genetic_variance,
        )
        self.assertAlmostEqual(
            original_fit.residual_variance,
            changed_fit.residual_variance,
        )

    def test_low_observation_marker_is_removed(self) -> None:
        train = np.array(
            [
                [-1.0, -1.0],
                [0.0, np.nan],
                [1.0, np.nan],
                [-1.0, np.nan],
                [1.0, np.nan],
            ]
        )
        test = np.zeros((2, 2))
        prepared = gblup.prepare_fold_relationships(
            train, test, min_observed_rate=0.5, maf_threshold=0.0
        )
        np.testing.assert_array_equal(
            prepared.retained_markers, np.array([True, False])
        )

    def test_reml_matches_dense_mixed_model_equations(self) -> None:
        relationship = np.array(
            [
                [1.1, 0.4, 0.2, 0.1],
                [0.4, 1.2, 0.3, 0.2],
                [0.2, 0.3, 1.0, 0.4],
                [0.1, 0.2, 0.4, 1.3],
            ]
        )
        phenotypes = np.array([2.0, 3.5, 4.0, 6.0])
        fit = gblup.fit_gblup_reml(relationship, phenotypes)
        covariance = relationship + fit.lambda_ratio * np.eye(4)
        covariance_inverse = np.linalg.inv(covariance)
        intercept_vector = np.ones(4)
        expected_intercept = float(
            (intercept_vector @ covariance_inverse @ phenotypes)
            / (intercept_vector @ covariance_inverse @ intercept_vector)
        )
        expected_coefficients = np.linalg.solve(
            covariance, phenotypes - expected_intercept
        )
        self.assertGreater(fit.lambda_ratio, 0.0)
        self.assertGreater(fit.genetic_variance, 0.0)
        self.assertGreater(fit.residual_variance, 0.0)
        self.assertAlmostEqual(fit.intercept, expected_intercept, places=9)
        np.testing.assert_allclose(
            fit.dual_coefficients,
            expected_coefficients,
            rtol=1e-9,
            atol=1e-10,
        )

    def test_test_phenotypes_do_not_affect_predictions(self) -> None:
        genotypes, phenotypes, train_indices, test_indices = _make_predictive_dataset()
        changed = phenotypes.copy()
        changed[test_indices] = [1000.0] * (test_indices.size - 1) + [-1000.0]
        original_predictions, _, _ = gblup.predict_gblup_fold(
            train_indices, test_indices, genotypes, phenotypes
        )
        changed_predictions, _, _ = gblup.predict_gblup_fold(
            train_indices, test_indices, genotypes, changed
        )
        np.testing.assert_allclose(original_predictions, changed_predictions)

    def test_predicts_known_signal_for_held_out_samples(self) -> None:
        genotypes, phenotypes, train_indices, test_indices = _make_predictive_dataset()
        predictions, _, _ = gblup.predict_gblup_fold(
            train_indices, test_indices, genotypes, phenotypes
        )
        observed = phenotypes[test_indices]
        baseline = np.full(test_indices.size, phenotypes[train_indices].mean())
        correlation = gblup.compute_pearson_correlation(observed, predictions)
        model_rmse = gblup.compute_rmse(observed, predictions)
        baseline_rmse = gblup.compute_rmse(observed, baseline)
        self.assertTrue(np.isfinite(predictions).all())
        self.assertGreater(correlation, 0.90)
        self.assertLess(model_rmse, 0.6 * baseline_rmse)

    def test_constant_training_phenotypes_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be constant"):
            gblup.fit_gblup_reml(np.eye(4), np.ones(4))

    def test_fold_preprocessing_record_matches_relationships(self) -> None:
        train = np.array(
            [
                [-1.0, -1.0, np.nan],
                [0.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
                [0.0, 1.0, 0.0],
            ]
        )
        test = np.array([[np.nan, 1.0, 1.0]])
        relationships = gblup.prepare_fold_relationships(
            train, test, min_observed_rate=0.25, maf_threshold=0.0
        )

        record, arrays = gblup.build_fold_preprocessing_record(
            2, "NAM03", relationships
        )

        self.assertEqual(record["fold_index"], 2)
        self.assertEqual(record["held_out_family"], "NAM03")
        self.assertEqual(
            record["total_marker_count"], int(relationships.retained_markers.size)
        )
        self.assertEqual(
            record["retained_marker_count"], int(relationships.retained_markers.sum())
        )
        self.assertAlmostEqual(record["denominator"], relationships.denominator)

        marker_mask_key = record["arrays"]["marker_mask_ref"]
        imputation_mean_key = record["arrays"]["imputation_mean_ref"]
        allele_frequency_key = record["arrays"]["allele_frequency_ref"]
        observed_rate_key = record["arrays"]["observed_rate_ref"]

        np.testing.assert_array_equal(
            arrays[marker_mask_key], relationships.retained_markers
        )
        np.testing.assert_allclose(
            arrays[imputation_mean_key], relationships.marker_means
        )
        np.testing.assert_allclose(
            arrays[allele_frequency_key], (relationships.marker_means + 1.0) / 2.0
        )
        np.testing.assert_allclose(
            arrays[observed_rate_key], relationships.observed_rates
        )


class GblupCliTest(unittest.TestCase):
    def test_defaults_preserve_the_previous_fixed_behaviour(self) -> None:
        args = gblup.parse_args([])

        self.assertEqual(args.data_dir, Path("data"))
        self.assertEqual(args.output_dir, Path("gblup_results"))
        self.assertEqual(args.expected_families, 16)

    def test_external_logging_is_off_by_default(self) -> None:
        self.assertEqual(gblup.parse_args([]).wandb_mode, "disabled")

    def test_arguments_are_read_from_the_command_line(self) -> None:
        args = gblup.parse_args(
            [
                "--data-dir",
                "/tmp/input",
                "--output-dir",
                "/tmp/output",
                "--expected-families",
                "3",
                "--wandb-mode",
                "offline",
            ]
        )

        self.assertEqual(args.data_dir, Path("/tmp/input"))
        self.assertEqual(args.output_dir, Path("/tmp/output"))
        self.assertEqual(args.expected_families, 3)
        self.assertEqual(args.wandb_mode, "offline")

    def test_family_count_below_two_is_rejected(self) -> None:
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            gblup.parse_args(["--expected-families", "1"])

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--expected-families must be at least 2", stderr.getvalue())

    def test_unknown_wandb_mode_is_rejected(self) -> None:
        stderr = io.StringIO()
        with (
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            gblup.parse_args(["--wandb-mode", "enabled"])

        self.assertEqual(raised.exception.code, 2)


class GblupRunLoggerTest(unittest.TestCase):
    def test_disabled_mode_never_initializes_wandb(self) -> None:
        fake = _fake_wandb_module()
        with (
            mock.patch.dict(sys.modules, {"wandb": fake}),
            mock.patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=False),
        ):
            logger = gblup.create_run_logger("disabled", config={"a": 1})
            logger.log({"metric": 1.0})
            logger.finish()

            self.assertIsInstance(logger, gblup.NullRunLogger)
            self.assertEqual(fake.calls, [])
            # A disabled run leaves the surrounding environment untouched
            # instead of reconfiguring a client it never starts.
            self.assertEqual(os.environ["WANDB_MODE"], "online")

    def test_offline_mode_is_not_escalated_by_the_environment(self) -> None:
        fake = _fake_wandb_module()
        with (
            mock.patch.dict(sys.modules, {"wandb": fake}),
            mock.patch.dict(os.environ, {"WANDB_MODE": "online"}, clear=False),
        ):
            logger = gblup.create_run_logger("offline", config={"family_count": 3})

            self.assertIsInstance(logger, gblup.WandbRunLogger)
            self.assertEqual(os.environ["WANDB_MODE"], "offline")

        self.assertEqual(len(fake.calls), 1)
        name, kwargs = fake.calls[0]
        self.assertEqual(name, "init")
        self.assertEqual(kwargs["mode"], "offline")
        self.assertEqual(kwargs["config"], {"family_count": 3})

    def test_online_mode_requires_an_explicit_selection(self) -> None:
        fake = _fake_wandb_module()
        with (
            mock.patch.dict(sys.modules, {"wandb": fake}),
            mock.patch.dict(os.environ, {"WANDB_MODE": "offline"}, clear=False),
        ):
            logger = gblup.create_run_logger("online", config={})
            logger.log({"gblup/r": 0.5})
            logger.finish()

            self.assertEqual(os.environ["WANDB_MODE"], "online")

        self.assertEqual(
            fake.calls,
            [
                ("init", fake.calls[0][1]),
                ("log", {"gblup/r": 0.5}),
                ("finish", None),
            ],
        )
        self.assertEqual(fake.calls[0][1]["mode"], "online")

    def test_unknown_mode_is_rejected_before_importing_wandb(self) -> None:
        fake = _fake_wandb_module()
        with (
            mock.patch.dict(sys.modules, {"wandb": fake}),
            self.assertRaisesRegex(ValueError, "unknown W&B mode"),
        ):
            gblup.create_run_logger("enabled", config={})

        self.assertEqual(fake.calls, [])


class GblupRunSettingsMetadataTest(unittest.TestCase):
    def test_metadata_records_the_resolved_run_settings(self) -> None:
        written: dict[str, object] = {}

        def fake_write_run_artifacts(**kwargs: object) -> Path:
            written.update(kwargs)
            return Path("gblup_results/artifacts/run")

        dataset = types.SimpleNamespace(
            genotypes=np.zeros((4, 2)),
            phenotypes=np.arange(4.0),
            family_ids=np.array(["A", "A", "B", "B"]),
            sample_names=np.array(["s1", "s2", "s3", "s4"]),
        )
        frame = gblup.pd.DataFrame(
            {
                "family_id": dataset.family_ids,
                "sample_name": dataset.sample_names,
                "observed_yield_kg_ha": dataset.phenotypes,
                "predicted_yield_kg_ha": dataset.phenotypes,
            }
        )

        with mock.patch.object(
            gblup.run_manifest, "write_run_artifacts", fake_write_run_artifacts
        ):
            gblup.save_run_artifacts(
                output_dir=Path("gblup_results"),
                dataset=dataset,
                predictions_frame=frame,
                fold_preprocessing_records=[],
                preprocessing_arrays={},
                fold_metric_records=[],
                macro_correlation=0.1,
                pooled_correlation=0.2,
                pooled_rmse=3.0,
                input_files=[],
                source_checksums={},
                expected_family_count=3,
                wandb_mode="offline",
                command_arguments=["--data-dir", "/private/input"],
            )

        metadata = written["metadata"]
        self.assertEqual(metadata["hyperparameters"]["expected_family_count"], 3)
        self.assertEqual(
            metadata["external_logging"], {"backend": "wandb", "mode": "offline"}
        )
        self.assertEqual(metadata["command"]["arguments"], ["--data-dir", "input"])


if __name__ == "__main__":
    unittest.main()
