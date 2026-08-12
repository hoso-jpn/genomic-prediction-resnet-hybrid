import unittest

import numpy as np
import torch

from resnet_baseline import (
    ResNetConfig,
    build_fold_preprocessing_entry,
    build_transform_record,
    fit_feature_transform,
    make_oof_frame,
    predict_resnet_fold,
    select_validation_family,
    transform_features,
)
from soynam_data import SoynamDataset


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

        record, arrays = build_transform_record("fold_000_final", transform)

        self.assertEqual(
            record["input_feature_count"], int(transform.retained_markers.size)
        )
        self.assertEqual(
            record["retained_marker_count"], int(transform.retained_markers.sum())
        )
        self.assertEqual(record["output_feature_count"], transform.pca.n_components_)
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


if __name__ == "__main__":
    unittest.main()
