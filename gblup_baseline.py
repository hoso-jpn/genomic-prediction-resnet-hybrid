"""Leakage-safe LOFO-CV GBLUP baseline implemented with NumPy and SciPy.

The model is y = 1 * mu + u + e, where u follows a genomic relationship
matrix. Raw SoyNAM genotypes are loaded without replacing missing values.
Marker filtering, mean imputation, and VanRaden relationship matrices are
computed from the training families independently in every LOFO fold.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from sklearn.model_selection import LeaveOneGroupOut

import run_manifest
import soynam_data
from soynam_data import list_family_files, load_soynam_dataset

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IndexArray = NDArray[np.int_]


DIAGONAL_JITTER = 1e-4
LOG_LAMBDA_BOUNDS = (-12.0, 12.0)
MIN_TRAINING_SAMPLES = 3
MIN_OBSERVED_RATE = 0.10
MAF_THRESHOLD = 0.05
VARIANCE_THRESHOLD = 1e-6
EXPECTED_FAMILY_COUNT = 16


@dataclass(frozen=True)
class FoldRelationships:
    """Training-derived marker statistics and relationship matrices."""

    relationship_train: FloatArray
    relationship_test_train: FloatArray
    retained_markers: BoolArray
    marker_means: FloatArray
    observed_rates: FloatArray
    denominator: float


@dataclass(frozen=True)
class GblupFit:
    """Fitted parameters for the one-kernel GBLUP model."""

    intercept: float
    lambda_ratio: float
    genetic_variance: float
    residual_variance: float
    dual_coefficients: FloatArray


def prepare_fold_relationships(
    genotypes_train: NDArray[Any],
    genotypes_test: NDArray[Any],
    *,
    min_observed_rate: float = MIN_OBSERVED_RATE,
    maf_threshold: float = MAF_THRESHOLD,
) -> FoldRelationships:
    """Build leakage-safe VanRaden matrices from training statistics only."""
    train = np.asarray(genotypes_train, dtype=np.float64)
    test = np.asarray(genotypes_test, dtype=np.float64)

    if train.ndim != 2 or test.ndim != 2:
        raise ValueError("genotypes must be two-dimensional arrays")
    if train.shape[1] == 0 or train.shape[1] != test.shape[1]:
        raise ValueError("training and test marker dimensions do not match")
    if train.shape[0] < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"at least {MIN_TRAINING_SAMPLES} training samples are required"
        )
    if np.isinf(train).any() or np.isinf(test).any():
        raise ValueError("genotypes must not contain infinite values")
    if not 0.0 < min_observed_rate <= 1.0:
        raise ValueError("min_observed_rate must be in (0, 1]")
    if not 0.0 <= maf_threshold < 0.5:
        raise ValueError("maf_threshold must be in [0, 0.5)")

    observed_counts = np.isfinite(train).sum(axis=0)
    observed_rates_all = observed_counts / train.shape[0]
    observed_mask = observed_rates_all > min_observed_rate
    if not observed_mask.any():
        raise ValueError("no markers pass the training observed-rate filter")

    candidate_train = train[:, observed_mask]
    candidate_means = np.nanmean(candidate_train, axis=0)
    imputed_candidate_train = np.where(
        np.isnan(candidate_train), candidate_means, candidate_train
    )
    candidate_variances = np.var(imputed_candidate_train, axis=0)
    allele_frequencies = (candidate_means + 1.0) / 2.0
    candidate_maf = np.minimum(allele_frequencies, 1.0 - allele_frequencies)
    candidate_keep = (
        np.isfinite(candidate_means)
        & (candidate_variances > VARIANCE_THRESHOLD)
        & (candidate_maf > maf_threshold)
    )
    if not candidate_keep.any():
        raise ValueError("no markers pass the training variance and MAF filters")

    retained_markers = np.zeros(train.shape[1], dtype=bool)
    retained_positions = np.flatnonzero(observed_mask)[candidate_keep]
    retained_markers[retained_positions] = True

    marker_means = candidate_means[candidate_keep]
    observed_rates = observed_rates_all[retained_markers]
    retained_train = train[:, retained_markers]
    retained_test = test[:, retained_markers]
    imputed_train = np.where(np.isnan(retained_train), marker_means, retained_train)
    imputed_test = np.where(np.isnan(retained_test), marker_means, retained_test)
    centered_train = imputed_train - marker_means
    centered_test = imputed_test - marker_means

    retained_frequencies = (marker_means + 1.0) / 2.0
    denominator = float(
        2.0 * np.sum(retained_frequencies * (1.0 - retained_frequencies))
    )
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("VanRaden denominator must be positive")

    relationship_train = centered_train @ centered_train.T / denominator
    relationship_train.flat[:: relationship_train.shape[0] + 1] += DIAGONAL_JITTER
    relationship_test_train = centered_test @ centered_train.T / denominator

    return FoldRelationships(
        relationship_train=relationship_train,
        relationship_test_train=relationship_test_train,
        retained_markers=retained_markers,
        marker_means=marker_means,
        observed_rates=observed_rates,
        denominator=denominator,
    )


def _validate_training_data(
    relationship_train: FloatArray,
    phenotypes_train: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    relationship = np.asarray(relationship_train, dtype=np.float64)
    phenotypes = np.asarray(phenotypes_train, dtype=np.float64).reshape(-1)

    if relationship.ndim != 2 or relationship.shape[0] != relationship.shape[1]:
        raise ValueError("relationship_train must be a square matrix")
    if relationship.shape[0] != phenotypes.size:
        raise ValueError("relationship_train and phenotypes_train sizes do not match")
    if phenotypes.size < MIN_TRAINING_SAMPLES:
        raise ValueError(
            f"at least {MIN_TRAINING_SAMPLES} training samples are required"
        )
    if not np.isfinite(relationship).all():
        raise ValueError("relationship_train must contain only finite values")
    if not np.isfinite(phenotypes).all():
        raise ValueError("phenotypes_train must contain only finite values")
    if not np.allclose(relationship, relationship.T, rtol=1e-8, atol=1e-10):
        raise ValueError("relationship_train must be symmetric")
    if np.ptp(phenotypes) <= np.finfo(np.float64).eps:
        raise ValueError("training phenotypes must not be constant")

    return relationship, phenotypes


def _profile_reml_objective(
    log_lambda: float,
    eigenvalues: FloatArray,
    rotated_phenotypes: FloatArray,
    rotated_intercept: FloatArray,
) -> float:
    lambda_ratio = float(np.exp(log_lambda))
    covariance_eigenvalues = eigenvalues + lambda_ratio
    inverse_weights = 1.0 / covariance_eigenvalues
    fixed_information = float(
        np.sum(rotated_intercept * rotated_intercept * inverse_weights)
    )
    if fixed_information <= 0.0:
        return float("inf")
    intercept = float(
        np.sum(rotated_intercept * rotated_phenotypes * inverse_weights)
        / fixed_information
    )
    rotated_residuals = rotated_phenotypes - intercept * rotated_intercept
    residual_quadratic = float(
        np.sum(rotated_residuals * rotated_residuals * inverse_weights)
    )
    if residual_quadratic <= 0.0:
        return float("inf")
    residual_degrees_of_freedom = rotated_phenotypes.size - 1
    return 0.5 * (
        float(np.sum(np.log(covariance_eigenvalues)))
        + np.log(fixed_information)
        + residual_degrees_of_freedom
        * np.log(residual_quadratic / residual_degrees_of_freedom)
    )


def fit_gblup_reml(
    relationship_train: FloatArray,
    phenotypes_train: FloatArray,
) -> GblupFit:
    """Fit one-kernel GBLUP variance components using profile REML."""
    relationship, phenotypes = _validate_training_data(
        relationship_train, phenotypes_train
    )
    eigenvalues, eigenvectors = np.linalg.eigh(relationship)
    minimum_eigenvalue = float(eigenvalues.min())
    if minimum_eigenvalue < -1e-8:
        raise ValueError(
            "relationship_train must be positive semidefinite; "
            f"minimum eigenvalue was {minimum_eigenvalue:.3e}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    rotated_phenotypes = eigenvectors.T @ phenotypes
    rotated_intercept = eigenvectors.T @ np.ones(phenotypes.size, dtype=np.float64)
    optimization = minimize_scalar(
        _profile_reml_objective,
        args=(eigenvalues, rotated_phenotypes, rotated_intercept),
        bounds=LOG_LAMBDA_BOUNDS,
        method="bounded",
        options={"xatol": 1e-8},
    )
    if not optimization.success or not np.isfinite(optimization.fun):
        raise RuntimeError(f"REML optimization failed: {optimization.message}")

    lambda_ratio = float(np.exp(optimization.x))
    covariance_eigenvalues = eigenvalues + lambda_ratio
    inverse_weights = 1.0 / covariance_eigenvalues
    fixed_information = float(
        np.sum(rotated_intercept * rotated_intercept * inverse_weights)
    )
    intercept = float(
        np.sum(rotated_intercept * rotated_phenotypes * inverse_weights)
        / fixed_information
    )
    rotated_residuals = rotated_phenotypes - intercept * rotated_intercept
    residual_degrees_of_freedom = phenotypes.size - 1
    residual_quadratic = float(
        np.sum(rotated_residuals * rotated_residuals * inverse_weights)
    )
    genetic_variance = residual_quadratic / residual_degrees_of_freedom
    residual_variance = lambda_ratio * genetic_variance
    dual_coefficients = eigenvectors @ (rotated_residuals / covariance_eigenvalues)
    return GblupFit(
        intercept=intercept,
        lambda_ratio=lambda_ratio,
        genetic_variance=genetic_variance,
        residual_variance=residual_variance,
        dual_coefficients=dual_coefficients,
    )


def predict_genetic_values(
    fit: GblupFit,
    relationship_test_train: FloatArray,
) -> FloatArray:
    """Predict held-out breeding values from test-training covariance."""
    cross_relationship = np.asarray(relationship_test_train, dtype=np.float64)
    if cross_relationship.ndim != 2:
        raise ValueError("relationship_test_train must be a two-dimensional array")
    if cross_relationship.shape[1] != fit.dual_coefficients.size:
        raise ValueError("relationship_test_train has an unexpected training dimension")
    if not np.isfinite(cross_relationship).all():
        raise ValueError("relationship_test_train must contain only finite values")
    return cross_relationship @ fit.dual_coefficients


def predict_gblup_fold(
    train_indices: IndexArray,
    test_indices: IndexArray,
    genotypes: FloatArray,
    phenotypes: FloatArray,
) -> tuple[FloatArray, GblupFit, FoldRelationships]:
    """Fit one leakage-safe LOFO split and predict held-out phenotypes."""
    fold_relationships = prepare_fold_relationships(
        genotypes[train_indices], genotypes[test_indices]
    )
    fit = fit_gblup_reml(
        fold_relationships.relationship_train,
        phenotypes[train_indices],
    )
    breeding_values = predict_genetic_values(
        fit, fold_relationships.relationship_test_train
    )
    predictions = fit.intercept + breeding_values
    return predictions, fit, fold_relationships


def compute_pearson_correlation(
    observed: FloatArray,
    predicted: FloatArray,
) -> float:
    observed_values = np.asarray(observed, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if observed_values.size != predicted_values.size:
        raise ValueError("observed and predicted sizes do not match")
    valid = np.isfinite(observed_values) & np.isfinite(predicted_values)
    if valid.sum() < 2:
        return float("nan")
    observed_valid = observed_values[valid]
    predicted_valid = predicted_values[valid]
    if (
        np.ptp(observed_valid) <= np.finfo(np.float64).eps
        or np.ptp(predicted_valid) <= np.finfo(np.float64).eps
    ):
        return 0.0
    return float(np.corrcoef(observed_valid, predicted_valid)[0, 1])


def compute_rmse(observed: FloatArray, predicted: FloatArray) -> float:
    observed_values = np.asarray(observed, dtype=np.float64).reshape(-1)
    predicted_values = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if observed_values.size != predicted_values.size:
        raise ValueError("observed and predicted sizes do not match")
    if (
        not np.isfinite(observed_values).all()
        or not np.isfinite(predicted_values).all()
    ):
        raise ValueError("RMSE inputs must contain only finite values")
    return float(np.sqrt(np.mean((observed_values - predicted_values) ** 2)))


def build_fold_preprocessing_record(
    fold_index: int,
    held_out_family: str,
    relationships: FoldRelationships,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Build one fold's preprocessing.json entry and its NPZ-backed arrays.

    Reuses the training-derived statistics already computed by
    ``prepare_fold_relationships`` without recomputing or altering them.
    """
    prefix = f"fold_{fold_index:03d}"
    allele_frequency = (relationships.marker_means + 1.0) / 2.0
    arrays: dict[str, FloatArray] = {
        f"{prefix}_marker_mask": relationships.retained_markers,
        f"{prefix}_imputation_mean": relationships.marker_means,
        f"{prefix}_allele_frequency": allele_frequency,
        f"{prefix}_observed_rate": relationships.observed_rates,
    }
    record = {
        "fold_index": fold_index,
        "held_out_family": held_out_family,
        "total_marker_count": int(relationships.retained_markers.size),
        "retained_marker_count": int(relationships.retained_markers.sum()),
        "denominator": float(relationships.denominator),
        "arrays": {
            "marker_mask_ref": f"{prefix}_marker_mask",
            "imputation_mean_ref": f"{prefix}_imputation_mean",
            "allele_frequency_ref": f"{prefix}_allele_frequency",
            "observed_rate_ref": f"{prefix}_observed_rate",
        },
    }
    return record, arrays


def save_run_artifacts(
    *,
    output_dir: Path,
    dataset: Any,
    predictions_frame: pd.DataFrame,
    fold_preprocessing_records: list[dict[str, Any]],
    preprocessing_arrays: dict[str, FloatArray],
    fold_metric_records: list[dict[str, Any]],
    macro_correlation: float,
    pooled_correlation: float,
    pooled_rmse: float,
    input_files: list[dict[str, str]],
    source_checksums: dict[str, str],
) -> Path:
    """Assemble this run's metadata/split/preprocessing/metrics and write them.

    ``input_files`` and ``source_checksums`` must be captured once at the
    start of the run (before training) and passed in as-is, so the recorded
    checksums describe what was actually read rather than whatever happens
    to be on disk when the run finishes.
    """
    families = sorted({str(family_id) for family_id in dataset.family_ids})
    split = {
        "schema_version": run_manifest.SCHEMA_VERSION,
        "outer": run_manifest.build_outer_split(
            dataset.sample_names.tolist(), dataset.family_ids.tolist()
        ),
        "inner": None,
    }
    preprocessing = {
        "schema_version": run_manifest.SCHEMA_VERSION,
        "model": "gblup",
        "config": {
            "min_observed_rate": MIN_OBSERVED_RATE,
            "maf_threshold": MAF_THRESHOLD,
            "imputation": "training_mean",
            "relationship": "VanRaden-1",
            "total_marker_count": int(dataset.genotypes.shape[1]),
        },
        "folds": fold_preprocessing_records,
    }
    metrics = {
        "schema_version": run_manifest.SCHEMA_VERSION,
        "model": "gblup",
        "folds": fold_metric_records,
        "summary": {
            "macro_family_r": macro_correlation,
            "pooled_r": pooled_correlation,
            "pooled_rmse": pooled_rmse,
        },
    }

    run_id = run_manifest.new_run_id()
    metadata = {
        "schema_version": run_manifest.SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": run_manifest.utc_now_iso(),
        "model_name": "gblup",
        "git_commit": run_manifest.git_commit_sha(Path(__file__).resolve().parent),
        "source_file_checksums": source_checksums,
        "command": run_manifest.sanitize_command(Path(sys.argv[0]).name, sys.argv[1:]),
        "seed": None,
        "python_version": run_manifest.python_version(),
        "library_versions": run_manifest.library_versions(
            ["numpy", "pandas", "scikit-learn", "scipy"]
        ),
        "hyperparameters": {
            "min_observed_rate": MIN_OBSERVED_RATE,
            "maf_threshold": MAF_THRESHOLD,
            "relationship": "VanRaden-1",
        },
        "input_files": input_files,
        "families": families,
        "split_ref": "split.json",
        "preprocessing_ref": "preprocessing.json",
        "preprocessing_arrays_ref": "preprocessing_arrays.npz",
        "metrics_ref": "metrics.json",
        "predictions_ref": "predictions.csv",
    }

    return run_manifest.write_run_artifacts(
        output_dir=output_dir,
        run_id=run_id,
        metadata=metadata,
        split=split,
        preprocessing=preprocessing,
        preprocessing_arrays=preprocessing_arrays,
        metrics=metrics,
        predictions=predictions_frame,
    )


def main() -> None:
    """Execute family-wise LOFO cross-validation."""
    import wandb

    data_dir = Path("data")
    # Fix the file list and its checksums once, before loading, so metadata
    # describes exactly what was read rather than whatever is on disk by
    # the time this (potentially long) run finishes.
    family_files = list_family_files(data_dir)
    input_files = run_manifest.describe_input_files(family_files)
    source_checksums = run_manifest.source_file_checksums(
        [Path(__file__), Path(soynam_data.__file__), Path(run_manifest.__file__)]
    )

    dataset = load_soynam_dataset(data_dir, family_files=family_files)
    splitter = LeaveOneGroupOut()
    total_folds = splitter.get_n_splits(
        dataset.genotypes, dataset.phenotypes, dataset.family_ids
    )
    expected_folds = np.unique(dataset.family_ids).size
    if total_folds != expected_folds:
        raise RuntimeError("LOFO fold count does not match the family count")
    if expected_folds != EXPECTED_FAMILY_COUNT:
        raise RuntimeError(
            f"expected {EXPECTED_FAMILY_COUNT} SoyNAM families, found {expected_folds}"
        )

    wandb.init(
        project="genomic-resnet-prediction-hy",
        job_type="gblup_baseline",
        name="gblup-lofo-leakage-safe",
        config={
            "sample_count": dataset.phenotypes.size,
            "marker_count": dataset.genotypes.shape[1],
            "family_count": expected_folds,
            "min_observed_rate": MIN_OBSERVED_RATE,
            "maf_threshold": MAF_THRESHOLD,
            "relationship": "VanRaden-1",
            "phenotype_scale": "raw-kg-per-ha",
        },
    )

    oof_predictions = np.full(dataset.phenotypes.size, np.nan, dtype=np.float64)
    fold_correlations: list[float] = []
    fold_count = 0
    fold_preprocessing_records: list[dict[str, Any]] = []
    preprocessing_arrays: dict[str, FloatArray] = {}
    fold_metric_records: list[dict[str, Any]] = []

    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(dataset.genotypes, dataset.phenotypes, dataset.family_ids)
    ):
        held_out_families = np.unique(dataset.family_ids[test_indices])
        if held_out_families.size != 1:
            raise RuntimeError("each LOFO fold must contain exactly one family")
        family_label = str(held_out_families[0])
        predictions, fit, relationships = predict_gblup_fold(
            train_indices,
            test_indices,
            dataset.genotypes,
            dataset.phenotypes,
        )
        oof_predictions[test_indices] = predictions
        correlation = compute_pearson_correlation(
            dataset.phenotypes[test_indices], predictions
        )
        rmse = compute_rmse(dataset.phenotypes[test_indices], predictions)
        if not np.isfinite(correlation):
            raise RuntimeError(f"non-finite correlation in {family_label}")
        fold_correlations.append(correlation)
        fold_count += 1
        print(
            f"{family_label:22s} r={correlation:.4f} rmse={rmse:.2f} "
            f"markers={relationships.retained_markers.sum()}"
        )
        wandb.log(
            {
                "fold": fold_index + 1,
                "gblup/r": correlation,
                "gblup/rmse": rmse,
                "gblup/lambda_ratio": fit.lambda_ratio,
                "gblup/genetic_variance": fit.genetic_variance,
                "gblup/residual_variance": fit.residual_variance,
                "gblup/retained_markers": int(relationships.retained_markers.sum()),
                "test_family": family_label,
            }
        )

        fold_record, fold_arrays = build_fold_preprocessing_record(
            fold_index, family_label, relationships
        )
        fold_preprocessing_records.append(fold_record)
        preprocessing_arrays.update(fold_arrays)
        fold_metric_records.append(
            {
                "fold_index": fold_index,
                "held_out_family": family_label,
                "pearson_r": correlation,
                "rmse": rmse,
            }
        )

    if fold_count != total_folds or not np.isfinite(oof_predictions).all():
        raise RuntimeError(
            f"LOFO-CV incomplete: {fold_count}/{total_folds} folds succeeded"
        )

    macro_correlation = float(np.mean(fold_correlations))
    pooled_correlation = compute_pearson_correlation(
        dataset.phenotypes, oof_predictions
    )
    pooled_rmse = compute_rmse(dataset.phenotypes, oof_predictions)

    predictions_frame = pd.DataFrame(
        {
            "family_id": dataset.family_ids,
            "sample_name": dataset.sample_names,
            "observed_yield_kg_ha": dataset.phenotypes,
            "predicted_yield_kg_ha": oof_predictions,
        }
    )
    run_manifest.verify_input_files_unchanged(family_files, input_files)
    run_dir = save_run_artifacts(
        output_dir=Path("gblup_results"),
        dataset=dataset,
        predictions_frame=predictions_frame,
        fold_preprocessing_records=fold_preprocessing_records,
        preprocessing_arrays=preprocessing_arrays,
        fold_metric_records=fold_metric_records,
        macro_correlation=macro_correlation,
        pooled_correlation=pooled_correlation,
        pooled_rmse=pooled_rmse,
        input_files=input_files,
        source_checksums=source_checksums,
    )
    print(f"run artifacts:             {run_dir}")

    print(f"\n{'=' * 58}")
    print(f"GBLUP LOFO macro family r: {macro_correlation:.4f}")
    print(f"GBLUP LOFO pooled OOF r:   {pooled_correlation:.4f}")
    print(f"GBLUP LOFO pooled RMSE:    {pooled_rmse:.2f} kg/ha")
    print(f"successful folds:          {fold_count}/{total_folds}")
    wandb.log(
        {
            "summary/macro_family_r": macro_correlation,
            "summary/pooled_oof_r": pooled_correlation,
            "summary/pooled_oof_rmse": pooled_rmse,
            "summary/successful_folds": fold_count,
            "summary/total_folds": total_folds,
        }
    )
    wandb.finish()


if __name__ == "__main__":
    main()
