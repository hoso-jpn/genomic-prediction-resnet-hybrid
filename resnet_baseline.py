"""Leakage-safe ResNet baseline for SoyNAM family-wise prediction."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import model
import run_manifest
import soynam_data
from model import GatedGenomicResNet
from soynam_data import SoynamDataset, list_family_files, load_soynam_dataset

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class ResNetConfig:
    """Hyperparameters for model selection and final refitting."""

    seed: int = 42
    hidden_dim: int = 64
    num_blocks: int = 3
    dropout_rate: float = 0.4
    kernel_size: int = 7
    pca_components: int = 64
    batch_size: int = 64
    max_epochs: int = 200
    patience: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    min_observed_rate: float = 0.9
    maf_threshold: float = 0.01


@dataclass(frozen=True)
class FeatureTransform:
    """Feature statistics fitted exclusively on one training partition."""

    retained_markers: BoolArray
    marker_means: FloatArray
    marker_scales: FloatArray
    pca: PCA


@dataclass(frozen=True)
class ResnetFoldRecord:
    """Everything recorded about one outer fold: split assignment, the two
    feature transforms fitted within it, the two target standardizations
    used to train and invert each stage's predictions, and its fold-level
    metric."""

    fold_index: int
    held_out_family: str
    validation_family: str
    fold_seed: int
    best_epoch: int
    selection_transform: FeatureTransform
    selection_target_mean: float
    selection_target_scale: float
    final_transform: FeatureTransform
    final_target_mean: float
    final_target_scale: float
    correlation: float


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def fit_feature_transform(
    train_genotypes: FloatArray,
    config: ResNetConfig,
    seed: int,
) -> FeatureTransform:
    """Fit marker filtering, imputation, scaling, and PCA on training data."""
    values = np.asarray(train_genotypes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("training genotypes must be a 2D array with at least 2 rows")

    observed = np.isfinite(values)
    observed_count = observed.sum(axis=0)
    observed_rate = observed_count / values.shape[0]
    marker_sums = np.where(observed, values, 0.0).sum(axis=0)
    marker_means = np.divide(
        marker_sums,
        observed_count,
        out=np.zeros(values.shape[1], dtype=np.float64),
        where=observed_count > 0,
    )
    imputed = np.where(observed, values, marker_means)
    marker_variances = imputed.var(axis=0)
    allele_frequency = np.clip((marker_means + 1.0) / 2.0, 0.0, 1.0)
    maf = np.minimum(allele_frequency, 1.0 - allele_frequency)
    retained = (
        (observed_rate >= config.min_observed_rate)
        & (maf >= config.maf_threshold)
        & (marker_variances > np.finfo(np.float64).eps)
    )
    if not retained.any():
        raise ValueError("no markers remain after training-only filtering")

    retained_values = imputed[:, retained]
    retained_means = marker_means[retained]
    retained_scales = retained_values.std(axis=0)
    standardized = (retained_values - retained_means) / retained_scales
    component_count = min(
        config.pca_components,
        standardized.shape[0],
        standardized.shape[1],
    )
    if component_count < 1:
        raise ValueError("pca_components must be positive")
    pca = PCA(
        n_components=component_count,
        svd_solver="randomized",
        random_state=seed,
    )
    pca.fit(standardized)
    return FeatureTransform(retained, retained_means, retained_scales, pca)


def transform_features(
    genotypes: FloatArray,
    transform: FeatureTransform,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Apply training-derived feature statistics without refitting."""
    values = np.asarray(genotypes, dtype=np.float64)[:, transform.retained_markers]
    imputed = np.where(np.isfinite(values), values, transform.marker_means)
    standardized = (imputed - transform.marker_means) / transform.marker_scales
    principal_components = transform.pca.transform(standardized)
    return standardized.astype(np.float32), principal_components.astype(np.float32)


def select_validation_family(
    outer_train_families: NDArray[np.str_],
    fold_index: int,
    seed: int,
) -> str:
    """Select one whole outer-training family deterministically."""
    families = np.unique(outer_train_families)
    if families.size < 2:
        raise ValueError("at least two outer-training families are required")
    return str(families[(seed + fold_index) % families.size])


def _new_model(
    snp_dim: int,
    pc_dim: int,
    config: ResNetConfig,
    device: torch.device,
) -> GatedGenomicResNet:
    return GatedGenomicResNet(
        snp_dim,
        hidden_dim=config.hidden_dim,
        num_blocks=config.num_blocks,
        dropout_rate=config.dropout_rate,
        pc_dim=pc_dim,
        kernel_size=config.kernel_size,
    ).to(device)


def _loader(
    snps: NDArray[np.float32],
    targets: NDArray[np.float32],
    pcs: NDArray[np.float32],
    config: ResNetConfig,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(snps),
        torch.from_numpy(targets),
        torch.from_numpy(pcs),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def _train_epoch(
    model: GatedGenomicResNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    model.train()
    loss_function = nn.MSELoss()
    for snps, targets, pcs in loader:
        optimizer.zero_grad(set_to_none=True)
        predictions = model(snps.to(device), pcs.to(device)).reshape(-1)
        loss = loss_function(predictions, targets.to(device))
        loss.backward()
        optimizer.step()


def _predict_standardized(
    model: GatedGenomicResNet,
    snps: NDArray[np.float32],
    pcs: NDArray[np.float32],
    device: torch.device,
) -> FloatArray:
    model.eval()
    with torch.no_grad():
        prediction = model(
            torch.from_numpy(snps).to(device),
            torch.from_numpy(pcs).to(device),
        )
    return prediction.cpu().numpy().reshape(-1).astype(np.float64)


def _target_scale(targets: FloatArray) -> tuple[float, float]:
    mean = float(np.mean(targets))
    scale = float(np.std(targets))
    if not np.isfinite(scale) or scale <= np.finfo(np.float64).eps:
        raise ValueError("training phenotypes must have non-zero finite variance")
    return mean, scale


def _select_epoch(
    fit_x: FloatArray,
    fit_y: FloatArray,
    validation_x: FloatArray,
    validation_y: FloatArray,
    config: ResNetConfig,
    seed: int,
    device: torch.device,
) -> tuple[int, FeatureTransform, float, float]:
    transform = fit_feature_transform(fit_x, config, seed)
    fit_snps, fit_pcs = transform_features(fit_x, transform)
    validation_snps, validation_pcs = transform_features(validation_x, transform)
    target_mean, target_scale = _target_scale(fit_y)
    fit_targets = ((fit_y - target_mean) / target_scale).astype(np.float32)
    validation_targets = (validation_y - target_mean) / target_scale

    seed_everything(seed)
    model = _new_model(fit_snps.shape[1], fit_pcs.shape[1], config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _loader(fit_snps, fit_targets, fit_pcs, config, seed)
    best_epoch = 1
    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, config.max_epochs + 1):
        _train_epoch(model, loader, optimizer, device)
        prediction = _predict_standardized(
            model, validation_snps, validation_pcs, device
        )
        validation_loss = float(np.mean((prediction - validation_targets) ** 2))
        if validation_loss < best_loss - 1e-12:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break
    return best_epoch, transform, target_mean, target_scale


def predict_resnet_fold(
    genotypes: FloatArray,
    phenotypes: FloatArray,
    family_ids: NDArray[np.str_],
    train_indices: NDArray[np.int_],
    test_indices: NDArray[np.int_],
    fold_index: int,
    config: ResNetConfig,
    device: torch.device,
) -> tuple[FloatArray, ResnetFoldRecord]:
    """Select an epoch without test data, refit, and predict one held-out family."""
    held_out_family = str(np.unique(family_ids[test_indices])[0])
    validation_family = select_validation_family(
        family_ids[train_indices], fold_index, config.seed
    )
    validation_mask = family_ids[train_indices] == validation_family
    fit_indices = train_indices[~validation_mask]
    validation_indices = train_indices[validation_mask]
    fold_seed = config.seed + fold_index * 100
    best_epoch, selection_transform, selection_target_mean, selection_target_scale = (
        _select_epoch(
            genotypes[fit_indices],
            phenotypes[fit_indices],
            genotypes[validation_indices],
            phenotypes[validation_indices],
            config,
            fold_seed,
            device,
        )
    )

    final_transform = fit_feature_transform(
        genotypes[train_indices], config, fold_seed + 1
    )
    train_snps, train_pcs = transform_features(
        genotypes[train_indices], final_transform
    )
    test_snps, test_pcs = transform_features(genotypes[test_indices], final_transform)
    final_target_mean, final_target_scale = _target_scale(phenotypes[train_indices])
    train_targets = (
        (phenotypes[train_indices] - final_target_mean) / final_target_scale
    ).astype(np.float32)

    seed_everything(fold_seed + 1)
    model = _new_model(train_snps.shape[1], train_pcs.shape[1], config, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loader = _loader(train_snps, train_targets, train_pcs, config, fold_seed + 1)
    for _ in range(best_epoch):
        _train_epoch(model, loader, optimizer, device)
    standardized = _predict_standardized(model, test_snps, test_pcs, device)
    predictions = standardized * final_target_scale + final_target_mean
    correlation = float(np.corrcoef(phenotypes[test_indices], predictions)[0, 1])
    record = ResnetFoldRecord(
        fold_index=fold_index,
        held_out_family=held_out_family,
        validation_family=validation_family,
        fold_seed=fold_seed,
        best_epoch=best_epoch,
        selection_transform=selection_transform,
        selection_target_mean=selection_target_mean,
        selection_target_scale=selection_target_scale,
        final_transform=final_transform,
        final_target_mean=final_target_mean,
        final_target_scale=final_target_scale,
        correlation=correlation,
    )
    return predictions, record


def run_lofo(
    dataset: SoynamDataset,
    config: ResNetConfig,
    device: torch.device,
) -> tuple[FloatArray, list[ResnetFoldRecord]]:
    """Generate one prediction for every sample using outer family-wise CV."""
    predictions = np.full(dataset.phenotypes.size, np.nan, dtype=np.float64)
    fold_records: list[ResnetFoldRecord] = []
    splitter = LeaveOneGroupOut()
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(dataset.genotypes, dataset.phenotypes, dataset.family_ids)
    ):
        fold_predictions, fold_record = predict_resnet_fold(
            dataset.genotypes,
            dataset.phenotypes,
            dataset.family_ids,
            train_indices,
            test_indices,
            fold_index,
            config,
            device,
        )
        predictions[test_indices] = fold_predictions
        fold_records.append(fold_record)
        print(
            f"{fold_record.held_out_family:22s} r={fold_record.correlation:.4f} "
            f"epoch={fold_record.best_epoch} validation={fold_record.validation_family}"
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("LOFO-CV did not produce every OOF prediction")
    return predictions, fold_records


def make_oof_frame(dataset: SoynamDataset, predictions: FloatArray) -> pd.DataFrame:
    """Build the same OOF output contract used by the GBLUP baseline."""
    return pd.DataFrame(
        {
            "family_id": dataset.family_ids,
            "sample_name": dataset.sample_names,
            "observed_yield_kg_ha": dataset.phenotypes,
            "predicted_yield_kg_ha": predictions,
        }
    )


def build_transform_record(
    prefix: str,
    transform: FeatureTransform,
    *,
    target_mean: float,
    target_scale: float,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Build one feature transform's preprocessing entry and NPZ-backed arrays.

    Reuses the training-derived statistics and fitted PCA already held by
    ``FeatureTransform``, plus the target mean/scale used to standardize and
    invert this stage's predictions, without recomputing or altering them.
    """
    arrays: dict[str, FloatArray] = {
        f"{prefix}_marker_mask": transform.retained_markers,
        f"{prefix}_imputation_mean": transform.marker_means,
        f"{prefix}_standardization_mean": transform.marker_means,
        f"{prefix}_standardization_scale": transform.marker_scales,
        f"{prefix}_pca_mean": transform.pca.mean_,
        f"{prefix}_pca_components": transform.pca.components_,
        f"{prefix}_pca_explained_variance_ratio": (
            transform.pca.explained_variance_ratio_
        ),
    }
    record = {
        "input_feature_count": int(transform.retained_markers.size),
        "retained_marker_count": int(transform.retained_markers.sum()),
        "output_feature_count": int(transform.pca.n_components_),
        "target_mean": float(target_mean),
        "target_scale": float(target_scale),
        "arrays": {
            "marker_mask_ref": f"{prefix}_marker_mask",
            "imputation_mean_ref": f"{prefix}_imputation_mean",
            "standardization_mean_ref": f"{prefix}_standardization_mean",
            "standardization_scale_ref": f"{prefix}_standardization_scale",
            "pca_mean_ref": f"{prefix}_pca_mean",
            "pca_components_ref": f"{prefix}_pca_components",
            "pca_explained_variance_ratio_ref": (
                f"{prefix}_pca_explained_variance_ratio"
            ),
        },
    }
    return record, arrays


def build_fold_preprocessing_entry(
    fold_record: ResnetFoldRecord,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Build one fold's preprocessing.json entry, covering both transforms."""
    selection_prefix = f"fold_{fold_record.fold_index:03d}_selection"
    final_prefix = f"fold_{fold_record.fold_index:03d}_final"
    selection_record, selection_arrays = build_transform_record(
        selection_prefix,
        fold_record.selection_transform,
        target_mean=fold_record.selection_target_mean,
        target_scale=fold_record.selection_target_scale,
    )
    final_record, final_arrays = build_transform_record(
        final_prefix,
        fold_record.final_transform,
        target_mean=fold_record.final_target_mean,
        target_scale=fold_record.final_target_scale,
    )
    entry = {
        "fold_index": fold_record.fold_index,
        "held_out_family": fold_record.held_out_family,
        "validation_family": fold_record.validation_family,
        "fold_seed": fold_record.fold_seed,
        "best_epoch": fold_record.best_epoch,
        "selection_transform": selection_record,
        "final_transform": final_record,
    }
    return entry, {**selection_arrays, **final_arrays}


def build_inner_split(
    fold_records: Sequence[ResnetFoldRecord],
    family_ids: Sequence[str],
    seed: int,
) -> dict[str, Any]:
    """Build the ResNet-specific inner validation-family split record."""
    unique_families = sorted({str(family_id) for family_id in family_ids})
    folds = []
    for fold_record in fold_records:
        outer_train_families = [
            family
            for family in unique_families
            if family != fold_record.held_out_family
        ]
        fit_family_ids = [
            family
            for family in outer_train_families
            if family != fold_record.validation_family
        ]
        folds.append(
            {
                "fold_index": fold_record.fold_index,
                "validation_family": fold_record.validation_family,
                "fit_family_ids": fit_family_ids,
            }
        )
    return {
        "strategy": "validation_family_selection",
        "seed": seed,
        "folds": folds,
    }


def _cuda_driver_api_version() -> str | None:
    """Read the CUDA driver API version (e.g. "12.1"), best-effort.

    This is the CUDA version the installed driver supports, not the
    display driver number. PyTorch exposes it only through a private
    helper whose availability differs between builds, so a missing value
    is recorded as ``None`` rather than failing an otherwise complete run.
    """
    getter = getattr(torch._C, "_cuda_getDriverVersion", None)
    if getter is None:
        return None
    try:
        raw = int(getter())
    except (RuntimeError, OSError, TypeError, ValueError):
        return None
    if raw <= 0:
        return None
    return f"{raw // 1000}.{(raw % 1000) // 10}"


def _nvidia_driver_version() -> str | None:
    """Read the NVIDIA display driver version via nvidia-smi, best-effort.

    PyTorch does not expose the display driver number, and it is what a
    GPU environment is actually pinned against, so it is read from
    nvidia-smi. Never raises: nvidia-smi may be absent even where CUDA
    runs (or present but unable to talk to the driver), in which case the
    field stays ``None``.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    first_line = result.stdout.strip().splitlines()
    if not first_line:
        return None
    return first_line[0].strip() or None


def _device_environment_info(requested: str, resolved: torch.device) -> dict[str, Any]:
    """Describe the compute device actually used, for reproducibility.

    ``--device auto`` (or an omitted flag) resolves to CPU or CUDA
    depending on availability at run time; recording both the request and
    the resolution is the only way to tell which one actually ran. On CUDA
    the GPU model, compute capability, drivers, and CUDA/cuDNN versions
    are recorded too, so a GPU result can be tied to the machine that
    produced it. ``environment_label`` comes from the ``GPRH_ENVIRONMENT``
    environment variable, which the CPU and CUDA images set to the pinned
    dependency environment they were built from.
    """
    info: dict[str, Any] = {
        "device_requested": requested,
        "device_resolved": str(resolved),
        "cuda_version": None,
        "cudnn_version": None,
        "gpu_name": None,
        "gpu_compute_capability": None,
        "cuda_driver_api_version": None,
        "nvidia_driver_version": None,
        "environment_label": os.environ.get("GPRH_ENVIRONMENT") or None,
    }
    if resolved.type == "cuda":
        info["cuda_version"] = torch.version.cuda
        if torch.backends.cudnn.is_available():
            info["cudnn_version"] = torch.backends.cudnn.version()
        properties = torch.cuda.get_device_properties(resolved)
        info["gpu_name"] = properties.name
        info["gpu_compute_capability"] = f"{properties.major}.{properties.minor}"
        info["cuda_driver_api_version"] = _cuda_driver_api_version()
        info["nvidia_driver_version"] = _nvidia_driver_version()
    return info


def save_run_artifacts(
    *,
    output_dir: Path,
    dataset: SoynamDataset,
    config: ResNetConfig,
    device: torch.device,
    device_requested: str,
    predictions_frame: pd.DataFrame,
    fold_records: list[ResnetFoldRecord],
    command_arguments: list[str],
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

    preprocessing_entries = []
    preprocessing_arrays: dict[str, FloatArray] = {}
    metric_folds = []
    for fold_record in fold_records:
        entry, arrays = build_fold_preprocessing_entry(fold_record)
        preprocessing_entries.append(entry)
        preprocessing_arrays.update(arrays)
        metric_folds.append(
            {
                "fold_index": fold_record.fold_index,
                "held_out_family": fold_record.held_out_family,
                # A short smoke run can yield near-constant predictions, making
                # the Pearson correlation undefined; represent that as JSON
                # null rather than letting a non-finite float reach write_json.
                "pearson_r": run_manifest.json_safe_float(fold_record.correlation),
            }
        )

    split = {
        "schema_version": run_manifest.SCHEMA_VERSION,
        "outer": run_manifest.build_outer_split(
            dataset.sample_names.tolist(), dataset.family_ids.tolist()
        ),
        "inner": build_inner_split(
            fold_records, dataset.family_ids.tolist(), config.seed
        ),
    }
    hyperparameters = {
        key: value for key, value in asdict(config).items() if key != "seed"
    }
    preprocessing = {
        "schema_version": run_manifest.SCHEMA_VERSION,
        "model": "resnet",
        "config": {
            "min_observed_rate": config.min_observed_rate,
            "maf_threshold": config.maf_threshold,
            "imputation": "training_mean",
            "standardization": "training_zscore",
            "pca": {"svd_solver": "randomized"},
            "hyperparameters": hyperparameters,
        },
        "folds": preprocessing_entries,
    }
    metrics = {
        "schema_version": run_manifest.SCHEMA_VERSION,
        "model": "resnet",
        "folds": metric_folds,
    }

    run_id = run_manifest.new_run_id()
    metadata = {
        "schema_version": run_manifest.SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": run_manifest.utc_now_iso(),
        "model_name": "resnet",
        "git_commit": run_manifest.git_commit_sha(Path(__file__).resolve().parent),
        "source_file_checksums": source_checksums,
        "command": run_manifest.sanitize_command(
            Path(sys.argv[0]).name, command_arguments
        ),
        "seed": config.seed,
        "python_version": run_manifest.python_version(),
        "library_versions": run_manifest.library_versions(
            ["numpy", "pandas", "scikit-learn", "torch", "torch-geometric"]
        ),
        "hyperparameters": hyperparameters,
        **_device_environment_info(device_requested, device),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("resnet_results"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--pca-components", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)
    config = ResNetConfig(
        seed=args.seed,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        pca_components=args.pca_components,
    )
    # Fix the file list and its checksums once, before loading, so metadata
    # describes exactly what was read rather than whatever is on disk by
    # the time this (potentially long) run finishes.
    family_files = list_family_files(args.data_dir)
    input_files = run_manifest.describe_input_files(family_files)
    source_checksums = run_manifest.source_file_checksums(
        [
            Path(__file__),
            Path(model.__file__),
            Path(soynam_data.__file__),
            Path(run_manifest.__file__),
        ]
    )

    dataset = load_soynam_dataset(args.data_dir, family_files=family_files)
    predictions, fold_records = run_lofo(dataset, config, device)
    predictions_frame = make_oof_frame(dataset, predictions)
    run_manifest.verify_input_files_unchanged(family_files, input_files)
    run_dir = save_run_artifacts(
        output_dir=args.output_dir,
        dataset=dataset,
        config=config,
        device=device,
        device_requested=args.device,
        predictions_frame=predictions_frame,
        fold_records=fold_records,
        command_arguments=sys.argv[1:],
        input_files=input_files,
        source_checksums=source_checksums,
    )
    print(f"run artifacts: {run_dir}")


if __name__ == "__main__":
    main()
