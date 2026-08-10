"""Leakage-safe ResNet baseline for SoyNAM family-wise prediction."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.decomposition import PCA
from sklearn.model_selection import LeaveOneGroupOut
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from model import GatedGenomicResNet
from soynam_data import SoynamDataset, load_soynam_dataset

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
) -> int:
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
    return best_epoch


def predict_resnet_fold(
    genotypes: FloatArray,
    phenotypes: FloatArray,
    family_ids: NDArray[np.str_],
    train_indices: NDArray[np.int_],
    test_indices: NDArray[np.int_],
    fold_index: int,
    config: ResNetConfig,
    device: torch.device,
) -> tuple[FloatArray, int, str]:
    """Select an epoch without test data, refit, and predict one held-out family."""
    validation_family = select_validation_family(
        family_ids[train_indices], fold_index, config.seed
    )
    validation_mask = family_ids[train_indices] == validation_family
    fit_indices = train_indices[~validation_mask]
    validation_indices = train_indices[validation_mask]
    fold_seed = config.seed + fold_index * 100
    best_epoch = _select_epoch(
        genotypes[fit_indices],
        phenotypes[fit_indices],
        genotypes[validation_indices],
        phenotypes[validation_indices],
        config,
        fold_seed,
        device,
    )

    transform = fit_feature_transform(genotypes[train_indices], config, fold_seed + 1)
    train_snps, train_pcs = transform_features(genotypes[train_indices], transform)
    test_snps, test_pcs = transform_features(genotypes[test_indices], transform)
    target_mean, target_scale = _target_scale(phenotypes[train_indices])
    train_targets = ((phenotypes[train_indices] - target_mean) / target_scale).astype(
        np.float32
    )

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
    return standardized * target_scale + target_mean, best_epoch, validation_family


def run_lofo(
    dataset: SoynamDataset,
    config: ResNetConfig,
    device: torch.device,
) -> FloatArray:
    """Generate one prediction for every sample using outer family-wise CV."""
    predictions = np.full(dataset.phenotypes.size, np.nan, dtype=np.float64)
    splitter = LeaveOneGroupOut()
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(dataset.genotypes, dataset.phenotypes, dataset.family_ids)
    ):
        fold_predictions, best_epoch, validation_family = predict_resnet_fold(
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
        held_out_family = str(np.unique(dataset.family_ids[test_indices])[0])
        correlation = np.corrcoef(dataset.phenotypes[test_indices], fold_predictions)[
            0, 1
        ]
        print(
            f"{held_out_family:22s} r={correlation:.4f} "
            f"epoch={best_epoch} validation={validation_family}"
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("LOFO-CV did not produce every OOF prediction")
    return predictions


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
    dataset = load_soynam_dataset(args.data_dir)
    predictions = run_lofo(dataset, config, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_oof_frame(dataset, predictions).to_csv(
        args.output_dir / "oof_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
