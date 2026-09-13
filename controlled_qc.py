"""Shared train-only marker admission for controlled model comparisons."""

import numpy as np

import run_manifest


def marker_mask(genotypes, *, min_observed_rate, maf_threshold):
    values = np.asarray(genotypes, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2 or np.isinf(values).any():
        raise ValueError("QC needs at least two rows and no infinite genotypes")
    if not 0 < min_observed_rate <= 1 or not 0 <= maf_threshold < 0.5:
        raise ValueError("invalid common QC thresholds")
    observed = np.isfinite(values)
    counts = observed.sum(axis=0)
    means = np.divide(
        np.nansum(values, axis=0),
        counts,
        out=np.zeros(values.shape[1]),
        where=counts > 0,
    )
    imputed = np.where(observed, values, means)
    maf = np.minimum((means + 1) / 2, (1 - means) / 2)
    mask = (
        (counts > 0)
        & (counts / len(values) >= min_observed_rate)
        & (maf >= maf_threshold)
        & (imputed.var(axis=0) > 1e-6)
    )
    if not mask.any():
        raise ValueError("no markers pass common training-only QC")
    return mask


def mask_record(marker_ids, mask, *, min_observed_rate, maf_threshold):
    ids = np.asarray(marker_ids)[mask].tolist()
    return {
        "marker_ids": ids,
        "marker_hash": run_manifest.canonical_json_hash(ids),
        "observed_rate": {"operator": ">=", "threshold": min_observed_rate},
        "maf": {"operator": ">=", "threshold": maf_threshold},
        "variance": {"operator": ">", "threshold": 1e-6},
    }


def ridge_predict(train, test, targets, mask, *, alpha=1.0):
    """Fixed-alpha centered ridge, solved in sample space; no outer selection."""
    x, z = train[:, mask], test[:, mask]
    means = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), means, x) - means
    z = np.where(np.isnan(z), means, z) - means
    intercept = float(np.mean(targets))
    coefficients = np.linalg.solve(
        x @ x.T + alpha * np.eye(len(x)), targets - intercept
    )
    return intercept + z @ x.T @ coefficients
