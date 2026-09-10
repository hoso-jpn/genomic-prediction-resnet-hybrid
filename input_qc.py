"""Predeclared sample QC and descriptive input missingness for both baselines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np

from soynam_data import SoynamDataset


def fraction(value: str) -> float:
    """Parse a finite proportion, including both endpoints."""
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("must be a finite value in [0, 1]")
    return number


def add_arguments(parser: argparse.ArgumentParser, *, marker_default: float) -> None:
    parser.add_argument(
        "--max-sample-missing-rate",
        type=fraction,
        default=1.0,
        help="exclude samples above this missing-call fraction (default: retain all)",
    )
    parser.add_argument(
        "--min-marker-observed-rate",
        type=fraction,
        default=marker_default,
        help="minimum observed fraction fitted separately on each training fold",
    )


@dataclass(frozen=True)
class InputQc:
    dataset: SoynamDataset
    report: dict[str, Any]
    arrays: dict[str, np.ndarray]


def apply_sample_qc(dataset: SoynamDataset, max_missing_rate: float = 1.0) -> InputQc:
    """Filter by each sample's own calls, without fitting a cohort statistic.

    Marker rates over all phenotyped RILs are descriptive only. Marker selection,
    mean imputation, and all learned transforms remain inside training folds.
    A threshold that eliminates a family fails rather than changing the CV task.
    """
    if not np.isfinite(max_missing_rate) or not 0 <= max_missing_rate <= 1:
        raise ValueError("max_missing_rate must be finite and in [0, 1]")
    if dataset.genotypes.ndim != 2 or 0 in dataset.genotypes.shape:
        raise ValueError("missingness QC requires a nonempty sample-by-marker matrix")
    missing = ~np.isfinite(dataset.genotypes)
    sample_rates = missing.mean(axis=1)
    marker_rates = missing.mean(axis=0)
    retained = sample_rates <= max_missing_rate
    lost_families = sorted(set(dataset.family_ids) - set(dataset.family_ids[retained]))
    if lost_families:
        raise ValueError(f"sample QC removes every sample in families: {lost_families}")
    filtered = SoynamDataset(
        genotypes=dataset.genotypes[retained],
        phenotypes=dataset.phenotypes[retained],
        family_ids=dataset.family_ids[retained],
        sample_names=dataset.sample_names[retained],
        marker_names=dataset.marker_names,
    )
    report = {
        "scope": "phenotyped RILs after loader exclusions, before sample QC",
        "max_sample_missing_rate": max_missing_rate,
        "original_sample_count": len(retained),
        "retained_sample_count": int(retained.sum()),
        "original_samples": [
            {"sample_id": str(sample), "family_id": str(family)}
            for sample, family in zip(dataset.sample_names, dataset.family_ids)
        ],
        "marker_order": "input genotype file order; see input_files checksums",
        "marker_rates_usage": "descriptive only; never used for marker selection",
        "arrays": {
            "sample_missing_rate": "input_qc_sample_missing_rate",
            "sample_retained": "input_qc_sample_retained",
            "marker_missing_rate": "input_qc_marker_missing_rate",
        },
    }
    arrays = {
        "input_qc_sample_missing_rate": sample_rates,
        "input_qc_sample_retained": retained,
        "input_qc_marker_missing_rate": marker_rates,
    }
    return InputQc(filtered, report, arrays)
