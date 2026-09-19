"""Shared QC policy: predeclared sample exclusions and fold-local marker fit."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_cpu_smoke import _write_synthetic_family

import controlled_qc
import gblup_baseline as gblup
import input_qc
from resnet_baseline import ResNetConfig, fit_feature_transform
from soynam_data import SoynamDataset


def dataset():
    return SoynamDataset(
        genotypes=np.array([[-1, np.nan], [0, 1], [np.nan, np.nan], [1, 0.0]]),
        phenotypes=np.arange(4.0),
        family_ids=np.array(["A", "A", "B", "B"]),
        sample_names=np.array(["a1", "a2", "b1", "b2"]),
        marker_names=np.array(["m1", "m2"]),
    )


def test_default_retains_all_and_audits_missingness():
    original = dataset()
    qc = input_qc.apply_sample_qc(original)
    np.testing.assert_equal(qc.dataset.genotypes, original.genotypes)
    np.testing.assert_equal(qc.arrays["input_qc_sample_missing_rate"], [0.5, 0, 1, 0])
    np.testing.assert_equal(qc.arrays["input_qc_marker_missing_rate"], [0.25, 0.5])
    assert qc.report["retained_sample_count"] == 4


def test_sample_threshold_is_inclusive_and_keeps_alignment():
    qc = input_qc.apply_sample_qc(dataset(), 0.5)
    assert list(qc.dataset.sample_names) == ["a1", "a2", "b2"]
    assert list(qc.dataset.phenotypes) == [0, 1, 3]
    assert list(qc.dataset.family_ids) == ["A", "A", "B"]
    assert qc.report["original_sample_count"] == 4


def test_eliminating_a_family_fails():
    original = dataset()
    original.genotypes[2:] = np.nan
    with pytest.raises(ValueError, match=r"every sample.*B"):
        input_qc.apply_sample_qc(original, 0.5)


@pytest.mark.parametrize("value", [-1, 2, np.nan, np.inf])
def test_invalid_threshold_fails(value):
    with pytest.raises(ValueError, match="finite"):
        input_qc.apply_sample_qc(dataset(), value)


def test_marker_threshold_uses_only_training_calls():
    train = np.array([[-1, -1], [0, np.nan], [1, 1.0]])
    heldout = np.full((2, 2), np.nan)
    strict = gblup.prepare_fold_relationships(train, heldout, min_observed_rate=0.9)
    loose = gblup.prepare_fold_relationships(train, heldout, min_observed_rate=0.5)
    assert strict.retained_markers.tolist() == [True, False]
    assert loose.retained_markers.tolist() == [True, True]
    transform = fit_feature_transform(train, ResNetConfig(min_observed_rate=1), 42)
    assert transform.retained_markers.tolist() == [True, False]


# Hand-computed MAF fixture on the internal additive scale (-1 / 0 / +1).
# Column 0 is a control that clears every threshold used below, so the
# "no markers pass" guard never fires and MAF stays the only reason a
# column is dropped.
#
#   column 0: 4x -1 and 4x +1      -> mean  0.00, MAF 0.500, observed 8/8
#   column 1: 3x -1, 1x +1, 4x NaN -> mean -0.50, MAF 0.250, observed 4/8
#   column 2: 7x -1, 1x +1         -> mean -0.75, MAF 0.125, observed 8/8
#
# MAF = min(af, 1 - af) with af = (mean + 1) / 2, the mean taken over
# observed calls only. Column 1 in dosage terms is [0, 0, 0, 2]: 2 minor
# alleles out of 2 * 4 observed calls = 0.25. Counting the four NaNs as 0
# on this scale (i.e. as heterozygotes) would give mean -0.25 and MAF
# 0.375 instead, which the thresholds below are chosen to expose.
MAF_FIXTURE = np.array(
    [
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [1.0, np.nan, -1.0],
        [1.0, np.nan, -1.0],
        [1.0, np.nan, -1.0],
        [1.0, np.nan, 1.0],
    ]
)
MAF_HELDOUT = np.zeros((2, 3))
# Observed rates are 1.0, 0.5, 1.0; 0.25 keeps every column admissible on
# the observed-rate filter so that it cannot mask a MAF decision.
OBSERVED_RATE = 0.25


@pytest.mark.parametrize(
    ("maf_threshold", "expected"),
    [
        (0.0, [True, True, True]),
        (0.2, [True, True, False]),
        (0.3, [True, False, False]),
    ],
)
def test_maf_threshold_changes_marker_admission(maf_threshold, expected):
    """Only the MAF threshold moves, and it alone decides admission."""
    relationships = gblup.prepare_fold_relationships(
        MAF_FIXTURE,
        MAF_HELDOUT,
        min_observed_rate=OBSERVED_RATE,
        maf_threshold=maf_threshold,
    )
    assert relationships.retained_markers.tolist() == expected

    transform = fit_feature_transform(
        MAF_FIXTURE,
        ResNetConfig(
            min_observed_rate=OBSERVED_RATE,
            maf_threshold=maf_threshold,
            use_pca=False,
        ),
        42,
    )
    assert transform.retained_markers.tolist() == expected


def test_maf_threshold_boundary_keeps_each_existing_operator():
    """At MAF exactly 0.25 the three paths disagree, and that is intended.

    GBLUP legacy admits on ``MAF > threshold``; ResNet legacy and the
    controlled QC admit on ``MAF >= threshold``. This pins the existing
    behaviour and must not be unified. The observed-rate filter is held
    away from its own boundary so the two comparisons stay independent.
    """
    exactly_the_column_maf = 0.25

    strict = gblup.prepare_fold_relationships(
        MAF_FIXTURE,
        MAF_HELDOUT,
        min_observed_rate=OBSERVED_RATE,
        maf_threshold=exactly_the_column_maf,
    )
    assert strict.retained_markers.tolist() == [True, False, False]

    inclusive = fit_feature_transform(
        MAF_FIXTURE,
        ResNetConfig(
            min_observed_rate=OBSERVED_RATE,
            maf_threshold=exactly_the_column_maf,
            use_pca=False,
        ),
        42,
    )
    assert inclusive.retained_markers.tolist() == [True, True, False]

    common = controlled_qc.marker_mask(
        MAF_FIXTURE,
        min_observed_rate=OBSERVED_RATE,
        maf_threshold=exactly_the_column_maf,
    )
    assert common.tolist() == [True, True, False]


def test_maf_and_imputation_means_ignore_missing_calls():
    """Missing calls are excluded from both the MAF and the training mean.

    Column 1 has MAF 0.25 over its four observed calls. Treating the four
    NaNs as 0 on the internal scale would raise it to 0.375 and keep the
    column at a 0.3 threshold, so each assertion below fails under that
    misreading.
    """
    dropped_by_maf = 0.3

    relationships = gblup.prepare_fold_relationships(
        MAF_FIXTURE,
        MAF_HELDOUT,
        min_observed_rate=OBSERVED_RATE,
        maf_threshold=dropped_by_maf,
    )
    assert relationships.retained_markers.tolist() == [True, False, False]

    transform = fit_feature_transform(
        MAF_FIXTURE,
        ResNetConfig(
            min_observed_rate=OBSERVED_RATE,
            maf_threshold=dropped_by_maf,
            use_pca=False,
        ),
        42,
    )
    assert transform.retained_markers.tolist() == [True, False, False]

    common = controlled_qc.marker_mask(
        MAF_FIXTURE,
        min_observed_rate=OBSERVED_RATE,
        maf_threshold=dropped_by_maf,
    )
    assert common.tolist() == [True, False, False]

    # The retained-marker training means carry the same exclusion: column 1
    # averages its four observed calls to -0.5, not the -0.25 that counting
    # the NaNs as 0 would give.
    admit_every_column = 0.0
    kept = gblup.prepare_fold_relationships(
        MAF_FIXTURE,
        MAF_HELDOUT,
        min_observed_rate=OBSERVED_RATE,
        maf_threshold=admit_every_column,
    )
    np.testing.assert_allclose(kept.marker_means, [0.0, -0.5, -0.75])

    kept_transform = fit_feature_transform(
        MAF_FIXTURE,
        ResNetConfig(
            min_observed_rate=OBSERVED_RATE,
            maf_threshold=admit_every_column,
            use_pca=False,
        ),
        42,
    )
    np.testing.assert_allclose(kept_transform.marker_means, [0.0, -0.5, -0.75])


@pytest.mark.parametrize("script", ["gblup_baseline.py", "resnet_baseline.py"])
def test_cli_records_qc_in_existing_artifact_bundle(tmp_path, script):
    data = tmp_path / "data"
    data.mkdir()
    for family in range(1, 4):
        _write_synthetic_family(data, family)
    output = tmp_path / "output"
    root = Path(__file__).resolve().parents[1]
    extra = (
        ["--expected-families", "3"]
        if script.startswith("gblup")
        else [
            "--device",
            "cpu",
            "--max-epochs",
            "1",
            "--patience",
            "1",
            "--pca-components",
            "2",
        ]
    )
    result = subprocess.run(
        [
            sys.executable,
            str(root / script),
            "--data-dir",
            str(data),
            "--output-dir",
            str(output),
            "--max-sample-missing-rate",
            "1",
            "--min-marker-observed-rate",
            ".8",
            *extra,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    run = next((output / "artifacts").iterdir())
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["hyperparameters"]["min_observed_rate"] == 0.8
    qc = metadata["input_qc"]
    assert qc["original_sample_count"] == qc["retained_sample_count"] == 18
    with np.load(run / "preprocessing_arrays.npz") as arrays:
        for reference in qc["arrays"].values():
            assert reference in arrays
    assert len(list(run.iterdir())) == 6
