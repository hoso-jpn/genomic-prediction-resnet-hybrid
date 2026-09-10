"""Shared QC policy: predeclared sample exclusions and fold-local marker fit."""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from test_cpu_smoke import _write_synthetic_family

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
