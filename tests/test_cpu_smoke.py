import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from soynam_data import GENOTYPE_SUFFIX, PHENOTYPE_SUFFIX

REQUIRED_ARTIFACT_FILES = (
    "metadata.json",
    "split.json",
    "preprocessing.json",
    "preprocessing_arrays.npz",
    "metrics.json",
    "predictions.csv",
)
REQUIRED_METADATA_KEYS = (
    "schema_version",
    "run_id",
    "created_at",
    "model_name",
    "git_commit",
    "source_file_checksums",
    "command",
    "seed",
    "python_version",
    "library_versions",
    "hyperparameters",
    "device_requested",
    "device_resolved",
    "cuda_version",
    "cudnn_version",
    "input_files",
    "families",
    "split_ref",
    "preprocessing_ref",
    "preprocessing_arrays_ref",
    "metrics_ref",
    "predictions_ref",
)


def _write_synthetic_family(
    data_dir: Path,
    family_index: int,
    marker_count: int = 12,
    sample_count: int = 6,
) -> str:
    founder = f"Founder{family_index}"
    family_id = f"{founder}_NAM{family_index:02d}"
    sample_names = [
        f"F{family_index:02d}_RIL_{index:03d}" for index in range(sample_count)
    ]
    marker_names = [f"marker_{index:03d}" for index in range(marker_count)]

    genotype_values = np.asarray(
        [
            [
                (-1.0, 0.0, 1.0)[(sample_index + marker_index + family_index) % 3]
                for sample_index in range(sample_count)
            ]
            for marker_index in range(marker_count)
        ]
    )
    genotype_symbols = np.select(
        [
            genotype_values == -1.0,
            genotype_values == 0.0,
            genotype_values == 1.0,
        ],
        ["A", "A/B", "B"],
    )

    phenotype = pd.DataFrame(
        {
            "Corrected Strain": sample_names,
            "Yld (kg/ha)": [
                3000.0 + family_index * 100.0 + sample_index * 25.0
                for sample_index in range(sample_count)
            ],
        }
    )
    genotype = pd.DataFrame(
        genotype_symbols,
        index=marker_names,
        columns=sample_names,
    )
    genotype.index.name = "marker"

    phenotype.to_csv(
        data_dir / f"{family_id}{PHENOTYPE_SUFFIX}",
        sep="\t",
        index=False,
        compression="gzip",
    )
    genotype.to_csv(
        data_dir / f"{family_id}_4312{GENOTYPE_SUFFIX}",
        sep="\t",
        compression="gzip",
    )
    return family_id


def test_resnet_cli_cpu_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()

    expected_families = {
        _write_synthetic_family(data_dir, family_index) for family_index in range(1, 4)
    }

    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-u",
            "resnet_baseline.py",
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--seed",
            "42",
            "--max-epochs",
            "1",
            "--patience",
            "1",
            "--batch-size",
            "4",
            "--pca-components",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    output_path = output_dir / "oof_predictions.csv"
    assert output_path.is_file(), result.stdout + result.stderr

    output = pd.read_csv(output_path)
    assert output.columns.tolist() == [
        "family_id",
        "sample_name",
        "observed_yield_kg_ha",
        "predicted_yield_kg_ha",
    ]
    assert len(output) == 18
    assert set(output["family_id"]) == expected_families
    assert output["sample_name"].is_unique
    assert output.notna().all().all()
    assert np.isfinite(
        output[
            [
                "observed_yield_kg_ha",
                "predicted_yield_kg_ha",
            ]
        ].to_numpy()
    ).all()

    # No W&B credentials were set in `environment`, and none were required
    # above: the run artifacts below are written by run_manifest.py alone.
    run_dirs = sorted((output_dir / "artifacts").iterdir())
    assert len(run_dirs) == 1, result.stdout + result.stderr
    run_dir = run_dirs[0]

    for name in REQUIRED_ARTIFACT_FILES:
        assert (run_dir / name).is_file(), name

    metadata = json.loads((run_dir / "metadata.json").read_text())
    for key in REQUIRED_METADATA_KEYS:
        assert key in metadata, key
    assert metadata["model_name"] == "resnet"
    assert metadata["run_id"] == run_dir.name
    assert len(metadata["families"]) == 3
    assert metadata["device_requested"] == "cpu"
    assert metadata["device_resolved"] == "cpu"
    assert metadata["cuda_version"] is None
    assert metadata["cudnn_version"] is None

    split = json.loads((run_dir / "split.json").read_text())
    assert split["outer"]["strategy"] == "leave_one_family_out"
    assert len(split["outer"]["folds"]) == 3
    assert split["inner"]["strategy"] == "validation_family_selection"

    preprocessing = json.loads((run_dir / "preprocessing.json").read_text())
    assert len(preprocessing["folds"]) == 3
    first_fold = preprocessing["folds"][0]
    assert "selection_transform" in first_fold
    assert "final_transform" in first_fold

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(metrics["folds"]) == 3

    with np.load(run_dir / "preprocessing_arrays.npz") as arrays:
        selection_ref = first_fold["selection_transform"]["arrays"]["marker_mask_ref"]
        final_ref = first_fold["final_transform"]["arrays"]["marker_mask_ref"]
        assert selection_ref in arrays
        assert final_ref in arrays

    run_predictions = pd.read_csv(run_dir / "predictions.csv")
    assert run_predictions.columns.tolist() == [
        "family_id",
        "sample_name",
        "observed_yield_kg_ha",
        "predicted_yield_kg_ha",
    ]
    assert len(run_predictions) == len(output)
