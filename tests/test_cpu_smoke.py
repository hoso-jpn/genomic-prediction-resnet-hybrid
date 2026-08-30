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
COMMON_METADATA_KEYS = (
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
    "input_files",
    "families",
    "split_ref",
    "preprocessing_ref",
    "preprocessing_arrays_ref",
    "metrics_ref",
    "predictions_ref",
)
REQUIRED_METADATA_KEYS = COMMON_METADATA_KEYS + (
    "device_requested",
    "device_resolved",
    "cuda_version",
    "cudnn_version",
)
GBLUP_METADATA_KEYS = COMMON_METADATA_KEYS + ("external_logging",)
OOF_COLUMNS = [
    "family_id",
    "sample_name",
    "observed_yield_kg_ha",
    "predicted_yield_kg_ha",
]
REPO_ROOT = Path(__file__).resolve().parent.parent


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


def _run_gblup_cli(
    arguments: list[str],
    *,
    cwd: Path,
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the GBLUP CLI from `cwd` with no W&B credentials in the environment.

    The script is addressed by absolute path so `cwd` can be a scratch
    directory: that keeps any file the run would create (a W&B run
    directory in particular) inside the test's own tmp_path instead of the
    repository, where a checked-in `wandb/` directory would hide it.
    """
    environment = os.environ.copy()
    environment.pop("WANDB_API_KEY", None)
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        }
    )
    if extra_environment:
        environment.update(extra_environment)
    return subprocess.run(
        [sys.executable, "-u", str(REPO_ROOT / "gblup_baseline.py"), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def test_gblup_cli_cpu_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    run_dir_cwd = tmp_path / "cwd"
    data_dir.mkdir()
    run_dir_cwd.mkdir()

    expected_families = {
        _write_synthetic_family(data_dir, family_index) for family_index in range(1, 4)
    }

    # WANDB_MODE=online in the surrounding environment must not turn this
    # run into an online one: the CLI default (disabled) decides.
    result = _run_gblup_cli(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--expected-families",
            "3",
        ],
        cwd=run_dir_cwd,
        extra_environment={"WANDB_MODE": "online"},
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # A disabled run initializes no W&B client, so no local run directory
    # is created next to the process working directory either.
    assert not (run_dir_cwd / "wandb").exists()
    assert list(run_dir_cwd.iterdir()) == []

    output_path = output_dir / "oof_predictions.csv"
    assert output_path.is_file(), result.stdout + result.stderr

    output = pd.read_csv(output_path)
    assert output.columns.tolist() == OOF_COLUMNS
    assert len(output) == 18
    assert set(output["family_id"]) == expected_families
    assert output["sample_name"].is_unique
    assert output.notna().all().all()
    assert np.isfinite(
        output[["observed_yield_kg_ha", "predicted_yield_kg_ha"]].to_numpy()
    ).all()

    run_dirs = sorted((output_dir / "artifacts").iterdir())
    assert len(run_dirs) == 1, result.stdout + result.stderr
    run_dir = run_dirs[0]

    for name in REQUIRED_ARTIFACT_FILES:
        assert (run_dir / name).is_file(), name

    metadata = json.loads((run_dir / "metadata.json").read_text())
    for key in GBLUP_METADATA_KEYS:
        assert key in metadata, key
    assert metadata["model_name"] == "gblup"
    assert metadata["run_id"] == run_dir.name
    assert sorted(metadata["families"]) == sorted(expected_families)
    assert metadata["external_logging"] == {"backend": "wandb", "mode": "disabled"}
    assert metadata["hyperparameters"]["expected_family_count"] == 3
    # The recorded command keeps the CLI options but never the paths they
    # pointed at (run_manifest.sanitize_command).
    assert metadata["command"]["arguments"] == [
        "--data-dir",
        data_dir.name,
        "--output-dir",
        output_dir.name,
        "--expected-families",
        "3",
    ]
    assert len(metadata["input_files"]) == 3

    split = json.loads((run_dir / "split.json").read_text())
    assert split["outer"]["strategy"] == "leave_one_family_out"
    assert len(split["outer"]["folds"]) == 3
    assert split["inner"] is None

    preprocessing = json.loads((run_dir / "preprocessing.json").read_text())
    assert len(preprocessing["folds"]) == 3
    assert {fold["held_out_family"] for fold in preprocessing["folds"]} == (
        expected_families
    )

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(metrics["folds"]) == 3
    assert {fold["held_out_family"] for fold in metrics["folds"]} == expected_families

    with np.load(run_dir / "preprocessing_arrays.npz") as arrays:
        for fold in preprocessing["folds"]:
            for array_ref in fold["arrays"].values():
                assert array_ref in arrays

    run_predictions = pd.read_csv(run_dir / "predictions.csv")
    assert run_predictions.columns.tolist() == OOF_COLUMNS
    pd.testing.assert_frame_equal(run_predictions, output)


def test_gblup_cli_family_count_mismatch_keeps_existing_artifacts(
    tmp_path: Path,
) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()
    output_dir.mkdir()

    for family_index in range(1, 4):
        _write_synthetic_family(data_dir, family_index)

    existing_csv = output_dir / "oof_predictions.csv"
    existing_csv.write_text("previous run output\n")

    result = _run_gblup_cli(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--expected-families",
            "16",
        ],
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "expected 16 families, found 3" in result.stderr
    assert existing_csv.read_text() == "previous run output\n"
    assert not (output_dir / "artifacts").exists()


def test_gblup_cli_rejects_family_count_below_two(tmp_path: Path) -> None:
    result = _run_gblup_cli(
        ["--data-dir", str(tmp_path), "--expected-families", "1"], cwd=tmp_path
    )

    assert result.returncode == 2
    assert "--expected-families must be at least 2" in result.stderr
    # Argument validation happens before any input is read, so a missing
    # data directory is never even reported.
    assert "no phenotype files" not in result.stderr
