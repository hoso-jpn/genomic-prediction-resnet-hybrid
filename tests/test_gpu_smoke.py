"""CUDA smoke tests for the verified ResNet baseline.

The CUDA cases are skipped unless a GPU is actually present, so this file
is safe to run in the CPU-only CI; a skipped run is never evidence that
the GPU path works. The "CUDA requested but unavailable" case is the
opposite: it only runs where there is no GPU, which is exactly where the
failure has to stay explicit instead of silently falling back to CPU.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

# pytest puts this file's directory on sys.path, so the synthetic fixture
# writer is shared with the CPU smoke test rather than duplicated.
from test_cpu_smoke import (
    OOF_COLUMNS,
    REPO_ROOT,
    REQUIRED_ARTIFACT_FILES,
    REQUIRED_METADATA_KEYS,
    _write_synthetic_family,
)

CUDA_AVAILABLE = torch.cuda.is_available()
# The GPU case is skipped on CPU-only machines, which means a green run
# proves nothing about the GPU path. Set GPRH_REQUIRE_CUDA=1 (the compose
# `gpu-smoke` service does) to turn that skip into a failure, so a GPU job
# that silently landed on a CPU-only host cannot look successful.
REQUIRE_CUDA = os.environ.get("GPRH_REQUIRE_CUDA") == "1"


def _run_resnet(arguments: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("WANDB_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-u", str(REPO_ROOT / "resnet_baseline.py"), *arguments],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


@pytest.mark.skipif(CUDA_AVAILABLE, reason="requires a machine without CUDA")
def test_cuda_request_fails_when_cuda_is_unavailable(tmp_path: Path) -> None:
    result = _run_resnet(
        ["--data-dir", str(tmp_path), "--device", "cuda"], cwd=tmp_path
    )

    assert result.returncode != 0
    assert "CUDA was requested but is unavailable" in result.stderr
    # The run must not quietly continue on the CPU instead.
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.skipif(
    not CUDA_AVAILABLE and not REQUIRE_CUDA, reason="requires a CUDA-capable GPU"
)
def test_resnet_cuda_smoke(tmp_path: Path) -> None:
    if not CUDA_AVAILABLE:
        pytest.fail("GPRH_REQUIRE_CUDA=1 was set but no CUDA device is available")

    data_dir = tmp_path / "data"
    output_dir = tmp_path / "output"
    data_dir.mkdir()

    expected_families = {
        _write_synthetic_family(data_dir, family_index) for family_index in range(1, 4)
    }

    result = _run_resnet(
        [
            "--data-dir",
            str(data_dir),
            "--output-dir",
            str(output_dir),
            "--device",
            "cuda",
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
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    output = pd.read_csv(output_dir / "oof_predictions.csv")
    assert output.columns.tolist() == OOF_COLUMNS
    assert len(output) == 18
    assert set(output["family_id"]) == expected_families
    assert output["sample_name"].is_unique
    assert np.isfinite(
        output[["observed_yield_kg_ha", "predicted_yield_kg_ha"]].to_numpy()
    ).all()

    run_dirs = sorted((output_dir / "artifacts").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for name in REQUIRED_ARTIFACT_FILES:
        assert (run_dir / name).is_file(), name

    metadata = json.loads((run_dir / "metadata.json").read_text())
    for key in REQUIRED_METADATA_KEYS:
        assert key in metadata, key
    # The device that actually ran the job is recorded, not the request
    # alone: a silent CPU fallback would be visible here.
    assert metadata["device_requested"] == "cuda"
    assert metadata["device_resolved"].startswith("cuda")
    assert metadata["cuda_version"] is not None
    assert metadata["gpu_name"]
    assert metadata["gpu_compute_capability"]

    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert len(metrics["folds"]) == 3
