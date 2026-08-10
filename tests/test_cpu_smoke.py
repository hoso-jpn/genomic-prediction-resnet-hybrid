import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from soynam_data import GENOTYPE_SUFFIX, PHENOTYPE_SUFFIX


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
