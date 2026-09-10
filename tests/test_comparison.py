"""Execute real baseline CLIs and reject mismatched paired experiments."""

import argparse
import gzip
import json

import numpy as np
import pytest
from test_cpu_smoke import _write_synthetic_family

import compare_baselines as comparison
import evaluation_split
import run_manifest


def arguments(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for family in range(1, 4):
        _write_synthetic_family(data, family)
    return argparse.Namespace(
        data_dir=data,
        experiment_dir=tmp_path / "experiment",
        dataset_id="synthetic-three-family",
        data_source="test_cpu_smoke fixture",
        data_kind="synthetic",
        seeds=[42, 43],
        device="cpu",
        max_epochs=1,
        patience=1,
        batch_size=8,
        pca_components=2,
        threads=1,
        max_sample_missing_rate=1.0,
        gblup_marker_rate=0.1,
        resnet_marker_rate=0.9,
    )


def test_fixed_split_roundtrip_and_tampering(tmp_path):
    args = arguments(tmp_path)
    comparison.plan_experiment(args)
    data, checksums = comparison.load_data(args.data_dir, 1.0)
    path = args.experiment_dir / "split-plan.json"
    splits, _ = evaluation_split.load_plan(path, data, checksums)
    seen = []
    for train, test in splits:
        assert not set(data.family_ids[train]) & set(data.family_ids[test])
        seen.extend(test.tolist())
    assert sorted(seen) == list(range(len(data.phenotypes)))
    original = json.loads(path.read_text())
    tampered = {**original, "max_sample_missing_rate": 0.5}
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="hash"):
        evaluation_split.load_plan(path, data, checksums)
    tampered["plan_hash"] = run_manifest.canonical_json_hash(
        {k: v for k, v in tampered.items() if k != "plan_hash"}
    )
    path.write_text(json.dumps(tampered))
    with pytest.raises(ValueError, match="QC"):
        evaluation_split.load_plan(path, data, checksums)
    path.write_text(json.dumps(original))
    data.sample_names[:] = data.sample_names[::-1]
    with pytest.raises(ValueError, match="sample order"):
        evaluation_split.load_plan(path, data, checksums)


def test_comparison_runs_both_clis_and_rejects_corrupted_oof(tmp_path):
    args = arguments(tmp_path)
    config = comparison.plan_experiment(args)
    report = comparison.run_experiment(args.experiment_dir, args.data_dir)
    assert report["status"] == "complete"
    assert report["experiment"]["data_kind"] == "synthetic"
    assert [record["seed"] for record in report["runs"]] == [None, 42, 43]
    assert (args.experiment_dir / "comparison.md").is_file()
    assert report["paired_family_bootstrap"]["rmse_kg_ha"]["families"] == 3
    for record in report["runs"]:
        assert len(record["families"]) == 3
        assert record["measurements"]["wall_seconds"] > 0
        assert record["measurements"]["cuda_peak_allocated_bytes"] is None
        assert np.isfinite(record["pooled"]["rmse_kg_ha"])
    assert comparison.report_experiment(args.experiment_dir, args.data_dir) == report
    with pytest.raises(FileExistsError):
        comparison.run_experiment(args.experiment_dir, args.data_dir)
    bundle = next(
        (args.experiment_dir / "runs" / "resnet-seed-42" / "artifacts").iterdir()
    )
    metadata = json.loads((bundle / "metadata.json").read_text())
    assert metadata["split_plan"]["plan_hash"] == config["split_plan_hash"]
    assert str(tmp_path) not in json.dumps(metadata["command"])
    metadata_path = bundle / "metadata.json"
    for field, value, message in [
        ("source_file_checksums", {}, "source checksum"),
        ("library_versions", {}, "environment mismatch"),
        (
            "hyperparameters",
            {**metadata["hyperparameters"], "learning_rate": 0.5},
            "hyperparameter mismatch",
        ),
    ]:
        metadata_path.write_text(json.dumps({**metadata, field: value}))
        with pytest.raises(ValueError, match=message):
            comparison.report_experiment(args.experiment_dir, args.data_dir)
    metadata_path.write_text(json.dumps(metadata))
    predictions = bundle / "predictions.csv"
    predictions.write_text(predictions.read_text().replace("F01_RIL_000", "unknown", 1))
    with pytest.raises(ValueError, match="OOF identity"):
        comparison.report_experiment(args.experiment_dir, args.data_dir)


def test_input_changed_before_run_fails_without_creating_runs(tmp_path):
    args = arguments(tmp_path)
    comparison.plan_experiment(args)
    genotype = next(args.data_dir.glob("*genotype*"))
    content = gzip.decompress(genotype.read_bytes())
    genotype.write_bytes(gzip.compress(content + b"\n"))
    with pytest.raises(ValueError, match="checksums"):
        comparison.run_experiment(args.experiment_dir, args.data_dir)
    assert not (args.experiment_dir / "runs").exists()


@pytest.mark.parametrize(
    "field,value", [("seeds", [42, 42]), ("max_epochs", 0), ("seeds", [-1, 42])]
)
def test_invalid_plan_is_rejected_before_writing(tmp_path, field, value):
    args = arguments(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        comparison.plan_experiment(args)
    assert not args.experiment_dir.exists()


def test_split_command_redaction():
    command = run_manifest.sanitize_command(
        "resnet_baseline.py",
        [
            "--split-file",
            "/private/project/split.json",
            "--split-file=/private/split2.json",
        ],
    )
    assert "/private" not in json.dumps(command)
    assert "split2.json" in json.dumps(command)


def test_bootstrap_reports_undefined_correlations():
    records = [
        {"families": {"A": {"r": None, "rmse_kg_ha": value}}}
        for value in [3.0, 2.0, 2.0]
    ]
    result = comparison.paired_uncertainty(
        records, {"bootstrap": {"seed": 1, "replicates": 100}}
    )
    assert result["r"]["mean_difference"] is None
    assert result["rmse_kg_ha"]["mean_difference"] == -1
