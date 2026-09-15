"""All individual datasets in this module are synthetic and redistributable."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_adzuki_gs_panel_data import _write_panel

import gs_dataset
import gs_evaluate
import resnet_baseline as resnet
import run_manifest


def refresh_checksums(path):
    manifest = json.loads(path.read_text())
    for key in ("panel_manifest", "sample_metadata", "phenotypes"):
        p = path.parent / manifest[key]
        manifest["checksums"][p.name] = "sha256:" + run_manifest.sha256_file(p)
    path.write_text(json.dumps(manifest))


@pytest.fixture
def individual_dataset(tmp_path):
    samples = [f"00{i}" for i in range(12)]
    rng = np.random.default_rng(27)
    variants = [f"Chr1:{i + 1}:A:T" for i in range(20)]
    _write_panel(
        tmp_path / "panel",
        samples=samples,
        variants=variants,
        rows=rng.integers(-1, 2, (20, 12)).astype(str).tolist(),
        manifest_overrides={"checksums": {"synthetic.fa": "sha256:" + "a" * 64}},
    )
    pd.DataFrame(
        {
            "sample_id": samples,
            "line_id": [f"L{i}" for i in range(12)],
            "family_id": [f"F{i // 4}" for i in range(12)],
        }
    ).to_csv(tmp_path / "samples.tsv", sep="\t", index=False)
    pd.DataFrame(
        {
            "observation_id": [f"O{i}" for i in range(12)],
            "sample_id": samples,
            "trait": "seed_mass",
            "unit": "g",
            "year": 2025,
            "site": "A",
            "replicate": "1",
            "value": [i % 4 + i / 10 for i in range(12)],
        }
    ).to_csv(tmp_path / "phenotypes.tsv", sep="\t", index=False)
    manifest = {
        "schema_version": 1,
        "kind": "gs_individual_dataset",
        "species": "Vigna angularis",
        "cohort_id": "cohort",
        "panel_manifest": "panel/cohort.gs_panel.manifest.json",
        "sample_metadata": "samples.tsv",
        "phenotypes": "phenotypes.tsv",
        "reference": {
            "assembly_id": "synthetic-v1",
            "fasta_file": "synthetic.fa",
            "fasta_sha256": "sha256:" + "a" * 64,
        },
        "effect_allele": "ALT",
        "trait": {
            "name": "seed_mass",
            "unit": "g",
            "method": "weigh",
            "scale": "mass",
            "type": "continuous",
            "direction": "higher",
        },
        "aggregation": {"method": "none", "measurement_type": "raw"},
        "checksums": {},
    }
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(manifest))
    refresh_checksums(path)
    return path


def test_ids_reorder_by_key_and_missing_are_audited(individual_dataset):
    path = individual_dataset
    p = path.parent / "phenotypes.tsv"
    frame = pd.read_csv(p, sep="\t", dtype=str)
    frame.loc[0, "value"] = "NA"
    frame.iloc[::-1].to_csv(p, sep="\t", index=False)
    refresh_checksums(path)
    dataset = gs_dataset.load_dataset(path)
    assert dataset.excluded_samples == ("000",)
    assert dataset.excluded_observations == ("O0",)
    assert dataset.observations.iloc[0].sample_id == "0011"
    np.testing.assert_equal(dataset.baseline.genotypes[0], dataset.panel.genotypes[11])
    assert dataset.provenance["aggregation_hash"]


@pytest.mark.parametrize(
    "column,value,match",
    [
        ("sample_id", "unknown", "sample ID"),
        ("unit", "kg", "unit mismatch"),
        ("observation_id", "O1", "duplicate observation_id"),
        ("value", "inf", "infinite"),
        ("year", "future", "year"),
    ],
)
def test_invalid_observations_are_rejected(individual_dataset, column, value, match):
    path = individual_dataset
    p = path.parent / "phenotypes.tsv"
    frame = pd.read_csv(p, sep="\t", dtype=str)
    frame.loc[0, column] = value
    frame.to_csv(p, sep="\t", index=False)
    refresh_checksums(path)
    with pytest.raises(ValueError, match=match):
        gs_dataset.load_dataset(path)


@pytest.mark.parametrize(
    "change,match",
    [
        ({"reference": {"assembly_id": "unknown"}}, "unknown assembly"),
        ({"effect_allele": "REF"}, "effect_allele"),
        ({"aggregation": {"method": "BLUE"}}, "only raw"),
        ({"kind": "summary_statistics"}, "individual dataset schema"),
    ],
)
def test_manifest_rejection(individual_dataset, change, match):
    content = json.loads(individual_dataset.read_text())
    content.update(change)
    individual_dataset.write_text(json.dumps(content))
    with pytest.raises(ValueError, match=match):
        gs_dataset.load_dataset(individual_dataset)


def test_checksum_and_all_missing_rejected(individual_dataset):
    path = individual_dataset
    p = path.parent / "phenotypes.tsv"
    frame = pd.read_csv(p, sep="\t", dtype=str)
    frame["value"] = "NA"
    frame.to_csv(p, sep="\t", index=False)
    with pytest.raises(ValueError, match="checksum"):
        gs_dataset.load_dataset(path)
    refresh_checksums(path)
    with pytest.raises(ValueError, match="all missing"):
        gs_dataset.load_dataset(path)


def test_generic_trait_runs_both_models_with_auditable_artifacts(individual_dataset):
    dataset = gs_dataset.load_dataset(individual_dataset)
    config = resnet.ResNetConfig(
        max_epochs=1, hidden_dim=4, num_blocks=1, pca_components=2
    )
    plan = gs_evaluate.make_plan(dataset, config)
    output = individual_dataset.parent / "evaluation"
    result = gs_evaluate.evaluate(dataset, plan, output)
    assert len(result) == 24
    assert set(result.model) == {"gblup", "resnet"}
    assert set(result.trait) == {"seed_mass"}
    assert np.isfinite(result.predicted).all()
    assert json.loads((output / "split.json").read_text()) == plan
    assert (output / "preprocessing.npz").is_file()
    changed = replace(
        dataset, provenance={**dataset.provenance, "dataset_hash": "changed"}
    )
    with pytest.raises(ValueError, match="split plan"):
        gs_evaluate.evaluate(changed, plan, output.parent / "invalid")


def test_duplicate_measurement_key_rejected(individual_dataset):
    path = individual_dataset.parent / "phenotypes.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str)
    frame.loc[1, "sample_id"] = frame.loc[0, "sample_id"]
    frame.to_csv(path, sep="\t", index=False)
    refresh_checksums(individual_dataset)
    with pytest.raises(ValueError, match="duplicate observation key"):
        gs_dataset.load_dataset(individual_dataset)


def test_duplicate_line_key_is_rejected_before_missing_value_exclusion(
    individual_dataset,
):
    samples_path = individual_dataset.parent / "samples.tsv"
    samples = pd.read_csv(samples_path, sep="\t", dtype=str)
    samples.loc[1, "line_id"] = samples.loc[0, "line_id"]
    samples.to_csv(samples_path, sep="\t", index=False)
    phenotypes_path = individual_dataset.parent / "phenotypes.tsv"
    phenotypes = pd.read_csv(phenotypes_path, sep="\t", dtype=str)
    phenotypes.loc[1, "value"] = "NA"
    phenotypes.to_csv(phenotypes_path, sep="\t", index=False)
    refresh_checksums(individual_dataset)
    with pytest.raises(ValueError, match="duplicate line observation key"):
        gs_dataset.load_dataset(individual_dataset)


def test_allele_metadata_must_agree_with_key(individual_dataset):
    path = individual_dataset.parent / "panel/cohort.gs_panel.variant_metadata.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str)
    frame.loc[0, "alt"] = "G"
    frame.to_csv(path, sep="\t", index=False)
    panel_path = path.parent / "cohort.gs_panel.manifest.json"
    panel = json.loads(panel_path.read_text())
    panel["checksums"][path.name] = "sha256:" + run_manifest.sha256_file(path)
    panel_path.write_text(json.dumps(panel))
    refresh_checksums(individual_dataset)
    with pytest.raises(ValueError, match="allele metadata mismatch"):
        gs_dataset.load_dataset(individual_dataset)
