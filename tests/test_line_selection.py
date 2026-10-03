"""Bounded, synthetic-only line-mean contract and held-out leakage regressions."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from test_gs_dataset import individual_dataset as individual_dataset
from test_gs_dataset import refresh_checksums

import gs_dataset
import gs_evaluate
import gs_model_bundle
import gs_selection
import resnet_baseline as resnet


def line_contract():
    return {
        "method": "within_trial_line_mean",
        "measurement_type": "raw",
        "trial": {"year": 2025, "site": "A"},
        "replicate_weighting": "equal",
        "line_weighting": "equal",
        "missing_policy": "exclude_observation",
        "unbalanced_replicates": "allow",
        "genotype_policy": "identical_including_missing",
    }


def enable_line_mode(path):
    manifest = json.loads(path.read_text())
    manifest.update(
        schema_version=2, selection_unit="line", aggregation=line_contract()
    )
    path.write_text(json.dumps(manifest))
    refresh_checksums(path)


def add_replicate(path, index=0, value="12.0"):
    phenotype = path.parent / "phenotypes.tsv"
    frame = pd.read_csv(phenotype, sep="\t", dtype=str)
    row = frame.iloc[index].copy()
    row["observation_id"] += "_rep2"
    row["replicate"] = "2"
    row["value"] = value
    pd.concat([frame, row.to_frame().T], ignore_index=True).to_csv(
        phenotype, sep="\t", index=False
    )
    refresh_checksums(path)


def test_issue_example_selects_two_lines_without_changing_observation_metrics():
    raw = pd.DataFrame(
        {
            "observation_id": ["A_rep1", "A_rep2", "B_rep1", "C_rep1"],
            "sample_id": ["A", "A", "B", "C"],
            "line_id": ["A", "A", "B", "C"],
            "family_id": ["F1", "F1", "F2", "F3"],
            "value": [10.0, 10.0, 9.0, 8.0],
            "year": 2025,
            "site": "A",
            "replicate": ["1", "2", "1", "1"],
        }
    )
    metadata = raw[["sample_id", "line_id", "family_id"]].drop_duplicates()
    panel = SimpleNamespace(
        sample_ids=np.array(["A", "B", "C"]),
        genotypes=np.array([[-1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
    )
    manifest = {
        "aggregation": line_contract(),
        "trait": {"direction": "higher"},
        "species": "synthetic",
        "cohort_id": "synthetic",
    }
    aggregated, processing = gs_dataset.aggregate_lines(
        raw, metadata, panel, manifest, source_count=4, missing_count=0
    )
    dataset = SimpleNamespace(
        selection_unit="line", phenotype_processing=processing, manifest=manifest
    )
    predictions = aggregated.rename(columns={"value": "observed"}).assign(
        model="gblup", predicted=[10.0, 9.0, 8.0]
    )
    plan = {
        "policy": {"scenario": "family_lofo", "selection_fraction": 0.5},
        "bootstrap_group": "family_id",
        "plan_hash": "synthetic-plan",
    }
    report = gs_selection.feasibility_report(predictions, dataset, plan)
    candidates = gs_selection.line_candidates(predictions, dataset, plan, report)
    assert candidates.loc[candidates.selected, "line_id"].tolist() == ["A", "B"]
    assert len(candidates) == 3
    assert report["models"]["gblup"]["metrics"]["selection_differential"] == 0.5
    assert processing["lineage"][0]["observation_weights"] == [0.5, 0.5]
    assert processing["counts"]["source_observations"] == 4
    original = gs_selection.selection_metrics(
        raw.value, raw.value, raw.observation_id, fraction=0.5, direction="higher"
    )
    assert original["selection_differential"] == 0.75
    assert original["n"] == 4


def test_unequal_replicates_have_equal_line_training_weight(individual_dataset):
    enable_line_mode(individual_dataset)
    add_replicate(individual_dataset)
    dataset = gs_dataset.load_dataset(individual_dataset)
    assert len(dataset.baseline.phenotypes) == 12
    assert (
        dataset.observations.loc[dataset.observations.line_id == "L0", "value"].item()
        == 6.0
    )
    processing = dataset.phenotype_processing
    assert processing["counts"] == {
        "source_observations": 13,
        "usable_observations": 13,
        "missing_observations": 0,
        "lines": 12,
        "families": 3,
        "genotype_samples": 12,
    }
    assert all(row["training_weight"] == 1.0 for row in processing["lineage"])
    assert processing["correction_fit_ids"] == []
    assert processing["environment_correction"] == "none"


def test_missing_replicate_is_audited_before_line_mean(individual_dataset):
    enable_line_mode(individual_dataset)
    add_replicate(individual_dataset, value="NA")
    dataset = gs_dataset.load_dataset(individual_dataset)
    assert dataset.excluded_observations == ("O0_rep2",)
    assert dataset.phenotype_processing["counts"]["missing_observations"] == 1
    assert (
        dataset.observations.loc[dataset.observations.line_id == "L0", "value"].item()
        == 0.0
    )


@pytest.mark.parametrize("column,value", [("year", "2026"), ("site", "B")])
def test_multiple_environments_are_refused_even_for_missing_observations(
    individual_dataset, column, value
):
    enable_line_mode(individual_dataset)
    path = individual_dataset.parent / "phenotypes.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str)
    frame.loc[0, column] = value
    frame.loc[0, "value"] = "NA"
    frame.to_csv(path, sep="\t", index=False)
    refresh_checksums(individual_dataset)
    with pytest.raises(ValueError, match="single year/site"):
        gs_dataset.load_dataset(individual_dataset)


def test_ambiguous_line_genotype_is_refused(individual_dataset):
    enable_line_mode(individual_dataset)
    samples_path = individual_dataset.parent / "samples.tsv"
    samples = pd.read_csv(samples_path, sep="\t", dtype=str)
    samples.loc[1, "line_id"] = "L0"
    samples.to_csv(samples_path, sep="\t", index=False)
    path = individual_dataset.parent / "phenotypes.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str)
    frame.loc[1, "replicate"] = "2"
    frame.to_csv(path, sep="\t", index=False)
    refresh_checksums(individual_dataset)
    with pytest.raises(ValueError, match="ambiguous genotype"):
        gs_dataset.load_dataset(individual_dataset)


def config():
    return resnet.ResNetConfig(
        max_epochs=1,
        hidden_dim=4,
        num_blocks=1,
        pca_components=2,
        qc_mode="controlled",
        min_observed_rate=0.9,
        maf_threshold=0.05,
    )


def test_line_evaluation_bundle_and_outer_test_values_do_not_leak(individual_dataset):
    enable_line_mode(individual_dataset)
    add_replicate(individual_dataset)
    dataset = gs_dataset.load_dataset(individual_dataset)
    plan = gs_evaluate.make_plan(dataset, config())
    assert plan["schema_version"] == 3
    output = individual_dataset.parent / "line-evaluation"
    gs_evaluate.evaluate(dataset, plan, output)
    candidate_path = output / "line_candidates.csv"
    candidates = pd.read_csv(candidate_path)
    assert not candidates.duplicated(["model", "line_id"]).any()
    assert set(candidates.plan_hash) == {plan["plan_hash"]}
    metadata = json.loads((output / "metadata.json").read_text())
    assert "line_candidates.csv" in metadata["artifact_checksums"]
    scope = {
        "cohort_ids": ["cohort"],
        "generations": ["trial-1"],
        "update_trigger": "new trial",
        "rollback_bundle": "none_initial",
    }
    bundle = gs_model_bundle.fit_final(
        dataset, output, output.parent / "line-model", scope=scope
    )
    assert bundle.metadata["phenotype_processing"] == dataset.phenotype_processing
    context = {
        "species": "Vigna angularis",
        "assembly_id": "synthetic-v1",
        "cohort_id": "cohort",
        "generation": "trial-1",
        "target_trial": {"year": 2025, "site": "A"},
    }
    predictions = gs_model_bundle.predict(
        output.parent / "line-model",
        individual_dataset.parent / "panel",
        context=context,
    )
    assert set(predictions.prediction_target) == {"within_trial_raw_line_mean"}
    with pytest.raises(ValueError, match="target trial"):
        gs_model_bundle.predict(
            output.parent / "line-model",
            individual_dataset.parent / "panel",
            context={**context, "target_trial": {"year": 2026, "site": "A"}},
        )
    # Mutate ONLY first outer test family's targets, rebuild the data-bound plan,
    # and compare that fold's train-fitted transforms and inner selection.
    path = individual_dataset.parent / "phenotypes.tsv"
    frame = pd.read_csv(path, sep="\t", dtype=str)
    held_out = frame.sample_id.isin(["000", "001", "002", "003"])
    frame.loc[held_out, "value"] = (
        frame.loc[held_out, "value"].astype(float) + 500
    ).astype(str)
    frame.to_csv(path, sep="\t", index=False)
    refresh_checksums(individual_dataset)
    changed = gs_dataset.load_dataset(individual_dataset)
    changed_plan = gs_evaluate.make_plan(changed, config())
    altered_output = output.parent / "changed-evaluation"
    gs_evaluate.evaluate(changed, changed_plan, altered_output)
    with (
        np.load(output / "preprocessing.npz") as before,
        np.load(altered_output / "preprocessing.npz") as after,
    ):
        for key in before.files:
            if key.startswith("fold_0_"):
                np.testing.assert_array_equal(before[key], after[key])
    changed_metadata = json.loads((altered_output / "metadata.json").read_text())
    for key in (
        "best_epoch",
        "gblup_lambda",
        "gblup_intercept",
        "target_mean",
        "target_scale",
    ):
        assert metadata["folds"][0][key] == changed_metadata["folds"][0][key]
    assert metadata["folds"][0]["phenotype_processing"]["correction_fit_ids"] == []


def test_unsupported_line_scenario_rejected(individual_dataset):
    enable_line_mode(individual_dataset)
    dataset = gs_dataset.load_dataset(individual_dataset)
    with pytest.raises(ValueError, match="family_lofo only"):
        gs_evaluate.make_plan(
            dataset, config(), {"scenario": "future_year", "selection_fraction": 0.5}
        )
