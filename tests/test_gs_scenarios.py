import copy
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch
from test_gs_dataset import individual_dataset as individual_dataset
from test_gs_dataset import refresh_checksums

import gs_dataset
import gs_evaluate
import gs_scenarios
import gs_selection
import resnet_baseline as resnet


@pytest.fixture
def repeated_dataset(individual_dataset):
    path = individual_dataset.parent / "phenotypes.tsv"
    original = pd.read_csv(path, sep="\t", dtype=str)
    years = []
    for year in (2024, 2025, 2026):
        frame = original.copy()
        frame["year"] = str(year)
        frame["observation_id"] += "-" + str(year)
        frame["value"] = frame.value.astype(float) + (year - 2024) * 0.25
        years.append(frame)
    pd.concat(years).to_csv(path, sep="\t", index=False)
    refresh_checksums(individual_dataset)
    return gs_dataset.load_dataset(individual_dataset)


def policy(**changes):
    return {
        "scenario": "future_year",
        "test_year": 2026,
        "validation_year": 2025,
        "genotype_status": "known",
        "environment_status": "known",
        "selection_fraction": 0.25,
        **changes,
    }


def small_config():
    return resnet.ResNetConfig(
        max_epochs=1, hidden_dim=4, num_blocks=1, pca_components=2
    )


def test_known_line_history_is_allowed_and_time_is_strict(repeated_dataset):
    dataset = repeated_dataset
    plan = gs_scenarios.make_plan(dataset, small_config(), policy())
    obs = dataset.observations.set_index("observation_id")
    fold = plan["folds"][0]
    assert set(obs.loc[fold["fit"]].year) == {2024}
    assert set(obs.loc[fold["validation"]].year) == {2025}
    assert set(obs.loc[fold["test"]].year) == {2026}
    with pytest.raises(ValueError, match="leaks IDs"):
        gs_scenarios.make_plan(dataset, small_config(), policy(genotype_status="new"))
    with pytest.raises(ValueError, match="temporal inversion"):
        gs_scenarios.make_plan(dataset, small_config(), policy(validation_year=2026))
    forged = copy.deepcopy(plan)
    forged["folds"][0]["train"].append(fold["test"][0])
    with pytest.raises(ValueError, match="split plan"):
        gs_scenarios.validate_plan(dataset, forged)


def test_future_year_both_models_use_saved_inner_and_outer(repeated_dataset, tmp_path):
    plan = gs_scenarios.make_plan(repeated_dataset, small_config(), policy())
    result = gs_evaluate.evaluate(repeated_dataset, plan, tmp_path / "run")
    assert set(result.year) == {2026}
    assert len(result) == 24
    saved = json.loads((tmp_path / "run/feasibility.json").read_text())
    assert saved["automatic_go"] is False
    assert saved["scenario"]["scenario"] == "future_year"


def test_test_changes_do_not_change_training_transforms_or_epoch(repeated_dataset):
    data = repeated_dataset.baseline
    fit = np.arange(12)
    valid = np.arange(12, 24)
    train = np.arange(24)
    test = np.arange(24, 36)
    x, y = data.genotypes.copy(), data.phenotypes.copy()
    args = (data.family_ids, train, test, 0, small_config(), torch.device("cpu"))
    first, a = resnet.predict_resnet_fold(x, y, *args, inner_indices=(fit, valid))
    x[test] *= -1
    y[test] += 1000
    second, b = resnet.predict_resnet_fold(x, y, *args, inner_indices=(fit, valid))
    assert a.best_epoch == b.best_epoch
    np.testing.assert_equal(
        a.selection_transform.marker_means, b.selection_transform.marker_means
    )
    np.testing.assert_equal(
        a.final_transform.marker_means, b.final_transform.marker_means
    )
    assert not np.array_equal(first, second)


def test_family_environment_cross_cells_are_excluded(individual_dataset):
    dataset = gs_dataset.load_dataset(individual_dataset)
    frames = []
    for site in ("A", "B", "C"):
        frame = dataset.observations.copy()
        frame["site"] = site
        frame["observation_id"] += site
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True)
    expanded = replace(dataset, observations=observations)
    plan = gs_scenarios.make_plan(
        expanded,
        small_config(),
        {
            "scenario": "family_environment",
            "selection_fraction": 0.25,
            "test_families": ["F2"],
            "test_sites": ["C"],
            "validation_families": ["F1"],
            "validation_sites": ["B"],
        },
    )
    indexed = observations.set_index("observation_id")
    fold = plan["folds"][0]
    assert not (set(indexed.loc[fold["train"]].family_id) & {"F2"})
    assert not (set(indexed.loc[fold["train"]].site) & {"C"})
    assert set(indexed.loc[fold["fit"]].family_id) == {"F0"}
    assert set(indexed.loc[fold["fit"]].site) == {"A"}
    assert fold["excluded"] and fold["inner_excluded"]


def test_selection_hand_calculations_and_ties():
    metric = gs_selection.selection_metrics(
        [1, 2, 3, 4], [1, 3, 2, 4], list("abcd"), fraction=0.5, direction="higher"
    )
    assert metric["k"] == 2
    assert metric["top_k_overlap"] == 0.5
    assert metric["selection_differential"] == 0.5
    assert metric["spearman"] == pytest.approx(0.8)
    tied = gs_selection.selection_metrics(
        [4, 3, 2, 1], [2, 2, 1, 0], list("abcd"), fraction=0.25, direction="higher"
    )
    assert tied["top_k_overlap"] == 1
    lower = gs_selection.selection_metrics(
        [1, 2, 3, 4], [1, 2, 3, 4], list("abcd"), fraction=0.25, direction="lower"
    )
    assert lower["selection_differential"] == -1.5
    assert lower["directional_differential"] == 1.5


def test_metrics_undefined_cases_and_group_shortage():
    for y, p in (([1, 2], [1, 1]), ([np.nan], [1]), ([1], [1])):
        metric = gs_selection.selection_metrics(
            y, p, [str(i) for i in range(len(y))], fraction=0.2, direction="higher"
        )
        assert metric["top_k_overlap"] is None
        assert metric["undefined_reason"]
    frame = pd.DataFrame({"line_id": ["a", "b"]})
    assert (
        gs_selection.group_interval(
            frame, group="line_id", fraction=0.2, direction="higher"
        )["interval"]
        is None
    )


def test_policy_cli_is_recorded(repeated_dataset, monkeypatch, tmp_path):
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(json.dumps(policy()))
    split_path = tmp_path / "split.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gs_evaluate.py",
            "plan",
            "--dataset",
            str(repeated_dataset.source_paths[0]),
            "--split",
            str(split_path),
            "--policy",
            str(policy_path),
            "--max-epochs",
            "1",
        ],
    )
    gs_evaluate.main()
    assert json.loads(split_path.read_text())["policy"]["scenario"] == "future_year"
