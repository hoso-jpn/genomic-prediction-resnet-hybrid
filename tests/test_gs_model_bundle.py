import json
import socket

import numpy as np
import pytest
from test_adzuki_gs_panel_data import _write_panel
from test_gs_dataset import individual_dataset as individual_dataset

import gblup_baseline as gblup
import gs_dataset
import gs_evaluate
import gs_model_bundle as bundles
import resnet_baseline as resnet
import run_manifest


@pytest.fixture
def fitted(individual_dataset):
    dataset = gs_dataset.load_dataset(individual_dataset)
    config = resnet.ResNetConfig(
        qc_mode="controlled",
        min_observed_rate=0.9,
        maf_threshold=0.05,
        max_epochs=1,
        hidden_dim=4,
        num_blocks=1,
        pca_components=2,
    )
    plan = gs_evaluate.make_plan(dataset, config)
    evaluation = individual_dataset.parent / "evaluation"
    gs_evaluate.evaluate(dataset, plan, evaluation)
    scope = {
        "cohort_ids": ["cohort", "new"],
        "generations": ["pilot-1"],
        "update_trigger": "before a new season or population change",
        "rollback_bundle": "none_initial",
    }
    path = individual_dataset.parent / "model"
    bundle = bundles.fit_final(
        dataset,
        evaluation,
        path,
        scope=scope,
        prediction_qc={
            "min_retained_marker_call_rate": 0.5,
            "basis": "synthetic test policy, not customer validated",
            "evidence_references": [],
        },
    )
    return dataset, evaluation, path, bundle


def new_panel(fitted, *, reverse=False, omit=False):
    dataset, _, path, _ = fitted
    keys = dataset.panel.variant_keys.tolist()
    rng = np.random.default_rng(30)
    rows = rng.integers(-1, 2, (len(keys), 3)).astype(str).tolist()
    rows[0][0] = "nan"
    if reverse:
        keys, rows = keys[::-1], rows[::-1]
    if omit:
        keys, rows = keys[:-1], rows[:-1]
    return _write_panel(
        path.parent / "new-panel",
        cohort_id="new",
        samples=["NA", "001", "unseen"],
        variants=keys,
        rows=rows,
        manifest_overrides={"checksums": {"synthetic.fa": "sha256:" + "a" * 64}},
    )


def context(**changes):
    return {
        "species": "Vigna angularis",
        "assembly_id": "synthetic-v1",
        "cohort_id": "new",
        "generation": "pilot-1",
        **changes,
    }


def test_serialization_matches_direct_gblup(fitted):
    dataset, _, path, bundle = fitted
    loaded = bundles.load_bundle(path)
    x = dataset.baseline.genotypes
    relation = gblup.prepare_fold_relationships(
        x, x, qc_mode="controlled", min_observed_rate=0.9, maf_threshold=0.05
    )
    fit = gblup.fit_gblup_reml(relation.relationship_train, dataset.baseline.phenotypes)
    expected = fit.intercept + gblup.predict_genetic_values(
        fit, relation.relationship_test_train
    )
    np.testing.assert_allclose(
        loaded.predict_matrix(x), expected, atol=1e-10, rtol=1e-10
    )
    np.testing.assert_allclose(
        loaded.predict_matrix(x), bundle.predict_matrix(x), atol=1e-10, rtol=1e-10
    )
    assert not loaded.metadata["model_card"]["fit_final_is_oof"]


def test_fit_final_rejects_a_split_from_another_evaluation(fitted, tmp_path):
    dataset, evaluation, _, bundle = fitted
    other_plan = gs_evaluate.make_plan(
        dataset,
        resnet.ResNetConfig(
            qc_mode="legacy",
            max_epochs=1,
            hidden_dim=4,
            num_blocks=1,
            pca_components=2,
        ),
    )
    (evaluation / "split.json").write_text(json.dumps(other_plan))
    with pytest.raises(ValueError, match="metadata/artifact binding"):
        bundles.fit_final(
            dataset,
            evaluation,
            tmp_path / "mixed-evidence-model",
            scope=bundle.metadata["model_card"]["scope"],
        )


def test_offline_new_individuals_need_no_phenotypes_or_training_files(
    fitted, monkeypatch
):
    dataset, evaluation, path, bundle = fitted
    panel = new_panel(fitted)
    before = {
        p: run_manifest.sha256_file(p)
        for p in (*dataset.source_paths, *evaluation.iterdir())
        if p.is_file()
    }

    def prohibited(*args, **kwargs):
        raise AssertionError("network/refit/training inputs used during predict")

    monkeypatch.setattr(socket, "socket", prohibited)
    monkeypatch.setattr(gs_dataset, "load_dataset", prohibited)
    monkeypatch.setattr(gblup, "fit_gblup_reml", prohibited)
    monkeypatch.setattr(np, "nanmean", prohibited)
    predictions = bundles.predict(path, panel, context=context())
    assert list(predictions.sample_id) == ["NA", "001", "unseen"]
    assert np.isfinite(predictions.prediction).all()
    assert before == {p: run_manifest.sha256_file(p) for p in before}
    assert list(predictions.bundle_hash) == [bundle.metadata["bundle_hash"]] * 3


def test_marker_and_sample_reorder_require_explicit_permission(fitted):
    panel = new_panel(fitted, reverse=True)
    path = fitted[2]
    with pytest.raises(ValueError, match="marker order"):
        bundles.predict(path, panel, context=context())
    desired = ["unseen", "001", "NA"]
    reordered = bundles.predict(
        path, panel, context=context(), expected_sample_ids=desired, allow_reorder=True
    )
    assert list(reordered.sample_id) == desired
    assert reordered.marker_reordered.all() and reordered.sample_reordered.all()
    canonical_panel = new_panel(fitted)
    canonical = bundles.predict(path, canonical_panel, context=context())
    np.testing.assert_allclose(reordered.prediction, canonical.prediction.iloc[::-1])
    with pytest.raises(ValueError, match="sample order"):
        bundles.predict(
            path, canonical_panel, context=context(), expected_sample_ids=desired
        )
    with pytest.raises(ValueError, match="sample ID set"):
        bundles.predict(
            path, canonical_panel, context=context(), expected_sample_ids=["absent"]
        )


def test_missing_markers_and_unknown_scope_rejected(fitted):
    panel = new_panel(fitted, omit=True)
    with pytest.raises(ValueError, match="missing or extra markers"):
        bundles.predict(fitted[2], panel, context=context())
    for change in (
        {"assembly_id": "unknown"},
        {"generation": "new-gen"},
        {"species": "other"},
    ):
        with pytest.raises(ValueError, match="scope/reference"):
            bundles.predict(fitted[2], panel, context=context(**change))


@pytest.mark.parametrize(
    "mutation,match",
    [
        ({"schema_version": 9}, "schema"),
        ({"encoding": "other"}, "encoding"),
        ({"marker_ids": ["duplicate", "duplicate"]}, "marker IDs"),
    ],
)
def test_bundle_schema_encoding_ids_rejected_even_with_updated_hash(
    fitted, mutation, match
):
    path = fitted[2] / "bundle.json"
    metadata = json.loads(path.read_text())
    metadata.update(mutation)
    metadata.pop("bundle_hash")
    metadata["bundle_hash"] = run_manifest.canonical_json_hash(metadata)
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=match):
        bundles.load_bundle(fitted[2])


def test_parameter_corruption_and_output_overwrite_rejected(fitted):
    dataset, evaluation, path, bundle = fitted
    with pytest.raises(FileExistsError):
        bundles.fit_final(
            dataset, evaluation, path, scope=bundle.metadata["model_card"]["scope"]
        )
    parameters = path / "parameters.npz"
    parameters.write_bytes(parameters.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="parameter checksum"):
        bundles.load_bundle(path)


def test_manifest_corruption_rejected(fitted):
    path = fitted[2] / "bundle.json"
    metadata = json.loads(path.read_text())
    metadata["fit"]["intercept"] += 1
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="manifest checksum"):
        bundles.load_bundle(fitted[2])


def test_predict_cli_writes_id_bound_csv_once(fitted, tmp_path, monkeypatch):
    panel = new_panel(fitted)
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context()))
    output = tmp_path / "predictions.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gs_model_bundle.py",
            "predict",
            "--bundle",
            str(fitted[2]),
            "--panel-dir",
            str(panel),
            "--context",
            str(context_path),
            "--output",
            str(output),
        ],
    )
    bundles.main()
    assert output.read_text().splitlines()[1].startswith("NA,")
    receipt = json.loads(output.with_suffix(".csv.metadata.json").read_text())
    assert not receipt["customer_release_authorized"]
    assert receipt["checksums"][output.name] == run_manifest.sha256_file(output)
    assert output.with_suffix(".csv.withheld.csv").is_file()
    with pytest.raises(FileExistsError):
        bundles.main()


def small_bundle(threshold=0.5, status="research_only"):
    return bundles.GblupBundle(
        {
            "schema_version": 2,
            "fit": {"intercept": 12.5},
            "prediction_qc": {
                "min_retained_marker_call_rate": threshold,
                "basis": "synthetic-only boundary probe",
                "evidence_references": [],
            },
            "model_card": {"status": status},
        },
        np.array([True, True, True, True, False]),
        np.array([0.25, -0.25, 0.0, 0.0]),
        np.array([2.0, -3.0, 1.0, 1.0]),
    )


def test_retained_calls_boundaries_and_batch_independence():
    bundle = small_bundle()
    matrix = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, 1.0],
            [1.0, np.nan, np.nan, np.nan, 1.0],
            [1.0, 0.0, np.nan, np.nan, np.nan],
            [1.0, 0.0, -1.0, np.nan, np.nan],
        ]
    )
    _, observed, rates, reasons = bundle.prediction_qc(matrix)
    np.testing.assert_array_equal(observed, [0, 0, 1, 2, 3])
    np.testing.assert_array_equal(rates, [0, 0, 0.25, 0.5, 0.75])
    assert "no_observed_retained_markers" in reasons[0]
    assert "no_observed_retained_markers" in reasons[1]
    predictions = bundle.predict_matrix(matrix)
    assert np.isnan(predictions[:3]).all()
    expected = 12.5 + (1.0 - 0.25) * 2 + (0.0 + 0.25) * -3
    assert predictions[3] == expected
    assert predictions[4] == expected - 1
    assert bundle.predict_matrix(matrix[[3]])[0] == predictions[3]
    assert np.isnan(bundle.predict_matrix(matrix[:3])).all()


@pytest.mark.parametrize(
    "threshold", [0, -0.1, 1.1, True, float("nan"), float("inf"), "0.5"]
)
def test_bad_prediction_qc_threshold_rejected(threshold):
    policy = small_bundle().metadata["prediction_qc"]
    policy["min_retained_marker_call_rate"] = threshold
    with pytest.raises(ValueError, match="call rate"):
        bundles.validate_prediction_qc(policy)


def test_legacy_policy_missing_and_rejected_models_are_withheld():
    bundle = small_bundle()
    bundle.metadata.pop("prediction_qc")
    assert np.isnan(bundle.predict_matrix(np.zeros((1, 5)))).all()
    assert bundle.prediction_qc(np.zeros((1, 5)))[3] == [
        ["legacy_qc_policy_unavailable"]
    ]
    rejected = small_bundle(status="rejected")
    assert rejected.prediction_qc(np.zeros((1, 5)))[3] == [["model_not_accepted"]]
    assert np.isnan(rejected.predict_matrix(np.zeros((1, 5)))).all()


@pytest.mark.parametrize("status", ["conditional", "rejected"])
def test_recorded_status_is_preserved_without_release_authorization(fitted, status):
    panel = new_panel(fitted)
    metadata_path = fitted[2] / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_card"]["status"] = status
    metadata["model_card"]["review_reference"] = "external-review-record; unverified"
    metadata.pop("bundle_hash")
    metadata["bundle_hash"] = run_manifest.canonical_json_hash(metadata)
    metadata_path.write_text(json.dumps(metadata))
    result = bundles.predict(fitted[2], panel, context=context())
    assert set(result.model_status) == {status}
    assert not result.attrs["delivery_metadata"]["customer_release_authorized"]
    if status == "rejected":
        assert result.prediction.isna().all()
        assert set(result.prediction_status) == {"withheld"}


def test_v1_bundle_reads_but_withholds_until_explicit_new_policy(fitted):
    panel = new_panel(fitted)
    metadata_path = fitted[2] / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["schema_version"] = 1
    metadata.pop("prediction_qc")
    metadata.pop("bundle_hash")
    metadata["bundle_hash"] = run_manifest.canonical_json_hash(metadata)
    metadata_path.write_text(json.dumps(metadata))
    result = bundles.predict(fitted[2], panel, context=context())
    assert result.prediction.isna().all()
    assert set(result.reason_codes) == {"legacy_qc_policy_unavailable"}


def test_qc_reorder_delivery_and_all_withheld(fitted, tmp_path):
    dataset, _, path, bundle = fitted
    keys = dataset.panel.variant_keys.tolist()
    rows = np.zeros((len(keys), 3)).astype(str)
    rows[:, 0] = "nan"
    rows[bundle.marker_mask, 1] = "nan"
    panel = _write_panel(
        tmp_path / "qc-panel",
        cohort_id="new",
        samples=["all-missing", "filtered-only", "valid"],
        variants=keys[::-1],
        rows=rows[::-1].tolist(),
        manifest_overrides={"checksums": {"synthetic.fa": "sha256:" + "a" * 64}},
    )
    result = bundles.predict(
        path,
        panel,
        context=context(),
        allow_reorder=True,
        expected_sample_ids=["valid", "filtered-only", "all-missing"],
    )
    assert list(result.prediction_status) == ["available", "withheld", "withheld"]
    assert result.prediction.iloc[1:].isna().all()
    assert set(result.model_status) == {"research_only"}
    output = tmp_path / "predictions.csv"
    bundles.write_predictions(result, output)
    receipt = json.loads(output.with_suffix(".csv.metadata.json").read_text())
    assert receipt["counts"] == {"samples": 3, "available": 1, "withheld": 2}
    assert receipt["prediction_qc"] == bundle.metadata["prediction_qc"]
    import pandas as pd

    held = pd.read_csv(output.with_suffix(".csv.withheld.csv"))
    assert set(held.sample_id) == {"filtered-only", "all-missing"}
    assert held.prediction.isna().all()


def test_publication_does_not_overwrite_or_leave_partial_files(
    fitted, tmp_path, monkeypatch
):
    panel = new_panel(fitted)
    result = bundles.predict(fitted[2], panel, context=context())
    output = tmp_path / "result.csv"
    existing = output.with_suffix(".csv.withheld.csv")
    existing.write_text("existing")
    with pytest.raises(FileExistsError):
        bundles.write_predictions(result, output)
    assert existing.read_text() == "existing"
    assert not output.exists()
    existing.unlink()
    real_link = bundles.os.link

    def fail_second(source, destination):
        if destination == existing:
            raise OSError("simulated publish failure")
        real_link(source, destination)

    monkeypatch.setattr(bundles.os, "link", fail_second)
    with pytest.raises(OSError, match="simulated"):
        bundles.write_predictions(result, output)
    assert not output.exists()
    assert not output.with_suffix(".csv.metadata.json").exists()
