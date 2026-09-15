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
    bundle = bundles.fit_final(dataset, evaluation, path, scope=scope)
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
    with pytest.raises(FileExistsError):
        bundles.main()
