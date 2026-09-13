import json

import numpy as np
import pytest
from test_gs_dataset import individual_dataset as individual_dataset

import controlled_qc
import gblup_baseline as gblup
import gs_dataset
import gs_evaluate
import resnet_baseline as resnet


def test_common_qc_boundaries_match_and_legacy_is_preserved():
    x = np.array(
        [[-1, -1, -1], [0, 1, -1], [1, np.nan, 1], [0, np.nan, 1]], dtype=float
    )
    z = np.array([[1, 0, -1]], dtype=float)
    config = resnet.ResNetConfig(
        qc_mode="controlled", min_observed_rate=0.5, maf_threshold=0, pca_components=2
    )
    relationships = gblup.prepare_fold_relationships(
        x, z, qc_mode="controlled", min_observed_rate=0.5, maf_threshold=0
    )
    transform = resnet.fit_feature_transform(x, config, 42)
    np.testing.assert_equal(relationships.retained_markers, [True, True, True])
    np.testing.assert_equal(relationships.retained_markers, transform.retained_markers)
    legacy = gblup.prepare_fold_relationships(
        x, z, min_observed_rate=0.5, maf_threshold=0
    )
    assert not legacy.retained_markers[1]
    changed_test = gblup.prepare_fold_relationships(
        x, -z, qc_mode="controlled", min_observed_rate=0.5, maf_threshold=0
    )
    np.testing.assert_equal(relationships.marker_means, changed_test.marker_means)


def test_ridge_equivalence_is_numerical_not_an_independent_architecture():
    rng = np.random.default_rng(2)
    x = rng.integers(-1, 2, (12, 8)).astype(float)
    y = np.arange(12, dtype=float) + x[:, 0]
    relationship = gblup.prepare_fold_relationships(x[:9], x[9:], qc_mode="controlled")
    fit = gblup.fit_gblup_reml(relationship.relationship_train, y[:9])
    prediction = fit.intercept + gblup.predict_genetic_values(
        fit, relationship.relationship_test_train
    )
    alpha = relationship.denominator * (fit.lambda_ratio + gblup.DIAGONAL_JITTER)
    ridge = controlled_qc.ridge_predict(
        x[:9], x[9:], y[:9], relationship.retained_markers, alpha=alpha
    )
    np.testing.assert_allclose(prediction, ridge, atol=1e-7, rtol=1e-7)


@pytest.mark.parametrize("use_pca", [True, False])
def test_controlled_run_records_shared_masks_and_all_candidates(
    individual_dataset, use_pca
):
    dataset = gs_dataset.load_dataset(individual_dataset)
    config = resnet.ResNetConfig(
        qc_mode="controlled",
        min_observed_rate=0.9,
        maf_threshold=0.05,
        max_epochs=1,
        hidden_dim=4,
        num_blocks=1,
        pca_components=2,
        use_pca=use_pca,
    )
    plan = gs_evaluate.make_plan(dataset, config)
    output = individual_dataset.parent / f"controlled-{use_pca}"
    predictions = gs_evaluate.evaluate(dataset, plan, output)
    assert set(predictions.model) == {"gblup", "resnet", "ridge_fixed_alpha_1"}
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["comparison"] == "controlled_common_qc"
    assert metadata["candidate_budget"]["resnet_candidates"] == 1
    assert metadata["candidate_budget"]["seeds"] == [42]
    for fold in metadata["folds"]:
        assert fold["common_qc"]["selection"]["marker_hash"]
        assert fold["common_qc"]["final"]["observed_rate"]["operator"] == ">="
    with np.load(output / "preprocessing.npz") as arrays:
        for i in range(3):
            np.testing.assert_equal(
                arrays[f"fold_{i}_gblup_mask"], arrays[f"fold_{i}_resnet_final_mask"]
            )
        if not use_pca:
            assert arrays["fold_0_resnet_pca_components"].size == 0


def test_failed_candidate_has_receipt_without_completed_run(
    individual_dataset, monkeypatch
):
    dataset = gs_dataset.load_dataset(individual_dataset)
    config = resnet.ResNetConfig(qc_mode="controlled", max_epochs=1)
    plan = gs_evaluate.make_plan(dataset, config)

    def failed(*args, **kwargs):
        raise RuntimeError("synthetic training failure")

    monkeypatch.setattr(gblup, "predict_gblup_fold", failed)
    output = individual_dataset.parent / "failed"
    with pytest.raises(RuntimeError, match="synthetic training failure"):
        gs_evaluate.evaluate(dataset, plan, output)
    assert not output.exists()
    (receipt,) = output.parent.glob("failed.attempt-*.json")
    assert json.loads(receipt.read_text())["status"] == "failed"
