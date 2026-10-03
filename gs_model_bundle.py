"""Fit-final and phenotype-free offline inference for a checksummed GBLUP bundle."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import adzuki_gs_panel_data as gs_panel
import controlled_qc
import gblup_baseline as gblup
import gs_dataset
import gs_evaluate
import run_manifest


@dataclass(frozen=True)
class GblupBundle:
    metadata: dict
    marker_mask: np.ndarray
    marker_means: np.ndarray
    marker_coefficients: np.ndarray

    def prediction_qc(self, matrix):
        """Evaluate each sample against the immutable retained-marker policy."""
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.marker_mask):
            raise ValueError("prediction marker dimension mismatch")
        if (
            np.isinf(values).any()
            or not np.isin(values[np.isfinite(values)], [-1, 0, 1]).all()
        ):
            raise ValueError("unknown genotype encoding in prediction matrix")
        retained = values[:, self.marker_mask]
        observed = np.isfinite(retained).sum(axis=1)
        rates = observed / int(self.marker_mask.sum())
        policy = self.metadata.get("prediction_qc")
        reasons = [[] for _ in observed]
        for i, count in enumerate(observed):
            if count == 0:
                reasons[i].append("no_observed_retained_markers")
            if policy is None:
                reasons[i].append("legacy_qc_policy_unavailable")
            elif rates[i] < policy["min_retained_marker_call_rate"]:
                reasons[i].append("retained_marker_call_rate_below_minimum")
            if self.metadata.get("model_card", {}).get("status") == "rejected":
                reasons[i].append("model_not_accepted")
        return retained, observed, rates, reasons

    def predict_matrix(self, matrix):
        retained, _, _, reasons = self.prediction_qc(matrix)
        centered = (
            np.where(np.isnan(retained), self.marker_means, retained)
            - self.marker_means
        )
        # No means, scales, frequencies, masks or model parameters are refitted.
        predictions = (
            self.metadata["fit"]["intercept"] + centered @ self.marker_coefficients
        )
        predictions[np.array([bool(reason) for reason in reasons], dtype=bool)] = np.nan
        return predictions


def validate_prediction_qc(policy):
    if not isinstance(policy, dict) or set(policy) != {
        "min_retained_marker_call_rate",
        "basis",
        "evidence_references",
    }:
        raise ValueError(
            "prediction QC requires call rate, basis and evidence_references"
        )
    threshold = policy["min_retained_marker_call_rate"]
    if (
        type(threshold) not in (int, float)
        or not np.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise ValueError("prediction QC call rate must be finite and in (0, 1]")
    gs_dataset.require_text(policy["basis"], "prediction QC basis")
    refs = policy["evidence_references"]
    if not isinstance(refs, list) or any(
        not isinstance(ref, str) or not ref.strip() for ref in refs
    ):
        raise ValueError(
            "prediction QC evidence_references must be a list of nonempty text"
        )


def default_prediction_qc():
    """Conservative research fallback, explicitly not a validated customer threshold."""
    return {
        "min_retained_marker_call_rate": 1.0,
        "basis": "research_default_complete_retained_markers; not pilot validated",
        "evidence_references": [],
    }


def validate_scope(scope):
    if not isinstance(scope, dict) or set(scope) != {
        "cohort_ids",
        "generations",
        "update_trigger",
        "rollback_bundle",
    }:
        raise ValueError(
            "scope must declare cohort_ids, generations, update_trigger and rollback_bundle"
        )
    for key in ("cohort_ids", "generations"):
        values = scope[key]
        if (
            not isinstance(values, list)
            or not values
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"scope.{key} must be unique nonempty IDs")
    for key in ("update_trigger", "rollback_bundle"):
        gs_dataset.require_text(scope[key], key)


def fit_final(
    dataset, evaluation_dir: Path, output_dir: Path, *, scope, prediction_qc=None
):
    """Refit the adopted GBLUP QC on all supplied training observations."""
    validate_scope(scope)
    prediction_qc = default_prediction_qc() if prediction_qc is None else prediction_qc
    validate_prediction_qc(prediction_qc)
    if output_dir.exists():
        raise FileExistsError("model bundle already exists")
    evidence_paths = [
        evaluation_dir / name
        for name in (
            "split.json",
            "metadata.json",
            "predictions.csv",
            "feasibility.json",
            "preprocessing.npz",
        )
    ]
    if dataset.selection_unit == "line":
        evidence_paths.extend(
            evaluation_dir / name
            for name in ("line_candidates.csv", "phenotype_processing.json")
        )
    evidence_hashes = {p.name: run_manifest.sha256_file(p) for p in evidence_paths}
    plan = json.loads(evidence_paths[0].read_text())
    config = gs_evaluate.validate_plan(dataset, plan)
    evaluation = json.loads(evidence_paths[1].read_text())
    artifact_hashes = {
        p.name: evidence_hashes[p.name]
        for p in evidence_paths
        if p.name != "metadata.json"
    }
    expected_comparison = (
        "controlled_common_qc"
        if config.qc_mode == "controlled"
        else "pipeline_vs_pipeline; model-specific QC"
    )
    if (
        evaluation.get("kind") != "held_out_evaluation"
        or evaluation.get("provenance") != dataset.provenance
        or evaluation.get("plan_hash") != plan["plan_hash"]
        or evaluation.get("artifact_checksums") != artifact_hashes
        or evaluation.get("comparison") != expected_comparison
    ):
        raise ValueError(
            "evaluation metadata/artifact binding does not match the training dataset and plan"
        )
    controlled = config.qc_mode == "controlled"
    kwargs = {
        "qc_mode": config.qc_mode,
        "min_observed_rate": config.min_observed_rate
        if controlled
        else gblup.MIN_OBSERVED_RATE,
        "maf_threshold": config.maf_threshold if controlled else gblup.MAF_THRESHOLD,
    }
    data = dataset.baseline
    # Cross matrix is empty: fitting does not fabricate an OOF performance result.
    relations = gblup.prepare_fold_relationships(
        data.genotypes, data.genotypes[:0], **kwargs
    )
    fit = gblup.fit_gblup_reml(relations.relationship_train, data.phenotypes)
    retained = data.genotypes[:, relations.retained_markers]
    centered = (
        np.where(np.isnan(retained), relations.marker_means, retained)
        - relations.marker_means
    )
    coefficients = centered.T @ fit.dual_coefficients / relations.denominator
    metadata = {
        "schema_version": 2,
        "kind": "gblup_model_bundle",
        "created_at": run_manifest.utc_now_iso(),
        "encoding": gs_panel.SUPPORTED_ENCODING_SCHEMA,
        "dtype": "float64",
        "species": dataset.manifest["species"],
        "reference": dataset.manifest["reference"],
        "effect_allele": "ALT",
        "trait": dataset.manifest["trait"],
        "marker_ids": data.marker_names.tolist(),
        "training_sample_ids": list(dict.fromkeys(dataset.observations.sample_id)),
        "training_observations": len(data.phenotypes),
        "provenance": dataset.provenance,
        "evaluation_checksums": evidence_hashes,
        "evaluation_plan_hash": plan["plan_hash"],
        "preprocessing": {**kwargs, "imputation": "fixed_training_mean", "pca": False},
        "prediction_qc": prediction_qc,
        "fit": {
            "intercept": fit.intercept,
            "lambda_ratio": fit.lambda_ratio,
            "denominator": relations.denominator,
            "diagonal_jitter": gblup.DIAGONAL_JITTER,
        },
        "versions": run_manifest.library_versions(["numpy", "scipy"]),
        "source_file_checksums": run_manifest.source_file_checksums(
            [
                Path(__file__),
                Path(gblup.__file__),
                Path(controlled_qc.__file__),
                Path(gs_dataset.__file__),
                Path(gs_panel.__file__),
                Path(run_manifest.__file__),
            ]
        ),
        "model_card": {
            "status": "research_only",
            "review_reference": None,
            "scope": scope,
            "validation_scenario": plan["policy"],
            "fit_final_is_oof": False,
            "real_adzuki_performance": "unverified",
            "unverified_conditions": "independent target cohorts/generations/environments and deployment performance",
            "privacy": "trained coefficients and training identifiers are protected like training data",
            "uncertainty": "point predictions only; no calibrated prediction interval",
        },
    }
    if dataset.selection_unit == "line":
        metadata["phenotype_processing"] = dataset.phenotype_processing
        metadata["model_card"]["prediction_target"] = "within_trial_raw_line_mean"
        metadata["model_card"]["unverified_conditions"] += (
            "; other trial years/sites, environmental correction and breeding values"
        )
    bundle = GblupBundle(
        metadata, relations.retained_markers, relations.marker_means, coefficients
    )
    gs_dataset.verify_unchanged(dataset)
    if {p.name: run_manifest.sha256_file(p) for p in evidence_paths} != evidence_hashes:
        raise RuntimeError("evaluation evidence changed during fit-final")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".gblup-bundle-", dir=output_dir.parent))
    try:
        np.savez_compressed(
            temporary / "parameters.npz",
            marker_mask=bundle.marker_mask,
            marker_means=bundle.marker_means,
            marker_coefficients=bundle.marker_coefficients,
        )
        metadata["checksums"] = {
            "parameters.npz": run_manifest.sha256_file(temporary / "parameters.npz")
        }
        metadata["bundle_hash"] = run_manifest.canonical_json_hash(metadata)
        gs_evaluate.write_json(temporary / "bundle.json", metadata)
        # Serialization validation happens before publication.
        loaded = load_bundle(temporary)
        np.testing.assert_allclose(
            loaded.predict_matrix(data.genotypes),
            bundle.predict_matrix(data.genotypes),
            atol=1e-10,
            rtol=1e-10,
        )
        (temporary / "model-card.md").write_text(
            "# GBLUP model card\n\nStatus: research only.\n\n"
            "Fit-final is a refit on the full supplied training set; it is not OOF evaluation. "
            "Real adzuki performance is unverified. Point predictions are not calibrated intervals.\n\n"
            "Scope, scenario, update trigger and rollback reference are recorded in bundle.json. "
            "Parameters and identifiers require the same protection as training data.\n",
            encoding="utf-8",
        )
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return bundle


def load_bundle(path: Path, *, max_bytes=512 * 1024**2) -> GblupBundle:
    metadata_path = path / "bundle.json"
    if metadata_path.stat().st_size > max_bytes:
        raise ValueError("bundle metadata exceeds memory budget")
    metadata = json.loads(metadata_path.read_text())
    if (
        not isinstance(metadata, dict)
        or type(metadata.get("schema_version")) is not int
        or metadata.get("schema_version") not in (1, 2)
        or metadata.get("kind") != "gblup_model_bundle"
    ):
        raise ValueError("unsupported model bundle schema")
    expected = metadata.get("bundle_hash")
    if (
        run_manifest.canonical_json_hash(
            {k: v for k, v in metadata.items() if k != "bundle_hash"}
        )
        != expected
    ):
        raise ValueError("bundle manifest checksum mismatch")
    if (
        metadata.get("encoding") != gs_panel.SUPPORTED_ENCODING_SCHEMA
        or metadata.get("dtype") != "float64"
        or metadata.get("effect_allele") != "ALT"
    ):
        raise ValueError("unknown bundle encoding/dtype/effect allele")
    if metadata["schema_version"] == 2:
        validate_prediction_qc(metadata.get("prediction_qc"))
    elif "prediction_qc" in metadata:
        raise ValueError("legacy bundle cannot silently acquire a prediction QC policy")
    if metadata.get("model_card", {}).get("status") not in (
        "research_only",
        "conditional",
        "rejected",
    ):
        raise ValueError(
            "unsupported model status; this path cannot authorize customer release"
        )
    marker_ids = metadata.get("marker_ids")
    if (
        not isinstance(marker_ids, list)
        or not marker_ids
        or any(not isinstance(x, str) or not x.strip() for x in marker_ids)
        or len(set(marker_ids)) != len(marker_ids)
    ):
        raise ValueError("invalid bundle marker IDs")
    try:
        validate_scope(metadata["model_card"]["scope"])
        intercept = metadata["fit"]["intercept"]
    except (KeyError, TypeError) as error:
        raise ValueError("incomplete model bundle contract") from error
    if type(intercept) not in (int, float) or not np.isfinite(intercept):
        raise ValueError("nonfinite model intercept")
    parameter_path = path / "parameters.npz"
    if run_manifest.sha256_file(parameter_path) != metadata.get("checksums", {}).get(
        "parameters.npz"
    ):
        raise ValueError("bundle parameter checksum mismatch")
    with zipfile.ZipFile(parameter_path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > max_bytes:
            raise ValueError("expanded model parameters exceed memory budget")
    with np.load(parameter_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"marker_mask", "marker_means", "marker_coefficients"}:
            raise ValueError("unexpected parameter arrays")
        mask, means, coefficients = (
            arrays[key]
            for key in ("marker_mask", "marker_means", "marker_coefficients")
        )
    if (
        mask.dtype != np.bool_
        or mask.shape != (len(marker_ids),)
        or not mask.any()
        or means.dtype != np.float64
        or coefficients.dtype != np.float64
        or means.shape != (int(mask.sum()),)
        or coefficients.shape != means.shape
        or not np.isfinite(means).all()
        or not np.isfinite(coefficients).all()
        or (np.abs(means) > 1).any()
    ):
        raise ValueError("invalid parameter shape/dtype/values")
    return GblupBundle(metadata, mask, means, coefficients)


def predict(
    bundle_path: Path,
    panel_dir: Path,
    *,
    context,
    expected_sample_ids=None,
    allow_reorder=False,
):
    bundle = load_bundle(bundle_path)
    metadata = bundle.metadata
    context_fields = {
        "species",
        "assembly_id",
        "cohort_id",
        "generation",
    }
    processing = metadata.get("phenotype_processing")
    if processing is not None:
        context_fields.add("target_trial")
    if not isinstance(context, dict) or set(context) != context_fields:
        raise ValueError(
            "prediction context requires species, assembly_id, cohort_id, generation and target_trial for a line-mean model"
        )
    if processing is not None and context["target_trial"] != processing["trial"]:
        raise ValueError(
            "line-mean model is restricted to its declared target trial; other years/sites are unvalidated"
        )
    scope = metadata["model_card"]["scope"]
    if (
        context["species"] != metadata["species"]
        or context["assembly_id"] != metadata["reference"]["assembly_id"]
        or context["cohort_id"] not in scope["cohort_ids"]
        or context["generation"] not in scope["generations"]
    ):
        raise ValueError(
            "prediction input is outside the declared model scope/reference"
        )
    panel = gs_panel.load_gs_panel(panel_dir, cohort_id=context["cohort_id"])
    gs_dataset.validate_reference(
        panel, metadata["reference"], expected_assembly=context["assembly_id"]
    )
    marker_ids = panel.variant_keys.tolist()
    expected_markers = metadata["marker_ids"]
    if set(marker_ids) != set(expected_markers):
        raise ValueError(
            "missing or extra markers; substitution/allele flips are not supported"
        )
    if marker_ids != expected_markers and not allow_reorder:
        raise ValueError("marker order mismatch; explicit allow_reorder is required")
    samples = panel.sample_ids.tolist()
    expected_samples = (
        samples if expected_sample_ids is None else list(expected_sample_ids)
    )
    if len(set(expected_samples)) != len(expected_samples) or set(samples) != set(
        expected_samples
    ):
        raise ValueError("prediction sample ID set mismatch")
    if samples != expected_samples and not allow_reorder:
        raise ValueError("sample order mismatch; explicit allow_reorder is required")
    marker_positions = {name: index for index, name in enumerate(marker_ids)}
    sample_positions = {name: index for index, name in enumerate(samples)}
    matrix = panel.genotypes[
        np.ix_(
            [sample_positions[s] for s in expected_samples],
            [marker_positions[m] for m in expected_markers],
        )
    ]
    values = bundle.predict_matrix(matrix)
    _, observed, rates, reasons = bundle.prediction_qc(matrix)
    withheld = np.array([bool(reason) for reason in reasons], dtype=bool)
    if not np.isfinite(values[~withheld]).all():
        raise RuntimeError("nonfinite bundle prediction")
    card = metadata["model_card"]
    result = pd.DataFrame(
        {
            "sample_id": expected_samples,
            "prediction": values,
            "trait": metadata["trait"]["name"],
            "unit": metadata["trait"]["unit"],
            "bundle_hash": metadata["bundle_hash"],
            "marker_reordered": marker_ids != expected_markers,
            "sample_reordered": samples != expected_samples,
            "prediction_status": np.where(withheld, "withheld", "available"),
            "reason_codes": ["|".join(reason) for reason in reasons],
            "observed_retained_markers": observed,
            "retained_marker_call_rate": rates,
            "model_status": card["status"],
            "review_reference": card.get("review_reference"),
            "uncertainty": "point_prediction_only; calibrated_interval_unavailable",
            "applicability": "operator_declared_scope; statistical_applicability_not_assessed",
            "validation_scenario": json.dumps(
                card.get("validation_scenario"), sort_keys=True
            ),
            "declared_scope": json.dumps(card["scope"], sort_keys=True),
            "prediction_target": card.get("prediction_target", "raw_observation"),
        }
    )
    result.attrs["delivery_metadata"] = {
        "schema_version": 1,
        "kind": "gs_prediction_delivery",
        "bundle_hash": metadata["bundle_hash"],
        "context": context,
        "model_card": card,
        "prediction_qc": metadata.get("prediction_qc"),
        "phenotype_processing": metadata.get("phenotype_processing"),
        "counts": {
            "samples": len(result),
            "available": int((~withheld).sum()),
            "withheld": int(withheld.sum()),
        },
        "statistical_applicability": "not_assessed",
        "calibrated_prediction_interval": "unavailable",
        "customer_release_authorized": False,
    }
    return result


def write_predictions(result, output):
    """Publish CSV, withheld table and checksum-bound receipt without overwrites."""
    output = Path(output)
    targets = [
        output,
        output.with_suffix(output.suffix + ".withheld.csv"),
        output.with_suffix(output.suffix + ".metadata.json"),
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("prediction delivery output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".prediction-delivery-", dir=output.parent
    ) as directory:
        staged = [Path(directory) / path.name for path in targets]
        result.to_csv(staged[0], index=False, na_rep="")
        result[result.prediction_status == "withheld"].to_csv(
            staged[1], index=False, na_rep=""
        )
        payload = {
            **result.attrs["delivery_metadata"],
            "checksums": {
                path.name: run_manifest.sha256_file(path) for path in staged[:2]
            },
        }
        gs_evaluate.write_json(staged[2], payload)
        published = []
        try:
            for source, destination in zip(staged, targets, strict=True):
                # Hard-link creation is exclusive; a concurrent writer is never overwritten.
                os.link(source, destination)
                published.append(destination)
        except BaseException:
            for path in published:
                path.unlink()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit-final")
    fit_parser.add_argument("--dataset", type=Path, required=True)
    fit_parser.add_argument("--expected-assembly")
    fit_parser.add_argument("--evaluation", type=Path, required=True)
    fit_parser.add_argument("--scope", type=Path, required=True)
    fit_parser.add_argument(
        "--prediction-qc",
        type=Path,
        help="Fixed prediction QC policy JSON; default requires complete retained-marker calls (research only)",
    )
    fit_parser.add_argument("--output-dir", type=Path, required=True)
    prediction_parser = sub.add_parser("predict")
    prediction_parser.add_argument("--bundle", type=Path, required=True)
    prediction_parser.add_argument("--panel-dir", type=Path, required=True)
    prediction_parser.add_argument("--context", type=Path, required=True)
    prediction_parser.add_argument(
        "--sample-ids",
        type=Path,
        help="Expected sample ID order, one opaque ID per line",
    )
    prediction_parser.add_argument("--allow-reorder", action="store_true")
    prediction_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit-final":
        dataset = gs_dataset.load_dataset(
            args.dataset, expected_assembly=args.expected_assembly
        )
        fit_final(
            dataset,
            args.evaluation,
            args.output_dir,
            scope=json.loads(args.scope.read_text()),
            prediction_qc=json.loads(args.prediction_qc.read_text())
            if args.prediction_qc
            else None,
        )
    else:
        result = predict(
            args.bundle,
            args.panel_dir,
            context=json.loads(args.context.read_text()),
            expected_sample_ids=args.sample_ids.read_text().splitlines()
            if args.sample_ids
            else None,
            allow_reorder=args.allow_reorder,
        )
        write_predictions(result, args.output)


if __name__ == "__main__":
    main()
