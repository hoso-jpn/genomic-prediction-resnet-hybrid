"""Fit-final and phenotype-free offline inference for a checksummed GBLUP bundle."""

from __future__ import annotations

import argparse
import json
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

    def predict_matrix(self, matrix):
        values = np.asarray(matrix, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.marker_mask):
            raise ValueError("prediction marker dimension mismatch")
        if (
            np.isinf(values).any()
            or not np.isin(values[np.isfinite(values)], [-1, 0, 1]).all()
        ):
            raise ValueError("unknown genotype encoding in prediction matrix")
        retained = values[:, self.marker_mask]
        centered = (
            np.where(np.isnan(retained), self.marker_means, retained)
            - self.marker_means
        )
        # No means, scales, frequencies, masks or model parameters are refitted.
        return self.metadata["fit"]["intercept"] + centered @ self.marker_coefficients


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


def fit_final(dataset, evaluation_dir: Path, output_dir: Path, *, scope):
    """Refit the adopted GBLUP QC on all supplied training observations."""
    validate_scope(scope)
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
    evidence_hashes = {p.name: run_manifest.sha256_file(p) for p in evidence_paths}
    plan = json.loads(evidence_paths[0].read_text())
    config = gs_evaluate.validate_plan(dataset, plan)
    evaluation = json.loads(evidence_paths[1].read_text())
    if (
        evaluation.get("kind") != "held_out_evaluation"
        or evaluation.get("provenance") != dataset.provenance
    ):
        raise ValueError("evaluation metadata does not match training dataset")
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
        "schema_version": 1,
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
            "scope": scope,
            "validation_scenario": plan["policy"],
            "fit_final_is_oof": False,
            "real_adzuki_performance": "unverified",
            "unverified_conditions": "independent target cohorts/generations/environments and deployment performance",
            "privacy": "trained coefficients and training identifiers are protected like training data",
            "uncertainty": "point predictions only; no calibrated prediction interval",
        },
    }
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
        or metadata.get("schema_version") != 1
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
    if not isinstance(context, dict) or set(context) != {
        "species",
        "assembly_id",
        "cohort_id",
        "generation",
    }:
        raise ValueError(
            "prediction context requires species, assembly_id, cohort_id and generation"
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
    if not np.isfinite(values).all():
        raise RuntimeError("nonfinite bundle prediction")
    return pd.DataFrame(
        {
            "sample_id": expected_samples,
            "prediction": values,
            "trait": metadata["trait"]["name"],
            "unit": metadata["trait"]["unit"],
            "bundle_hash": metadata["bundle_hash"],
            "marker_reordered": marker_ids != expected_markers,
            "sample_reordered": samples != expected_samples,
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit_parser = sub.add_parser("fit-final")
    fit_parser.add_argument("--dataset", type=Path, required=True)
    fit_parser.add_argument("--expected-assembly")
    fit_parser.add_argument("--evaluation", type=Path, required=True)
    fit_parser.add_argument("--scope", type=Path, required=True)
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
        with args.output.open("x", encoding="utf-8", newline="") as handle:
            result.to_csv(handle, index=False)


if __name__ == "__main__":
    main()
