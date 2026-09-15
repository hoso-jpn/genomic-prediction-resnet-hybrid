"""Plan and evaluate a continuous individual-level GS dataset offline on CPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import controlled_qc
import gblup_baseline as gblup
import gs_dataset
import gs_scenarios
import gs_selection
import resnet_baseline as resnet
import run_manifest

# Public aliases preserve the initial generic input API.
make_plan = gs_scenarios.make_plan
validate_plan = gs_scenarios.validate_plan


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def evaluate(dataset, plan, output_dir: Path):
    """Record every executed candidate, including failures, without overwriting runs."""
    validate_plan(dataset, plan)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("output directory already exists")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    receipt = output_dir.parent / f"{output_dir.name}.attempt-{uuid.uuid4().hex}.json"
    attempt = {
        "schema_version": 1,
        "plan_hash": plan["plan_hash"],
        "configuration": plan["resnet_config"],
        "started_at": run_manifest.utc_now_iso(),
        "status": "running",
    }
    write_json(receipt, attempt)
    start = time.perf_counter()
    try:
        result = _evaluate(dataset, plan, output_dir)
    except Exception as error:
        attempt.update(
            status="failed", error_type=type(error).__name__, error=str(error)
        )
        raise
    else:
        attempt["status"] = "completed"
        return result
    finally:
        attempt["wall_seconds"] = time.perf_counter() - start
        pending = receipt.with_suffix(".pending")
        write_json(pending, attempt)
        os.replace(pending, receipt)


def _evaluate(dataset, plan, output_dir: Path):
    config = validate_plan(dataset, plan)
    if output_dir.exists():
        raise FileExistsError(
            "output directory already exists; use a new run directory"
        )
    source_checksums = run_manifest.source_file_checksums(
        [
            Path(__file__),
            Path(controlled_qc.__file__),
            Path(gs_dataset.__file__),
            Path(gs_scenarios.__file__),
            Path(gs_selection.__file__),
            Path(gblup.__file__),
            Path(resnet.__file__),
            Path(resnet.model.__file__),
            Path(run_manifest.__file__),
            Path(gs_dataset.gs_panel.__file__),
        ]
    )
    ids = {name: i for i, name in enumerate(dataset.baseline.sample_names)}
    data = dataset.baseline
    controlled = config.qc_mode == "controlled"
    qc_kwargs = (
        {
            "qc_mode": "controlled",
            "min_observed_rate": config.min_observed_rate,
            "maf_threshold": config.maf_threshold,
        }
        if controlled
        else {}
    )
    rows, records, arrays = [], [], {}
    for fold_index, fold in enumerate(plan["folds"]):
        train = np.array([ids[name] for name in fold["train"]])
        test = np.array([ids[name] for name in fold["test"]])
        start = time.perf_counter()
        gp, fit, relationships = gblup.predict_gblup_fold(
            train, test, data.genotypes, data.phenotypes, **qc_kwargs
        )
        gblup_seconds = time.perf_counter() - start
        start = time.perf_counter()
        rp, record = resnet.predict_resnet_fold(
            data.genotypes,
            data.phenotypes,
            data.family_ids,
            train,
            test,
            fold_index,
            config,
            torch.device("cpu"),
            inner_indices=(
                np.array([ids[name] for name in fold["fit"]]),
                np.array([ids[name] for name in fold["validation"]]),
            ),
        )
        resnet_seconds = time.perf_counter() - start
        qc_records = None
        model_predictions = [("gblup", gp), ("resnet", rp)]
        if controlled:
            inner_fit = np.array([ids[name] for name in fold["fit"]])
            inner_validation = np.array([ids[name] for name in fold["validation"]])
            inner_relationships = gblup.prepare_fold_relationships(
                data.genotypes[inner_fit], data.genotypes[inner_validation], **qc_kwargs
            )
            if not np.array_equal(
                relationships.retained_markers, record.final_transform.retained_markers
            ) or not np.array_equal(
                inner_relationships.retained_markers,
                record.selection_transform.retained_markers,
            ):
                raise RuntimeError("controlled baseline masks differ")
            qc_records = {
                stage: controlled_qc.mask_record(
                    data.marker_names,
                    mask,
                    min_observed_rate=config.min_observed_rate,
                    maf_threshold=config.maf_threshold,
                )
                for stage, mask in (
                    ("selection", inner_relationships.retained_markers),
                    ("final", relationships.retained_markers),
                )
            }
            ridge = controlled_qc.ridge_predict(
                data.genotypes[train],
                data.genotypes[test],
                data.phenotypes[train],
                relationships.retained_markers,
            )
            model_predictions.append(("ridge_fixed_alpha_1", ridge))
        prefix = f"fold_{fold_index}"
        arrays.update(
            {
                prefix + "_gblup_mask": relationships.retained_markers,
                prefix + "_gblup_means": relationships.marker_means,
                prefix
                + "_resnet_selection_mask": record.selection_transform.retained_markers,
                prefix
                + "_resnet_selection_means": record.selection_transform.marker_means,
                prefix
                + "_resnet_selection_scales": record.selection_transform.marker_scales,
                prefix + "_resnet_selection_pca_components": resnet.pca_arrays(
                    record.selection_transform
                )["components"],
                prefix + "_resnet_selection_pca_mean": resnet.pca_arrays(
                    record.selection_transform
                )["mean"],
                prefix + "_resnet_final_mask": record.final_transform.retained_markers,
                prefix + "_resnet_final_means": record.final_transform.marker_means,
                prefix + "_resnet_scales": record.final_transform.marker_scales,
                prefix + "_resnet_pca_components": resnet.pca_arrays(
                    record.final_transform
                )["components"],
                prefix + "_resnet_pca_mean": resnet.pca_arrays(record.final_transform)[
                    "mean"
                ],
            }
        )
        records.append(
            {
                "fold": fold_index,
                "common_qc": qc_records,
                "wall_seconds": {"gblup": gblup_seconds, "resnet": resnet_seconds},
                "best_epoch": record.best_epoch,
                "validation_family": record.validation_family,
                "gblup_lambda": fit.lambda_ratio,
                "gblup_intercept": fit.intercept,
                "gblup_denominator": relationships.denominator,
                "target_mean": record.final_target_mean,
                "target_scale": record.final_target_scale,
            }
        )
        for model, predictions in model_predictions:
            for position, prediction in zip(test, predictions, strict=True):
                observation = dataset.observations.iloc[position]
                rows.append(
                    {
                        "model": model,
                        "fold": fold_index,
                        "observation_id": observation.observation_id,
                        "sample_id": observation.sample_id,
                        "line_id": observation.line_id,
                        "family_id": observation.family_id,
                        "year": int(observation.year),
                        "site": observation.site,
                        "observed": float(data.phenotypes[position]),
                        "predicted": float(prediction),
                        "trait": dataset.manifest["trait"]["name"],
                        "unit": dataset.manifest["trait"]["unit"],
                    }
                )
    predictions = pd.DataFrame(rows)
    if not np.isfinite(predictions.predicted).all():
        raise RuntimeError("nonfinite model predictions")
    report = gs_selection.feasibility_report(predictions, dataset, plan)
    gs_dataset.verify_unchanged(dataset)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".gs-evaluation-", dir=output_dir.parent))
    try:
        write_json(temporary / "split.json", plan)
        write_json(temporary / "feasibility.json", report)
        (temporary / "feasibility.md").write_text(
            "# GS feasibility review\n\nScenario: "
            + plan["policy"]["scenario"]
            + "\n\n"
            + "\n".join(
                f"- {model}: {result['decision']} ({result['reason']})"
                for model, result in report["models"].items()
            )
            + "\n\nReal adzuki performance is unverified. See feasibility.json for metrics, "
            "group uncertainty, applicable population, further trials and update review. "
            "Observed selection differential is not future genetic gain.\n",
            encoding="utf-8",
        )
        write_json(
            temporary / "metadata.json",
            {
                "schema_version": 1,
                "kind": "held_out_evaluation",
                "provenance": dataset.provenance,
                "manifest": dataset.manifest,
                "excluded_samples_no_phenotype": list(dataset.excluded_samples),
                "excluded_missing_observations": list(dataset.excluded_observations),
                "git_commit": run_manifest.git_commit_sha(Path(__file__).parent),
                "source_file_checksums": source_checksums,
                "versions": run_manifest.library_versions(
                    ["numpy", "pandas", "scipy", "torch", "scikit-learn"]
                ),
                "device": "cpu",
                "real_adzuki_performance": "unverified",
                "comparison": "controlled_common_qc"
                if controlled
                else "pipeline_vs_pipeline; model-specific QC",
                "candidate_budget": {
                    "resnet_candidates": 1,
                    "seeds": [config.seed],
                    "max_epochs": config.max_epochs,
                    "ridge_alpha": 1.0 if controlled else None,
                },
                "model_adoption": "deferred_pending_independent_trials; GBLUP sufficient remains valid",
                "folds": records,
            },
        )
        np.savez_compressed(temporary / "preprocessing.npz", **arrays)
        predictions.to_csv(temporary / "predictions.csv", index=False)
        temporary.rename(output_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "run"))
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-assembly")
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--policy", type=Path, help="Predeclared scenario JSON (plan only)"
    )
    parser.add_argument(
        "--comparison-mode", choices=("legacy", "controlled"), default="legacy"
    )
    parser.add_argument("--no-pca", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = gs_dataset.load_dataset(
        args.dataset, expected_assembly=args.expected_assembly
    )
    if args.command == "plan":
        plan = make_plan(
            dataset,
            resnet.ResNetConfig(
                max_epochs=args.max_epochs,
                seed=args.seed,
                qc_mode=args.comparison_mode,
                use_pca=not args.no_pca,
                maf_threshold=0.05 if args.comparison_mode == "controlled" else 0.01,
            ),
            json.loads(args.policy.read_text()) if args.policy else None,
        )
        validate_plan(dataset, plan)
        write_json(args.split, plan)
    else:
        if args.output_dir is None:
            parser.error("run requires --output-dir")
        evaluate(dataset, json.loads(args.split.read_text()), args.output_dir)


if __name__ == "__main__":
    main()
