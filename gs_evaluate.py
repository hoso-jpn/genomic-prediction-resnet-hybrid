"""Plan and evaluate a continuous individual-level GS dataset offline on CPU."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

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
    config = validate_plan(dataset, plan)
    if output_dir.exists():
        raise FileExistsError(
            "output directory already exists; use a new run directory"
        )
    source_checksums = run_manifest.source_file_checksums(
        [
            Path(__file__),
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
    rows, records, arrays = [], [], {}
    for fold_index, fold in enumerate(plan["folds"]):
        train = np.array([ids[name] for name in fold["train"]])
        test = np.array([ids[name] for name in fold["test"]])
        gp, fit, relationships = gblup.predict_gblup_fold(
            train, test, data.genotypes, data.phenotypes
        )
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
                prefix
                + "_resnet_selection_pca_components": record.selection_transform.pca.components_,
                prefix
                + "_resnet_selection_pca_mean": record.selection_transform.pca.mean_,
                prefix + "_resnet_final_mask": record.final_transform.retained_markers,
                prefix + "_resnet_final_means": record.final_transform.marker_means,
                prefix + "_resnet_scales": record.final_transform.marker_scales,
                prefix
                + "_resnet_pca_components": record.final_transform.pca.components_,
                prefix + "_resnet_pca_mean": record.final_transform.pca.mean_,
            }
        )
        records.append(
            {
                "fold": fold_index,
                "best_epoch": record.best_epoch,
                "validation_family": record.validation_family,
                "gblup_lambda": fit.lambda_ratio,
                "gblup_intercept": fit.intercept,
                "gblup_denominator": relationships.denominator,
                "target_mean": record.final_target_mean,
                "target_scale": record.final_target_scale,
            }
        )
        for model, predictions in (("gblup", gp), ("resnet", rp)):
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
                "comparison": "pipeline_vs_pipeline; model-specific QC",
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
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    dataset = gs_dataset.load_dataset(
        args.dataset, expected_assembly=args.expected_assembly
    )
    if args.command == "plan":
        plan = make_plan(
            dataset,
            resnet.ResNetConfig(max_epochs=args.max_epochs, seed=args.seed),
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
