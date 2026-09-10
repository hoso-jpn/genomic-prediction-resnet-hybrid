"""Predeclare, execute, and audit a paired GBLUP/ResNet LOFO experiment.

This runner uses the validated SoyNAM file contract and yield trait only.
It does not infer biological performance from synthetic smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import evaluation_split
import input_qc
import run_manifest
from soynam_data import list_family_files, load_soynam_dataset

ROOT = Path(__file__).resolve().parent
SOURCES = [
    "compare_baselines.py",
    "evaluation_split.py",
    "input_qc.py",
    "run_manifest.py",
    "run_measurements.py",
    "soynam_data.py",
    "model.py",
    "gblup_baseline.py",
    "resnet_baseline.py",
    "external_logging.py",
]
PACKAGES = ["numpy", "pandas", "scipy", "scikit-learn", "torch", "torch-geometric"]


def source_checksums():
    return run_manifest.source_file_checksums([ROOT / name for name in SOURCES])


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_data(data_dir, threshold):
    files = list_family_files(data_dir)
    checksums = run_manifest.describe_input_files(files)
    data = load_soynam_dataset(data_dir, family_files=files)
    qc = input_qc.apply_sample_qc(data, threshold)
    run_manifest.verify_input_files_unchanged(files, checksums)
    if len(set(qc.dataset.family_ids)) < 3:
        raise ValueError("comparison requires at least three families for nested LOFO")
    return qc.dataset, checksums


def plan_experiment(args):
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("declare at least two distinct ResNet seeds")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("seeds must be in [0, 2**32)")
    if any(
        getattr(args, name) < 1
        for name in (
            "max_epochs",
            "patience",
            "batch_size",
            "pca_components",
            "threads",
        )
    ):
        raise ValueError("training budgets and thread count must be positive")
    if not 0 < args.gblup_marker_rate < 1:
        raise ValueError("GBLUP marker observed-rate threshold must be in (0, 1)")
    if not args.dataset_id.strip() or not args.data_source.strip():
        raise ValueError("dataset identity and source must be nonempty")
    from dataclasses import asdict

    from resnet_baseline import ResNetConfig

    candidate = asdict(
        ResNetConfig(
            max_epochs=args.max_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            pca_components=args.pca_components,
            min_observed_rate=args.resnet_marker_rate,
        )
    )
    candidate.pop("seed")
    data, checksums = load_data(args.data_dir, args.max_sample_missing_rate)
    split = evaluation_split.make_plan(data, checksums, args.max_sample_missing_rate)
    config = {
        "schema_version": 1,
        "dataset_id": args.dataset_id,
        "data_source": args.data_source,
        "trait": "Yld (kg/ha)",
        "data_kind": args.data_kind,
        "created_at": run_manifest.utc_now_iso(),
        "git_commit": run_manifest.git_commit_sha(ROOT),
        "source_file_checksums": source_checksums(),
        "python_version": run_manifest.python_version(),
        "library_versions": run_manifest.library_versions(PACKAGES),
        "split_plan_hash": split["plan_hash"],
        "seeds": args.seeds,
        "resnet_search_space": {key: [value] for key, value in candidate.items()},
        "selection_policy": "single predeclared configuration; nested family validation chooses epoch only",
        "device": args.device,
        "max_sample_missing_rate": args.max_sample_missing_rate,
        "gblup_marker_rate": args.gblup_marker_rate,
        "resnet_marker_rate": args.resnet_marker_rate,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "pca_components": args.pca_components,
        "threads": args.threads,
        "bootstrap": {"seed": 42, "replicates": 2000, "unit": "family"},
        "comparison_scope": "existing pipelines; preprocessing differs, not architecture-only",
    }
    config["config_hash"] = run_manifest.canonical_json_hash(config)
    args.experiment_dir.mkdir(parents=True, exist_ok=False)
    run_manifest.write_json(args.experiment_dir / "split-plan.json", split)
    run_manifest.write_json(args.experiment_dir / "experiment.json", config)
    return config


def validate_experiment(experiment_dir, data_dir):
    config = read_json(experiment_dir / "experiment.json")
    content = {key: value for key, value in config.items() if key != "config_hash"}
    if run_manifest.canonical_json_hash(content) != config.get("config_hash"):
        raise ValueError("experiment configuration changed after declaration")
    if config["source_file_checksums"] != source_checksums():
        raise ValueError(
            "source code changed after declaration; create a new experiment"
        )
    if config["python_version"] != run_manifest.python_version() or config[
        "library_versions"
    ] != run_manifest.library_versions(PACKAGES):
        raise ValueError("Python or library versions changed after declaration")
    data, checksums = load_data(data_dir, config["max_sample_missing_rate"])
    split_path = experiment_dir / "split-plan.json"
    _, provenance = evaluation_split.load_plan(
        split_path, data, checksums, config["max_sample_missing_rate"]
    )
    if provenance["plan_hash"] != config["split_plan_hash"]:
        raise ValueError("split plan differs from the declared experiment")
    return config, data, checksums, provenance


def jobs(config):
    return [("gblup", None)] + [
        (f"resnet-seed-{seed}", seed) for seed in config["seeds"]
    ]


def run_experiment(experiment_dir, data_dir):
    config, data, _, _ = validate_experiment(experiment_dir, data_dir)
    if config["device"] == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA experiment requested but no CUDA device is available"
            )
    runs = experiment_dir / "runs"
    runs.mkdir(exist_ok=False)
    env = {
        **os.environ,
        "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": str(config["threads"]),
        "MKL_NUM_THREADS": str(config["threads"]),
        "OPENBLAS_NUM_THREADS": str(config["threads"]),
    }
    for name, seed in jobs(config):
        model = "gblup" if seed is None else "resnet"
        command = [
            sys.executable,
            str(ROOT / f"{model}_baseline.py"),
            "--data-dir",
            str(data_dir.resolve()),
            "--output-dir",
            str((runs / name).resolve()),
            "--split-file",
            str((experiment_dir / "split-plan.json").resolve()),
            "--max-sample-missing-rate",
            str(config["max_sample_missing_rate"]),
            "--min-marker-observed-rate",
            str(config[f"{model}_marker_rate"]),
        ]
        if seed is None:
            command += [
                "--expected-families",
                str(len(set(data.family_ids))),
                "--wandb-mode",
                "disabled",
            ]
        else:
            command += ["--device", config["device"], "--seed", str(seed)]
            for key in ("max_epochs", "patience", "batch_size", "pca_components"):
                command += [f"--{key.replace('_', '-')}", str(config[key])]
        with (runs / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                cwd=ROOT,
            )
        if result.returncode:
            raise RuntimeError(
                f"{name} failed; inspect its log; no complete report was written"
            )
    return report_experiment(experiment_dir, data_dir)


def metrics(observed, predicted):
    correlation = None
    if len(observed) > 1 and np.std(observed) > 0 and np.std(predicted) > 0:
        correlation = float(np.corrcoef(observed, predicted)[0, 1])
    return {
        "r": correlation,
        "rmse_kg_ha": float(np.sqrt(np.mean((observed - predicted) ** 2))),
    }


def audit_run(run_dir, name, seed, config, data, checksums, provenance):
    bundles = list((run_dir / "artifacts").iterdir())
    if len(bundles) != 1 or not bundles[0].is_dir():
        raise ValueError(f"{name}: expected exactly one artifact bundle")
    bundle = bundles[0]
    metadata = read_json(bundle / "metadata.json")
    split = read_json(bundle / "split.json")
    expected_outer = run_manifest.build_outer_split(data.sample_names, data.family_ids)
    model = "gblup" if seed is None else "resnet"
    if (
        metadata["input_files"] != checksums
        or metadata["split_plan"] != provenance
        or split["outer"] != expected_outer
        or metadata["model_name"] != model
        or metadata["seed"] != seed
    ):
        raise ValueError(f"{name}: input, split, model, or seed mismatch")
    required_sources = {
        f"{model}_baseline.py",
        "soynam_data.py",
        "input_qc.py",
        "evaluation_split.py",
        "run_measurements.py",
        "run_manifest.py",
        "external_logging.py" if model == "gblup" else "model.py",
    }
    expected_sources = {
        filename: config["source_file_checksums"][filename]
        for filename in required_sources
    }
    if metadata["source_file_checksums"] != expected_sources:
        raise ValueError(f"{name}: source checksum mismatch or missing source")
    required_packages = {"numpy", "pandas", "scikit-learn"} | (
        {"scipy"} if model == "gblup" else {"torch", "torch-geometric"}
    )
    expected_versions = {
        package: config["library_versions"][package] for package in required_packages
    }
    if metadata["library_versions"] != expected_versions:
        raise ValueError(f"{name}: environment mismatch or missing library")
    if metadata["python_version"] != config["python_version"]:
        raise ValueError(f"{name}: Python version mismatch")
    hyper = metadata["hyperparameters"]
    if hyper["min_observed_rate"] != config[f"{model}_marker_rate"]:
        raise ValueError(f"{name}: marker threshold mismatch")
    if (
        metadata["input_qc"]["max_sample_missing_rate"]
        != config["max_sample_missing_rate"]
    ):
        raise ValueError(f"{name}: sample threshold mismatch")
    if seed is not None:
        expected_hyper = {
            key: values[0] for key, values in config["resnet_search_space"].items()
        }
        if metadata["device_resolved"] != config["device"] or hyper != expected_hyper:
            raise ValueError(f"{name}: device or hyperparameter mismatch")
    frame = pd.read_csv(
        bundle / "predictions.csv",
        dtype={"family_id": str, "sample_name": str},
        keep_default_na=False,
        float_precision="round_trip",
    )
    keys = ["family_id", "sample_name"]
    if frame.duplicated(keys).any():
        raise ValueError(f"{name}: duplicate OOF sample")
    frame = frame.set_index(keys).sort_index()
    expected = (
        pd.DataFrame(
            {
                "family_id": data.family_ids,
                "sample_name": data.sample_names,
                "observed_yield_kg_ha": data.phenotypes,
            }
        )
        .set_index(keys)
        .sort_index()
    )
    observed = frame["observed_yield_kg_ha"].to_numpy(dtype=float)
    predicted = frame["predicted_yield_kg_ha"].to_numpy(dtype=float)
    if (
        not frame.index.equals(expected.index)
        or not np.isfinite(predicted).all()
        or not np.array_equal(
            observed, expected["observed_yield_kg_ha"].to_numpy(dtype=float)
        )
    ):
        raise ValueError(
            f"{name}: OOF identity, observed yield, or finite prediction mismatch"
        )
    per_family = {}
    for family in sorted(set(data.family_ids)):
        mask = frame.index.get_level_values("family_id") == family
        per_family[family] = metrics(observed[mask], predicted[mask])
    return {
        "name": name,
        "seed": seed,
        "pooled": metrics(observed, predicted),
        "families": per_family,
        "measurements": metadata["measurements"],
        "hyperparameters": hyper,
        "artifact_checksums": {
            p.name: run_manifest.sha256_file(p)
            for p in sorted(bundle.iterdir())
            if p.is_file()
        },
    }


def paired_uncertainty(records, config):
    """Descriptive family bootstrap of seed-mean ResNet minus GBLUP metrics."""
    result = {}
    for metric in ("r", "rmse_kg_ha"):
        differences = []
        for family, baseline in records[0]["families"].items():
            values = [run["families"][family][metric] for run in records[1:]]
            if baseline[metric] is not None and all(
                value is not None for value in values
            ):
                differences.append(float(np.mean(values) - baseline[metric]))
        if not differences:
            result[metric] = {
                "families": 0,
                "mean_difference": None,
                "interval_95": None,
            }
            continue
        rng = np.random.default_rng(config["bootstrap"]["seed"])
        means = rng.choice(
            differences, size=(config["bootstrap"]["replicates"], len(differences))
        ).mean(axis=1)
        result[metric] = {
            "families": len(differences),
            "mean_difference": float(np.mean(differences)),
            "interval_95": np.quantile(means, [0.025, 0.975]).tolist(),
        }
    return result


def report_experiment(experiment_dir, data_dir):
    config, data, checksums, provenance = validate_experiment(experiment_dir, data_dir)
    records = [
        audit_run(
            experiment_dir / "runs" / name,
            name,
            seed,
            config,
            data,
            checksums,
            provenance,
        )
        for name, seed in jobs(config)
    ]
    report = {
        "schema_version": 1,
        "status": "complete",
        "experiment": config,
        "split_plan": provenance,
        "runs": records,
        "paired_family_bootstrap": paired_uncertainty(records, config),
        "limitations": [
            "Synthetic results verify execution only and do not measure biological utility.",
            "Existing GBLUP and ResNet pipelines use different MAF, observed-rate comparisons, and transforms.",
            "Bootstrap treats held-out families as units; overlapping training sets and few families limit interpretation.",
            "GBLUP is deterministic and run once; every declared ResNet seed is reported without best-seed selection.",
            "Peak RSS is per process; CUDA memory covers the PyTorch allocator only.",
        ],
    }
    lines = [
        "# GBLUP / ResNet comparison",
        "",
        f"Data kind: **{config['data_kind']}**; trait: Yld (kg/ha).",
        "",
        "All declared runs completed and passed input/split/OOF alignment checks.",
        "",
        "| Run | Pooled r | Pooled RMSE (kg/ha) | Wall seconds | Peak RSS bytes | CUDA allocated bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        m = record["measurements"]
        lines.append(
            f"| {record['name']} | {record['pooled']['r']} | {record['pooled']['rmse_kg_ha']:.4f} | {m['wall_seconds']:.3f} | {m['process_peak_rss_bytes']} | {m['cuda_peak_allocated_bytes']} |"
        )
    lines += [
        "",
        "## Per-family results",
        "",
        "| Run | Family | r | RMSE (kg/ha) |",
        "|---|---|---:|---:|",
    ]
    for record in records:
        for family, values in record["families"].items():
            lines.append(
                f"| {record['name']} | {family.replace('|', '/')} | {values['r']} | {values['rmse_kg_ha']:.4f} |"
            )
    lines += [
        "",
        "## Paired uncertainty",
        "",
        "ResNet seed-mean minus GBLUP; descriptive 95% family bootstrap intervals.",
        "",
        "```json",
        json.dumps(report["paired_family_bootstrap"], indent=2),
        "```",
        "",
        "## Limits",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]
    run_manifest.write_json(experiment_dir / "comparison.json", report)
    temporary = experiment_dir / "comparison.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(experiment_dir / "comparison.md")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    plan = sub.add_parser("plan")
    for action in (plan, sub.add_parser("run"), sub.add_parser("report")):
        action.add_argument("--data-dir", type=Path, required=True)
        action.add_argument("--experiment-dir", type=Path, required=True)
    plan.add_argument("--dataset-id", required=True)
    plan.add_argument("--data-source", required=True)
    plan.add_argument("--data-kind", choices=("synthetic", "real"), required=True)
    plan.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    plan.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    for flag, default in (
        ("max-epochs", 200),
        ("patience", 20),
        ("batch-size", 64),
        ("pca-components", 64),
        ("threads", 1),
    ):
        plan.add_argument(f"--{flag}", type=int, default=default)
    for flag, default in (
        ("max-sample-missing-rate", 1.0),
        ("gblup-marker-rate", 0.1),
        ("resnet-marker-rate", 0.9),
    ):
        plan.add_argument(f"--{flag}", type=input_qc.fraction, default=default)
    args = parser.parse_args()
    if args.action == "plan":
        plan_experiment(args)
    elif args.action == "run":
        run_experiment(args.experiment_dir, args.data_dir)
    else:
        report_experiment(args.experiment_dir, args.data_dir)


if __name__ == "__main__":
    main()
