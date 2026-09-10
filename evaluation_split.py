"""Checksummed LOFO plans shared by the comparison runner and both baselines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

import run_manifest


def make_plan(dataset, input_files, max_sample_missing_rate=1.0):
    plan = {
        "schema_version": 1,
        "kind": "fixed_lofo_plan",
        "input_files": input_files,
        "max_sample_missing_rate": max_sample_missing_rate,
        "outer": run_manifest.build_outer_split(
            dataset.sample_names.tolist(), dataset.family_ids.tolist()
        ),
    }
    plan["plan_hash"] = run_manifest.canonical_json_hash(plan)
    return plan


def load_plan(path: Path, dataset, input_files, max_sample_missing_rate=1.0):
    """Validate exact data, sample order, QC policy, and every family partition.

    Equality with the canonical LOFO contract also rejects omitted/duplicated
    folds, training/test overlap, reordered individuals, and unknown fields.
    The returned partitions are taken from the saved plan, not regenerated.
    """
    raw = path.read_bytes()
    plan = json.loads(raw)
    if not isinstance(plan, dict) or type(plan.get("schema_version")) is not int:
        raise ValueError("invalid fixed LOFO plan schema")
    content = {key: value for key, value in plan.items() if key != "plan_hash"}
    if run_manifest.canonical_json_hash(content) != plan.get("plan_hash"):
        raise ValueError("split plan hash does not match its content")
    expected = make_plan(dataset, input_files, max_sample_missing_rate)
    if plan != expected:
        raise ValueError(
            "split plan does not match input checksums, sample order, QC, or LOFO folds"
        )
    splits = []
    for fold in plan["outer"]["folds"]:
        train = np.flatnonzero(np.isin(dataset.family_ids, fold["train_family_ids"]))
        test = np.flatnonzero(np.isin(dataset.family_ids, fold["test_family_ids"]))
        splits.append((train, test))
    provenance = {
        "filename": path.name,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "plan_hash": plan["plan_hash"],
    }
    return splits, provenance


def verify_unchanged(path: Path | None, provenance) -> None:
    if path is not None and run_manifest.sha256_file(path) != provenance["sha256"]:
        raise RuntimeError("split plan changed during execution")
