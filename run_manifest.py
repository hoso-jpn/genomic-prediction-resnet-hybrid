"""Shared, model-agnostic helpers for saving auditable run artifacts.

This module only provides generic building blocks (run identifiers,
checksums, git metadata, the outer leave-one-family-out split record, and
atomic JSON/NPZ writing). It intentionally holds no GBLUP- or ResNet-specific
preprocessing or metric logic: callers build their own preprocessing and
metrics documents and pass them in for writing.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCHEMA_VERSION = 1


def new_run_id(now: datetime | None = None, suffix: str | None = None) -> str:
    """Build a sortable, human-readable run identifier."""
    moment = now if now is not None else datetime.now(UTC)
    timestamp = moment.strftime("%Y%m%dT%H%M%SZ")
    token = suffix if suffix is not None else uuid.uuid4().hex[:8]
    return f"{timestamp}-{token}"


def utc_now_iso(now: datetime | None = None) -> str:
    """Format a UTC timestamp as an ISO-8601 string with a 'Z' suffix."""
    moment = now if now is not None else datetime.now(UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    """Compute a file's SHA-256 hex digest without loading it fully into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_file_checksums(paths: Sequence[Path]) -> dict[str, str]:
    """Checksum the source files that make up the running code, keyed by filename.

    Raises if two paths share a filename: silently keeping only the last one
    would drop a checksum without any indication that it happened.
    """
    checksums: dict[str, str] = {}
    for path in paths:
        name = Path(path).name
        if name in checksums:
            raise ValueError(f"duplicate source file name in checksum set: '{name}'")
        checksums[name] = sha256_file(Path(path))
    return checksums


def describe_input_files(
    family_files: Sequence[tuple[str, Path, Path]],
) -> list[dict[str, str]]:
    """Describe input data files by name and checksum, never by absolute path."""
    return [
        {
            "family_id": family_id,
            "phenotype_file": Path(phenotype_path).name,
            "phenotype_sha256": sha256_file(Path(phenotype_path)),
            "genotype_file": Path(genotype_path).name,
            "genotype_sha256": sha256_file(Path(genotype_path)),
        }
        for family_id, phenotype_path, genotype_path in family_files
    ]


def verify_input_files_unchanged(
    family_files: Sequence[tuple[str, Path, Path]],
    expected: list[dict[str, str]],
) -> None:
    """Re-checksum the same files and raise if any changed since first read.

    ``expected`` should be the ``describe_input_files`` result captured
    before loading. This only catches a change to the same on-disk file
    paths (content replaced in place); it cannot detect files that were
    added or removed under a differently-resolved directory listing, which
    is why the file list itself must be fixed once up front rather than
    re-resolved here.
    """
    current = describe_input_files(family_files)
    if current != expected:
        raise RuntimeError(
            "input files changed while the run was in progress; "
            "re-run against a stable data directory"
        )


def git_commit_sha(repo_dir: Path | None = None) -> str | None:
    """Resolve the current git commit SHA, preferring GIT_COMMIT_SHA, else None.

    Never raises: git may be unavailable (e.g. inside a container image that
    excludes .git from its build context), in which case None is returned.
    """
    env_value = os.environ.get("GIT_COMMIT_SHA")
    if env_value:
        return env_value.strip() or None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    commit = result.stdout.strip()
    return commit or None


def python_version() -> str:
    """Return the running Python interpreter's version string."""
    return platform.python_version()


def library_versions(package_names: Sequence[str]) -> dict[str, str | None]:
    """Read installed versions for the given distribution names, best-effort."""
    versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


PATH_OPTIONS = frozenset({"--data-dir", "--output-dir"})


def _safe_path_identifier(value: str) -> str:
    """Reduce a filesystem path to a basename-only identifier.

    Works regardless of the original spelling (absolute, relative, or a
    "./"-prefixed path) since it only ever looks at ``Path(value).name``; the
    original path is never retained anywhere.
    """
    return Path(value).name or "unnamed-path"


def sanitize_command(executable: str, argv: Sequence[str]) -> dict[str, Any]:
    """Build a structured command record with no path or credential leakage.

    Recognizes the GBLUP/ResNet CLI's known path-bearing options
    (``--data-dir``, ``--output-dir``) in both "--option value" and
    "--option=value" forms and replaces their value with a basename-only
    identifier. Unrecognized tokens are kept as-is. Extending this to a
    future sensitive option only requires adding its name to
    ``PATH_OPTIONS`` (or a similarly small, explicit allowlist) rather than
    guessing at every possible spelling after the fact.
    """
    arguments: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        option, separator, inline_value = token.partition("=")
        if option in PATH_OPTIONS and separator:
            arguments.append(f"{option}={_safe_path_identifier(inline_value)}")
            index += 1
            continue
        if token in PATH_OPTIONS:
            arguments.append(token)
            if index + 1 < len(argv):
                arguments.append(_safe_path_identifier(argv[index + 1]))
                index += 2
                continue
            index += 1
            continue
        arguments.append(token)
        index += 1
    return {"executable": executable, "arguments": arguments}


def json_safe_float(value: float) -> float | None:
    """Convert a non-finite float to ``None`` so it survives strict JSON output.

    Use this only where a value is legitimately allowed to be undefined
    (e.g. a correlation computed from near-constant predictions); it must
    never be used to paper over an unexpected non-finite value elsewhere,
    which ``allow_nan=False`` below will still raise on.
    """
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def canonical_json_hash(payload: dict[str, Any]) -> str:
    """Hash a JSON-serializable document deterministically, order-sensitive."""
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def build_outer_split(
    sample_ids: Sequence[str], family_ids: Sequence[str]
) -> dict[str, Any]:
    """Build the shared leave-one-family-out outer split record and its hash.

    Both GBLUP and ResNet call this with the same (sample_ids, family_ids)
    ordering used to construct their sklearn.LeaveOneGroupOut folds, so the
    resulting ``outer_split_hash`` can be compared across models directly.
    """
    if len(sample_ids) != len(family_ids):
        raise ValueError("sample_ids and family_ids must have the same length")

    ordered_samples = [
        {"sample_id": str(sample_id), "family_id": str(family_id)}
        for sample_id, family_id in zip(sample_ids, family_ids)
    ]
    unique_families = sorted({str(family_id) for family_id in family_ids})
    if len(unique_families) < 2:
        raise ValueError("at least two families are required for leave-one-family-out")

    folds = []
    for fold_index, held_out_family in enumerate(unique_families):
        train_family_ids = [
            family for family in unique_families if family != held_out_family
        ]
        folds.append(
            {
                "fold_index": fold_index,
                "held_out_family": held_out_family,
                "train_family_ids": train_family_ids,
                "test_family_ids": [held_out_family],
            }
        )

    outer = {
        "strategy": "leave_one_family_out",
        "ordered_samples": ordered_samples,
        "folds": folds,
    }
    outer["outer_split_hash"] = canonical_json_hash(outer)
    return outer


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON document with stable formatting for diffability.

    ``allow_nan=False`` makes an unexpected NaN/Infinity fail loudly at
    write time instead of silently producing non-standard JSON; callers
    must convert any legitimately-undefined float (see
    ``json_safe_float``) before it reaches this function.
    """
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
        default=_json_default,
    )
    Path(path).write_text(text + "\n")


def write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write named NumPy arrays to a compressed .npz archive."""
    np.savez_compressed(Path(path), **arrays)


def _atomic_write_csv(path: Path, frame: pd.DataFrame, unique_suffix: str) -> None:
    """Write a CSV via a same-filesystem temp file, then ``os.replace`` it in.

    ``unique_suffix`` (the run_id) is embedded in the temp filename so two
    concurrent writers targeting the same ``path`` never share one temp
    file. The temp file is always removed: a successful ``os.replace``
    leaves nothing to clean up, and ``unlink(missing_ok=True)`` handles both
    that case and a failed ``to_csv`` without raising a second error.
    """
    path = Path(path)
    tmp_path = path.with_name(f".{path.name}.{unique_suffix}.tmp")
    try:
        frame.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_run_artifacts(
    *,
    output_dir: Path,
    run_id: str,
    metadata: dict[str, Any],
    split: dict[str, Any],
    preprocessing: dict[str, Any],
    preprocessing_arrays: Mapping[str, np.ndarray],
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    compat_csv_name: str = "oof_predictions.csv",
) -> Path:
    """Write one run's artifacts atomically, then refresh the compat OOF CSV.

    The run directory only ever becomes visible at its final path via a
    single rename once every file has been written successfully; a failure
    partway through leaves no partial run directory behind. Reusing an
    existing run_id is rejected before anything is written.
    """
    output_dir = Path(output_dir)
    artifacts_dir = output_dir / "artifacts"
    final_dir = artifacts_dir / run_id
    if final_dir.exists():
        raise FileExistsError(
            f"run artifacts already exist for run_id '{run_id}': {final_dir}"
        )

    tmp_dir = artifacts_dir / f".tmp-{run_id}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        write_json(tmp_dir / "metadata.json", metadata)
        write_json(tmp_dir / "split.json", split)
        write_json(tmp_dir / "preprocessing.json", preprocessing)
        write_npz(tmp_dir / "preprocessing_arrays.npz", preprocessing_arrays)
        write_json(tmp_dir / "metrics.json", metrics)
        predictions.to_csv(tmp_dir / "predictions.csv", index=False)
        tmp_dir.rename(final_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

    # The run directory above is now complete and public; a failure past
    # this point only affects the compat CSV refresh, not the run artifacts.
    _atomic_write_csv(output_dir / compat_csv_name, predictions, run_id)
    return final_dir
