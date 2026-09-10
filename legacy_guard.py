"""Explicit opt-in gate for the legacy/experimental scripts.

`preprocess.py`, `main.py`, and `train_gnn.py` predate the verified
GBLUP/ResNet baselines and still carry the preprocessing and evaluation
problems listed below. They stay in the repository for reference, but a
plain `python main.py` must not quietly run them: without
``--allow-legacy`` each script stops here, before reading any input,
writing any file, or initializing external logging, and points at the
verified command instead.

The gate is deliberately a command-line flag rather than an environment
variable or a Compose profile, so neither `docker compose --profile
legacy` nor a W&B sweep agent can satisfy it as a side effect of being
launched.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

import external_logging

ALLOW_LEGACY_FLAG = "--allow-legacy"
EXPERIMENTAL_BANNER = (
    "[EXPERIMENTAL] legacy path: results are not a verified baseline. "
    "Do not report them as verified OOF performance."
)

VERIFIED_ALTERNATIVES = (
    "python gblup_baseline.py --data-dir data --output-dir gblup_results",
    "python resnet_baseline.py --data-dir data --output-dir resnet_results --device cpu",
)

KNOWN_ISSUES: dict[str, tuple[str, ...]] = {
    "preprocess.py": (
        "phenotype/genotype files are paired by sorted filename order, not by family ID",
        "only samples common to both files are kept and duplicates are dropped silently",
        "unknown or missing genotype symbols become 0 via fillna(0)",
        "low-variance and MAF filters are computed over all individuals, not per fold",
        "founder parents are not excluded and marker IDs are not validated",
        (
            "phenotypes are standardized within each family before splitting, "
            "so a held-out family is scored on statistics derived from itself "
            "and the result is not comparable with the raw kg/ha baselines"
        ),
    ),
    "main.py": (
        "reads processed_data_hy/, which is produced by the legacy preprocess.py",
        (
            "PCA is fitted inside the outer training partition but before the "
            "inner validation split is separated"
        ),
        "metrics are computed on family-standardized phenotypes, not raw kg/ha",
        "no run artifacts (metadata/split/preprocessing/metrics) are written",
    ),
    "train_gnn.py": (
        (
            "reads processed_data_hy/ plus dummy graph data, which is not a "
            "real SNP-to-gene mapping"
        ),
        "metrics are computed on family-standardized phenotypes, not raw kg/ha",
        "no run artifacts (metadata/split/preprocessing/metrics) are written",
    ),
}


def _refusal_message(script_name: str) -> str:
    lines = [
        (
            f"{script_name} is a legacy/experimental script and is not a "
            "verified baseline."
        ),
        "",
        (
            f"Re-run it with {ALLOW_LEGACY_FLAG} only if you intend to use "
            "the legacy path knowingly."
        ),
        "",
        "Verified commands:",
    ]
    lines.extend(f"  {command}" for command in VERIFIED_ALTERNATIVES)
    issues = KNOWN_ISSUES.get(script_name, ())
    if issues:
        lines.extend(["", f"Known issues in {script_name}:"])
        lines.extend(f"  - {issue}" for issue in issues)
    return "\n".join(lines)


def build_parser(script_name: str, description: str) -> argparse.ArgumentParser:
    """Build the shared legacy CLI parser for one script."""
    parser = argparse.ArgumentParser(prog=script_name, description=description)
    parser.add_argument(
        ALLOW_LEGACY_FLAG,
        action="store_true",
        help=(
            "acknowledge that this legacy/experimental path is not a verified "
            "baseline and run it anyway"
        ),
    )
    external_logging.add_wandb_mode_argument(parser)
    return parser


def require_opt_in(
    script_name: str,
    description: str,
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse the legacy CLI and stop unless legacy use was acknowledged.

    Unknown arguments are ignored rather than rejected so a W&B sweep
    agent's hyperparameter arguments still reach the script; the
    ``--allow-legacy`` requirement itself is unaffected by them.

    Raises ``SystemExit(2)`` before the caller touches any input, output,
    or external service. Legacy acknowledgement is independent of the W&B
    mode: passing ``--allow-legacy`` leaves external logging disabled
    unless ``--wandb-mode`` also asks for it.
    """
    parser = build_parser(script_name, description)
    args, _ = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    if not args.allow_legacy:
        print(_refusal_message(script_name), file=sys.stderr)
        raise SystemExit(2)
    print(EXPERIMENTAL_BANNER)
    for issue in KNOWN_ISSUES.get(script_name, ()):
        print(f"[EXPERIMENTAL] known issue: {issue}")
    return args
