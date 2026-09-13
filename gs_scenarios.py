"""Explicit outer/inner observation partitions for deployment questions."""

from __future__ import annotations

from dataclasses import asdict

import resnet_baseline as resnet
import run_manifest


def _ids(frame):
    return frame.observation_id.tolist()


def _status_check(train, test, column, status):
    known = set(train[column])
    queried = set(test[column])
    if status == "known" and not queried <= known:
        raise ValueError(f"{column}: known status contains unseen IDs")
    if status == "new" and queried & known:
        raise ValueError(f"{column}: new status leaks IDs across partitions")
    if status not in ("known", "new"):
        raise ValueError("known/new status must be explicitly declared")


def _fold(train, test, fit, validation, all_observations):
    if min(len(train), len(test), len(fit), len(validation)) == 0 or len(fit) < 3:
        raise ValueError("empty partition or fewer than three inner-fit observations")
    for left, right in (
        (train, test),
        (fit, validation),
        (fit, test),
        (validation, test),
    ):
        if set(_ids(left)) & set(_ids(right)):
            raise ValueError("observation leakage across split partitions")
    if not set(_ids(fit) + _ids(validation)) <= set(_ids(train)):
        raise ValueError("inner partition is outside outer training")
    used = set(_ids(train) + _ids(test))
    inner_used = set(_ids(fit) + _ids(validation))
    return {
        "train": _ids(train),
        "test": _ids(test),
        "fit": _ids(fit),
        "validation": _ids(validation),
        "excluded": [i for i in _ids(all_observations) if i not in used],
        "inner_excluded": [i for i in _ids(train) if i not in inner_used],
    }


def make_plan(dataset, config, policy=None):
    policy = dict(policy or {"scenario": "family_lofo", "selection_fraction": 0.2})
    scenario = policy.get("scenario")
    fraction = policy.get("selection_fraction")
    if type(fraction) not in (float, int) or not 0 < fraction <= 1:
        raise ValueError("selection_fraction must be in (0, 1]")
    if config.max_epochs < 1 or config.patience < 1 or config.batch_size < 2:
        raise ValueError("epochs/patience must be positive and batch_size >= 2")
    obs = dataset.observations
    folds = []
    if scenario == "family_lofo":
        if set(policy) != {"scenario", "selection_fraction"}:
            raise ValueError("unexpected family_lofo policy fields")
        families = sorted(set(obs.family_id))
        if len(families) < 3:
            raise ValueError("at least three families are required")
        for index, family in enumerate(families):
            train, test = obs[obs.family_id != family], obs[obs.family_id == family]
            validation_family = resnet.select_validation_family(
                train.family_id.to_numpy(dtype=str), index, config.seed
            )
            fit = train[train.family_id != validation_family]
            validation = train[train.family_id == validation_family]
            folds.append(_fold(train, test, fit, validation, obs))
        bootstrap_group = "family_id"
    elif scenario == "future_year":
        if set(policy) != {
            "scenario",
            "selection_fraction",
            "test_year",
            "validation_year",
            "genotype_status",
            "environment_status",
        }:
            raise ValueError(
                "future_year requires years and known/new genotype/environment status"
            )
        test_year, validation_year = policy["test_year"], policy["validation_year"]
        if (
            type(test_year) is not int
            or type(validation_year) is not int
            or validation_year >= test_year
        ):
            raise ValueError(
                "temporal inversion: validation year must precede test year"
            )
        train, test = obs[obs.year < test_year], obs[obs.year == test_year]
        if train.empty or int(train.year.max()) != validation_year:
            raise ValueError("validation year must be the latest outer training year")
        fit, validation = (
            train[train.year < validation_year],
            train[train.year == validation_year],
        )
        for earlier, later in ((train, test), (fit, validation)):
            _status_check(earlier, later, "line_id", policy["genotype_status"])
            _status_check(earlier, later, "site", policy["environment_status"])
        folds.append(_fold(train, test, fit, validation, obs))
        bootstrap_group = "line_id"
    elif scenario == "family_environment":
        if set(policy) != {
            "scenario",
            "selection_fraction",
            "test_families",
            "test_sites",
            "validation_families",
            "validation_sites",
        }:
            raise ValueError("family_environment requires held-out family/site lists")
        for key, column in (
            ("test_families", "family_id"),
            ("validation_families", "family_id"),
            ("test_sites", "site"),
            ("validation_sites", "site"),
        ):
            values = policy[key]
            if (
                not isinstance(values, list)
                or not values
                or len(values) != len(set(values))
                or not set(values) <= set(obs[column])
            ):
                raise ValueError(f"invalid or unknown {key}")
        if set(policy["test_families"]) & set(policy["validation_families"]) or set(
            policy["test_sites"]
        ) & set(policy["validation_sites"]):
            raise ValueError("held-out test/validation groups overlap")
        tf, te = (
            obs.family_id.isin(policy["test_families"]),
            obs.site.isin(policy["test_sites"]),
        )
        test, train = obs[tf & te], obs[~tf & ~te]
        vf, ve = (
            train.family_id.isin(policy["validation_families"]),
            train.site.isin(policy["validation_sites"]),
        )
        validation, fit = train[vf & ve], train[~vf & ~ve]
        for earlier, later in ((train, test), (fit, validation)):
            _status_check(earlier, later, "line_id", "new")
            _status_check(earlier, later, "site", "new")
        folds.append(_fold(train, test, fit, validation, obs))
        bootstrap_group = "family_id"
    else:
        raise ValueError("unsupported prediction scenario")
    plan = {
        "schema_version": 2,
        "kind": "gs_scenario_plan",
        "policy": policy,
        "dataset_hash": dataset.provenance["dataset_hash"],
        "preprocessing_fit_scope": {"selection": "fit", "final": "train"},
        "phenotype_correction": "none",
        "resnet_config": asdict(config),
        "bootstrap_group": bootstrap_group,
        "folds": folds,
    }
    plan["plan_hash"] = run_manifest.canonical_json_hash(plan)
    return plan


def validate_plan(dataset, plan):
    try:
        config = resnet.ResNetConfig(**plan["resnet_config"])
        expected = make_plan(dataset, config, plan["policy"])
    except (KeyError, TypeError) as error:
        raise ValueError("invalid GS split plan") from error
    if expected != plan:
        raise ValueError(
            "GS split plan differs from data, policy or canonical partitions"
        )
    return config
