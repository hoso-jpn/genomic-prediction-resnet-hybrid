"""Selection diagnostics with declared direction, tie policy and group uncertainty."""

import math

import numpy as np
from scipy.stats import rankdata


def selection_metrics(observed, predicted, ids, *, fraction, direction):
    if not 0 < fraction <= 1 or direction not in ("higher", "lower"):
        raise ValueError("invalid selection fraction or direction")
    y, p = np.asarray(observed, dtype=float), np.asarray(predicted, dtype=float)
    names = np.asarray(ids, dtype=str)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or y.shape != names.shape
        or len(set(names)) != len(names)
    ):
        raise ValueError("metrics require aligned unique observation IDs")
    valid = np.isfinite(y) & np.isfinite(p)
    y, p, names = y[valid], p[valid], names[valid]
    n = len(y)
    result = {
        "n": n,
        "excluded_nonfinite": int((~valid).sum()),
        "pearson": None,
        "rmse": None,
        "spearman": None,
        "k": math.ceil(fraction * n),
        "top_k_overlap": None,
        "selection_differential": None,
        "directional_differential": None,
        "undefined_reason": None,
    }
    if not n:
        result["undefined_reason"] = "no_finite_pairs"
        return result
    result["rmse"] = float(np.sqrt(np.mean((y - p) ** 2)))
    if n < 2 or np.ptp(p) == 0 or np.ptp(y) == 0:
        result["undefined_reason"] = "too_few_or_constant_values"
        return result
    result["pearson"] = float(np.corrcoef(y, p)[0, 1])
    result["spearman"] = float(np.corrcoef(rankdata(y), rankdata(p))[0, 1])
    sign = 1 if direction == "higher" else -1
    k = result["k"]
    # Exact k, stable opaque-ID tie breaking; correlation uses average ranks.
    selected = np.lexsort((names, -sign * p))[:k]
    observed_top = np.lexsort((names, -sign * y))[:k]
    result["top_k_overlap"] = len(set(selected) & set(observed_top)) / k
    result["selection_differential"] = float(y[selected].mean() - y.mean())
    result["directional_differential"] = sign * result["selection_differential"]
    return result


def group_interval(frame, *, group, fraction, direction, seed=42, replicates=200):
    groups = sorted(set(frame[group]))
    if len(groups) < 3:
        return {
            "interval": None,
            "reason": "fewer_than_three_groups",
            "groups": len(groups),
        }
    rng = np.random.default_rng(seed)
    values = []
    blocks = {g: frame[frame[group] == g] for g in groups}
    for _ in range(replicates):
        sampled = [blocks[g] for g in rng.choice(groups, len(groups), replace=True)]
        y = np.concatenate([b.observed.to_numpy() for b in sampled])
        p = np.concatenate([b.predicted.to_numpy() for b in sampled])
        names = [
            f"draw{j}:{name}"
            for j, b in enumerate(sampled)
            for name in b.observation_id
        ]
        metric = selection_metrics(y, p, names, fraction=fraction, direction=direction)
        if metric["directional_differential"] is not None:
            values.append(metric["directional_differential"])
    if len(values) < 0.8 * replicates:
        return {
            "interval": None,
            "reason": "too_many_undefined_resamples",
            "groups": len(groups),
        }
    return {
        "interval": np.quantile(values, [0.025, 0.975]).tolist(),
        "groups": len(groups),
        "replicates": replicates,
        "valid_replicates": len(values),
        "unit": group,
        "scope": "conditional_observed_selection_differential",
    }


def feasibility_report(predictions, dataset, plan):
    fraction = plan["policy"]["selection_fraction"]
    direction = dataset.manifest["trait"]["direction"]
    models = {}
    for model, frame in predictions.groupby("model", sort=True):
        metrics = selection_metrics(
            frame.observed,
            frame.predicted,
            frame.observation_id,
            fraction=fraction,
            direction=direction,
        )
        uncertainty = group_interval(
            frame, group=plan["bootstrap_group"], fraction=fraction, direction=direction
        )
        models[model] = {
            "metrics": metrics,
            "uncertainty": uncertainty,
            "decision": "no-go"
            if metrics["spearman"] is None or uncertainty["interval"] is None
            else "conditional",
            "reason": "insufficient_evidence"
            if uncertainty["interval"] is None
            else "requires_independent_validation_and_domain_review",
        }
    return {
        "schema_version": 1,
        "scenario": plan["policy"],
        "models": models,
        "population": {
            "species": dataset.manifest["species"],
            "cohort": dataset.manifest["cohort_id"],
        },
        "real_adzuki_performance": "unverified",
        "automatic_go": False,
        "limitations": [
            "Observed selection differential is not future genetic gain.",
            "Intervals resample declared groups within observed conditions; not prediction intervals.",
            "Year/environment generalization requires additional independent held-out trials.",
            "Replicate rows are retained; selection ranks observation units, not independently validated breeding lines.",
        ],
        "next_trial": "independent target population/year/environment with individual phenotypes",
        "update_review": "before each new season, generation or changed target population",
        "tie_policy": "average Spearman ranks; exact-k ties broken by opaque observation_id",
    }
