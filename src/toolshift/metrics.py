"""Episode metrics and paired, provenance-group bootstrap intervals."""
from __future__ import annotations

from collections import defaultdict
import random
from typing import Any, Sequence


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(rows: Sequence[dict]) -> dict[str, Any]:
    if not rows:
        return {"episodes": 0}
    successes = [r for r in rows if r["success"]]
    feasible = [r for r in rows if r.get("oracle_feasible", True)]
    impossible = [r for r in rows if not r.get("oracle_feasible", True)]
    return {
        "episodes": len(rows),
        "groups": len({r["group_id"] for r in rows}),
        "success_rate": mean([float(r["success"]) for r in rows]),
        "robust_success_rate": mean([float(r.get("robust_success", r["success"])) for r in rows]),
        "feasible_robust_success_rate": mean([float(r.get("robust_success", r["success"])) for r in feasible]),
        "feasible_success_rate": mean([float(r["success"]) for r in feasible]),
        "invariant_violation_rate": mean([float(not r["invariant_ok"]) for r in rows]),
        "mean_query_cost": mean([float(r["query_cost"]) for r in rows]),
        "mean_query_count": mean([float(r["query_count"]) for r in rows]),
        "query_cost_per_success": sum(r["query_cost"] for r in rows) / len(successes) if successes else None,
        "mean_successful_query_cost": mean([float(r["query_cost"]) for r in successes]),
        "abstention_rate": mean([float(r["termination"] == "abstained") for r in rows]),
        "impossible_case_action_rate": mean([float(r["action_count"] > 0) for r in impossible]),
        "invalid_decision_rate": mean([float(r["termination"] == "invalid_decision") for r in rows]),
        "policy_error_rate": mean([float(r["termination"] == "policy_error") for r in rows]),
        "uncertified_action_episode_rate": mean([float(r.get("uncertified_actions", 0) > 0) for r in rows]),
        "mean_wall_seconds": mean([float(r.get("wall_seconds", 0)) for r in rows]),
        "mean_policy_seconds": mean([float(r.get("policy_seconds", 0)) for r in rows]),
        "mean_audit_seconds": mean([float(r.get("audit_seconds", 0)) for r in rows]),
    }


def grouped_summary(rows: Sequence[dict], key: str) -> dict:
    groups: dict[str, list] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarize(group) for name, group in sorted(groups.items())}


def paired_bootstrap(left: Sequence[dict], right: Sequence[dict], metric: str = "success",
                     seed: int = 17, resamples: int = 2000) -> dict:
    """Resample entire independent task groups; paired variants stay together.

    Each source group has equal weight, avoiding inflated precision from many
    correlated version variants. The point estimate is a group-mean difference.
    """
    if resamples < 20:
        raise ValueError("At least 20 bootstrap resamples are required")
    a = {r["episode_id"]: r for r in left}
    b = {r["episode_id"]: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or set(a) != set(b):
        raise ValueError("Paired runs require unique, matching episode IDs")
    groups: dict[str, list[float]] = defaultdict(list)
    for key in sorted(a):
        if a[key]["group_id"] != b[key]["group_id"]:
            raise ValueError("Provenance groups differ")
        groups[a[key]["group_id"]].append(float(a[key][metric]) - float(b[key][metric]))
    differences = [sum(v) / len(v) for v in groups.values()]
    if not differences:
        raise ValueError("No pairs")
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(differences) for _ in differences) / len(differences)
                     for _ in range(resamples))
    return {
        "metric": metric, "difference_left_minus_right": sum(differences) / len(differences),
        "ci95": [samples[int(0.025 * (resamples - 1))], samples[int(0.975 * (resamples - 1))]],
        "independent_groups": len(differences), "resamples": resamples, "seed": seed,
        "warning": "Few independent groups; interval is descriptive" if len(differences) < 20 else None,
    }

