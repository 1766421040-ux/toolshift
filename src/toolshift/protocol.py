"""The explicit public information boundary shared by every policy."""
from __future__ import annotations

from copy import deepcopy
from itertools import product
from typing import Sequence

from .domains import QUERY_COSTS
from .io import canonical
from .schema import OracleDecision, WorldSpec


STOP = {"type": "stop", "status": "pause"}


def public_support(worlds: Sequence[WorldSpec]) -> dict:
    """Publish the *support*, never the sampled true world or a hidden suite ID.

    The current benchmark gives agents a migration notice specifying which
    semantics may have changed. This is a substantive deployment assumption.
    Cartesian supports are compact; coupled supports retain their coupling.
    """
    if not worlds:
        raise ValueError("An uncertainty support cannot be empty")
    fields = ("unit_factor", "update_mode", "default_scope", "observable")
    tuples = {tuple(getattr(world, key) for key in fields) for world in worlds}
    values = [sorted({row[i] for row in tuples}) for i in range(len(fields))]
    result = {"assumption": "The current service belongs to this declared support.",
              "allowed": dict(zip(fields, values))}
    result["cartesian"] = set(product(*values)) == tuples
    if not result["cartesian"]:
        result["joint_semantics"] = [dict(zip(fields, row)) for row in sorted(tuples)]
    return result


def observation(context: dict, history: list, worlds: Sequence[WorldSpec],
                remaining_budget: int) -> dict:
    context = deepcopy(context)
    context.pop("task_id", None)
    candidates = context.pop("candidates")
    context["uncertainty_contract"] = public_support(worlds)
    # Only remove calls whose unavailability follows from PUBLIC acknowledgments
    # or PUBLIC remaining budget. Do not use private verification or world state.
    locked = {h["call"]["tool"] for h in history
              if h["kind"] == "act" and h["observation"].get("ok") is True}
    candidates = [c for c in candidates
                  if not (c["type"] == "act" and c["name"] in locked)
                  and not (c["type"] == "query" and QUERY_COSTS[c["name"]] > remaining_budget)]
    candidates = sorted(candidates, key=canonical)
    result = {"context": context, "history": deepcopy(history),
              "candidates": candidates, "remaining_budget": remaining_budget}
    validate_public_observation(result)
    return result


def validate_public_observation(value: dict) -> None:
    """Catch accidental oracle/identity fields; not an OS security boundary."""
    forbidden = {"world_id", "actual_world", "oracle", "oracle_decision",
                 "acceptable_decisions", "ground_truth", "suite", "split",
                 "group_id", "episode_id", "task_id"}

    def walk(node):
        if isinstance(node, dict):
            if forbidden.intersection(node):
                raise ValueError("Private or identifying fields reached policy input")
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
    walk(value)


def oracle_labels(decision: OracleDecision) -> list[dict]:
    # Done records are omitted by the generator. Pause has no private reason.
    if decision.kind in ("pause", "done"):
        return [dict(STOP)]
    return decision.decisions()


def messages(observed: dict) -> list[dict]:
    from .model_policy import decision_messages
    return decision_messages(observed)

