"""Self-authored finite task families; no downloaded benchmark fixtures."""

from itertools import product
import random
from typing import Dict, Tuple

from .schema import Action, Query, TaskSpec, WorldSpec


DOMAINS: Dict[str, dict] = {
    "travel": {
        "table": "travel_records", "numeric_tool": "set_duration",
        "items_tool": "update_stops", "field": "duration_minutes",
        "items_field": "stops", "canonical_unit": "minutes",
        "alternate_unit": "hours", "alternate_factor": 60,
        "target_id": "trip_main", "protected_id": "trip_other",
        "initial_items": ("origin",), "new_item": "museum",
    },
    "catalog": {
        "table": "catalog_records", "numeric_tool": "set_price",
        "items_tool": "update_tags", "field": "price_cents",
        "items_field": "tags", "canonical_unit": "cents",
        "alternate_unit": "dollars", "alternate_factor": 100,
        "target_id": "product_main", "protected_id": "product_other",
        "initial_items": ("verified",), "new_item": "seasonal",
    },
    "calendar": {
        "table": "calendar_records", "numeric_tool": "set_length",
        "items_tool": "update_attendees", "field": "duration_minutes",
        "items_field": "attendees", "canonical_unit": "minutes",
        "alternate_unit": "hours", "alternate_factor": 60,
        "target_id": "event_main", "protected_id": "event_other",
        "initial_items": ("organizer",), "new_item": "guest",
    },
}

QUERY_COSTS = {
    "inspect_units": 1,
    "inspect_update_mode": 1,
    "inspect_default_scope": 1,
    "read_state": 1,
    "read_contract": 3,
}


def make_task(domain: str, seed: int = 0, query_budget: int = 4) -> TaskSpec:
    if domain not in DOMAINS:
        raise ValueError("Unknown domain: " + str(domain))
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    d = DOMAINS[domain]
    # A local RNG gives repeatable instances without mutating global training
    # randomness. Vary actual goals, not merely task IDs: the former modular
    # construction had only 28 distinct amount/item combinations.
    # Domain-specific streams prevent train/test domains with the same source
    # seed from reusing identical latent amounts and item-index combinations.
    rng = random.Random("{}:{}".format(domain, seed))
    amount = rng.randint(2, 300)
    target = amount * d["alternate_factor"]
    initial = d["initial_items"]
    additions = tuple(d["new_item"] + "_" + str(i)
                      for i in rng.sample(range(1000), rng.randint(2, 3)))
    required = initial + additions
    instruction = (
        "For {record}, commit {field}={value} {unit} and make {items_field} "
        "exactly {items}. Preserve its existing items without duplicates. "
        "Do not change {protected} or any other record. Each field's commit "
        "is write-once: an accepted write locks that field, even if its "
        "business value is wrong. Both fields must be committed."
    ).format(record=d["target_id"], field=d["field"], value=target,
             unit=d["canonical_unit"], items_field=d["items_field"],
             items=list(required), protected=d["protected_id"])
    return TaskSpec(
        task_id="{}-{:06d}".format(domain, seed), domain=domain,
        instruction=instruction, target_id=d["target_id"],
        target_value=target, required_items=required, initial_items=initial,
        initial_value=d["alternate_factor"], protected_id=d["protected_id"],
        protected_value=777, protected_items=("protected",),
        query_budget=query_budget, seed=seed,
    )


def candidate_worlds(domain: str, mode: str = "mixed") -> Tuple[WorldSpec, ...]:
    """Build the finite candidate set for a migration condition.

    World identities and the actual candidate stay private. The protocol
    exposes the de-identified semantic support to every policy, so different
    conditions may have different public uncertainty contracts. Within one
    candidate set, the public input is identical across actual worlds.
    no_drift has distinct IDs with identical behavior: identifying an exact
    version is unnecessary.
    """
    if domain not in DOMAINS:
        raise ValueError("Unknown domain.")
    alt = DOMAINS[domain]["alternate_factor"]
    aliases = {"unit": "units", "unit_change": "units",
               "append-replace": "append_replace", "scope": "default_scope",
               "none": "no_drift", "unidentifiable": "unidentifiable",
               "indistinguishable": "unidentifiable"}
    mode = aliases.get(mode, mode)
    if mode == "units":
        settings = [(1, "append", "current", True),
                    (alt, "append", "current", True)]
    elif mode == "append_replace":
        settings = [(1, "append", "current", True),
                    (1, "replace", "current", True)]
    elif mode == "default_scope":
        settings = [(1, "append", "current", True),
                    (1, "append", "all", True)]
    elif mode == "no_drift":
        settings = [(1, "append", "current", True)] * 2
    elif mode == "unidentifiable":
        settings = [(1, "append", "current", False),
                    (alt, "append", "current", False)]
    elif mode == "mixed":
        settings = [(u, m, s, True)
                    for u, m, s in product((1, alt), ("append", "replace"),
                                           ("current", "all"))]
    else:
        raise ValueError("Unknown variation mode: " + str(mode))
    return tuple(WorldSpec("{}-{}-{}".format(domain, mode, i), domain, *s)
                 for i, s in enumerate(settings))


def action_space(task: TaskSpec) -> Tuple[Action, ...]:
    """Finite public menu, constructed without a world or oracle.

    Explicit scope is a universally available way to avoid a default-scope
    ambiguity; units and append/replace semantics cannot be overridden.
    """
    d = DOMAINS[task.domain]
    alt = d["alternate_factor"]
    values = [task.target_value]
    if task.target_value % alt == 0:
        values.append(task.target_value // alt)
    additions = [x for x in task.required_items if x not in task.initial_items]
    item_options = [additions, list(task.required_items)]
    result = []
    for scope in (None, "current"):
        base = {"record_id": task.target_id}
        if scope is not None:
            base["scope"] = scope
        for value in values:
            result.append(Action(d["numeric_tool"], dict(base, value=value)))
        for items in item_options:
            result.append(Action(d["items_tool"], dict(base, items=items)))
    # Stable keys deduplicate custom tasks whose candidate values coincide.
    return tuple({a.key(): a for a in result}.values())


def query_space(task: TaskSpec) -> Tuple[Query, ...]:
    return tuple(Query(name) for name in QUERY_COSTS)


def query_cost(query: Query) -> int:
    if query.name not in QUERY_COSTS:
        raise ValueError("Unknown query.")
    if query.args:
        raise ValueError("These query endpoints accept no arguments.")
    return QUERY_COSTS[query.name]


def legacy_plan(task: TaskSpec) -> Tuple[Action, ...]:
    d = DOMAINS[task.domain]
    additions = [x for x in task.required_items if x not in task.initial_items]
    return (
        Action(d["numeric_tool"], {"record_id": task.target_id,
                                   "value": task.target_value}),
        Action(d["items_tool"], {"record_id": task.target_id,
                                 "items": additions}),
    )


def tool_schemas(task: TaskSpec) -> list:
    d = DOMAINS[task.domain]
    shared = {
        "record_id": {"type": "string"},
        "scope": {"type": "string", "enum": ["current", "all"],
                  "description": "Optional. Omission uses the service default."},
    }
    return [
        {"name": d["numeric_tool"], "write": True,
         "arguments": dict(shared, value={"type": "integer", "minimum": 1}),
         "required": ["record_id", "value"],
         "effect": "Commit the numeric field once; accepted writes lock it."},
        {"name": d["items_tool"], "write": True,
         "arguments": dict(shared, items={"type": "array", "items": {"type": "string"}}),
         "required": ["record_id", "items"],
         "effect": "Commit the item collection once; accepted writes lock it."},
    ]


def legacy_docs(task: TaskSpec) -> dict:
    d = DOMAINS[task.domain]
    return {
        "numeric_tool": d["numeric_tool"], "numeric_input_unit": d["canonical_unit"],
        "items_tool": d["items_tool"], "collection_update": "append",
        "default_scope": "current",
        "notice": "Cached documentation; the current service semantics may differ.",
        "write_once": True,
    }
