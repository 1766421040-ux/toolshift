"""Finite, public-model diagnostic trees; actual plans execute in SQLite.

This independent extension does not alter the v1 tool environment or training.
Only the evaluator knows the actual support row. Compilers receive public_model.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import time
from typing import Optional, Sequence, Tuple

from .domains import DOMAINS, make_task
from .sandbox import Sandbox
from .schema import Action, TaskSpec, WorldSpec, canonical


PROTOCOL = "toolshift.diagnostic.v1"
METHODS = ("exact_action", "exact_version", "greedy_action",
           "greedy_version", "full_refresh")


def _cost(value) -> Fraction:
    """Finite JSON numbers are interpreted by their decimal spelling."""
    return Fraction(str(value))


def _cost_fields(value: Optional[Fraction]) -> dict:
    return {"worst_cost": None if value is None else float(value),
            "worst_cost_fraction": None if value is None else [value.numerator, value.denominator]}


def _node_cost(node: dict) -> Optional[Fraction]:
    value = node["worst_cost_fraction"]
    return None if value is None else Fraction(*value)


def _positive_number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(name + " must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be a finite positive number")
    return float(value)


def semantic_row(world: WorldSpec) -> dict:
    return {key: getattr(world, key) for key in
            ("unit_factor", "update_mode", "default_scope", "observable")}


@dataclass(frozen=True)
class DiagnosticQuery:
    name: str
    cost: float
    answers: Tuple[str, ...]
    full_refresh: bool = False

    def __post_init__(self):
        if not self.name or not isinstance(self.name, str):
            raise ValueError("Query name must be a nonempty string")
        _positive_number(self.cost, "Query cost")
        if not self.answers or any(not isinstance(a, str) for a in self.answers):
            raise ValueError("Partition answers must be strings")

    def to_dict(self) -> dict:
        return {"name": self.name, "cost": self.cost,
                "answers": list(self.answers), "read_only": True,
                "state_independent": True, "full_refresh": self.full_refresh}


@dataclass(frozen=True)
class DiagnosticProblem:
    case_id: str
    task: TaskSpec
    worlds: Tuple[WorldSpec, ...]
    queries: Tuple[DiagnosticQuery, ...]

    def __post_init__(self):
        if not self.worlds:
            raise ValueError("Support must be nonempty")
        if any(w.domain != self.task.domain or not w.observable for w in self.worlds):
            raise ValueError("Support requires same-domain observable worlds")
        if len({canonical(semantic_row(w)) for w in self.worlds}) != len(self.worlds):
            raise ValueError("Duplicate semantic worlds cannot inflate versions")
        if len({q.name for q in self.queries}) != len(self.queries):
            raise ValueError("Query names must be unique")
        if not math.isfinite(sum(q.cost for q in self.queries)):
            raise ValueError("Total query cost must be representable as a finite number")
        for q in self.queries:
            if len(q.answers) != len(self.worlds):
                raise ValueError("Every query must define every support row")
            if q.full_refresh and len(set(q.answers)) != len(self.worlds):
                raise ValueError("Full refresh must identify every semantic row")
        if sum(q.full_refresh for q in self.queries) > 1:
            raise ValueError("Only one designated full refresh is allowed")
        for world in self.worlds:
            if self.task.target_value % world.unit_factor:
                raise ValueError("Task must be exactly representable in every world")

    def public_model(self) -> dict:
        # Same object content for every actual world; no actual-row argument.
        return {
            "protocol": PROTOCOL,
            "task": {k: deepcopy(getattr(self.task, k)) for k in
                     ("domain", "target_id", "target_value", "initial_items",
                      "required_items", "protected_id")},
            "support": [semantic_row(w) for w in self.worlds],
            "queries": [q.to_dict() for q in self.queries],
            "assumptions": {
                "known_finite_support": True, "state_independent_queries": True,
                "explicit_scope_current": True, "write_acknowledgments_uninformative": True,
                "commuting_write_operators": True, "query_cost_only": True,
            },
        }


def complete_plan(model: dict, row: int) -> list:
    """Derive a plan from PUBLIC candidate semantics, not actual-world labels."""
    task, semantic = model["task"], model["support"][row]
    domain = DOMAINS[task["domain"]]
    target = task["target_value"]
    if target % semantic["unit_factor"]:
        raise ValueError("Unrepresentable numeric goal")
    items = list(task["required_items"])
    if semantic["update_mode"] == "append":
        items = [item for item in items if item not in task["initial_items"]]
    base = {"record_id": task["target_id"], "scope": "current"}
    return [
        Action(domain["numeric_tool"],
               dict(base, value=target // semantic["unit_factor"])).to_dict(),
        Action(domain["items_tool"], dict(base, items=items)).to_dict(),
    ]


def _classes(model: dict) -> tuple:
    return tuple(canonical(complete_plan(model, i))
                 for i in range(len(model["support"])))


def _partition(query: dict, belief: tuple) -> dict:
    branches = {}
    for row in belief:
        branches.setdefault(query["answers"][row], []).append(row)
    return {answer: tuple(rows) for answer, rows in sorted(branches.items())}


def _terminal(belief: tuple, classes: tuple, objective: str) -> bool:
    return (len(belief) == 1 if objective == "version"
            else len({classes[i] for i in belief}) == 1)


def _leaf(model: dict, belief: tuple) -> dict:
    return {"type": "plan", "plan": complete_plan(model, belief[0]),
            "remaining_rows": list(belief), **_cost_fields(Fraction(0))}


def compile_tree(public_model: dict, method: str) -> dict:
    """Compile using only the same public information supplied to every method.

    public_model must come from a validated DiagnosticProblem.public_model();
    this internal compiler is not a parser for untrusted external model files.
    Greedy tie break: larger cost-normalized worst-class reduction, then larger
    raw reduction, then lower cost, then lexical query name. A zero-gain split
    remains eligible. Exact ties use the lexical query name.
    """
    if method not in METHODS:
        raise ValueError("Unknown diagnostic method")
    started = time.perf_counter()
    model = deepcopy(public_model)
    classes = _classes(model)
    queries = sorted(model["queries"], key=lambda q: q["name"])
    objective = "version" if method in ("exact_version", "greedy_version") else "action"
    stats = {"states_examined": 0}

    @lru_cache(None)
    def solve(belief: tuple):
        stats["states_examined"] += 1
        if _terminal(belief, classes, objective):
            return _leaf(model, belief)
        eligible = [(q, _partition(q, belief)) for q in queries
                    if len(_partition(q, belief)) > 1]
        if not eligible:
            return {"type": "pause", "reason": "unidentifiable",
                    "remaining_rows": list(belief), **_cost_fields(None)}
        if method.startswith("greedy"):
            def count(rows):
                return len(rows) if objective == "version" else len({classes[i] for i in rows})
            initial = count(belief)

            def greedy_key(item):
                q, parts = item
                gain = initial - max(count(rows) for rows in parts.values())
                return (-Fraction(gain) / _cost(q["cost"]), -gain, _cost(q["cost"]), q["name"])
            eligible = [min(eligible, key=greedy_key)]
        candidates = []
        for query, parts in eligible:
            children = {answer: solve(rows) for answer, rows in parts.items()}
            values = [_node_cost(child) for child in children.values()]
            cost = (None if any(value is None for value in values)
                    else _cost(query["cost"]) + max(values))
            node = {"type": "query", "query": query["name"],
                    "cost": query["cost"], "children": children,
                    "remaining_rows": list(belief), **_cost_fields(cost)}
            candidates.append(node)
        return min(candidates, key=lambda node: (
            math.inf if _node_cost(node) is None else _node_cost(node),
            node["query"]))

    initial = tuple(range(len(classes)))
    if method == "full_refresh":
        refresh = next((q for q in queries if q["full_refresh"]), None)
        if refresh is None:
            tree = {"type": "pause", "reason": "refresh_unavailable",
                    "remaining_rows": list(initial), **_cost_fields(None)}
        else:
            tree = {"type": "query", "query": refresh["name"],
                    "cost": refresh["cost"], "remaining_rows": list(initial),
                    **_cost_fields(_cost(refresh["cost"])),
                    "children": {answer: _leaf(model, rows)
                                 for answer, rows in _partition(refresh, initial).items()}}
        stats["states_examined"] = 1
    else:
        tree = solve(initial)

    def count_nodes(node):
        return 1 + sum(count_nodes(child) for child in node.get("children", {}).values())
    return {"method": method, "objective": objective, "tree": deepcopy(tree),
            "worst_cost": tree["worst_cost"], "tree_nodes": count_nodes(tree),
            "states_examined": stats["states_examined"],
            "compile_seconds": time.perf_counter() - started}


class DiagnosticEnvironment:
    """Evaluator-side query service; compilers never receive this object."""

    def __init__(self, problem: DiagnosticProblem, actual_row: int,
                 budget: Optional[float] = None, db_path: str = ":memory:"):
        if type(actual_row) is not int or not 0 <= actual_row < len(problem.worlds):
            raise ValueError("Invalid actual row")
        if budget is None:
            exact_budget = sum((_cost(q.cost) for q in problem.queries), Fraction(0))
        else:
            if (isinstance(budget, bool) or not isinstance(budget, (int, float))
                    or not math.isfinite(budget) or budget < 0):
                raise ValueError("Budget must be finite and nonnegative")
            exact_budget = _cost(budget)
        self._problem = problem
        self._actual_row = actual_row
        self._queries = {q.name: q for q in problem.queries}
        self._budget = exact_budget
        self.budget = float(exact_budget)
        self._spent = Fraction(0)
        self.history = []
        self.sandbox = Sandbox(problem.task, problem.worlds[actual_row], db_path)

    def public_model(self) -> dict:
        return deepcopy(self._problem.public_model())

    @property
    def spent(self) -> float:
        return float(self._spent)

    def query(self, name: str) -> dict:
        if name not in self._queries:
            raise ValueError("Undeclared diagnostic query")
        query = self._queries[name]
        if self._spent + _cost(query.cost) > self._budget:
            raise ValueError("Diagnostic query budget exceeded")
        before = self.sandbox._snapshot()
        answer = query.answers[self._actual_row]
        self._spent += _cost(query.cost)
        result = {"answer": answer}
        self.history.append({"kind": "query", "name": name,
                             "cost": query.cost, "observation": result})
        if self.sandbox._snapshot() != before:
            raise AssertionError("A diagnostic query mutated SQLite")
        return result

    def close(self):
        self.sandbox.close()


def execute_plan(task: TaskSpec, world: WorldSpec, plan: Sequence[dict],
                 reverse: bool = False) -> dict:
    with Sandbox(task, world) as env:
        for step in (list(reversed(plan)) if reverse else plan):
            env.act(Action.from_dict(step))
        return {"verification": env.verify(), "state": env._snapshot(),
                "history": deepcopy(env.history)}


def audit_sqlite_model(problem: DiagnosticProblem) -> dict:
    """Independent SQL check of every proposed plan in every support world."""
    model = problem.public_model()
    plans = {canonical(complete_plan(model, i)): complete_plan(model, i)
             for i in range(len(problem.worlds))}
    comparisons = 0
    wrong_plans = 0
    for key, plan in plans.items():
        for i, world in enumerate(problem.worlds):
            forward = execute_plan(problem.task, world, plan)
            reverse = execute_plan(problem.task, world, plan, reverse=True)
            expected = key == canonical(complete_plan(model, i))
            if forward["verification"]["success"] != expected:
                raise AssertionError("Public plan semantics disagree with SQLite")
            if reverse["verification"]["success"] != expected:
                raise AssertionError("Write order changed plan correctness")
            if forward["state"]["records"] != reverse["state"]["records"]:
                raise AssertionError("Write operators failed to commute")
            if forward["verification"]["invariant_ok"] != reverse["verification"]["invariant_ok"]:
                raise AssertionError("Write order changed process invariants")
            if any(h["observation"] != {"ok": True, "tool": h["call"]["tool"], "affected": 1}
                   for h in forward["history"] + reverse["history"]):
                raise AssertionError("A write supplied diagnostic information")
            comparisons += 1
            wrong_plans += not expected
    return {"plan_world_pairs": comparisons, "write_orders_checked": comparisons * 2,
            "wrong_plan_pairs_checked": wrong_plans, "passed": True}


def run_episode(problem: DiagnosticProblem, compiled: dict, actual_row: int,
                budget: Optional[float] = None) -> dict:
    """Traverse compiled public tree, obtain paid observations, execute SQL."""
    env = DiagnosticEnvironment(problem, actual_row, budget)
    started = time.perf_counter()
    try:
        model = env.public_model()
        node = compiled["tree"]
        belief = tuple(range(len(problem.worlds)))
        reason = None
        queries = {q["name"]: q for q in model["queries"]}
        # Refuse a tree that cannot robustly finish within this budget, even if
        # the actual (hidden) branch happens to be cheap.
        if _node_cost(node) is None or _node_cost(node) > env._budget:
            reason = "no_robust_completion_within_budget"
        while reason is None and node["type"] == "query":
            result = env.query(node["query"])
            belief = _partition(queries[node["query"]], belief)[result["answer"]]
            node = node["children"][result["answer"]]
        certified = False
        if reason is None and node["type"] == "plan":
            # Evaluator-only certificate: never sent back to the compiled policy.
            plan_key = canonical(node["plan"])
            certified = all(plan_key == canonical(complete_plan(model, i)) for i in belief)
            for action in node["plan"]:
                env.sandbox.act(Action.from_dict(action))
        elif reason is None:
            reason = node.get("reason", "pause")
        verification = env.sandbox.verify()
        snapshot = env.sandbox._snapshot()
        return {
            "case_id": problem.case_id, "method": compiled["method"],
            "actual_row_evaluator_only": actual_row, "query_cost": env.spent,
            "query_count": len(env.history), "remaining_versions": len(belief),
            "success": verification["success"],
            "robust_success": verification["success"] and certified,
            "complete_plan_certified": certified,
            "invariant_ok": verification["invariant_ok"],
            "violations": verification["violations"], "action_count": env.sandbox.action_count,
            "paused": reason is not None, "pause_reason": reason,
            "budget": env.budget, "query_history": deepcopy(env.history),
            "write_history": deepcopy(env.sandbox.history), "final_sql_state": snapshot,
            "wall_seconds": time.perf_counter() - started,
        }
    finally:
        env.close()


def worked_example(costs=(1, 2, 2), refresh_cost=4,
                   case_id="worked_coupled") -> DiagnosticProblem:
    task = make_task("travel", seed=901)
    worlds = tuple(WorldSpec("private-" + str(i), "travel", *values)
                   for i, values in enumerate(((1, "append", "current"),
                                               (60, "append", "current"),
                                               (1, "append", "all"),
                                               (1, "replace", "all"))))
    queries = (
        DiagnosticQuery("inspect_route", costs[0], ("A", "A", "B", "B")),
        DiagnosticQuery("inspect_units", costs[1], ("1", "60", "1", "1")),
        DiagnosticQuery("inspect_update_mode", costs[2], ("append", "append", "append", "replace")),
        DiagnosticQuery("full_refresh", refresh_cost, tuple("v" + str(i) for i in range(4)), True),
    )
    return DiagnosticProblem(case_id, task, worlds, queries)


def random_problem(seed: int, index: int) -> DiagnosticProblem:
    rng = random.Random("{}:{}".format(seed, index))
    domain = tuple(DOMAINS)[index % len(DOMAINS)]
    alt = DOMAINS[domain]["alternate_factor"]
    pool = [(u, m, s) for u in (1, alt) for m in ("append", "replace")
            for s in ("current", "all")]
    selected = sorted(rng.sample(pool, rng.randint(3, 8)))
    worlds = tuple(WorldSpec("hidden-" + str(i), domain, *values)
                   for i, values in enumerate(selected))
    n = len(worlds)
    queries = []
    for field in ("unit_factor", "update_mode", "default_scope"):
        queries.append(DiagnosticQuery("field_" + field, rng.randint(1, 6),
                                       tuple(str(getattr(w, field)) for w in worlds)))
    # These partitions model diagnostic endpoints, not physical new operators.
    for q in range(rng.randint(2, 5)):
        bins = rng.randint(2, min(n, 4))
        labels = [str(i) for i in range(bins)]
        answers = labels + [rng.choice(labels) for _ in range(n - bins)]
        rng.shuffle(answers)
        queries.append(DiagnosticQuery("probe_" + str(q), rng.randint(1, 6),
                                       tuple(answers)))
    # Price is sampled independently, even when refresh dominates other queries.
    queries.append(DiagnosticQuery("full_refresh", rng.randint(1, 10),
                                   tuple("semantic_" + str(i) for i in range(n)), True))
    return DiagnosticProblem("random-{:04d}".format(index), make_task(domain, seed=1000 + index),
                             worlds, tuple(queries))


def _structure_key(model: dict) -> str:
    """Labeled structure: excludes costs, task values, and endpoint names.

    This is NOT graph-isomorphism canonicalization; report the weaker count.
    """
    support = [(r["unit_factor"] == 1, r["update_mode"], r["default_scope"])
               for r in model["support"]]
    partitions = []
    for query in model["queries"]:
        cells = sorted(sorted(rows) for rows in _partition(
            query, tuple(range(len(support)))).values())
        partitions.append(cells)
    return canonical({"support": support, "partitions": sorted(partitions)})


def run_benchmark(output_dir, random_cases: int = 64, seed: int = 20261020) -> dict:
    if type(random_cases) is not int or random_cases < 0:
        raise ValueError("random_cases must be a nonnegative integer")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    problems = [worked_example(), worked_example((1, 1, 1), 4, "worked_equal_cost"),
                worked_example((3, 2, 2), 4, "worked_expensive_route")]
    problems += [random_problem(seed, i) for i in range(random_cases)]
    episodes, cases, trees, models = [], [], [], []
    audits = []
    for problem in problems:
        model = problem.public_model()
        models.append({"case_id": problem.case_id, "public_model": model})
        audits.append(dict(case_id=problem.case_id, **audit_sqlite_model(problem)))
        for method in METHODS:
            compiled = compile_tree(model, method)
            trees.append(dict(case_id=problem.case_id, **compiled))
            rows = [run_episode(problem, compiled, i) for i in range(len(problem.worlds))]
            episodes.extend(rows)
            worst = max(row["query_cost"] for row in rows)
            if method.startswith("exact") and compiled["worst_cost"] != worst:
                raise AssertionError("Compiled worst cost disagrees with execution")
            cases.append({
                "case_id": problem.case_id, "method": method,
                "worlds": len(rows), "worst_query_cost": worst,
                "mean_query_cost": sum(row["query_cost"] for row in rows) / len(rows),
                "success_rate": sum(row["success"] for row in rows) / len(rows),
                "robust_success_rate": sum(row["robust_success"] for row in rows) / len(rows),
                "invariant_violations": sum(not row["invariant_ok"] for row in rows),
                "mean_remaining_versions": sum(row["remaining_versions"] for row in rows) / len(rows),
                "mean_query_count": sum(row["query_count"] for row in rows) / len(rows),
                "compile_seconds": compiled["compile_seconds"],
                "tree_nodes": compiled["tree_nodes"], "states_examined": compiled["states_examined"],
            })
    by_case = {(row["case_id"], row["method"]): row for row in cases}
    by_method = {}
    for method in METHODS:
        rows = [row for row in cases if row["method"] == method]
        eps = [row for row in episodes if row["method"] == method]
        by_method[method] = {
            "cases": len(rows), "episodes": len(eps),
            "robust_success_rate": sum(row["robust_success"] for row in eps) / len(eps),
            "invariant_violations": sum(not row["invariant_ok"] for row in eps),
            "mean_case_worst_cost": sum(row["worst_query_cost"] for row in rows) / len(rows),
            "mean_episode_query_cost": sum(row["query_cost"] for row in eps) / len(eps),
            "total_compile_seconds": sum(row["compile_seconds"] for row in rows),
            "max_tree_nodes": max(row["tree_nodes"] for row in rows),
        }
    comparisons = {}
    for other in METHODS[1:]:
        outcomes = Counter()
        deltas = []
        for problem in problems:
            exact = by_case[problem.case_id, "exact_action"]["worst_query_cost"]
            candidate = by_case[problem.case_id, other]["worst_query_cost"]
            delta = candidate - exact
            outcomes["lower" if delta > 1e-10 else "higher" if delta < -1e-10 else "equal"] += 1
            deltas.append(delta)
        comparisons["exact_action_vs_" + other] = {
            "meaning": "Counts compare exact_action worst cost to the named method.",
            "lower": outcomes["lower"], "equal": outcomes["equal"], "higher": outcomes["higher"],
            "mean_other_minus_exact": sum(deltas) / len(deltas),
        }
    summary = {
        "protocol": PROTOCOL, "seed": seed, "random_cases": random_cases,
        "fixed_cases": 3, "total_cases": len(problems),
        "distinct_labeled_support_partition_structures": len({
            _structure_key(p.public_model()) for p in problems}),
        "structure_count_caveat": "Not canonicalized for graph isomorphism; do not call independent graph families.",
        "semantic_world_instances": sum(len(p.worlds) for p in problems),
        "policy_episodes": len(episodes), "methods": by_method, "comparisons": comparisons,
        "sql_audit": {"passed": True, "plan_world_pairs": sum(a["plan_world_pairs"] for a in audits),
                      "write_orders_checked": sum(a["write_orders_checked"] for a in audits),
                      "wrong_plan_pairs_checked": sum(a["wrong_plan_pairs_checked"] for a in audits)},
        "claims": {"real_sqlite_execution": True, "learned_policy": False,
                   "new_dynamic_programming_theory": False,
                   "external_benchmark_reproduction": False},
    }
    artifacts = {
        "public_cases.jsonl": models, "compiled_trees.jsonl": trees,
        "episodes.jsonl": episodes, "case_results.jsonl": cases, "sql_audits.jsonl": audits,
    }
    hashes = {}
    for name, rows in artifacts.items():
        payload = "".join(json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                          for row in rows)
        (out / name).write_text(payload, encoding="utf-8")
        hashes[name] = hashlib.sha256(payload.encode()).hexdigest()
    summary_payload = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (out / "summary.json").write_text(summary_payload, encoding="utf-8")
    hashes["summary.json"] = hashlib.sha256(summary_payload.encode()).hexdigest()
    module_path = Path(__file__)
    manifest = {
        "protocol": PROTOCOL, "config": {"seed": seed, "random_cases": random_cases},
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
        "module_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "core_source_sha256": {name: hashlib.sha256((module_path.parent / name).read_bytes()).hexdigest()
                               for name in ("schema.py", "domains.py", "sandbox.py")},
        "artifact_sha256": hashes,
        "sampling": "All generated cases retained; all support rows executed for all methods.",
        "cost_objective": "minimax query cost; separate uniform-world means are descriptive only",
        "information": "Same complete public finite model to every compiler; actual row only evaluator",
    }
    locked_protocol = out / "protocol_locked.md"
    if locked_protocol.exists():
        manifest["locked_protocol_sha256"] = hashlib.sha256(locked_protocol.read_bytes()).hexdigest()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary
