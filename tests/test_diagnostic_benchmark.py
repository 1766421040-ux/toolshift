import json
import math
from pathlib import Path
import sqlite3
import tempfile
import unittest

from toolshift.diagnostic_benchmark import (
    METHODS, DiagnosticEnvironment, DiagnosticProblem, DiagnosticQuery,
    audit_sqlite_model, compile_tree, complete_plan, execute_plan,
    random_problem, run_benchmark, run_episode, worked_example,
)
from toolshift.domains import make_task
from toolshift.schema import Action, WorldSpec, canonical


def independent_brute_force(problem, objective):
    """Small exhaustive recurrence using SQL-derived successful plan sets.

    No compiler helper, plan class helper, partition helper, or cache is used.
    """
    model = problem.public_model()
    plans = {canonical(complete_plan(model, i)): complete_plan(model, i)
             for i in range(len(problem.worlds))}
    valid = []
    for world in problem.worlds:
        valid.append({key for key, plan in plans.items()
                      if execute_plan(problem.task, world, plan)["verification"]["success"]})

    def recurse(rows):
        common = set.intersection(*(valid[i] for i in rows))
        if (len(rows) == 1 if objective == "version" else bool(common)):
            return 0
        candidates = []
        for query in problem.queries:
            cells = {}
            for row in rows:
                cells.setdefault(query.answers[row], set()).add(row)
            if len(cells) <= 1:
                continue
            candidates.append(query.cost + max(recurse(cell) for cell in cells.values()))
        return min(candidates, default=math.inf)
    return recurse(set(range(len(problem.worlds))))


class DiagnosticBenchmarkTests(unittest.TestCase):
    def test_worked_cost_regimes_and_full_refresh(self):
        for costs, expected in [((1, 2, 2), 3), ((1, 1, 1), 2), ((3, 2, 2), 4)]:
            problem = worked_example(costs)
            compiled = compile_tree(problem.public_model(), "exact_action")
            self.assertEqual(compiled["worst_cost"], expected)
            for row in range(4):
                result = run_episode(problem, compiled, row)
                self.assertTrue(result["robust_success"])
                self.assertEqual(result["action_count"], 2)
                self.assertEqual(len(result["final_sql_state"]["effects"]), 2)
            self.assertEqual(compile_tree(problem.public_model(), "full_refresh")["worst_cost"], 4)

    def test_exact_dp_matches_independent_sql_bruteforce(self):
        problems = [worked_example(), worked_example((3, 2, 2))]
        problems += [random_problem(733, i) for i in range(3)]
        for problem in problems:
            for objective in ("action", "version"):
                with self.subTest(case=problem.case_id, objective=objective):
                    expected = independent_brute_force(problem, objective)
                    observed = compile_tree(problem.public_model(), "exact_" + objective)["worst_cost"]
                    self.assertEqual(observed, expected)

    def test_public_models_do_not_depend_on_actual_world(self):
        problem = worked_example()
        models = []
        for row in range(4):
            env = DiagnosticEnvironment(problem, row)
            try:
                models.append(env.public_model())
                serialized = json.dumps(models[-1])
                self.assertNotIn("world_id", serialized)
                self.assertNotIn("actual_row", serialized)
                self.assertNotIn(problem.worlds[row].world_id, serialized)
                model = env.public_model()
                model["support"].clear()
                self.assertEqual(len(env.public_model()["support"]), 4)
            finally:
                env.close()
        self.assertTrue(all(model == models[0] for model in models))

    def test_positive_fractional_costs_and_invalid_costs(self):
        for bad in (0, -1, float("inf"), float("nan"), True, "1"):
            with self.assertRaises(ValueError):
                DiagnosticQuery("bad", bad, ("a", "b"))
        problem = worked_example((0.25, 0.5, 0.5), refresh_cost=1.0)
        self.assertEqual(compile_tree(problem.public_model(), "exact_action")["worst_cost"], 0.75)
        self.assertEqual(run_episode(problem, compile_tree(problem.public_model(), "exact_action"), 0)
                         ["query_cost"], 0.75)

    def test_query_readonly_charging_and_declaration_boundary(self):
        problem = worked_example()
        env = DiagnosticEnvironment(problem, 0, budget=1)
        try:
            before = env.sandbox._snapshot()
            self.assertEqual(env.query("inspect_route"), {"answer": "A"})
            self.assertEqual(env.spent, 1)
            self.assertEqual(before, env.sandbox._snapshot())
            for forbidden in ("read_state", "inspect_units", "private_world"):
                with self.assertRaises(ValueError):
                    env.query(forbidden)
            self.assertEqual(env.spent, 1)
            self.assertEqual(before, env.sandbox._snapshot())
        finally:
            env.close()

    def test_tiny_positive_query_cannot_bypass_zero_budget(self):
        problem = worked_example((1e-12, 2e-12, 2e-12), refresh_cost=4e-12)
        env = DiagnosticEnvironment(problem, 0, budget=0)
        try:
            with self.assertRaises(ValueError):
                env.query("inspect_route")
            self.assertEqual(env.spent, 0)
        finally:
            env.close()
        result = run_episode(problem, compile_tree(problem.public_model(), "exact_action"), 0, budget=0)
        self.assertTrue(result["paused"])
        self.assertEqual(result["action_count"], 0)

    def test_decimal_costs_use_exact_rational_budget_accounting(self):
        problem = worked_example((0.1, 0.2, 0.2), refresh_cost=0.4)
        compiled = compile_tree(problem.public_model(), "exact_action")
        self.assertEqual(compiled["worst_cost"], 0.3)
        self.assertEqual(compiled["tree"]["worst_cost_fraction"], [3, 10])
        for row in range(4):
            result = run_episode(problem, compiled, row, budget=0.3)
            self.assertTrue(result["robust_success"])
            self.assertEqual(result["query_cost"], 0.3)

    def test_budget_failure_cannot_select_lucky_actual_branch(self):
        problem = worked_example((3, 2, 2))
        compiled = compile_tree(problem.public_model(), "exact_action")
        self.assertEqual(compiled["worst_cost"], 4)
        for actual_row in range(4):
            result = run_episode(problem, compiled, actual_row, budget=2)
            self.assertTrue(result["paused"])
            self.assertEqual(result["action_count"], 0)
            self.assertEqual(result["query_cost"], 0)
            self.assertFalse(result["success"])

    def test_nuisance_scope_needs_no_query_but_identification_can_be_impossible(self):
        task = make_task("catalog")
        worlds = (WorldSpec("a", "catalog", default_scope="current"),
                  WorldSpec("b", "catalog", default_scope="all"))
        problem = DiagnosticProblem("nuisance", task, worlds, ())
        action = compile_tree(problem.public_model(), "exact_action")
        version = compile_tree(problem.public_model(), "exact_version")
        self.assertEqual(action["worst_cost"], 0)
        self.assertIsNone(version["worst_cost"])
        for row in range(2):
            self.assertTrue(run_episode(problem, action, row)["robust_success"])
            self.assertTrue(run_episode(problem, version, row)["paused"])

    def test_unidentifiable_action_ambiguity_pauses(self):
        task = make_task("travel")
        worlds = (WorldSpec("a", "travel", 1), WorldSpec("b", "travel", 60))
        problem = DiagnosticProblem("unidentifiable", task, worlds,
                                    (DiagnosticQuery("uninformative", 1, ("same", "same")),))
        compiled = compile_tree(problem.public_model(), "exact_action")
        self.assertIsNone(compiled["worst_cost"])
        for row in range(2):
            result = run_episode(problem, compiled, row)
            self.assertTrue(result["paused"])
            self.assertEqual(result["action_count"], 0)

    def test_wrong_plans_sql_effects_and_order_assumption(self):
        problem = worked_example()
        audit = audit_sqlite_model(problem)
        self.assertEqual(audit["plan_world_pairs"], 12)
        self.assertEqual(audit["write_orders_checked"], 24)
        self.assertGreater(audit["wrong_plan_pairs_checked"], 0)
        wrong = complete_plan(problem.public_model(), 1)
        execution = execute_plan(problem.task, problem.worlds[0], wrong)
        self.assertFalse(execution["verification"]["success"])
        self.assertEqual(len(execution["state"]["effects"]), 2)

    def test_sqlite_persistence_visible_from_independent_connection(self):
        problem = worked_example()
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "case.sqlite")
            env = DiagnosticEnvironment(problem, 2, db_path=path)
            try:
                env.query("inspect_route")
                env.query("inspect_update_mode")
                for step in complete_plan(problem.public_model(), 2):
                    env.sandbox.act(Action.from_dict(step))
                with sqlite3.connect(path) as conn:
                    actual = conn.execute("SELECT value,value_locked,items_locked "
                                          "FROM travel_records WHERE id=?",
                                          (problem.task.target_id,)).fetchone()
                    effects = conn.execute("SELECT COUNT(*) FROM effect_log").fetchone()[0]
                self.assertEqual(actual, (problem.task.target_value, 1, 1))
                self.assertEqual(effects, 2)
            finally:
                env.close()

    def test_duplicate_worlds_and_malformed_partitions_rejected(self):
        problem = worked_example()
        with self.assertRaises(ValueError):
            DiagnosticProblem("duplicate", problem.task, (problem.worlds[0], problem.worlds[0]), ())
        with self.assertRaises(ValueError):
            DiagnosticProblem("short", problem.task, problem.worlds,
                              (DiagnosticQuery("short", 1, ("a",)),))
        with self.assertRaises(ValueError):
            DiagnosticProblem("fake_refresh", problem.task, problem.worlds,
                              (DiagnosticQuery("refresh", 1, ("a",) * 4, True),))

    def test_all_baselines_have_same_public_input_and_succeed(self):
        problem = random_problem(417, 0)
        public = problem.public_model()
        snapshot = canonical(public)
        for method in METHODS:
            compiled = compile_tree(public, method)
            self.assertEqual(canonical(public), snapshot)
            results = [run_episode(problem, compiled, row) for row in range(len(problem.worlds))]
            self.assertTrue(all(r["robust_success"] for r in results))
            self.assertEqual(max(r["query_cost"] for r in results), compiled["worst_cost"])

    def test_small_run_writes_inspectable_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_benchmark(tmp, random_cases=2, seed=11)
            self.assertEqual(summary["total_cases"], 5)
            self.assertFalse(summary["claims"]["learned_policy"])
            self.assertTrue(summary["sql_audit"]["passed"])
            manifest = json.loads((Path(tmp) / "manifest.json").read_text())
            self.assertEqual(len(manifest["artifact_sha256"]), 6)
            episodes = [json.loads(line) for line in (Path(tmp) / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(episodes), summary["policy_episodes"])
            self.assertTrue(all(e["robust_success"] for e in episodes))


if __name__ == "__main__":
    unittest.main()
