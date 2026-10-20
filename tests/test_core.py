import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from toolshift.domains import (DOMAINS, action_space, candidate_worlds, legacy_plan,
                               make_task, query_space)
from toolshift.oracle import ExactOracle
from toolshift.protocol import observation as public_observation, public_support
from toolshift.sandbox import Sandbox
from toolshift.schema import Action, Query, TaskSpec, WorldSpec


def run_oracle(task, worlds, actual, max_actions=4):
    oracle = ExactOracle(task, worlds, max_actions=max_actions)
    with Sandbox(task, actual) as env:
        for _ in range(12):
            decision = oracle.decide(env.history)
            if decision.kind == "act":
                env.act(decision.actions[0])
            elif decision.kind == "query":
                env.query(decision.queries[0])
            else:
                return decision, env.verify(), list(env.history)
        raise AssertionError("Oracle did not terminate.")


class DomainAndSandboxTests(unittest.TestCase):
    def test_task_round_trip_and_seed_determinism(self):
        for domain in DOMAINS:
            task = make_task(domain, 13)
            self.assertEqual(task, make_task(domain, 13))
            self.assertEqual(task, TaskSpec.from_dict(json.loads(json.dumps(task.to_dict()))))
            self.assertNotEqual(task.target_value, make_task(domain, 14).target_value)

    def test_seed_changes_business_content_beyond_small_modular_cycle(self):
        tasks = [make_task("catalog", seed) for seed in range(200)]
        semantics = {(t.target_value, t.required_items) for t in tasks}
        self.assertEqual(len(semantics), 200)
        self.assertTrue(all(len(t.required_items) in (3, 4) for t in tasks))
        self.assertGreater(len({t.target_value for t in tasks}), 100)

    def test_domains_do_not_share_latent_random_content_for_the_same_seed(self):
        for seed in range(100):
            latent_signatures = set()
            for domain, config in DOMAINS.items():
                task = make_task(domain, seed)
                amount = task.target_value // config["alternate_factor"]
                item_indices = tuple(item.rsplit("_", 1)[-1]
                                     for item in task.required_items
                                     if item not in task.initial_items)
                latent_signatures.add((amount, item_indices))
            self.assertEqual(len(latent_signatures), len(DOMAINS))

    def test_two_real_sql_writes_and_reset_in_every_domain(self):
        for domain in DOMAINS:
            with self.subTest(domain=domain), tempfile.TemporaryDirectory() as tmp:
                task = make_task(domain)
                path = str(Path(tmp) / "world.sqlite")
                with Sandbox(task, candidate_worlds(domain, "no_drift")[0], path) as env:
                    self.assertFalse(env.verify()["success"])
                    first, second = legacy_plan(task)
                    env.act(first)
                    self.assertFalse(env.verify()["success"])
                    # Confirm persistence in an independent SQLite connection.
                    with sqlite3.connect(path) as conn:
                        row = conn.execute(
                            "SELECT value,value_locked FROM {} WHERE id=?".format(DOMAINS[domain]["table"]),
                            (task.target_id,)).fetchone()
                        self.assertEqual(row, (task.target_value, 1))
                    env.act(second)
                    self.assertTrue(env.verify()["success"])
                    self.assertEqual(env.verify()["action_count"], 2)
                    env.reset()
                    self.assertFalse(env.verify()["success"])
                    self.assertEqual(env.query_spent, 0)
                    self.assertEqual(env.action_count, 0)
                    self.assertEqual(len(env._snapshot()["effects"]), 0)

    def test_queries_are_read_only_and_positive_cost(self):
        task = make_task("catalog", query_budget=4)
        with Sandbox(task, candidate_worlds("catalog")[0]) as env:
            before = env._snapshot()
            env.query(Query("read_state"))
            env.query(Query("read_contract"))
            self.assertEqual(before, env._snapshot())
            self.assertEqual(env.query_spent, 4)
            result = env.query(Query("inspect_units"))
            self.assertEqual(result.error, "query_budget_exceeded")
            self.assertEqual(before, env._snapshot())
            self.assertEqual(env.query_spent, 4)

    def test_public_context_and_candidate_menu_do_not_leak_world(self):
        for domain in DOMAINS:
            task = make_task(domain, 4)
            contexts = []
            for world in candidate_worlds(domain, "mixed") + candidate_worlds(domain, "unidentifiable"):
                with Sandbox(task, world) as env:
                    contexts.append(env.public_context())
                    self.assertNotIn(world.world_id, json.dumps(env.public_context()))
            self.assertTrue(all(c == contexts[0] for c in contexts))
            self.assertNotIn("world_id", json.dumps(contexts[0]))
            self.assertNotIn("oracle", contexts[0])

    def test_write_output_does_not_contain_correctness_labels(self):
        task = make_task("travel")
        for world in candidate_worlds("travel", "units"):
            with Sandbox(task, world) as env:
                result = env.act(legacy_plan(task)[0])
                self.assertIsNone(result.success)
                self.assertIsNone(result.invariant_ok)
                self.assertNotIn("success", result.to_dict())
                self.assertNotIn("invariant_ok", result.observation)
                self.assertNotIn("world_id", result.observation)

    def test_default_scope_has_real_protected_side_effect(self):
        task = make_task("calendar")
        world = candidate_worlds("calendar", "default_scope")[1]
        with Sandbox(task, world) as env:
            result = env.act(legacy_plan(task)[0])
            self.assertEqual(result.observation["affected"], 2)
            report = env.verify()
            self.assertFalse(report["invariant_ok"])
            self.assertIn("non_target_changed", report["violations"])

    def test_append_replace_and_duplicate_effects_are_not_simulated_labels(self):
        task = make_task("catalog")
        append, replace = candidate_worlds("catalog", "append_replace")
        with Sandbox(task, replace) as env:
            env.act(legacy_plan(task)[1])
            self.assertIn("initial_items_removed", env.verify()["violations"])
            target = next(r for r in env._records() if r["id"] == task.target_id)
            self.assertNotIn(task.initial_items[0], target["items"])
        with Sandbox(task, append) as env:
            env.act(Action("update_tags", {"record_id": task.target_id,
                                          "items": list(task.required_items)}))
            self.assertIn("duplicate_items", env.verify()["violations"])

    def test_wrong_numeric_commit_can_preserve_invariants_but_is_irreversible(self):
        task = make_task("travel")
        world = candidate_worlds("travel", "units")[0]
        with Sandbox(task, world) as env:
            wrong = Action("set_duration", {"record_id": task.target_id,
                                            "value": task.target_value // 60})
            env.act(wrong)
            self.assertTrue(env.verify()["invariant_ok"])
            self.assertFalse(env.verify()["goal_ok"])
            result = env.act(legacy_plan(task)[0])
            self.assertEqual(result.error, "field_locked")
            env.act(legacy_plan(task)[1])
            self.assertFalse(env.verify()["success"])

    def test_invalid_input_is_atomic_and_sql_safe(self):
        task = make_task("catalog")
        with Sandbox(task, candidate_worlds("catalog")[0]) as env:
            before = env._snapshot()
            for args in [
                {"record_id": "'; DROP TABLE catalog_records;--", "value": 100},
                {"record_id": task.target_id, "value": True},
                {"record_id": task.target_id, "value": -1},
                {"record_id": task.target_id, "value": 20, "unknown": 7},
            ]:
                result = env.act(Action("set_price", args))
                self.assertIsNotNone(result.error)
                self.assertEqual(before, env._snapshot())


class OracleTests(unittest.TestCase):
    def test_no_drift_completes_with_two_actions_and_no_queries(self):
        for domain in DOMAINS:
            task = make_task(domain)
            worlds = candidate_worlds(domain, "no_drift")
            decision, report, history = run_oracle(task, worlds, worlds[0])
            self.assertEqual(decision.kind, "done")
            self.assertTrue(report["success"])
            self.assertEqual(report["query_spent"], 0)
            self.assertEqual(len(history), 2)

    def test_unknown_default_scope_does_not_require_identifying_version(self):
        task = make_task("travel")
        worlds = candidate_worlds("travel", "default_scope")
        oracle = ExactOracle(task, worlds)
        decision = oracle.decide()
        self.assertEqual(decision.kind, "act")
        self.assertEqual(decision.worst_cost, 0)
        self.assertEqual(decision.compatible_count, 2)
        self.assertTrue(decision.actions)
        self.assertTrue(all(a.args.get("scope") == "current" for a in decision.actions))
        for world in worlds:
            _, report, _ = run_oracle(task, worlds, world)
            self.assertTrue(report["success"])
            self.assertEqual(report["query_spent"], 0)

    def test_exact_query_cost_for_each_change_and_combination(self):
        for mode, expected in [("units", 1), ("append_replace", 1), ("mixed", 2)]:
            with self.subTest(mode=mode):
                task = make_task("catalog")
                worlds = candidate_worlds("catalog", mode)
                self.assertEqual(ExactOracle(task, worlds).decide().worst_cost, expected)
                for world in worlds:
                    decision, report, _ = run_oracle(task, worlds, world)
                    self.assertEqual(decision.kind, "done")
                    self.assertTrue(report["success"])
                    self.assertEqual(report["query_spent"], expected)

    def test_optimal_query_ties_are_sets_not_single_teacher_order(self):
        task = make_task("travel")
        worlds = candidate_worlds("travel", "mixed")
        decision = ExactOracle(task, worlds).decide()
        self.assertEqual(decision.kind, "query")
        self.assertEqual({q.name for q in decision.queries},
                         {"inspect_units", "inspect_update_mode"})
        self.assertNotIn("read_state", {q.name for q in decision.queries})
        self.assertNotIn("inspect_default_scope", {q.name for q in decision.queries})

    def test_action_first_tie_break_does_not_erase_full_optimal_set(self):
        task = make_task("travel")
        decision = ExactOracle(task, candidate_worlds("travel", "units")).decide()
        self.assertEqual(decision.kind, "act")
        self.assertTrue(all(d["type"] == "act" for d in decision.decisions()))
        self.assertIn("query", {d["type"] for d in decision.all_optimal_decisions()})
        self.assertIn("act", {d["type"] for d in decision.all_optimal_decisions()})

    def test_query_observation_filters_candidates(self):
        task = make_task("calendar")
        worlds = candidate_worlds("calendar", "mixed")
        oracle = ExactOracle(task, worlds)
        with Sandbox(task, worlds[-1]) as env:
            env.query(Query("inspect_units"))
            compatible = oracle.compatible_worlds(env.history)
            self.assertEqual(len(compatible), 4)
            self.assertTrue(all(w.unit_factor == 60 for w in compatible))

    def test_future_completion_required_not_just_current_invariants(self):
        task = make_task("travel")
        worlds = candidate_worlds("travel", "units")
        oracle = ExactOracle(task, worlds)
        # With unit uncertainty, only the collection write has a robust future;
        # neither numeric value is correct in both one-shot worlds.
        decision = oracle.decide()
        self.assertTrue(decision.safe_actions)
        self.assertTrue(all(a.tool == "update_stops" for a in decision.safe_actions))
        with Sandbox(task, worlds[0]) as env:
            env.act(Action("set_duration", {"record_id": task.target_id,
                                            "value": task.target_value // 60}))
            self.assertTrue(env.verify()["invariant_ok"])
            broken = ExactOracle(task, (worlds[0],)).decide(env.history)
            self.assertEqual(broken.kind, "pause")
            self.assertIsNone(broken.worst_cost)

    def test_individual_world_solvability_is_not_joint_solvability(self):
        task = make_task("catalog")
        worlds = candidate_worlds("catalog", "unidentifiable")
        for world in worlds:
            self.assertNotEqual(ExactOracle(task, (world,)).decide().kind, "pause")
        joint = ExactOracle(task, worlds).decide()
        self.assertEqual(joint.kind, "pause")
        self.assertFalse(joint.safe_actions)
        self.assertIsNone(joint.worst_cost)
        for query in query_space(task):
            with Sandbox(task, worlds[0]) as left, Sandbox(task, worlds[1]) as right:
                self.assertEqual(left._query_observation(query), right._query_observation(query))

    def test_pausing_is_never_zero_cost_success(self):
        task = make_task("travel", query_budget=0)
        worlds = candidate_worlds("travel", "units")
        decision = ExactOracle(task, worlds).decide()
        self.assertEqual(decision.kind, "pause")
        self.assertIsNone(decision.worst_cost)
        with Sandbox(task, worlds[0]) as env:
            self.assertFalse(env.verify()["success"])

    def test_action_horizon_and_query_budget_both_enforced(self):
        task = make_task("calendar", query_budget=1)
        self.assertEqual(ExactOracle(task, candidate_worlds("calendar", "mixed")).decide().kind, "pause")
        self.assertEqual(ExactOracle(task, candidate_worlds("calendar", "no_drift"),
                                     max_actions=1).decide().kind, "pause")
        self.assertEqual(ExactOracle(task, candidate_worlds("calendar", "units")).decide().worst_cost, 1)

    def test_budget_override_cannot_grant_environment_extra_calls(self):
        task = make_task("catalog", query_budget=0)
        decision = ExactOracle(task, candidate_worlds("catalog", "units")).decide(query_budget=100)
        self.assertEqual(decision.kind, "pause")

    def test_exceeded_history_horizon_is_not_retroactive_success(self):
        task = make_task("travel")
        worlds = candidate_worlds("travel", "no_drift")
        with Sandbox(task, worlds[0]) as env:
            env.act(Action("missing_tool", {}))
            for action in legacy_plan(task):
                env.act(action)
            self.assertTrue(env.verify()["success"])
            decision = ExactOracle(task, worlds, max_actions=2).decide(env.history)
            self.assertEqual(decision.kind, "pause")

    def test_history_is_observation_compatible_not_oracle_informed(self):
        task = make_task("catalog")
        worlds = candidate_worlds("catalog", "mixed")
        fake = [{"kind": "query", "call": Query("inspect_units").to_dict(),
                 "observation": {"numeric_input_unit": "parsecs"}}]
        decision = ExactOracle(task, worlds).decide(fake)
        self.assertEqual(decision.kind, "pause")
        self.assertEqual(decision.compatible_count, 0)


class PublicSupportTests(unittest.TestCase):
    def test_public_support_identical_across_actual_worlds(self):
        task = make_task("calendar", 72)
        worlds = candidate_worlds("calendar", "mixed")
        observations = []
        for world in worlds:
            with Sandbox(task, world) as env:
                observations.append(public_observation(
                    env.public_context(), [], worlds, task.query_budget))
        self.assertTrue(all(o == observations[0] for o in observations))
        text = json.dumps(observations[0])
        self.assertNotIn("world_id", text)
        self.assertTrue(all(w.world_id not in text for w in worlds))

    def test_support_is_order_independent_and_semantic_duplicate_invariant(self):
        worlds = candidate_worlds("travel", "mixed")
        first = worlds[0]
        duplicate = WorldSpec("another-private-id", first.domain, first.unit_factor,
                              first.update_mode, first.default_scope, first.observable)
        self.assertEqual(public_support(worlds),
                         public_support(tuple(reversed(worlds)) + (duplicate,)))

    def test_availability_support_does_not_publish_actual_availability(self):
        worlds = (
            WorldSpec("private-visible", "travel", observable=True),
            WorldSpec("private-hidden", "travel", unit_factor=60, observable=False),
        )
        support = public_support(worlds)
        self.assertEqual(support["allowed"]["observable"], [False, True])
        self.assertFalse(support["cartesian"])
        self.assertEqual(len(support["joint_semantics"]), 2)

    def test_same_public_evidence_and_support_implies_same_labels(self):
        task = make_task("catalog", 18)
        worlds = candidate_worlds("catalog", "mixed")
        first = worlds[0]
        renamed = tuple(WorldSpec("renamed-" + str(i), w.domain, w.unit_factor,
                                  w.update_mode, w.default_scope, w.observable)
                        for i, w in enumerate(reversed(worlds)))
        with Sandbox(task, first) as env:
            env.query(Query("inspect_units"))
            left = public_observation(env.public_context(), env.history, worlds, 3)
            right = public_observation(env.public_context(), env.history, renamed, 3)
            self.assertEqual(left, right)
            self.assertEqual(ExactOracle(task, worlds).decide(env.history).decisions(),
                             ExactOracle(task, renamed).decide(env.history).decisions())


if __name__ == "__main__":
    unittest.main()
