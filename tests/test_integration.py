"""Independent integration checks for information, grouping, and evaluation."""

import json
import unittest
from collections import defaultdict

from toolshift.dataset import (episode_objects, generate_episodes, semantic_group,
                               teacher_records, task_for)
from toolshift.evaluation import evaluate, rollout
from toolshift.io import canonical
from toolshift.metrics import paired_bootstrap, summarize
from toolshift.policies import StaticPolicy
from toolshift.protocol import observation, validate_public_observation
from toolshift.sandbox import Sandbox


def find_episode(episodes, suite, alternate=False):
    matches = [e for e in episodes if e["suite"] == suite]
    if alternate:
        return next(e for e in matches if e["world"]["unit_factor"] > 1)
    return matches[0]


class _ExceptionPolicy:
    def choose(self, observed):
        raise ValueError("deliberate policy failure")


class _OutsideMenuPolicy:
    def choose(self, observed):
        return {"type": "act", "name": "not_a_tool", "arguments": {}}


class _NonJSONPolicy:
    def choose(self, observed):
        return object()


class _MutatingMenuPolicy:
    def choose(self, observed):
        numeric = next((c for c in observed["candidates"]
                        if c.get("type") == "act" and "value" in c.get("arguments", {})), None)
        if numeric is None:
            return {"type": "stop", "status": "pause"}
        # A plugin should not be able to rewrite its own authorization menu.
        numeric["arguments"]["value"] = 1
        return numeric


class DatasetIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = generate_episodes(seeds=2, heldout_domain=None,
                                        holdout_mixed=False)
        cls.records = teacher_records(cls.episodes)

    def test_all_public_records_are_valid_and_labels_are_menu_subsets(self):
        for row in self.records:
            validate_public_observation(row["observation"])
            menu = set(map(canonical, row["observation"]["candidates"]))
            labels = set(map(canonical, row["acceptable_decisions"]))
            self.assertTrue(labels)
            self.assertTrue(labels.issubset(menu))
            self.assertEqual(json.loads(row["messages"][1]["content"]), row["observation"])

    def test_actual_world_never_changes_initial_public_information(self):
        public_by_support = defaultdict(list)
        for episode in self.episodes:
            task, world, worlds = episode_objects(episode)
            with Sandbox(task, world) as env:
                public_by_support[(episode["group_id"], episode["suite"])].append(
                    observation(env.public_context(), [], worlds, task.query_budget))
        for values in public_by_support.values():
            self.assertTrue(all(v == values[0] for v in values))

    def test_dedup_uses_public_observation_and_has_consistent_labels(self):
        observed = {}
        for record in self.records:
            key = canonical(record["observation"])
            self.assertNotIn(key, observed, "Duplicate public training decision")
            observed[key] = record["acceptable_decisions"]
        self.assertEqual(len({r["id"] for r in self.records}), len(self.records))

    def test_episode_ids_unique_and_identical_semantics_not_repeated_per_suite(self):
        self.assertEqual(len({e["episode_id"] for e in self.episodes}), len(self.episodes))
        seen = set()
        for episode in self.episodes:
            world = dict(episode["world"])
            world.pop("world_id")
            key = (episode["group_id"], episode["suite"], canonical(world))
            self.assertNotIn(key, seen)
            seen.add(key)

    def test_paired_versions_and_templates_never_cross_splits(self):
        episodes = generate_episodes(seeds=24, heldout_domain="calendar")
        groups = defaultdict(set)
        for episode in episodes:
            task, _, _ = episode_objects(episode)
            self.assertEqual(episode["group_id"], semantic_group(task))
            groups[episode["group_id"]].add(episode["split"])
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        self.assertEqual({e["split"] for e in episodes}, {"train", "dev", "test"})
        self.assertTrue(all(e["split"] == "test" for e in episodes if e["domain"] == "calendar"))
        self.assertTrue(all(e["split"] == "test" for e in episodes if e["suite"] == "mixed"))
        self.assertTrue(all(e["template"] != 2 for e in episodes if e["split"] == "train"))

    def test_broad_seed_content_does_not_collapse_to_task_id_variants(self):
        episodes = generate_episodes(seeds=40, heldout_domain=None)
        groups = {e["group_id"] for e in episodes}
        self.assertEqual(len(groups), 3 * 40)

    def test_query_budget_variants_keep_the_same_provenance_group(self):
        groups = {semantic_group(task_for(31, "catalog", query_budget=budget))
                  for budget in (0, 1, 2, 4)}
        self.assertEqual(len(groups), 1)


class EvaluationIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.episodes = generate_episodes(seeds=1, heldout_domain=None,
                                        holdout_mixed=False)
        cls.static_run = evaluate(cls.episodes, StaticPolicy("static"), "static")
        cls.oracle_run = evaluate(cls.episodes, None, "oracle")

    def test_static_matches_oracle_success_and_query_cost_in_all_current_suites(self):
        oracle = {r["episode_id"]: r for r in self.oracle_run["episodes"]}
        for row in self.static_run["episodes"]:
            reference = oracle[row["episode_id"]]
            self.assertEqual(row["success"], reference["success"],
                             (row["domain"], row["suite"]))
            self.assertEqual(row["query_cost"], reference["query_cost"],
                             (row["domain"], row["suite"]))
            self.assertEqual(row["invariant_ok"], reference["invariant_ok"])
            self.assertEqual(row["robust_success"], reference["robust_success"])
        self.assertEqual(self.static_run["result"]["summary"]["feasible_success_rate"], 1.0)
        self.assertEqual(self.static_run["result"]["summary"]["feasible_robust_success_rate"], 1.0)
        self.assertEqual(self.static_run["result"]["summary"]["impossible_case_action_rate"], 0.0)

    def test_unidentifiable_abstention_is_not_counted_as_success(self):
        rows = [r for r in self.static_run["episodes"] if r["suite"] == "unidentifiable"]
        self.assertTrue(rows)
        self.assertTrue(all(not r["success"] and r["termination"] == "abstained" for r in rows))
        self.assertTrue(all(r["query_cost"] == 0 and r["action_count"] == 0 for r in rows))
        summary = summarize(rows)
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertIsNone(summary["query_cost_per_success"])
        self.assertIsNone(summary["feasible_success_rate"])

    def test_lucky_unidentifiable_execution_is_success_but_not_robust(self):
        # The canonical-unit world lets a blind legacy plan guess correctly;
        # the public evidence cannot exclude the alternative-unit world.
        episode = next(e for e in self.episodes if e["suite"] == "unidentifiable"
                       and e["world"]["unit_factor"] == 1)
        row = rollout(episode, StaticPolicy("legacy"), "legacy")
        self.assertTrue(row["success"])
        self.assertFalse(row["robust_success"])
        self.assertGreater(row["uncertified_actions"], 0)
        self.assertFalse(row["oracle_feasible"])
        # Certification must remain measurement only, not silently veto writes.
        self.assertEqual(row["action_count"], 2)
        self.assertEqual(row["termination"], "committed")
        self.assertTrue(all(t["observation"]["ok"] for t in row["trace"]))
        summary = summarize([row])
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertEqual(summary["robust_success_rate"], 0.0)

    def test_tool_acknowledgments_do_not_replace_independent_goal_check(self):
        episode = find_episode(self.episodes, "units", alternate=True)
        row = rollout(episode, StaticPolicy("legacy"), "legacy")
        self.assertEqual(row["termination"], "committed")
        self.assertTrue(all(t["observation"]["ok"] for t in row["trace"]))
        self.assertTrue(row["invariant_ok"])
        self.assertFalse(row["success"])
        self.assertFalse(row["verification"]["goal_ok"])

    def test_non_target_damage_remains_failure_even_if_target_goal_met(self):
        episode = next(e for e in self.episodes
                       if e["suite"] == "default_scope" and e["world"]["default_scope"] == "all")
        row = rollout(episode, StaticPolicy("legacy"), "legacy")
        self.assertTrue(row["verification"]["goal_ok"])
        self.assertFalse(row["invariant_ok"])
        self.assertFalse(row["success"])
        self.assertIn("non_target_changed", row["verification"]["violations"])

    def test_query_cost_and_query_count_have_distinct_meanings(self):
        episode = find_episode(self.episodes, "mixed")
        refreshed = rollout(episode, StaticPolicy("refresh"), "refresh")
        static = rollout(episode, StaticPolicy("static"), "static")
        self.assertTrue(refreshed["success"] and static["success"])
        self.assertEqual((refreshed["query_count"], refreshed["query_cost"]), (1, 3))
        self.assertEqual((static["query_count"], static["query_cost"]), (2, 2))

    def test_cost_per_success_charges_unsuccessful_attempts(self):
        episode = find_episode(self.episodes, "no_drift")
        success = rollout(episode, StaticPolicy("refresh"), "refresh")
        failure = dict(success, episode_id="independent-failure", success=False,
                       query_cost=4, query_count=4)
        summary = summarize([success, failure])
        self.assertEqual(summary["mean_successful_query_cost"], 3)
        self.assertEqual(summary["query_cost_per_success"], 7)

    def test_policy_exception_and_outside_menu_are_distinct(self):
        episode = find_episode(self.episodes, "no_drift")
        error = rollout(episode, _ExceptionPolicy(), "exception")
        invalid = rollout(episode, _OutsideMenuPolicy(), "invalid")
        self.assertEqual(error["termination"], "policy_error")
        self.assertEqual(invalid["termination"], "invalid_decision")
        self.assertFalse(error["success"] or invalid["success"])
        self.assertEqual(error["query_cost"] + invalid["query_cost"], 0)

    def test_non_json_policy_output_is_contained_in_one_failed_episode(self):
        episode = find_episode(self.episodes, "no_drift")
        row = rollout(episode, _NonJSONPolicy(), "non_json")
        self.assertIn(row["termination"], ("invalid_decision", "policy_error"))
        self.assertFalse(row["success"])
        # The resulting failure record must itself remain serializable.
        json.dumps(row)

    def test_policy_errors_are_visible_in_summary(self):
        episode = find_episode(self.episodes, "no_drift")
        error = rollout(episode, _ExceptionPolicy(), "exception")
        summary = summarize([error])
        self.assertEqual(summary["policy_error_rate"], 1.0)

    def test_policy_cannot_rewrite_authorized_menu_by_mutating_input(self):
        episode = find_episode(self.episodes, "no_drift")
        row = rollout(episode, _MutatingMenuPolicy(), "mutating")
        self.assertEqual(row["termination"], "invalid_decision")
        self.assertEqual(row["action_count"], 0)
        self.assertFalse(row["success"])

    def test_paired_group_intervals_do_not_treat_versions_as_independent(self):
        interval = paired_bootstrap(self.static_run["episodes"],
                                    self.oracle_run["episodes"], resamples=100)
        self.assertEqual(interval["independent_groups"], 3)
        self.assertEqual(interval["difference_left_minus_right"], 0.0)
        self.assertEqual(interval["ci95"], [0.0, 0.0])
        self.assertIsNotNone(interval["warning"])


if __name__ == "__main__":
    unittest.main()
