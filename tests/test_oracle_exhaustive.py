"""Finite coverage plus a non-DP Cartesian enumeration of tiny query trees."""

import itertools
import unittest

from toolshift.domains import (DOMAINS, action_space, candidate_worlds, legacy_plan,
                               make_task, query_cost, query_space)
from toolshift.oracle import ExactOracle
from toolshift.sandbox import Sandbox
from toolshift.schema import canonical


MANUAL_COST = {
    "no_drift": 0, "units": 1, "append_replace": 1,
    "default_scope": 0, "mixed": 2, "unidentifiable": None,
}


def enumerate_tiny_query_trees(task, worlds, budget):
    """Enumerate concrete trees with 0, 1, or 2 reads then one numeric commit.

    A common collection commit has already happened. Terminal-action outcomes
    and query response tables come from fresh SQLite executions. The code does
    not call oracle internals, recurse over value functions, cache subproblems,
    or discard no-information queries.

    A tree may stop after its first query in one branch but take a second query
    in another. Only query trees that complete in all actual worlds are counted.
    """
    prefix = legacy_plan(task)[1]
    actions = tuple(a for a in action_space(task)
                    if a.tool == DOMAINS[task.domain]["numeric_tool"])
    queries = query_space(task)
    success = {}
    response = {}
    history = None
    for i, world in enumerate(worlds):
        for action in actions:
            with Sandbox(task, world) as env:
                env.act(prefix)
                if history is None:
                    history = list(env.history)
                env.act(action)
                success[i, action.key()] = env.verify()["success"]
        for query in queries:
            with Sandbox(task, world) as env:
                env.act(prefix)
                result = env.query(query)
                if result.error:
                    raise AssertionError("Response-table query unexpectedly failed.")
                response[i, query.name] = canonical(result.observation)

    indices = tuple(range(len(worlds)))
    successful_costs = []
    uninformative_root_successes = 0
    # Zero-query trees consist only of a constant terminal action.
    for action in actions:
        if all(success[i, action.key()] for i in indices):
            successful_costs.append(0)

    for first_query in queries:
        first_cost = query_cost(first_query)
        if first_cost > budget:
            continue
        first_observations = sorted({response[i, first_query.name] for i in indices})
        first_groups = {
            answer: tuple(i for i in indices if response[i, first_query.name] == answer)
            for answer in first_observations
        }
        options_per_branch = []
        for first_answer in first_observations:
            group = first_groups[first_answer]
            options = []
            # A branch may commit immediately after observing the first answer.
            for action in actions:
                if all(success[i, action.key()] for i in group):
                    options.append(("act", action))
            # Or choose any affordable second query, including repeats and
            # non-splitting queries, and a concrete terminal action per answer.
            for second_query in queries:
                if first_cost + query_cost(second_query) > budget:
                    continue
                second_answers = sorted({response[i, second_query.name] for i in group})
                terminal_options = []
                for answer in second_answers:
                    compatible = [i for i in group if response[i, second_query.name] == answer]
                    terminal_options.append([
                        action for action in actions
                        if all(success[i, action.key()] for i in compatible)
                    ])
                for choices in itertools.product(*terminal_options):
                    options.append(("query", second_query,
                                    dict(zip(second_answers, choices))))
            options_per_branch.append(options)

        # This is explicit enumeration of complete policies, not minimization
        # of child values. Evaluate every resulting policy in every true world.
        for branches in itertools.product(*options_per_branch):
            tree = dict(zip(first_observations, branches))
            costs = []
            for i in indices:
                branch = tree[response[i, first_query.name]]
                if branch[0] == "act":
                    action = branch[1]
                    cost = first_cost
                else:
                    second_query, leaves = branch[1:]
                    action = leaves[response[i, second_query.name]]
                    cost = first_cost + query_cost(second_query)
                if cost > budget or not success[i, action.key()]:
                    raise AssertionError("Enumeration constructed an invalid purported policy.")
                costs.append(cost)
            successful_costs.append(max(costs))
            if len(first_groups) == 1:
                uninformative_root_successes += 1
    return {
        "minimum": min(successful_costs) if successful_costs else None,
        "successful_policies": len(successful_costs),
        "uninformative_root_successes": uninformative_root_successes,
        "history": history,
    }


class ExhaustiveOracleTests(unittest.TestCase):
    def test_hand_derived_costs_and_completion_for_all_worlds_budgets_and_domains(self):
        configurations = 0
        rollouts = 0
        for domain in DOMAINS:
            for suite, required in MANUAL_COST.items():
                for budget in range(5):
                    with self.subTest(domain=domain, suite=suite, budget=budget):
                        configurations += 1
                        task = make_task(domain, seed=47, query_budget=budget)
                        worlds = candidate_worlds(domain, suite)
                        oracle = ExactOracle(task, worlds, max_actions=2)
                        feasible = required is not None and budget >= required
                        first = oracle.decide()
                        self.assertEqual(first.worst_cost, required if feasible else None)
                        self.assertEqual(first.kind == "pause", not feasible)
                        for world in worlds:
                            rollouts += 1
                            with Sandbox(task, world) as env:
                                for _ in range(8):
                                    choice = oracle.decide(env.history)
                                    if choice.kind == "act":
                                        result = env.act(choice.actions[0])
                                    elif choice.kind == "query":
                                        result = env.query(choice.queries[0])
                                    else:
                                        break
                                    self.assertIsNone(result.error)
                                    self.assertTrue(env.verify()["invariant_ok"])
                                else:
                                    self.fail("Finite-horizon policy did not terminate.")
                                report = env.verify()
                                self.assertEqual(report["success"], feasible)
                                if feasible:
                                    self.assertEqual(choice.kind, "done")
                                    self.assertEqual(report["query_spent"], required)
                                    self.assertEqual(report["action_count"], 2)
                                else:
                                    self.assertEqual(choice.kind, "pause")
                                    self.assertEqual(report["query_spent"], 0)
                                    self.assertEqual(report["action_count"], 0)
        self.assertEqual(configurations, 90)
        self.assertEqual(rollouts, 270)

    def test_independent_cartesian_tree_enumeration_matches_dp(self):
        task = make_task("travel", seed=17, query_budget=4)
        for suite, expected in (("no_drift", 0), ("units", 1), ("unidentifiable", None)):
            worlds = candidate_worlds("travel", suite)
            for budget in range(3):
                with self.subTest(suite=suite, budget=budget):
                    enumeration = enumerate_tiny_query_trees(task, worlds, budget)
                    oracle = ExactOracle(task, worlds, max_actions=2)
                    decision = oracle.decide(enumeration["history"], query_budget=budget)
                    manual = expected if expected is not None and budget >= expected else None
                    self.assertEqual(enumeration["minimum"], manual)
                    self.assertEqual(decision.worst_cost, enumeration["minimum"])
                    if suite == "units" and budget == 2:
                        # The brute enumerator explicitly includes cost-2 plans
                        # that first ask a useless question. Their presence does
                        # not lower the cost-1 optimum; DP may safely prune them.
                        self.assertGreater(enumeration["uninformative_root_successes"], 0)
                        self.assertGreater(enumeration["successful_policies"], 1)


if __name__ == "__main__":
    unittest.main()
