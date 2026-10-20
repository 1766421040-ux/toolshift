"""Exact privileged upper bound for a small, finite deterministic sandbox.

This enumerates real SQLite transitions. It is not a scalable planning claim.
The objective is worst-case *query cost*, subject to robust task completion,
process invariants, query budget, and a finite remaining write-attempt horizon.
Stopping without successful completion has infinite cost, not zero cost.
"""

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple
import math

from .domains import action_space, query_cost, query_space
from .sandbox import Sandbox
from .schema import Action, OracleDecision, Query, TaskSpec, WorldSpec, canonical


@dataclass(frozen=True)
class _Solution:
    cost: float
    actions: Tuple[Action, ...] = ()
    queries: Tuple[Query, ...] = ()
    safe_actions: Tuple[Action, ...] = ()


class ExactOracle:
    """Evaluation/labeling service with private hypothesis access.

    `history` uses Sandbox.history entries. An explicit query_budget denotes
    the REMAINING budget; None deducts observed query costs from TaskSpec.
    Both actual history and all simulated futures must use public observations.
    """

    def __init__(self, task: TaskSpec, worlds: Sequence[WorldSpec],
                 max_actions: int = 4) -> None:
        if not worlds:
            raise ValueError("At least one candidate world is required.")
        if any(w.domain != task.domain for w in worlds):
            raise ValueError("Mixed domains in candidate worlds.")
        if len({w.world_id for w in worlds}) != len(worlds):
            raise ValueError("Candidate world IDs must be unique.")
        if type(max_actions) is not int or max_actions < 0:
            raise ValueError("max_actions must be a nonnegative integer.")
        self.task = task
        self.worlds = tuple(worlds)
        self.max_actions = max_actions
        self._actions = action_space(task)
        self._queries = query_space(task)
        self._memo: Dict[Any, _Solution] = {}
        self._transition_cache: Dict[Any, Any] = {}
        self._query_cache: Dict[Any, Any] = {}
        self._verify_cache: Dict[Any, Any] = {}
        self._states_examined = 0

    @staticmethod
    def _entry_call(entry: dict) -> Any:
        return entry.get("call", entry.get("decision", {}))

    def _replay(self, history: Sequence[dict]) -> Tuple[tuple, int, int]:
        states = []
        spent = 0
        attempts = 0
        for entry in history:
            kind = entry.get("kind", entry.get("type"))
            if kind == "query":
                spent += query_cost(Query.from_dict(self._entry_call(entry)))
            elif kind == "act":
                attempts += 1
            else:
                raise ValueError("History kind must be query or act.")
        for i, world in enumerate(self.worlds):
            with Sandbox(self.task, world) as sandbox:
                compatible = True
                for entry in history:
                    kind = entry.get("kind", entry.get("type"))
                    call = self._entry_call(entry)
                    if kind == "query":
                        observation = sandbox._query_observation(Query.from_dict(call))
                    else:
                        observation = sandbox.act(Action.from_dict(call)).observation
                    if canonical(observation) != canonical(entry.get("observation")):
                        compatible = False
                        break
                if compatible:
                    states.append((i, canonical(sandbox._snapshot())))
        return tuple(states), spent, attempts

    def compatible_worlds(self, history: Sequence[dict] = ()) -> Tuple[WorldSpec, ...]:
        """Privileged debugging helper. Never include its result in a prompt."""
        states, _, _ = self._replay(history)
        return tuple(self.worlds[i] for i, _ in states)

    def _open(self, state: tuple) -> Sandbox:
        import json
        i, snapshot = state
        return Sandbox._from_snapshot(self.task, self.worlds[i], json.loads(snapshot))

    def _verify(self, state: tuple) -> dict:
        if state not in self._verify_cache:
            with self._open(state) as sandbox:
                self._verify_cache[state] = sandbox.verify()
        return self._verify_cache[state]

    def _transition(self, state: tuple, action: Action) -> tuple:
        key = (state, action.key())
        if key not in self._transition_cache:
            with self._open(state) as sandbox:
                result = sandbox.act(action)
                self._transition_cache[key] = (
                    result.error, canonical(result.observation),
                    (state[0], canonical(sandbox._snapshot())))
        return self._transition_cache[key]

    def _observation(self, state: tuple, query: Query) -> str:
        key = (state, query.key())
        if key not in self._query_cache:
            with self._open(state) as sandbox:
                self._query_cache[key] = canonical(sandbox._query_observation(query))
        return self._query_cache[key]

    @staticmethod
    def _groups(pairs: Sequence[tuple]) -> tuple:
        partitions: Dict[str, list] = {}
        for observation, state in pairs:
            partitions.setdefault(observation, []).append(state)
        return tuple(tuple(values) for _, values in sorted(partitions.items()))

    def _solve(self, states: tuple, budget: int, remaining_actions: int) -> _Solution:
        key = (states, budget, remaining_actions)
        if key in self._memo:
            return self._memo[key]
        self._states_examined += 1
        reports = [self._verify(s) for s in states]
        if not states or any(not r["invariant_ok"] for r in reports):
            solution = _Solution(math.inf)
        elif all(r["success"] for r in reports):
            solution = _Solution(0)
        elif remaining_actions <= 0:
            solution = _Solution(math.inf)
        else:
            action_values = []
            for action in self._actions:
                transitions = [self._transition(s, action) for s in states]
                if any(error for error, _, _ in transitions):
                    continue
                successors = self._groups([(obs, s) for _, obs, s in transitions])
                # A shared action is safe only when every observable successor
                # belief admits a shared completing strategy. Individual-world
                # reachability alone is deliberately insufficient.
                costs = [self._solve(group, budget, remaining_actions - 1).cost
                         for group in successors]
                value = max(costs)
                if math.isfinite(value):
                    action_values.append((action, value))
            query_values = []
            for query in self._queries:
                cost = query_cost(query)
                if cost > budget:
                    continue
                groups = self._groups([(self._observation(s, query), s) for s in states])
                if len(groups) == 1:
                    # A deterministic query that never separates this belief
                    # cannot help a history-aware policy; prune its repeat loop.
                    continue
                tail = max(self._solve(g, budget - cost, remaining_actions).cost
                           for g in groups)
                value = cost + tail
                if math.isfinite(value):
                    query_values.append((query, value))
            values = [v for _, v in action_values] + [v for _, v in query_values]
            best = min(values) if values else math.inf
            solution = _Solution(
                best,
                tuple(a for a, v in action_values if v == best),
                tuple(q for q, v in query_values if v == best),
                tuple(a for a, _ in action_values),
            )
        self._memo[key] = solution
        return solution

    def decide(self, history: Sequence[dict] = (),
               query_budget: Any = None) -> OracleDecision:
        states, spent, attempts = self._replay(history)
        available = self.task.query_budget - spent
        budget = available if query_budget is None else query_budget
        if type(budget) is not int or budget < 0 or available < 0:
            return OracleDecision("pause", reason="Query budget is exhausted or invalid.",
                                  compatible_count=len(states))
        # An explicit tighter budget is useful for budget sweeps. It cannot
        # grant more calls than the actual environment still permits.
        budget = min(budget, available)
        if not states:
            return OracleDecision("pause", reason="History matches no candidate world.")
        remaining = self.max_actions - attempts
        if remaining < 0:
            return OracleDecision("pause", reason="Write-attempt horizon was exceeded.",
                                  compatible_count=len(states))
        solution = self._solve(states, budget, remaining)
        if not math.isfinite(solution.cost):
            return OracleDecision(
                "pause", reason="No invariant-preserving completing policy exists within the budgets.",
                states_examined=self._states_examined, compatible_count=len(states))
        if all(self._verify(s)["success"] for s in states):
            kind = "done"
        elif solution.actions:
            # Acting wins equal-cost ties: full version identification is not
            # the goal when a common completing next action already exists.
            kind = "act"
        else:
            kind = "query"
        return OracleDecision(
            kind, actions=solution.actions, queries=solution.queries,
            safe_actions=solution.safe_actions, worst_cost=int(solution.cost),
            reason="Exact finite-model minimax query cost; privileged upper bound.",
            states_examined=self._states_examined, compatible_count=len(states))

    def common_correct_actions(self, history: Sequence[dict] = (),
                               query_budget: Any = None) -> Tuple[Action, ...]:
        """All robustly completing shared actions under the remaining budgets."""
        return self.decide(history, query_budget).safe_actions
