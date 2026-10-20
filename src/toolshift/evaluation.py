"""Closed-loop evaluation. Private verification never enters policy input."""
from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import subprocess
import platform
import time

from .dataset import episode_objects
from .io import canonical, read_jsonl, write_json, write_jsonl
from .metrics import grouped_summary, summarize
from .oracle import ExactOracle
from .protocol import observation, oracle_labels
from .sandbox import Sandbox
from .schema import Action, Query


def rollout(episode: dict, policy, policy_name: str = "unknown", max_steps: int = 10,
            oracle: ExactOracle | None = None) -> dict:
    task, world, worlds = episode_objects(episode)
    oracle = oracle or ExactOracle(task, worlds)
    feasible = oracle.decide().kind != "pause"
    started = time.perf_counter()
    with Sandbox(task, world) as env:
        termination = "horizon"
        trace = []
        robust_actions = True
        uncertified_actions = 0
        policy_seconds = 0.0
        audit_seconds = 0.0
        environment_seconds = 0.0
        for step in range(max_steps):
            observed = observation(env.public_context(), env.history, worlds,
                                   task.query_budget - env.query_spent)
            authorized_menu = frozenset(map(canonical, observed["candidates"]))
            # oracle is a separate labeled experimental control. It is NEVER
            # an implicit guardrail/correction for a learned policy.
            if policy_name == "oracle":
                decision_started = time.perf_counter()
                reference = oracle.decide(env.history, query_budget=observed["remaining_budget"])
                policy_seconds += time.perf_counter() - decision_started
                if reference.kind == "done":
                    termination = "completed"
                    break
                decision = sorted(oracle_labels(reference), key=canonical)[0]
            else:
                try:
                    decision_started = time.perf_counter()
                    decision = policy.choose(deepcopy(observed))
                    policy_seconds += time.perf_counter() - decision_started
                except Exception as exc:
                    policy_seconds += time.perf_counter() - decision_started
                    trace.append({"step": step, "policy_error": type(exc).__name__ + ": " + str(exc)})
                    termination = "policy_error"
                    break
            try:
                valid = canonical(decision) in authorized_menu
            except (ValueError, TypeError, OverflowError):
                valid = False
            if not valid:
                # Never try to persist a non-serializable policy object.
                trace.append({"step": step, "error": "outside_public_menu",
                              "returned_type": type(decision).__name__})
                termination = "invalid_decision"
                break
            if decision["type"] == "stop":
                termination = "abstained"
                trace.append({"step": step, "decision": decision})
                break
            if decision["type"] == "query":
                execution_started = time.perf_counter()
                result = env.query(Query.from_dict(decision))
                environment_seconds += time.perf_counter() - execution_started
            else:
                audit_started = time.perf_counter()
                certificate = oracle.decide(env.history, query_budget=observed["remaining_budget"])
                certified = canonical(decision) in {
                    canonical(a.decision()) for a in certificate.safe_actions
                }
                robust_actions = robust_actions and certified
                uncertified_actions += not certified
                audit_seconds += time.perf_counter() - audit_started
                execution_started = time.perf_counter()
                result = env.act(Action.from_dict(decision))
                environment_seconds += time.perf_counter() - execution_started
            trace.append({"step": step, "decision": decision, "observation": result.observation})
            # Two acknowledged field commits form a public terminal condition.
            # This does not reveal whether their business effects were correct.
            committed = {h["call"]["tool"] for h in env.history
                         if h["kind"] == "act" and h["observation"].get("ok") is True}
            if len(committed) == 2:
                termination = "committed"
                break
        report = env.verify()
        row = {
            **{key: episode[key] for key in ("episode_id", "group_id", "domain", "suite",
                                             "split", "evaluation_category")},
            "policy": policy_name, "success": bool(report["success"]),
            "robust_success": bool(report["success"]) and robust_actions,
            "uncertified_actions": uncertified_actions,
            "invariant_ok": bool(report["invariant_ok"]), "oracle_feasible": feasible,
            "termination": termination, "query_cost": env.query_spent,
            "query_count": sum(h["kind"] == "query" for h in env.history),
            "action_count": env.action_count, "wall_seconds": time.perf_counter() - started,
            "policy_seconds": policy_seconds, "audit_seconds": audit_seconds,
            "environment_seconds": environment_seconds,
            "trace": trace, "verification": report,
            "domain_suite": episode["domain"] + "/" + episode["suite"],
        }
    return row


def evaluate(episodes: list[dict], policy, name: str, output: str | Path | None = None) -> dict:
    rows = []
    current_key = None
    oracle = None
    for episode in episodes:
        key = canonical([episode["task"], episode["support"]])
        if key != current_key:
            task, _, worlds = episode_objects(episode)
            oracle = ExactOracle(task, worlds)
            current_key = key
        rows.append(rollout(episode, policy, name, oracle=oracle))
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                           stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        revision = None
    result = {
        "policy": name, "summary": summarize(rows), "by_suite": grouped_summary(rows, "suite"),
        "by_category": grouped_summary(rows, "evaluation_category"),
        "by_domain": grouped_summary(rows, "domain"),
        "by_domain_and_suite": grouped_summary(rows, "domain_suite"),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "git_revision": revision},
    }
    if output is not None:
        output = Path(output)
        result["trace_sha256"] = write_jsonl(output / "episodes.jsonl", rows)
        write_json(output / "summary.json", result)
    return {"result": result, "episodes": rows}

