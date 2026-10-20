"""Generate independent synthetic fixtures and set-valued decision examples."""
from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .domains import DOMAINS, candidate_worlds, make_task
from .io import canonical, digest, read_jsonl, write_json, write_jsonl
from .oracle import ExactOracle
from .protocol import messages, observation, oracle_labels
from .sandbox import Sandbox
from .schema import Action, Query, TaskSpec, WorldSpec


SUITES = ("no_drift", "units", "append_replace", "default_scope", "mixed", "unidentifiable")
SCHEMA_VERSION = "toolshift.dataset.v1"


def semantic_group(task: TaskSpec) -> str:
    content = task.to_dict()
    for key in ("task_id", "seed", "instruction", "query_budget"):
        content.pop(key)
    return digest(content)[:24]


def partition(group_id: str) -> str:
    bucket = int(digest({"partition": group_id})[:8], 16) % 10
    return "train" if bucket < 6 else "dev" if bucket < 8 else "test"


def task_for(seed: int, domain: str, query_budget: int = 4, template: int = 0) -> TaskSpec:
    task = make_task(domain, seed, query_budget)
    if template == 1:
        task = replace(task, instruction="Please execute this migration-sensitive request: " + task.instruction)
    elif template == 2:
        task = replace(task, instruction="Required final state and preservation constraints follow. " + task.instruction)
    elif template != 0:
        raise ValueError("Unknown template")
    return task


def generate_episodes(seeds: int = 24, heldout_domain: str | None = "calendar",
                      holdout_mixed: bool = True, query_budget: int = 4) -> list[dict]:
    if seeds < 1 or heldout_domain not in (*DOMAINS, None):
        raise ValueError("Invalid generation settings")
    rows = []
    seen_groups = set()
    for domain in DOMAINS:
        for seed in range(seeds):
            task = task_for(seed, domain, query_budget)
            group = semantic_group(task)
            if group in seen_groups:
                continue
            seen_groups.add(group)
            split = "test" if domain == heldout_domain else partition(group)
            template = 2 if split == "test" else seed % 2
            task = task_for(seed, domain, query_budget, template)
            for suite in SUITES:
                # Never place a mixed variant of a training provenance group in
                # test: holdout combinations use entirely held-out task groups.
                if holdout_mixed and suite == "mixed" and split != "test":
                    continue
                worlds = candidate_worlds(domain, suite)
                seen_semantics = set()
                for world in worlds:
                    spec = world.to_dict()
                    spec.pop("world_id")
                    key = canonical(spec)
                    if key in seen_semantics:
                        continue
                    seen_semantics.add(key)
                    category = ("domain" if domain == heldout_domain else
                                "composition" if suite == "mixed" else
                                "instance_and_prefix" if split == "test" else split)
                    rows.append({
                        "schema_version": SCHEMA_VERSION,
                        "episode_id": digest([group, suite, key])[:24], "group_id": group,
                        "split": split, "evaluation_category": category,
                        "domain": domain, "suite": suite, "template": template,
                        "task": task.to_dict(), "world": world.to_dict(),
                        "support": [w.to_dict() for w in worlds],
                    })
    return rows


def episode_objects(episode: dict):
    return (TaskSpec.from_dict(episode["task"]), WorldSpec.from_dict(episode["world"]),
            tuple(WorldSpec.from_dict(w) for w in episode["support"]))


def teacher_records(episodes: Iterable[dict]) -> list[dict]:
    rows: dict[str, dict] = {}
    oracle_cache = {}
    for episode in episodes:
        task, world, worlds = episode_objects(episode)
        cache_key = digest([task.to_dict(), [w.to_dict() for w in worlds]])
        if cache_key not in oracle_cache:
            # Bound RAM while retaining reuse across every version of one task.
            oracle_cache = {cache_key: ExactOracle(task, worlds)}
        oracle = oracle_cache[cache_key]
        with Sandbox(task, world) as env:
            for _ in range(10):
                choice = oracle.decide(env.history, query_budget=task.query_budget - env.query_spent)
                if choice.kind == "done":
                    break
                observed = observation(env.public_context(), env.history, worlds,
                                       task.query_budget - env.query_spent)
                labels = sorted(oracle_labels(choice), key=canonical)
                if not set(map(canonical, labels)).issubset(set(map(canonical, observed["candidates"]))):
                    raise AssertionError("Teacher emitted a decision outside the public menu")
                identity = digest([episode["group_id"], observed])
                row = {
                    "schema_version": SCHEMA_VERSION, "id": identity,
                    "group_id": episode["group_id"], "split": episode["split"],
                    "observation": observed, "messages": messages(observed),
                    "acceptable_decisions": labels,
                }
                if identity in rows and rows[identity]["acceptable_decisions"] != labels:
                    raise AssertionError("Identical public information has conflicting teacher labels")
                rows[identity] = row
                decision = labels[0]
                if decision["type"] == "stop":
                    break
                if decision["type"] == "query":
                    env.query(Query.from_dict(decision))
                else:
                    env.act(Action.from_dict(decision))
            else:
                raise AssertionError("Teacher exceeded bounded horizon")
    return list(rows.values())


def build_dataset(output: str | Path, seeds: int = 24,
                  heldout_domain: str | None = "calendar", query_budget: int = 4) -> dict:
    output = Path(output)
    episodes = generate_episodes(seeds, heldout_domain, query_budget=query_budget)
    records = teacher_records(episodes)
    groups = {}
    for row in episodes:
        previous = groups.setdefault(row["group_id"], row["split"])
        if previous != row["split"]:
            raise AssertionError("Provenance group leakage")
    hashes = {"episodes": write_jsonl(output / "episodes.jsonl", episodes)}
    for split in ("train", "dev", "test"):
        hashes[split] = write_jsonl(output / f"{split}.jsonl",
                                   (row for row in records if row["split"] == split))
    manifest = {
        "schema_version": SCHEMA_VERSION, "seeds_requested_per_domain": seeds,
        "heldout_domain": heldout_domain, "query_budget": query_budget,
        "episodes": len(episodes), "decision_records": len(records),
        "unique_semantic_groups": len(groups),
        "episode_splits": dict(Counter(row["split"] for row in episodes)),
        "record_splits": dict(Counter(row["split"] for row in records)),
        "evaluation_categories": dict(Counter(row["evaluation_category"] for row in episodes)),
        "sha256": hashes,
        "assumptions": [
            "Every policy receives the same declared finite semantic support.",
            "Three domains share two underlying operator families.",
            "Prefix holdout is superficial wording variation, not new language templates.",
            "Teacher uses act-first tie-breaking among minimum-worst-cost policies.",
            "Test fixtures and labels are evaluator-owned; they are never training inputs.",
        ],
    }
    write_json(output / "manifest.json", manifest)
    return manifest

