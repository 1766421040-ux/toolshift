"""Locked CPU experiment suite. Run from the repository root."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

from toolshift.evaluation import evaluate
from toolshift.io import read_jsonl, write_json, write_jsonl
from toolshift.metrics import paired_bootstrap
from toolshift.model_policy import LinearPolicy
from toolshift.policies import RandomPolicy, StaticPolicy
from toolshift.training import train_linear, assert_group_disjoint


def publish_result(name, evaluated):
    root = Path("experiments/cpu") / name
    compact = [{k: v for k, v in row.items() if k not in ("trace", "verification")}
               for row in evaluated["episodes"]]
    write_jsonl(root / "episode_metrics.jsonl", compact)
    # Keep one actual trajectory per suite; every full trajectory remains in runs/.
    examples = {}
    for row in evaluated["episodes"]:
        examples.setdefault(row["suite"], row)
    write_jsonl(root / "example_traces.jsonl", examples.values())
    write_json(root / "summary.json", evaluated["result"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["baselines", "learned", "all"], default="all")
    args = parser.parse_args()
    data = Path("data/generated/main")
    train, dev, test = [read_jsonl(data / f"{s}.jsonl") for s in ("train", "dev", "test")]
    assert_group_disjoint(train, dev, test)
    episodes = [r for r in read_jsonl(data / "episodes.jsonl") if r["split"] == "test"]
    manifest = json.loads((data / "manifest.json").read_text())
    protocol = {
        "dataset_sha256": manifest["sha256"],
        "seeds": [17, 29, 43], "epochs": 12, "learning_rate": 0.25,
        "evaluation": "all test episodes, no test-based selection",
        "ablation": "single first valid teacher label, same observations, seed17 and epochs12",
        "primary": "robust_success_rate; realized success is also reported",
        "baselines": ["oracle", "static", "identify", "refresh", "legacy", "schema", "random"],
    }
    protocol_path = Path("experiments/cpu/protocol.json")
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("Protocol changed; use a separate experiment directory")
    write_json(protocol_path, protocol)
    write_json("experiments/cpu/dataset_manifest.json", manifest)
    if args.stage in ("baselines", "all"):
        for name in protocol["baselines"]:
            policy = None if name == "oracle" else RandomPolicy(17) if name == "random" else StaticPolicy(name)
            result = evaluate(episodes, policy, name, Path("runs/cpu") / name)
            publish_result(name, result)
            print(json.dumps({"policy": name, **result["result"]["summary"]}), flush=True)
    if args.stage in ("learned", "all"):
        for seed, single in [(17, False), (29, False), (43, False), (17, True)]:
            name = f"linear_{'single' if single else 'set'}_{seed}"
            rows = deepcopy(train) if single else train
            if single:
                for row in rows:
                    row["acceptable_decisions"] = row["acceptable_decisions"][:1]
            directory = Path("runs/cpu") / name
            summary = train_linear(rows, directory, {"seed": seed, "epochs": 12})
            write_json(Path("experiments/cpu") / name / "training_summary.json", summary)
            model = LinearPolicy.load(directory / "model.json")
            result = evaluate(episodes, model, name, directory / "evaluation")
            publish_result(name, result)
            print(json.dumps({"policy": name, **result["result"]["summary"]}), flush=True)
    root = Path("experiments/cpu")
    oracle_path = root / "oracle/episode_metrics.jsonl"
    if oracle_path.exists():
        oracle = read_jsonl(oracle_path)
        comparisons = {}
        for path in sorted(root.glob("*/episode_metrics.jsonl")):
            rows = read_jsonl(path)
            comparisons[path.parent.name] = {
                metric: paired_bootstrap(rows, oracle, metric=metric)
                for metric in ("success", "robust_success", "query_cost")
            }
        write_json(root / "paired_comparisons_to_oracle.json", comparisons)


if __name__ == "__main__":
    main()
