"""Project the existing CPU static evaluation onto registered model-test IDs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from toolshift.io import read_jsonl, write_json, write_jsonl
from toolshift.metrics import summarize, paired_bootstrap


def main() -> None:
    destination = Path(__file__).resolve().parent
    protocol_path = destination / "evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    cpu_protocol_path = PROJECT / "experiments/cpu/protocol.json"
    cpu_protocol = json.loads(cpu_protocol_path.read_text())
    if cpu_protocol["dataset_sha256"]["episodes"] != protocol["episodes_data_sha256"]:
        raise ValueError("Static and model runs use different episode datasets")
    source = PROJECT / "experiments/cpu/static/episode_metrics.jsonl"
    existing = read_jsonl(source)
    indexed = {row["episode_id"]: row for row in existing}
    if len(indexed) != len(existing):
        raise ValueError("Duplicate episode IDs in source static evaluation")
    selected = [indexed[episode_id] for episode_id in protocol["episode_ids"]]
    if any(row["split"] != "test" for row in selected):
        raise ValueError("Static reference contains non-test data")
    if len({row["group_id"] for row in selected}) != protocol["independent_groups"]:
        raise ValueError("Static source groups differ from registered evaluation")
    target = destination / "static"
    subset_hash = write_jsonl(target / "episode_metrics.jsonl", selected)
    provenance = {
        "method": "Exact episode-ID projection from the existing full test evaluation; no new rollout.",
        "source": str(source.relative_to(PROJECT)),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_episodes": len(existing),
        "episodes_data_sha256": protocol["episodes_data_sha256"],
        "cpu_protocol_sha256": hashlib.sha256(cpu_protocol_path.read_bytes()).hexdigest(),
        "evaluation_protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    }
    result = {"policy": "static", "summary": summarize(selected),
              "episode_metrics_sha256": subset_hash, "provenance": provenance}
    write_json(target / "summary.json", result)
    comparison_path = destination / "comparison.json"
    comparison = json.loads(comparison_path.read_text())
    adapter = read_jsonl(destination / "adapter/episodes.jsonl")
    comparison["static_reference"] = {
        "provenance": provenance,
        "summary": result["summary"],
        "adapter_minus_static_robust_success": paired_bootstrap(
            adapter, selected, metric="robust_success"),
        "adapter_minus_static_actual_success": paired_bootstrap(
            adapter, selected, metric="success"),
        "weighting_note": "Bootstrap deltas average source-group means equally. "
                          "The summary rates weight episodes equally, so their differences may differ.",
    }
    write_json(comparison_path, comparison)
    print(json.dumps(comparison["static_reference"], indent=2))


if __name__ == "__main__":
    main()
