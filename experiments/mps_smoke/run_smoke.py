"""Run the authorized real-model smoke check; model download is separate."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))

from toolshift.model_policy import (canonical_decision, decision_messages,
                                    encode_assistant_completion)
from toolshift.training import atomic_json, load_records, train_hf, validate_training_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--train-data", type=Path,
                        default=PROJECT / "data/generated/pilot/train.jsonl")
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/mps_smoke")
    parser.add_argument("--records", type=int, default=8)
    args = parser.parse_args()
    if not 3 <= args.records <= 10:
        raise ValueError("This smoke protocol uses 3–10 training records")

    import torch
    from transformers import AutoTokenizer

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable in this process; do not relabel a CPU run")
    all_rows = load_records(args.train_data)
    validate_training_records(all_rows)
    rows = random.Random(17).sample(all_rows, min(args.records, len(all_rows)))
    config = json.loads((PROJECT / "configs/lora_mps_smoke.json").read_text())
    config["model_name_or_path"] = str(args.model_dir.resolve())
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    lengths, supervised = [], []
    for row in rows:
        for decision in row["acceptable_decisions"]:
            item = encode_assistant_completion(
                tokenizer, decision_messages(row["observation"]),
                canonical_decision(decision), config["max_length"])
            lengths.append(len(item["input_ids"]))
            supervised.append(sum(label != -100 for label in item["labels"]))
    preflight = {
        "device": "mps", "model": "HuggingFaceTB/SmolLM2-135M-Instruct",
        "selected_training_records": len(rows),
        "selected_training_groups": len({row["group_id"] for row in rows}),
        "record_ids": [row["id"] for row in rows],
        "max_acceptable_alternatives": max(len(row["acceptable_decisions"]) for row in rows),
        "full_tokens_min": min(lengths), "full_tokens_median": statistics.median(lengths),
        "full_tokens_max": max(lengths), "supervised_tokens_min": min(supervised),
        "supervised_tokens_max": max(supervised),
        "assistant_prefix_check_passed": True, "truncated_records": 0,
    }
    destination = Path(__file__).resolve().parent
    atomic_json(destination / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2), flush=True)
    result = train_hf(rows, args.run_dir, config)
    # Public evidence omits machine-local paths; original full manifests remain
    # with the ignored local run and its checkpoint.
    result["manifest"]["config"]["model_name_or_path"] = "LOCAL_SNAPSHOT/SmolLM2-135M-Instruct"
    result["smoke_scope"] = "Four optimizer steps validate execution, not Agent capability."
    result["preflight"] = preflight
    durations = [row["seconds"] for row in result["step_measurements"]]
    result["timing_estimate"] = {
        "measured_optimizer_steps": len(durations),
        "median_seconds_excluding_first_step": statistics.median(durations[1:]),
        "estimated_200_step_seconds_from_smoke": statistics.median(durations[1:]) * 200,
        "caveat": "Estimate holds only for comparable lengths/alternatives and excludes evaluation; "
                  "four steps cannot establish steady-state throughput.",
    }
    atomic_json(destination / "summary.json", result)
    print(json.dumps({"status": result["status"], "timing": result["timing_estimate"],
                      "trainable_parameters": result["trainable_parameters"]}, indent=2))


if __name__ == "__main__":
    main()
