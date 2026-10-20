"""Execute the registered 200-step MPS protocol; never download a model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from toolshift.model_policy import canonical_decision, decision_messages, encode_assistant_completion
from toolshift.training import (
    assert_group_disjoint, atomic_json, load_records, train_hf, validate_training_records,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/generated/main")
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/lora_main")
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    protocol_path = destination / "protocol_config.json"
    protocol = json.loads(protocol_path.read_text())
    evaluation_protocol = json.loads((destination / "evaluation_protocol.json").read_text())
    config = dict(protocol["training_config"])
    config["model_name_or_path"] = str(args.model_dir.resolve())
    rows = load_records(args.data_dir / "train.jsonl")
    raw_data_hash = hashlib.sha256((args.data_dir / "train.jsonl").read_bytes()).hexdigest()
    if raw_data_hash != evaluation_protocol["training_data_sha256"]:
        raise ValueError("Training data changed after protocol registration")
    validate_training_records(rows)
    # Only group identifiers from the two holdouts participate in this check.
    holdout_groups = []
    for split in ("dev", "test"):
        holdout_groups.append([{"group_id": row["group_id"]}
                               for row in load_records(args.data_dir / f"{split}.jsonl")])
    assert_group_disjoint(rows, *holdout_groups)

    import torch
    from transformers import AutoTokenizer
    if not torch.backends.mps.is_available():
        raise RuntimeError("This registered experiment requires MPS")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True)
    lengths, targets, alternatives = [], [], []
    for row in rows:
        alternatives.append(len(row["acceptable_decisions"]))
        for decision in row["acceptable_decisions"]:
            item = encode_assistant_completion(
                tokenizer, decision_messages(row["observation"]), canonical_decision(decision),
                config["max_length"])
            lengths.append(len(item["input_ids"]))
            targets.append(sum(label != -100 for label in item["labels"]))
    preflight = {
        "training_records": len(rows),
        "training_jsonl_sha256": raw_data_hash,
        "training_groups": len({row["group_id"] for row in rows}),
        "family_overlap_with_dev_or_test": False,
        "acceptable_alternatives_max": max(alternatives),
        "full_tokens_min": min(lengths), "full_tokens_median": statistics.median(lengths),
        "full_tokens_max": max(lengths),
        "supervised_tokens_min": min(targets), "supervised_tokens_max": max(targets),
        "assistant_prefix_check_passed": True, "truncated_records": 0,
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "protocol_commit": subprocess.check_output(
            ["git", "log", "-1", "--format=%H", "--", "experiments/lora_main/protocol_config.json"],
            cwd=PROJECT, text=True).strip(),
        "runtime_sources_sha256": {
            name: hashlib.sha256((PROJECT / "src/toolshift" / name).read_bytes()).hexdigest()
            for name in ("training.py", "model_policy.py")},
    }
    if max(alternatives) > config["max_alternatives"]:
        raise ValueError("Too many acceptable labels; do not silently remove alternatives")
    atomic_json(destination / "preflight.json", preflight)
    print(json.dumps(preflight, indent=2), flush=True)
    result = train_hf(rows, args.run_dir, config, resume_from=args.resume_from)
    result["manifest"]["config"]["model_name_or_path"] = "LOCAL_SNAPSHOT/SmolLM2-135M-Instruct"
    result["preflight"] = preflight
    result["interpretation"] = protocol["interpretation"]
    atomic_json(destination / "training_summary.json", result)
    print(json.dumps({"status": result["status"], "metrics": result["metrics"],
                      "step_count_this_invocation": len(result["step_measurements"])}, indent=2))


if __name__ == "__main__":
    main()
