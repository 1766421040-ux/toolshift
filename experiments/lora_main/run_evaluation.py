"""Closed-loop base/adapter evaluation on the registered identical test IDs."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from toolshift.evaluation import evaluate
from toolshift.metrics import paired_bootstrap
from toolshift.model_policy import HFCausalPolicy
from toolshift.training import atomic_json, load_records, _base_model_identity


class ProgressPolicy(HFCausalPolicy):
    def choose(self, observation, candidates=None):
        decision = super().choose(observation, candidates)
        if self.usage["score_calls"] % 5 == 0:
            print(json.dumps({"progress_score_calls": self.usage["score_calls"],
                              "candidate_completions": self.usage["candidate_completions"],
                              "input_tokens": self.usage["nonpadding_input_tokens"]}), flush=True)
        return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data/generated/main")
    parser.add_argument("--run-dir", type=Path, default=PROJECT / "runs/lora_main")
    parser.add_argument("--phase", choices=("both", "base", "adapter"), default="both")
    args = parser.parse_args()
    destination = Path(__file__).resolve().parent
    protocol_path = destination / "evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    source = args.data_dir / "episodes.jsonl"
    if hashlib.sha256(source.read_bytes()).hexdigest() != protocol["episodes_data_sha256"]:
        raise ValueError("Evaluation episode data differs from the registered artifact")
    indexed = {row["episode_id"]: row for row in load_records(source)}
    selected = [indexed[episode_id] for episode_id in protocol["episode_ids"]]
    if any(row["split"] != "test" for row in selected):
        raise ValueError("Registered model evaluation includes a non-test episode")
    train_summary = json.loads((destination / "training_summary.json").read_text())
    if _base_model_identity(str(args.model_dir), None) != train_summary["manifest"]["base_model_identity"]:
        raise ValueError("Base-model weights differ from the actual training manifest")
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("The registered local evaluation requires MPS")
    phases = ("base", "adapter") if args.phase == "both" else (args.phase,)
    for phase in phases:
        print(json.dumps({"phase": phase, "episodes": len(selected),
                          "groups": protocol["independent_groups"]}), flush=True)
        started = time.perf_counter()
        policy = ProgressPolicy(
            str(args.model_dir), adapter_path=str(args.run_dir / "adapter") if phase == "adapter" else None,
            load_in_4bit=False, device="mps", max_length=2048,
            max_batch_size=protocol["candidate_batch_size"], completion_logits_only=True)
        result = evaluate(selected, policy, f"hf_135m_{phase}", output=destination / phase)
        result["result"].update(
            model_usage=dict(policy.usage),
            model_loading_and_rollout_seconds=time.perf_counter() - started,
            evaluation_protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
            note="Same public observations and candidate menu; no oracle correction of model choices.",
        )
        atomic_json(destination / phase / "summary.json", result["result"])
        print(json.dumps({"phase": phase, "summary": result["result"]["summary"],
                          "usage": policy.usage}), flush=True)
        del policy
        gc.collect()
        torch.mps.empty_cache()
    base_file = destination / "base" / "episodes.jsonl"
    adapter_file = destination / "adapter" / "episodes.jsonl"
    if base_file.exists() and adapter_file.exists():
        base, adapter = load_records(base_file), load_records(adapter_file)
        comparison = {
            "protocol_version": protocol["protocol_version"],
            "episodes": len(selected), "independent_groups": protocol["independent_groups"],
            "robust_success": paired_bootstrap(adapter, base, metric="robust_success"),
            "actual_world_success": paired_bootstrap(adapter, base, metric="success"),
            "interpretation": "Paired group-bootstrap intervals are descriptive with only "
                              f"{protocol['independent_groups']} independent groups and one training seed.",
        }
        atomic_json(destination / "comparison.json", comparison)
        print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
