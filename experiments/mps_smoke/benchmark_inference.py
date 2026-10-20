"""Compare candidate batching on one fixed local adapter without hidden labels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "src"))
from toolshift.model_policy import HFCausalPolicy
from toolshift.training import atomic_json, load_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS unavailable")
    # Only public observations enter inference; labels are never inspected.
    observations = [row["observation"] for row in load_records(
        PROJECT / "data/generated/pilot/train.jsonl")[:3]]
    policy = HFCausalPolicy(str(args.model_dir),
                            adapter_path=str(PROJECT / "runs/mps_smoke/adapter"),
                            load_in_4bit=False, device="mps", max_batch_size=1,
                            completion_logits_only=False)
    reference = policy.scores(observations[0])
    results = []
    for batch_size in (1, 2, 4):
        policy.max_batch_size = batch_size
        policy.completion_logits_only = True
        torch.mps.empty_cache()
        probe = policy.scores(observations[0])
        delta = max(abs(a[1] - b[1]) for a, b in zip(reference, probe))
        same_decision = max(reference, key=lambda x: x[1])[0] == max(probe, key=lambda x: x[1])[0]
        durations = []
        for observation in observations:
            torch.mps.synchronize()
            start = time.perf_counter()
            policy.scores(observation)
            torch.mps.synchronize()
            durations.append(time.perf_counter() - start)
        result = {
            "max_batch_size": batch_size, "observation_count": len(observations),
            "candidates_per_observation": [len(o["candidates"]) for o in observations],
            "seconds_per_observation": durations, "median_seconds": statistics.median(durations),
            "max_score_absolute_difference_from_full_logits": delta,
            "same_selected_decision_as_full_logits": same_decision,
            "allocated_bytes_after": torch.mps.current_allocated_memory(),
            "driver_bytes_after": torch.mps.driver_allocated_memory(),
        }
        results.append(result)
        print(json.dumps(result), flush=True)
    atomic_json(Path(__file__).resolve().parent / "inference_benchmark.json", {
        "device": "mps", "model": "SmolLM2-135M-Instruct plus 4-step smoke adapter",
        "protocol": "One warmup, then three public observations per batch size, fixed model.",
        "results": results,
        "limitations": [
            "Three observations are a speed smoke check, not a latency distribution.",
            "After-call memory counters are not peak memory.",
            "This benchmark does not measure task success or use correctness labels.",
        ],
    })


if __name__ == "__main__":
    main()
