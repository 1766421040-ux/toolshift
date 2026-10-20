"""Run the independent public-model diagnostic graph CPU experiment."""
import argparse
import json
from pathlib import Path

from toolshift.diagnostic_benchmark import run_benchmark


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(Path(__file__).parent))
    parser.add_argument("--random-cases", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args()
    summary = run_benchmark(args.output_dir, args.random_cases, args.seed)
    print(json.dumps(summary, indent=2))
