"""Command-line entry points; all network/model loading is explicit."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="toolshift")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Build deterministic fixtures and teacher trajectories")
    generate.add_argument("--output", default="data/generated/main")
    generate.add_argument("--seeds", type=int, default=24)
    generate.add_argument("--heldout-domain", choices=["travel", "catalog", "calendar", "none"], default="calendar")
    generate.add_argument("--query-budget", type=int, default=4)
    evaluate = commands.add_parser("evaluate", help="Run actual closed-loop evaluations")
    evaluate.add_argument("--episodes", required=True)
    evaluate.add_argument("--split", choices=["train", "dev", "test", "all"], default="test")
    evaluate.add_argument("--policy", choices=["oracle", "static", "identify", "refresh", "legacy", "schema", "random", "linear", "hf"], required=True)
    evaluate.add_argument("--model")
    evaluate.add_argument("--adapter")
    evaluate.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    evaluate.add_argument("--max-length", type=int, default=2048)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--limit", type=int)
    evaluate.add_argument("--seed", type=int, default=17)
    for kind in ("train-linear", "train-hf"):
        train = commands.add_parser(kind)
        train.add_argument("--data", required=True)
        train.add_argument("--output", required=True)
        train.add_argument("--config")
        train.add_argument("--resume")
    report = commands.add_parser("report", help="Render measured summaries")
    report.add_argument("summaries", nargs="+")
    report.add_argument("--output", required=True)
    replay = commands.add_parser("replay", help="Print one measured trajectory")
    replay.add_argument("--episodes", required=True)
    replay.add_argument("--index", type=int, default=0)
    serve = commands.add_parser("serve", help="Start the local SQLite replay application")
    serve.add_argument("--port", type=int, default=8765)
    diagnostic = commands.add_parser("diagnostic-benchmark", help="Run independent coupled evidence graphs")
    diagnostic.add_argument("--output", default="runs/diagnostic-graph")
    diagnostic.add_argument("--random-cases", type=int, default=64)
    diagnostic.add_argument("--seed", type=int, default=20261020)
    args = parser.parse_args(argv)
    from .io import read_jsonl
    if args.command == "generate":
        from .dataset import build_dataset
        result = build_dataset(args.output, args.seeds,
                               None if args.heldout_domain == "none" else args.heldout_domain,
                               args.query_budget)
    elif args.command == "evaluate":
        from .evaluation import evaluate as run_evaluation
        from .policies import RandomPolicy, StaticPolicy
        episodes = [e for e in read_jsonl(args.episodes) if args.split == "all" or e["split"] == args.split]
        if args.limit:
            episodes = episodes[:args.limit]
        if not episodes:
            parser.error("No episodes matched")
        if args.policy == "oracle":
            policy = None
        elif args.policy == "linear":
            if not args.model:
                parser.error("--model is required")
            from .model_policy import LinearPolicy
            policy = LinearPolicy.load(args.model)
        elif args.policy == "hf":
            if not args.model:
                parser.error("--model is required")
            from .model_policy import HFCausalPolicy
            policy = HFCausalPolicy(args.model, adapter_path=args.adapter, device=args.device,
                                    load_in_4bit=args.device == "cuda", max_length=args.max_length)
        elif args.policy == "random":
            policy = RandomPolicy(args.seed)
        else:
            policy = StaticPolicy(args.policy)
        result = run_evaluation(episodes, policy, args.policy, args.output)["result"]
    elif args.command.startswith("train-"):
        from .training import train_linear, train_hf
        config = json.loads(Path(args.config).read_text()) if args.config else None
        fn = train_linear if args.command == "train-linear" else train_hf
        result = fn(args.data, args.output, config=config, resume_from=args.resume)
    elif args.command == "report":
        from .report import render_report
        result = {"report": str(render_report([json.loads(Path(p).read_text()) for p in args.summaries],
                                               args.output))}
    elif args.command == "replay":
        result = read_jsonl(args.episodes)[args.index]
    elif args.command == "diagnostic-benchmark":
        from .diagnostic_benchmark import run_benchmark
        result = run_benchmark(args.output, random_cases=args.random_cases, seed=args.seed)
    else:
        from .server import serve
        serve(args.port)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))

