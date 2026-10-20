# Coupled diagnostic queries: executed CPU experiment

This is an independent extension of ToolShift. It uses the original real
SQLite executor but a separate public query-partition model. It trains no
model and must not be mixed with the original LoRA evaluation.

The locked protocol and full interpretation are in
[docs/diagnostic_graph.md](../../docs/diagnostic_graph.md).
The library entry point is `toolshift.diagnostic_benchmark.run_benchmark`.

Run from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -p test_diagnostic_benchmark.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/diagnostic_graph/run.py --random-cases 64 --seed 20261020
```

The saved run has 67 cases, 374 worlds and 1,870 policy episodes. Five methods
all complete successfully. Exact action-sufficient trees have lower worst
query cost than complete version-identification trees in 17 cases, and lower
cost than action-aware greedy in 15 cases. The other cases are retained ties.
Full evidence and hashes are in `summary.json` and `manifest.json`.

The exact action trees contain at most 15 nodes and are cheap to compile.
The result supports a distinction between diagnosis objectives and a gap for
some greedy decisions. It does not establish a need for learning or a new DP
theory. Public supports and synthetic deterministic partitions are substantive
assumptions; these are not measurements of real API reliability.

`episodes.jsonl` contains evaluator-only actual-row indices and database
verification. They are audit outputs, never policy features. All policy
compilers receive only `public_cases.jsonl`'s public model.
