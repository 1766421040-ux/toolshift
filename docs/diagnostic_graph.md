# Coupled diagnostic graph benchmark

Protocol `toolshift.diagnostic.v1`, locked before implementation on 2026-10-20.
This is a separate CPU extension. It does not change the original training
dataset, SQLite transition code, or LoRA results. No model is trained here.

## Question and information boundary

Does cost-sensitive, adaptive acquisition of coupled evidence require more than
the original fixed field-inspection rule? When can a common correct complete
plan be selected before identifying the complete semantic version?

Every method receives exactly the same public task, anonymous finite semantic
support, deterministic query partitions and positive query costs. A support
row contains all candidate `WorldSpec` semantics, but no world ID. A row's
index is a public hypothesis reference; the actual selected row remains in
the evaluator and is never passed to a policy or tree compiler. Duplicate
semantic rows are rejected rather than counted as distinct versions.

All worlds use the existing `WorldSpec` and real SQLite `Sandbox`. The public
transition model is known: numeric input is multiplied by the unit factor;
collections append or replace; explicit scope `current` affects one record.
This is a finite model-based diagnosis experiment, not unknown API discovery.
Methods may derive candidate complete plans from this public model. Correctness
is independently checked by executing those plans in SQLite, including
intermediate invariant checks. Public plan computations are not hidden-world
labels supplied by the evaluator.

Queries are deterministic, read-only and independent of database state or
previous writes. The query service exposes only the selected partition answer
and charges its declared cost. It cannot execute the original Sandbox's
`read_state` or any other undeclared query. The two write operators commute
under explicit `scope=current`; both write-once acknowledgements are identical
across worlds. Writes provide no diagnostic information. Therefore queries
can be moved before writes without increasing cost or changing feasibility,
and the diagnostic phase may stop when one complete plan works in every
remaining world. The implementation must validate this assumption by executing
both write orders. This reduction is invalid with state-dependent queries,
informative writes, concurrency, or irreversible actions that change available
evidence.

## Objectives, algorithms and termination

For a nonempty compatible set B, the action-sufficient terminal condition is
that all worlds in B share a complete correct plan. Version identification
instead requires one distinct semantic row. For either objective:

`V(B) = 0` at a terminal set; otherwise
`V(B) = min_q [cost(q) + max_answer V(B ∩ answer_cell(q))]`.

Only queries that strictly split B are considered. If no finite continuation
exists, V is infinity and the policy pauses. Every child then has fewer worlds,
so recursion terminates in at most n−1 query levels, with at most 2^n−1 nonempty
belief sets. Positive finite costs exclude free loops; repeated noninformative
queries cannot improve a solution. This is standard finite decision-tree
dynamic programming, not a new theorem.

Required baselines share the public model and have unrestricted access to all
declared queries:

- `exact_action`: compile the full minimax action-sufficient decision tree.
- `exact_version`: compile the minimax full semantic-identification tree.
- `greedy_action`: cost-normalized reduction of the worst remaining number of
  distinct complete-plan classes; stop once a common plan exists.
- `greedy_version`: cost-normalized reduction of the worst remaining number of
  distinct semantic rows; stop only at a singleton.
- `full_refresh`: pay for the explicitly declared full semantic refresh.

Greedy queries with zero immediate objective gain remain eligible when they
split the support, so a useful first query is not forbidden merely because
its benefit appears later. Ties are deterministic and documented in code.
The exact compiled tree is a strong deployable non-learning control, not
excluded to make learning appear useful. No learning advantage is implied.

## Fixed worked example

Four anonymous worlds:

| Row | Unit factor | Update mode | Default scope |
|---|---:|---|---|
| 0 | 1 | append | current |
| 1 | 60 | append | current |
| 2 | 1 | append | all |
| 3 | 1 | replace | all |

The public `inspect_route` partition `{0,1}/{2,3}` costs 1. Unit and mode
inspection cost 2 each; full refresh costs 4. The default scope can always be
overridden, so rows 0 and 2 admit the same complete plan. A route-first
branching tree costs 3 in the worst case: inspect units on the first branch,
update mode on the other. Directly inspecting both fields costs 4. This is
a truth-table acceptance example, not evidence of learning improvement.

Also execute the cost regimes `(route, units, mode) = (1,1,1)` and `(3,2,2)`.
The first permits a tie with direct inspection; in the latter direct
inspection has worst cost 4, lower than route-first's 5. Full-refresh costs
are explicit. No baseline is forced to choose the route query.

## Experiment design and acceptance

Run the worked example and at least 50 deterministic random cases spanning
support subsets, cost vectors and multiple query partitions. Random generators
must not filter cases based on whether the proposed objective wins. All
distinct support worlds are exhaustively executed for every algorithm; no
sampling only easy actual worlds. Report graph cases, unique public support/
partition structures, semantic worlds, and episodes separately. Random
synthetic structures do not represent real API frequencies.

Report each case's worst realized query cost, average cost over its enumerated
worlds, database success, process-invariant failures, robust complete-plan
certification, query count, remaining semantic rows, compilation time and tree
size. Compare minimax algorithms by worst case, not by pretending they optimize
average cost. Preserve no-gain and worse cases. The full-refresh baseline is
expected to succeed whenever affordable; prices must not be selected after
seeing results. Main comparisons have sufficient budget; dedicated tests
cover budgets too small for robust completion and unidentifiable supports.

Before interpreting results:

1. Cross-world initial public-model equality and query read-only checks pass.
2. Analytically derived plan correctness agrees with actual SQLite execution
   for every candidate plan/world pair, including wrong-plan counterexamples.
3. Both write orders preserve final state and success for valid plans.
4. Tiny independent brute-force enumeration agrees with compiled DP.
5. For each exact tree, worst cost over all actual worlds equals its DP value.
6. Budget failure pauses without any write; no lucky hidden-world execution is
   relabelled as robust completion.

If exact and strong greedy methods tie, report that the extension remains
easy for those rules. If action sufficiency lowers cost relative to full
identification, it supports the objective distinction only. Further learning
requires independent structured train/test splits and trained policies; this
extension has neither and must not be mixed with the original LoRA scoreboard.

## Executed implementation and numerical details

The independent implementation is
[diagnostic_benchmark.py](../src/toolshift/diagnostic_benchmark.py).
Its validated `DiagnosticProblem` provides the public model; compilers take
only that model. `DiagnosticEnvironment` owns the private actual row and the
real `Sandbox`. It records query spending separately from the original
fixed-cost v1 query API, which is not exposed to these policies.

Finite positive integer or decimal query costs are accepted. DP comparisons,
greedy ratios and budget accounting use `Fraction(str(value))`, interpreting
JSON numbers by their decimal spelling. Compiled nodes retain rational
numerator/denominator pairs; decimal `worst_cost` fields are display values.
This avoids treating `0.1 + 0.2` as more expensive than a budget of `0.3`.
The public-model compiler expects a model from the validated constructor, not
an arbitrary untrusted file.

The query-before-write reduction also assumes an adequate two-write horizon,
no cost discount and no write-dependent permissions. All executed writes use
explicit target scope. SQLite audits verify both orders' process invariants,
final records and identical non-diagnostic acknowledgements. State-reading
queries and trial-write probes remain excluded.

## Reproduction and actual results

From the project root:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m unittest discover -s tests -p test_diagnostic_benchmark.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python experiments/diagnostic_graph/run.py --random-cases 64 --seed 20261020
```

Library entry point:
`run_benchmark(output_dir, random_cases=64, seed=20261020) -> summary`.

The executed run contains 67 configurations: 3 fixed cost examples and 64
unfiltered random cases, 374 semantic world instances, and 1,870 episodes over
five algorithms. There are 65 distinct labeled support/partition structures
after excluding costs, task values and endpoint names. This is not a
graph-isomorphism count or evidence of 65 independent problem families.

All five algorithms completed every enumerated world without invariant
violation. This verifies execution on the declared finite models under an
ample budget; the synthetic generator supplies a full-refresh endpoint, so
success alone is an expected sanity check, not the main performance claim.

| Method | Mean of per-case worst query cost | Worst-cost improvement of exact_action: cases lower / equal / higher |
|---|---:|---:|
| exact_action compiled tree | 3.7015 | — |
| exact_version compiled tree | 4.1343 | 17 / 50 / 0 |
| greedy_action | 3.9552 | 15 / 52 / 0 |
| greedy_version | 4.6418 | 32 / 35 / 0 |
| full_refresh | 5.8806 | 37 / 30 / 0 |

The fixed cost examples produce action-sufficient minima of 3, 2 and 4.
For an inspectable counterexample, `random-0022` has worst cost 2 for
action sufficiency versus 7 for version identification; `random-0013` has
worst cost 9 for the exact action tree versus 11 for action-aware greedy.
These are illustrations from the complete retained results, not a filtered
benchmark. The exact full-refresh option is available to every planner.

The exact action trees have at most 15 nodes. Their measured combined compile
time for 67 cases is below one tenth of a second on this run; exact timings
are in the summary and are not stable performance estimates. The strongest
compiled non-learning method is therefore already inexpensive. This extension
demonstrates an objective distinction and some greedy gaps, not that an LLM
is necessary or a learned policy is superior.

`mean_case_worst_cost` weights cases equally. The separately stored
`mean_episode_query_cost` weights each semantic world equally; unequal support
sizes make those aggregate weights different. Minimax trees do not minimize
average costs, and a version-greedy method can have a lower average than a
version-minimax tree without contradicting exactness. No statistical claim
about real API workloads or out-of-distribution learning is made.

Validation executed 15 tests, including SQL-derived independent brute-force
recurrence, zero/tiny/decimal cost boundaries, information isolation,
unidentifiable supports, budget refusal and persistence through a separate
SQLite connection. The full experiment additionally checked 1,396 candidate
plan/world pairs in both write orders: 2,792 order checks, including 1,022
wrong-plan pairs that were required to fail. Successful policy execution uses
the actual core SQL executor rather than replaying stored success labels.

Artifacts are under [experiments/diagnostic_graph](../experiments/diagnostic_graph):

- [protocol_locked.md](../experiments/diagnostic_graph/protocol_locked.md):
  protocol text preserved before execution.
- [manifest.json](../experiments/diagnostic_graph/manifest.json):
  module/core source hashes, protocol hash, configuration and artifact hashes.
- [summary.json](../experiments/diagnostic_graph/summary.json):
  actual aggregate measurements.
- [public_cases.jsonl](../experiments/diagnostic_graph/public_cases.jsonl):
  public semantic supports, query partitions and costs.
- [compiled_trees.jsonl](../experiments/diagnostic_graph/compiled_trees.jsonl):
  inspectable exact and baseline decision trees.
- [case_results.jsonl](../experiments/diagnostic_graph/case_results.jsonl):
  every case/method, including ties.
- [episodes.jsonl](../experiments/diagnostic_graph/episodes.jsonl):
  every query, actual write acknowledgement, final SQL state and verification.
  Actual row indices here are evaluator outputs and must not become future
  policy inputs.
- [sql_audits.jsonl](../experiments/diagnostic_graph/sql_audits.jsonl):
  per-case transition/ordering checks.

No original LoRA data, model scores or holdout splits were changed. This
extension is a completed CPU mechanism benchmark. Its future use for learning
still requires a new structural split and independently trained policies.
