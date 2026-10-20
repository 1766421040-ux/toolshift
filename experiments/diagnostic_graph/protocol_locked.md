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

## Reproduction and results

Implementation, exact command, artifact hashes and executed results will be
appended after the locked protocol is run. Planned numbers above are not
experiment results.
