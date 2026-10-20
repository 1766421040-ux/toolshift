# Exact finite-world oracle: semantics and limits

This document describes the implementation in
[`oracle.py`](../src/toolshift/oracle.py), the environment in
[`sandbox.py`](../src/toolshift/sandbox.py), and the trace-level evaluation in
[`evaluation.py`](../src/toolshift/evaluation.py). The dynamic program is a
standard finite minimax / AND–OR planning construction. This project does not
claim a new dynamic-programming theorem.

## What is known, and by whom

A task fixes a public initial database, requested final values, preservation
constraints, a finite action menu, and positive-cost read-only queries. An
episode samples one concrete service world from a declared finite support.
Worlds may differ in numeric input units, collection-update semantics, default
scope, and semantic-metadata availability.

Every deployable policy receives the **same declared support** through the
public `uncertainty_contract`. It does not receive the sampled world, private
world IDs, oracle labels, or evaluator certificates. The public support is
constructed from the entire hypothesis set, not from the sampled world.
Coupled supports retain their coupling; they are not replaced with a larger
Cartesian product without notice.

The oracle is privileged because it can simulate the transition function and
verify goals in every candidate world. It does **not** get the sampled true
world as an argument. Its hypothesis set is narrowed only by replaying the
public query/write observations. This is a model-based reference for the
declared support, not a system that infers arbitrary real API semantics.

Policies get JSON observations, not environment/oracle objects. Python-private
attributes are not an operating-system security boundary. An untrusted plugin
would require process/filesystem isolation in addition to this API separation.

## State and admissible policy class

Let \(h\) denote a candidate service world and \(x\) its current database and
effect-log state. A belief is a finite nonempty set

\[
  B = \{(h,x_h): h \text{ is compatible with the public history}\}.
\]

Keeping \(x_h\), not only a version ID, matters: the same attempted write can
produce different business states. The effect log is also part of the state.
The independent verifier checks preservation constraints at historical
intermediate states, so a past violation cannot be hidden by a later repair.

The other planning state variables are:

- \(b\): remaining integer query-cost budget;
- \(m\): remaining write-attempt horizon;
- \(\mathcal A\): the fixed finite public action menu;
- \(\mathcal Q\): the fixed finite read-only query menu.

The oracle optimizes over deterministic policies with the following limits:

1. Queries and transitions are deterministic for a fixed world and state.
2. The world does not change while the episode runs.
3. Queries are read-only and have integer cost \(c(q)\geq1\).
4. A chosen write must be accepted in every currently compatible world;
   error-producing writes are not used as diagnostic probes.
5. Every realized prefix must satisfy the process invariants.
6. Every possible observation branch must eventually satisfy the task goal
   within the remaining budgets.
7. Actions come from the finite menu. “Exact” does not mean optimal over all
   syntactically possible JSON calls or over an unknown external API.

Writes have zero **query cost**, but consume one unit of \(m\). This does not
mean writes are free in runtime, risk, or storage. Those quantities are not
the objective optimized here. The default oracle write horizon is four; the
current tasks require two accepted commits.

The policy class excludes error probing deliberately. In another environment,
a harmless rejected request might reveal useful information. Such a method is
outside this oracle's guarantee. In the current public menu, successful
trajectories do not need it.

## Independent terminal and invariant predicates

Write \(I(h,x)\) for the independent process-invariant predicate and \(G(h,x)\)
for the independent final-goal predicate.

The current verifier requires:

- the target numeric value equals the requested canonical value;
- the target collection equals the requested collection;
- both requested fields were actually committed;
- all initial collection items remain, without duplicates;
- non-target records, including their lock flags, remain unchanged;
- historical intermediate states also satisfy the preservation predicates.

These predicates inspect the SQLite records and effect log. They do not use
the oracle's selected action, teacher labels, or a policy's completion claim.
Two successful tool acknowledgments are a public episode stopping condition,
not proof of business success.

## Belief transitions

For a query \(q\), let \(O_q(h,x)\) be its public result. For each possible result
\(o\), the successor belief is

\[
  B^{q,o} = \{(h,x)\in B: O_q(h,x)=o\}.
\]

Only nonempty branches are included.

For a write \(a\), let \(F_a(h,x)\) be the successor state and
\(O_a(h,x)\) its public acknowledgment. After simulating that same write in
every candidate world, partition the successor states by their **public**
acknowledgments:

\[
  B^{a,o} =
  \{(h,F_a(h,x)):(h,x)\in B,\ O_a(h,x)=o\}.
\]

This is exactly why the code groups the pairs `(observation, successor_state)`
after a write. It neither assumes that all write observations are identical,
nor gives a policy a private state to select its branch. If any candidate
world rejects the proposed write, that write is excluded from the admissible
next-action set.

## Bellman recurrence

Let \(V(B,b,m)\) be the minimum worst-case additional **query cost** among
admissible policies that complete the task in every world in \(B\).
An infeasible problem has value \(+\infty\).

Boundary cases are:

\[
V(B,b,m)=
\begin{cases}
\infty, & B=\varnothing\ \text{or some }I(h,x)\text{ is false},\\
0, & I(h,x)\land G(h,x)\text{ holds for every }(h,x)\in B,\\
\infty, & m=0\text{ and the common goal has not been reached}.
\end{cases}
\]

The API separately rejects an invalid or already exceeded budget/horizon.
An explicitly supplied remaining query budget can tighten, but cannot enlarge,
the actual remaining environment budget.

For an admissible write,

\[
  C_a(B,b,m)
  = \max_{o:B^{a,o}\ne\varnothing} V(B^{a,o},b,m-1).
\]

For a query with \(c(q)\leq b\),

\[
  C_q(B,b,m)
  = c(q)+
    \max_{o:B^{q,o}\ne\varnothing} V(B^{q,o},b-c(q),m).
\]

At a nonterminal state,

\[
  V(B,b,m) =
  \min\left\{
    \min_{a\in\mathcal A} C_a(B,b,m),
    \min_{\substack{q\in\mathcal Q\\c(q)\leq b}} C_q(B,b,m)
  \right\}.
\]

Inadmissible choices have value \(+\infty\). Empty minima also mean infinity.
The implementation does not include “pause” as an alternative with cost zero.
It returns `pause` with `worst_cost=None` precisely when no completing policy
is available under its assumptions and budgets. `done` has cost zero only
after every compatible world already satisfies the goal and invariants.

The maximum is over possible observation branches, not over independently
optimized world-specific actions that the policy could not distinguish.
There must be **one shared policy at each public information state**.

## Why the recursion terminates

Use the nonnegative integer rank

\[
  \rho(B,b,m)=b+m.
\]

Every simulated write reduces \(m\) by one. Every simulated query reduces \(b\)
by at least one. Thus every recursive edge strictly decreases \(\rho\).
The menus and initial support are finite; bounded write depth also makes the
set of reachable SQLite states finite. Memoization reuses subproblems but is
not needed to establish termination.

A path contains at most
\(m+\lfloor b/\min_q c(q)\rfloor\) nonterminal decisions. The number of belief
states may still grow rapidly with the number of hypotheses, effects, and
available actions. No polynomial-complexity or large-environment scalability
claim follows from this implementation.

## Why the value is optimal within the stated class

The argument is induction on \(\rho=b+m\).

**Base cases.** A shared terminal goal requires no further query. An invariant
violation, absent candidate world, or exhausted write horizon before completion
cannot be repaired into a valid admissible trajectory; its value is infinity.

**Inductive step.** Any completing policy must first choose one admissible
write or affordable query. Its continuation after each public observation is
a policy on the corresponding successor belief. By the inductive hypothesis,
that continuation cannot cost less than the appropriate successor \(V\).
Taking the worst observation gives the stated \(C_a\) or \(C_q\); minimizing
over possible first choices is a lower bound on every completing policy.

Conversely, choose a finite minimizing first choice and, on each observation
branch, a successor policy attaining its already established value. These
branch policies are selected using public observations, so they combine into
one admissible policy. It attains the recurrence value, proving the bound
is tight.

Randomization is not used to obtain a probabilistic success guarantee. The
benchmark requires completion and invariant preservation for every candidate
world. Permitting a nonzero failure probability would define a different
optimization problem.

## Why a no-information query may be removed

If a deterministic query returns the same result in every state in the current
belief, its only successor is the same \(B\). It changes neither the database,
the admissible tools, nor any world belief, and has strictly positive cost.

Any successful policy that asks it can be transformed into one that skips it
and follows the same continuation. That transformed policy has more remaining
budget and lower query cost. Therefore the query cannot be required by an
optimal policy. `_solve` removes it when the partition has one group.

This argument does **not** apply to a query that changes state, refreshes a
credential, waits for a time-dependent event, has history-dependent results,
or unlocks a future tool. Those effects are excluded by the present model.
A query uninformative before a write is not globally disabled: after a write,
the new state is evaluated again and the query may have a different partition.

## Safe, optimal, and teacher-selected sets

The implementation exposes distinct sets:

\[
  \operatorname{Safe}(B,b,m)
  = \{a\in\mathcal A: C_a(B,b,m)<+\infty\}.
\]

`safe_actions` means that taking the action still admits a single completing
continuation strategy on every observable successor branch. It is stronger
than “the next state passes the current invariants” and stronger than “each
world separately has some way to finish.”

- `actions`: writes whose \(C_a\) equals \(V\);
- `queries`: queries whose \(C_q\) equals \(V\);
- `safe_actions`: all feasible shared next writes, including potentially
  higher-cost feasible continuations;
- `all_optimal_decisions()`: the union of both minimum-cost sets;
- `decisions()`: the teacher set after the declared secondary tie-break.

The secondary tie-break prefers an optimal write over an optimal query when
both have the same total worst-case query cost. This encodes “act once the
required evidence is already sufficient” without requiring complete version
identification. It is a convention, not a separate minimax theorem.
All tied writes within the selected action class, or tied queries when the
query class is selected, remain valid labels.

For example, when only the default scope might have changed, an explicit
`scope="current"` can be correct in both worlds. The oracle can act at zero
query cost without discovering which default the service uses.

## Hand-derived values for the current suites

These values assume the initial task state, a write horizon of at least two,
and the finite public menu implemented in `domains.py`.

| Suite | Minimum query cost if feasible | Reason |
|---|---:|---|
| `no_drift` | 0 | Candidate IDs may differ, but their business semantics are identical. |
| `default_scope` | 0 | Explicit current-record scope is correct in both worlds. |
| `units` | 1 | One unit query separates incompatible one-shot numeric encodings. |
| `append_replace` | 1 | One update-mode query separates incompatible collection arguments. |
| `mixed` | 2 | Unit and update-mode uncertainties both affect required writes. Two focused queries cost 2; the full contract costs 3. |
| `unidentifiable` | infeasible | All read-only semantic evidence is identical/unavailable, but correct numeric commits differ. |

If the remaining query budget is below the required cost, the corresponding
task is infeasible under the robust completion criterion. Guessing may still
succeed in one actual world; that is not robust completion.

The lower bounds use the write-once semantics. An incorrect numeric commit
cannot be corrected later even when it has not yet damaged a protected record.
The incompatible collection action either removes an initial item or creates
a duplicate. Writes therefore cannot safely substitute for the required
diagnostic evidence. The mixed support contains all combinations, so learning
one axis does not determine the other.

## Meaning of `robust_success` in a realized rollout

The evaluator records ordinary `success` from the independent SQLite verifier.
It additionally checks, **before each actual write**, whether that write
belongs to the oracle's `safe_actions` for the current public history, declared
support, and remaining budgets.

`robust_success` is true when:

1. the realized episode succeeds; and
2. every realized write passed that check.

The check is evaluator-only. Its answer is not included in policy observations,
does not change the chosen action, and does not veto a write. A blind legacy
policy can therefore get `success=True` by guessing correctly in an
unidentifiable world, while getting `robust_success=False`.

This is a **successful realized trace with action certificates under the
declared finite model**. It does not prove that the policy will behave correctly
on every unexecuted observation branch. A safe action establishes that some
completing continuation exists; it does not prove the evaluated policy will
choose that continuation everywhere. Nor does it guarantee correctness under
omitted worlds, inaccurate transition models, stochastic tools, unexpected
permissions, or external state changes.

The evaluator reports policy, environment, and certificate-computation times
separately. Total harness wall time is not pure deployed-policy latency.

## Validation and reproducibility

Run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -p 'test_oracle_exhaustive.py' -v
```

The validation has two independent layers:

1. Across three domains, all six suites, and query budgets 0–4, check the
   hand-derived value table and replay the selected oracle policy in **every**
   candidate actual world. This covers 90 support/budget configurations and
   270 world rollouts, including infeasible cases.
2. In a tiny one-write-remaining instance, enumerate concrete query trees
   of depth at most two by Cartesian products, with no dynamic-programming
   recurrence or memoization. Query observations and terminal write success
   are collected by actual SQLite executions. Include repeated and
   uninformative queries, and compare the minimum cost with the oracle.

The latter is a bounded independent check, not a proof for arbitrary tasks.
The induction above supplies the conditional mathematical argument; the tests
check that the concrete implementation agrees on small executable instances.
Other core/integration tests check future unreachability, write-observation
compatibility, public-support consistency, cost accounting, and the distinction
between lucky success and certified realized actions.

The executed budget-2 tiny enumeration found:

| Support | Successful concrete trees | Of which start with an uninformative query | Minimum cost |
|---|---:|---:|---:|
| `no_drift` | 42 | 40 | 0 |
| `units` | 112 | 12 | 1 |
| `unidentifiable` | 0 | 0 | infeasible |

These are counts of syntactically distinct bounded policies, not independent
experimental samples. The redundant successful trees demonstrate directly why
retaining no-information queries cannot improve the optimum in this model.
