# Research log

## 2026-10-20 — user-selected execution route

The user selected proposal P2 and authorized implementation, local experiments,
and eventual public publication under `1766421040-ux`. P1 and P3 are not active.
The design target is one 24 GB GPU. The available host is an Apple M4 Pro with
24 GB unified memory; this is not a CUDA GPU.

Before experiments: H1 concerns action sufficiency versus full version
identification; H2 concerns set-valued supervision; H3 concerns improvement
over deployable static policies. A successful synthetic oracle is evidence
for the finite task construction, not evidence for learned LLM generalization.

Reference audit identified that AppWorld bundles have redistribution conditions
distinct from its public infrastructure. ToolShift will use independently
generated synthetic tasks and SQLite fixtures. No third-party task database
or private evaluation labels are being redistributed.

## Information and data audit

An initial teacher design could narrow hypotheses by a private suite, making
identical public histories receive incompatible labels. The final protocol
publishes the complete anonymous allowed support equally to all policies.
Teacher records are deduplicated by observation, with set-consistency checks.
Known finite support is now an explicit deployment assumption.

Initial small periodic task variations were replaced before the frozen
experiments by domain-specific deterministic random streams, content hashing
and group splits. Mixed cases are generated only from already-held-out source
groups. The final manifest has 180 source groups; changed prefixes are not
called held-out template families.

## CPU experiment and outer reflection

All 1,258 test episodes / 74 source groups ran under the frozen protocol.
Static task-relevant inspection equals the finite oracle: 1,110 feasible
successes and 148 justified pauses. At equal success, mean query cost is
60.78% below full refresh. The three learned linear seeds have only 19.87%
mean robust success. This rejects an incremental learning claim on v1.

The strong competing explanation was retained, not weakened. A separate
coupled evidence-graph protocol was then locked and implemented: 67 cases,
374 worlds, 1,870 real SQL episodes. Action sufficiency lowers worst cost
relative to exact version identification in 17 cases, and relative to
action-aware greedy in 15. An exact compiled tree remains cheap (at most
15 nodes), so this extension still does not establish a need for learning.
It does not alter the original dataset or model experiment.

## Actual MPS training and failure recovery

After a four-step real model smoke run, a 200-step SmolLM2-135M LoRA protocol
and held-out IDs were registered. Selection was revised before any model
test results to improve group coverage: 51 episodes / 13 groups, stratified
by domain and suite. Training uses one seed, rank 8 and assistant-only set
supervision, with no truncation.

The first attempt failed with MPS OOM at step 37. Only assistant prediction
positions now materialize full-vocabulary logits; numerical loss and adapter
update equivalence were checked. Cache release at step boundaries was added.
The same seeded protocol restarted from step zero and completed 200 steps
in 554.40 seconds. Failure and corrective evidence are preserved.

The locked model rollout completed: base paused on all 51 episodes. The adapter
achieved 23 actual successes but only 8 audited robust successes, violated
protected state in 15 episodes, and never queried. All six impossible cases
still received writes. This is a negative result for evidence acquisition,
even though execution frequency and actual-world success increased.
Scoring was not changed after observing test results. The exported adapter
was reloaded and used for real CPU inference; its weights/tokenizer match the
training artifact by hash.

## Delivery validation

84 standard-library tests ran: 83 passed, one optional HF test skipped in the
default environment. Independent tiny policy-tree enumeration and 270
world/parameter rollouts match finite-oracle calculations. Browser testing
verified both successful adapted writes and failed legacy writes. The public
repository includes actual metrics and limitations, not only success demos.

