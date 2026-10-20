# Claims

## C01 / H1: Action sufficiency can cost less than version identification

- Statement: Within the declared finite support, a common completing action
  can eliminate the need to identify every semantic-version component.
- Status: supported within the controlled benchmark.
- Provenance: ai-suggested; supporting experiments ai-executed.
- Evidence: [CPU results](../../experiments/cpu/),
  [coupled results](../../experiments/diagnostic_graph/summary.json).
- Falsification: independently recomputing the state/decision trees removes
  the claimed cost advantage or exposes an incorrect final/protected state.
- Boundary: exact planning is established technology; these measurements
  do not establish global novelty or a need for machine learning.

## C02 / H2: Immediate invariants alone are insufficient

- Statement: A permitted write must also preserve a common feasible
  completion under the remaining public uncertainty.
- Status: supported by finite counterexamples and independent checks.
- Provenance: ai-suggested; implementation and validation ai-executed.
- Evidence: [oracle semantics](../../docs/oracle_semantics.md),
  [test suite](../../tests/).
- Falsification: a planner-certified action loses every valid completion in
  a remaining world under the declared action model.
- Principle revision: the initial immediate-state safety condition was
  strengthened to include future common completion. This revision is
  validated only in the enumerated finite setting.

## C03 / H3: Learned acquisition beats strong deployable static policies

- Statement: Learning improves evidence cost at matched safe completion.
- Status: weakened; not supported by the delivered experiments.
- Provenance: ai-suggested; negative evidence ai-executed.
- Evidence: three linear seeds average 19.87% robust success on v1;
  [LoRA comparison](../../experiments/lora_main/comparison.json) has
  8/51 robust successes, 23/51 actual successes and zero queries.
  Static achieves 45/51 robust successes on those same tasks.
- Failure kind: condition-matched counterevidence for these runs; not a
  universal impossibility result for all models or training procedures.
- Reopen if: a protocol fixed before evaluation tests independently held-out
  evidence structures, multiple seeds and teacher ablations against cheap
  exact compiled trees with identical public information and matched safety.
- Resource boundary: the actual MPS run is 200 optimizer steps, one seed
  and a 135M model. No NVIDIA/CUDA result is inferred.
