# Heuristics

## H01: Materialize language-model logits only at supervised positions

- Provenance: ai-suggested implementation repair, ai-executed validation.
- Rationale: the first MPS run ran out of memory at step 37. Full attention
  is retained; vocabulary-sized logits are produced only where the
  assistant-target loss requires them. Unused MPS cache is released at
  step boundaries.
- Sensitivity: high for backend memory consumption; equivalence depends on
  the supported causal-model implementation and correct prediction indices.
- Evidence: [first failure](../../../experiments/lora_main/attempt_1_failure.json),
  [validation](../../../experiments/lora_main/validation.json),
  [training summary](../../../experiments/lora_main/training_summary.json).
- Code: [training implementation](../../../src/toolshift/training.py).
- Boundaries: numerical loss/update equivalence was tested. This does not
  justify changing hardware memory limits or extrapolating CUDA throughput.
