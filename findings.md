# Findings

## Current understanding

The approved project studies acquiring enough evidence to choose a correct
action after a tool's semantics change. Identifying every version detail is
unnecessary when all remaining possible worlds permit a common action.
Correctness must include future task reachability and protected state.

## Evidence status

The implemented finite system has been exercised on 1,258 held-out v1
episodes and a separate 1,870-episode coupled diagnostic extension.
Task-relevant rules match the v1 oracle and reduce benchmark query cost
60.78% versus full refresh at the same completion rate. Learned linear
policies are substantially worse. See [measured results](docs/results.md).

H1 is supported within the declared finite model: action-sufficient
information can cost less than complete semantic-version identification.
H2 is implemented and checked with irreversible wrong-write counterexamples:
no immediate side effect alone does not establish future completion.
H3 is not established. A small actual LoRA run is evidence that the training
and evaluation machinery works, not evidence that learning is necessary.

## Strong competing explanation

This competing explanation holds in v1. It remains strong in the coupled
extension: exact trees are cheap and small. The resulting contribution is
the auditable implementation, information-consistent set supervision,
reproducible controlled comparison and candid failure analysis.

## Open questions

- Can a learned policy add value on independently held-out evidence graphs,
  given the same information as cheap compiled trees?
- Can training escape premature stopping and evidence-free guessing without
  spending more queries than a static policy?
- How does the result change when candidate supports are incomplete or
  observations noisy, where the present exact certificate no longer applies?
- Do larger models and multiple seeds change the conclusion? CUDA experiments
  and the full planned teacher ablations remain unrun.
