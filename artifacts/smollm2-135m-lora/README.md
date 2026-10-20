---
license: apache-2.0
base_model: HuggingFaceTB/SmolLM2-135M-Instruct
library_name: peft
tags:
- lora
- tool-use
- experimental
---

# ToolShift SmolLM2-135M LoRA adapter

This is the final adapter from an actual local experiment, including a negative
result: it learned to execute candidate actions but did not learn to obtain
information or pause when the tool semantics could not be identified reliably.
It is a research artifact for ToolShift's synthetic sandbox, not a production
tool-calling policy.

## Contents and identity

The package contains the final LoRA adapter and tokenizer. It excludes the
base-model weights, optimizer, scheduler, RNG state and intermediate checkpoints.
The adapter has **2,442,240 trainable parameters**; the weight file is approximately
9.8 MB. The base model is
[HuggingFaceTB/SmolLM2-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/tree/12fd25f77366fa6b3b4b768ec3050bf629380bac),
pinned to revision `12fd25f77366fa6b3b4b768ec3050bf629380bac`.

The base-weight SHA256 is
`5af571cbf074e6d21a03528d2330792e532ca608f24ac70a143f6b369968ab8c`.
The adapter-weight SHA256 is
`7db5f902b6e56b5441aa72ea2e31480ebc9205fd7ea8c939afa0204df3372df2`.
[artifact_manifest.json](artifact_manifest.json) records all asset hashes,
runtime versions, the data hash and training settings.
[SHA256SUMS](SHA256SUMS) also covers this model card.

The adapter and inherited tokenizer assets are distributed under Apache-2.0.
See [LICENSE](LICENSE) and [NOTICE](NOTICE). The
[pinned upstream model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/blob/12fd25f77366fa6b3b4b768ec3050bf629380bac/README.md)
declares the base model's Apache-2.0 license. This directory's license is
explicit and separate from the repository's code license.

## Actual training

Training ran on an Apple M4 Pro with 24 GB unified memory, using **fp32 MPS
LoRA**, not CUDA QLoRA. The CUDA 1.5B configuration in this repository has not
been executed.

| Item | Executed value |
|---|---|
| Eligible training data | 1,053 decision records from 81 source groups |
| Schedule | 200 optimizer steps; 400 original decision exposures, 0.380 epoch |
| Seed | 17; one training seed |
| Batch | 1 original record, gradient accumulation 2 |
| LoRA | rank 8, alpha 16, dropout 0; all linear attention/MLP projections |
| Optimizer | AdamW through Transformers Trainer; learning rate 0.0002 with linear decay |
| Input length | 961–1,162 tokens; maximum allowed 2,048; no truncation |
| Acceptable answers | At most 4 per training record; all retained |
| Loss mask | Final assistant decision and end markers only |
| Objective | Negative log total sequence probability of acceptable decisions |
| Trainer runtime | 554.40 seconds |
| Average training loss | 2.63748; not a held-out score |
| Runtime | Python 3.12.14, PyTorch 2.14.1, Transformers 4.57.6, PEFT 0.21.2, Accelerate 1.15.0 |

Only training labels were used. Source groups are disjoint from development and
test groups. The synthetic data contains migration notices describing possible
tool semantics, public task/history fields, public candidate decisions and
oracle-generated training targets. Hidden actual worlds and oracle outputs do
not appear in inference observations.

The initial formal attempt ran out of MPS memory at step 37. The successful
rerun kept the same seed and learning hyperparameters, materialized LM-head
logits only at answer-prediction positions and released unused MPS cache between
optimizer steps. The full attention context and objective were retained; a
small-model parameter-update equivalence test passed. The memory limit was not
disabled. The failure and repair are preserved in
[the experiment record](../../experiments/lora_main/README.md).

## Held-out closed-loop results

The evaluation sample was fixed before model results were observed: **51
episodes in 13 independent source groups**, covering 18 domain-by-suite strata.
Each selected stratum includes all of its possible actual worlds. The base and
adapter were evaluated on identical episode IDs, with no oracle correction.
Inference ranks the complete public candidate menu by joint answer probability;
it is not unrestricted text generation.

| Metric | Base model | Final adapter | Static query-order policy |
|---|---:|---:|---:|
| Actual-world success | 0/51 (0.00%) | 23/51 (45.10%) | 45/51 (88.24%) |
| Robust success | 0/51 (0.00%) | 8/51 (15.69%) | 45/51 (88.24%) |
| Invariant violations | 0/51 (0.00%) | 15/51 (29.41%) | 0/51 (0.00%) |
| Episodes with an uncertified action | 0/51 (0.00%) | 43/51 (84.31%) | 0/51 (0.00%) |
| Abstentions | 51/51 (100.00%) | 0/51 (0.00%) | 6/51 (11.76%) |
| Query calls | 0 | 0 | 60 |
| Acting in oracle-infeasible cases | 0/6 | 6/6 | 0/6 |
| Candidate completions scored | 714 | 1,224 | Not a language model |
| Input tokens processed across candidates | 796,773 | 1,305,672 | Not applicable |
| Candidate-scoring time | 78.44 s | 165.35 s | Not applicable |

Actual-world success checks the final state for the sampled actual semantics.
Robust success additionally requires every executed action to be safe across
the worlds still compatible with public evidence. Thus 23 actual successes
do not establish reliable handling of uncertainty. The adapter executed two
actions in every episode and never queried, including six cases where reliable
completion was impossible from available observations.

The group-balanced robust-success difference was +0.2137, with a descriptive
group-bootstrap interval [0.0342, 0.4402]. This weights groups equally and is
different from the episode-weighted +0.1569 above. With only 13 groups and one
training seed, the interval does not establish broad generalization.
The adapter is substantially worse than the static query-order policy on the
same 51 episodes: 8 versus 45 robust successes. Static reference records were
projected by exact episode ID from the pre-existing full CPU evaluation, without
new rollouts or changing the model sample. The source hash is recorded in
[static/summary.json](../../experiments/lora_main/static/summary.json).
Full traces, selection protocol, timing and comparisons are under
[experiments/lora_main](../../experiments/lora_main/).

## Use through the tested policy API

Install the repository's optional HF dependencies in an isolated environment.
Download the exact base revision separately; this adapter alone cannot run:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="HuggingFaceTB/SmolLM2-135M-Instruct",
    revision="12fd25f77366fa6b3b4b768ec3050bf629380bac",
    local_dir="models/SmolLM2-135M-Instruct",
)
```

From the repository root, with `toolshift` installed or `PYTHONPATH=src`:

```python
from toolshift.model_policy import HFCausalPolicy

policy = HFCausalPolicy(
    "models/SmolLM2-135M-Instruct",
    adapter_path="artifacts/smollm2-135m-lora",
    local_files_only=True,
    load_in_4bit=False,
    device="mps",              # "cpu" was also checked for the exported artifact
    max_length=2048,
    max_batch_size=1,
)
# public_observation must follow docs/training.md: context, history,
# remaining_budget, and public candidates; never include acceptable labels.
decision = policy.choose(public_observation)
print(decision, policy.usage)
```

The exported weights, tokenizer and chat template match the training artifact.
A real CPU reload and candidate inference passed; the exported weights also
match the final step-200 checkpoint. See
[export_load_check.json](../../experiments/lora_main/export_load_check.json).
Full checkpoint resumption requires local checkpoint state, which this package
intentionally omits.

## Limits and next experiments

This is a small synthetic benchmark with an explicit finite candidate menu and
a supplied semantic-support notice. It does not establish performance on open
APIs, uncontrolled text, new task domains, general chat, or real services.
The 400 training exposures cover less than one epoch. A larger model, longer
training, further seeds and length-calibration comparisons have not been run.
Joint sequence probability may favor short answers; that possible cause of
base-model abstention has not been isolated experimentally.

The observed failure calls for separate tests of uncertainty-sensitive
query/pause learning and same-observation decision calibration. Changing
scoring or sampling after seeing these test results requires a new protocol and
a new held-out set. The adapter should not be used for unattended external
writes or represented as a verified safe agent.
