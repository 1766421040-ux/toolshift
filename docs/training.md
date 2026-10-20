# Training and inference

ToolShift has two separate learners:

* `LinearPolicy` is a CPU feature-hashed linear candidate ranker. It uses only
  Python's standard library. It is **not an LLM**.
* `train_hf` is an optional Transformers/PEFT entry for a real causal language
  model with LoRA on MPS/CPU, or 4-bit QLoRA on CUDA. Code availability does not
  establish that a training run or a hardware profile has succeeded. Actual run
  summaries and evaluation artifacts are the evidence.

The current local machine uses Apple M4 Pro with 24GB unified memory. This is
different from a CUDA GPU with 24GB VRAM. `lora_mps_smoke.json` therefore uses a
135M model, fp32, four optimizer steps and a small adapter. That configuration
tests the end-to-end training path; it is not sufficient evidence of improved
Agent capability. `qlora_1_5b_24gb.json` is a separate unprofiled CUDA design.

The executed formal run is recorded in
[`experiments/lora_main/training_summary.json`](../experiments/lora_main/training_summary.json):
SmolLM2-135M-Instruct, 200 optimizer steps, two original records per update,
rank-8 LoRA, fp32 MPS, 554.40 seconds of Trainer runtime. There were 1,053 eligible
training decisions in 81 source groups, but the fixed schedule consumed 400
decision exposures (0.380 epoch). Training did not process every eligible record.
The exported final adapter is in
[`artifacts/smollm2-135m-lora`](../artifacts/smollm2-135m-lora/README.md);
its model card gives the actual held-out results and limitations.

## Input contract

Each JSONL record contains:

```json
{
  "id": "unique-training-state-id",
  "group_id": "source-task-family",
  "split": "train",
  "observation": {
    "context": {},
    "history": [],
    "remaining_budget": 3,
    "candidates": [
      {"type":"query","name":"inspect_mode","arguments":{}},
      {"type":"stop","status":"pause"}
    ]
  },
  "acceptable_decisions": [
    {"type":"query","name":"inspect_mode","arguments":{}}
  ]
}
```

Candidates are the public tool menu, not an oracle-filtered set. The oracle may
produce the acceptable label set for training. It never appears in observation,
features, prompts, inference, or rollout correction. `messages` may be present
for interchange, but both implemented learners construct inputs from the public
observation. This prevents accidentally trusting a message field containing the
oracle answer.

The training functions reject any row whose split is not `train`, duplicate
IDs, empty labels, labels outside the menu, incompatible labels for identical
public inputs, and explicitly forbidden privileged
containers. This is a defensive schema check, not a proof against deliberately
renamed secrets. The data producer must keep its hidden state out of all public
fields. Call `assert_group_disjoint(train, dev, test)` before any training;
same-prefix sibling worlds must share their source family and split.

## CPU baseline

```python
from toolshift.training import train_linear
from toolshift.model_policy import LinearPolicy

summary = train_linear("data/train.jsonl", "runs/linear",
                       {"seed": 17, "epochs": 12})
policy = LinearPolicy.load("runs/linear/model.json")
decision = policy.choose(public_observation)
```

The score is a linear function of candidate fields and their hashed interactions
with public task/history text. Bookkeeping IDs are excluded. Hashing uses stable
BLAKE2-derived hashes; Python's process-randomized `hash()` is not used. The
version-2 feature map reserves separate budgets for static context, the declared
semantic support and recent observations; long documents cannot erase the
public uncertainty support from the feature vector. Old v1 checkpoints are
rejected rather than interpreted under a different feature map. The
learner receives no handcrafted interpretation of the sandbox's hidden world.
It can still exploit template regularities, so family and wording holdouts matter.

For acceptable set \(A^*\), loss is
\(-\log\sum_{a\in A^*}\operatorname{softmax}(s)_a\). This accepts any optimal next
decision and does not invent a unique teacher order. Sparse AdaGrad minimizes
the loss. If all candidates are acceptable, loss and gradient are zero.

`model.json` stores inference weights. `linear_checkpoint.json` additionally
stores AdaGrad accumulators, Python RNG state, epoch count, dataset fingerprint,
configuration and feature version. Checkpoints occur at epoch boundaries.
`epochs` is the target total:

```python
train_linear("data/train.jsonl", "runs/linear",
             {"seed": 17, "epochs": 20}, resume_from="runs/linear")
```

An unchanged configuration and dataset are required; only target epoch count
may increase. An exact resumed-versus-continuous test is included. Reported
training accuracy is not held-out performance.

## Optional language-model training

Use an isolated virtual environment. The project does not install dependencies
or download a model when imported. The current integration uses Transformers
4.46+ (the executed environment should record its exact version), PEFT,
Accelerate and PyTorch; only CUDA QLoRA needs bitsandbytes. Prefer the repository
environment instructions and capture installed versions with each run.

```python
import json
from toolshift.training import train_hf

config = json.load(open("configs/lora_mps_smoke.json"))
config["model_name_or_path"] = "/path/to/local/model"
summary = train_hf("data/train.jsonl", "runs/mps-smoke", config)
```

`local_files_only` defaults to `true`. Downloading and selecting a model is a
separate explicit operation. Pin the repository revision or retain a manifest
of the local snapshot, and verify the exact model's license before use. The
Qwen 1.5B config points to the official Apache-2.0 model card; it is not evidence
that any 1.5B–3B model will fit or achieve the same behavior.

Prompts are built using the tokenizer's native chat template. Only the final
assistant decision and its end markers carry loss; system/user text and
padding receive `-100`. The implementation checks that full-message tokens
actually begin with the generation-prefix tokens. Unsupported templates fail
explicitly. Examples exceeding `max_length` fail; the code does not silently
truncate away the answer or train on all-`-100` labels.

Every acceptable answer is encoded separately. The custom Trainer sums token
log probabilities for each complete answer, then minimizes the negative log
sum across acceptable answers for each original record. It averages across
records, not across alternatives. The loss is a per-microbatch mean and Trainer
performs gradient accumulation. Different answer lengths can affect sequence
probabilities; this is reported as a limitation, not hidden by a different
inference normalization. Alternative expansion increases memory and runtime;
`max_alternatives` is a hard limit, never silent label deletion.

When the model supports `logits_to_keep`, training materializes the LM-head
output only from the position predicting the first assistant answer onward.
The prompt still participates in attention. A full-logit versus answer-logit
parameter-update comparison is included in the optional integration test.
On MPS, unused allocator cache is released after each optimizer step. This
followed an actual out-of-memory failure at step 37 of the first formal attempt;
the failure is retained in `experiments/lora_main/attempt_1_failure.json`.
The memory limit was not disabled and learning hyperparameters were unchanged.

MPS/CPU paths use ordinary fp32 LoRA and freeze base parameters through PEFT.
CUDA uses NF4, double quantization, an explicit compute dtype, and
`prepare_model_for_kbit_training`. The first profile should record peak memory,
tokens, alternatives per record, optimizer steps and elapsed time. CUDA numbers
cannot stand in for MPS unified-memory measurements.

## Full checkpoint resume

For the optional model, resume from `checkpoint-N`, not the exported adapter:

```python
train_hf("data/train.jsonl", "runs/mps-smoke", config,
         resume_from="runs/mps-smoke/checkpoint-2")
```

The checkpoint must include Trainer state, optimizer, scheduler, RNG and the
ToolShift manifest. The manifest compares dataset, config, chat template,
tokenizer vocabulary, local base-weight hashes and package versions.
Configuration changes, including a changed total schedule,
are rejected. Trainer handles optimizer-step boundaries and restores its RNG,
sampler position and precision state. CUDA fp16 additionally uses Trainer's
scaler checkpoint. An adapter export alone is not full training state.

A resumed job's numerical equivalence must be measured for its actual device;
the exact stdlib resume test is not proof of CUDA/MPS bitwise determinism.
Never label a partially restored optimizer as exact continuation.
The public final adapter contains no optimizer, scheduler or RNG state; the
local full checkpoints under `runs/` are intentionally separate.

## Model policy

```python
from toolshift.model_policy import HFCausalPolicy

policy = HFCausalPolicy("/path/to/local/model",
                        adapter_path="runs/mps-smoke/adapter",
                        load_in_4bit=False, device="mps", max_batch_size=2)
decision = policy.choose(public_observation)
```

This policy scores all public candidate JSON completions by joint log
probability, in bounded batches (default one candidate). For models exposing
`logits_to_keep`, it materializes only answer-prediction positions in the LM
head; the full input still passes through attention. Padding is masked and
score equivalence with the full-logit path is tested. A larger batch must be
profiled on the target hardware. It does not read acceptable labels,
enumerate hidden worlds or call the oracle after a mistake. Scoring a finite
menu is a constrained policy; it must not be described as unrestricted
free-form tool use. Candidate-scoring inference cost belongs in the report.

`policy.usage` counts score calls, candidates, forward passes, nonpadding input
tokens, padded forward tokens, supervised-answer positions and forward-scoring
wall time. Tokenization/model-loading time is excluded from this timer; report
total rollout wall time separately. `policy.reset_usage()` resets counters
without changing model state.

## Verified official API references

Checked 2026-10-20; package versions are logged separately at execution.

* [Transformers 4.57 Trainer](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/trainer)
  documents custom loss, processing classes, gradient accumulation and
  training/checkpoint configuration.
* [Transformers chat templates](https://huggingface.co/docs/transformers/main/chat_templating)
  defines generation prompts and training template use.
* [PEFT quantization](https://huggingface.co/docs/peft/developer_guides/quantization)
  documents k-bit preparation and LoRA configuration.
* [Transformers bitsandbytes](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
  documents NF4, nested quantization and compute dtypes.
* [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  provides the exact model identity and license.
* [SmolLM2-135M-Instruct model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
  identifies the smaller model used by the separate local smoke configuration.

Documentation inspection is not a substitute for running the selected package
versions and hardware path. `training_summary.json` records a completed run;
configuration files alone make no claim of completed training or improvement.
