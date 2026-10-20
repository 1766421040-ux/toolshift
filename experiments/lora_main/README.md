# Actual 135M MPS LoRA experiment

This directory records a completed 200-step local training run and a closed-loop
comparison on a sample registered before model evaluation. The primary finding
is mixed: the adapter executes actions more often, but it never queries and
violates invariants in 15 of 51 test episodes. This does not validate the
uncertainty-sensitive policy hypothesis.

## Protocol and data

[protocol_config.json](protocol_config.json) fixed the learning schedule before
formal training: SmolLM2-135M-Instruct, fp32 MPS LoRA, rank 8 / alpha 16 /
dropout 0, all linear projections, seed 17, microbatch 1, accumulation 2,
learning rate 0.0002, 200 optimizer steps, maximum length 2,048.

The training JSONL hash is
`f43f99ef5d6f42962bdb00a88c89263087eda571590032450f0d7a2729924f2e`.
There are 1,053 eligible records in 81 source groups. The fixed schedule consumes
400 original-record exposures, or 0.3799 epoch. No development/test labels are
used, and source groups are disjoint. All acceptable answer alternatives are
retained; only assistant answer tokens and their end markers carry loss.
[preflight.json](preflight.json) records token lengths, hashes and checks.

[evaluation_protocol.json](evaluation_protocol.json) freezes 51 test episode IDs,
13 independent source groups and 18 domain-by-suite strata. The initial
lexicographic sample would have covered only three groups. Before any model
evaluation, it was revised to select the minimum salted SHA256 group per stratum
and retain all selected actual worlds. This improves independent-group coverage
without choosing cases using observed model performance.

## Execution and repair

The host was Apple M4 Pro with 24 GB unified memory. An initial attempt failed
with MPS out-of-memory after step 37, before the first checkpoint; see
[attempt_1_failure.json](attempt_1_failure.json) and the preserved first
[preflight](attempt_1_preflight.json).

The successful rerun retains all learning hyperparameters and the seed. It
computes LM-head outputs only for answer-prediction positions and releases
unused MPS cache after updates. Prompt attention remains intact. A numerical
parameter-update comparison against the full-logit objective passed before
rerunning. The MPS memory limit was not disabled.

[training_summary.json](training_summary.json) confirms 200 completed updates,
2,442,240 trainable parameters, 554.40 seconds of Trainer runtime and average
training loss 2.63748. Per-step MPS allocations in this summary are measurements
after each step, **not peak memory**. The actual environment was Python 3.12.14,
PyTorch 2.14.1, Transformers 4.57.6, PEFT 0.21.2 and Accelerate 1.15.0. No CUDA
QLoRA run is reported.
The 20 interval log entries, including loss, gradient norm and learning rate,
are preserved in [training_log.json](training_log.json). These are training
diagnostics, not evaluation scores.

## Held-out results

| Metric | Base | Adapter | Static query order |
|---|---:|---:|---:|
| Actual-world success | 0/51 | 23/51 | 45/51 |
| Robust success | 0/51 | 8/51 | 45/51 |
| Invariant violations | 0/51 | 15/51 | 0/51 |
| Episodes with an uncertified action | 0/51 | 43/51 | 0/51 |
| Abstention | 51/51 | 0/51 | 6/51 |
| Query calls | 0 | 0 | 60 |
| Acting in oracle-infeasible episodes | 0/6 | 6/6 | 0/6 |
| Forward passes | 714 | 1,224 | Not a language model |
| Nonpadding input tokens | 796,773 | 1,305,672 | Not applicable |
| Candidate-scoring time | 78.44 seconds | 165.35 seconds | Not applicable |

See [base summary](base/summary.json), [adapter summary](adapter/summary.json),
their complete `episodes.jsonl` traces and [comparison.json](comparison.json).
The [static reference](static/summary.json) comes from exact episode-ID
projection of the existing 1,258-episode CPU test evaluation; there was no
additional rollout. Source and subset hashes are retained. This same-sample
comparison is preferable to comparing the 51-episode model rates against a
different full-test denominator.
The oracle only audits actions and supplies the separately labeled control; it
does not replace or correct model decisions. The model has access to public
semantic support and a public finite menu. Joint-probability menu ranking is a
constrained policy, not unrestricted tool generation.

The adapter makes two writes in each episode. Some happen to achieve the goal
in the actual world, but most are not justified under the remaining possible
semantics. Robust success therefore differs from actual-world success.
The zero-query result and 6/6 unsafe attempts on oracle-infeasible tasks are
central failures, not omitted cases.

The paired bootstrap resamples source groups, so its group-balanced robust
delta (+0.2137; descriptive 95% interval [0.0342, 0.4402]) differs from the
episode-weighted delta (+0.1569). Thirteen independent groups and one seed are
insufficient for a general capability claim.
Against static query order, the group-balanced robust-success delta is -0.5940
(descriptive interval [-0.8205, -0.3462]); the episode-weighted difference is
-0.7255. The static policy's explicit querying and pause behavior outperforms
this small adapter.

## Reproduction and distribution

Run from the repository root after generating the exact main dataset and
downloading the pinned model:

```sh
python experiments/lora_main/run_training.py --model-dir models/SmolLM2-135M-Instruct
python experiments/lora_main/run_evaluation.py --model-dir models/SmolLM2-135M-Instruct
python experiments/lora_main/add_static_reference.py
```

The scripts validate hashes and source-group isolation. The training script
binds local model hashes, source hashes and package versions in the manifest.
An unmodified full checkpoint supports continuation with the same configuration;
an exported adapter is insufficient to resume optimizer state. Tiny-model
CPU resume equivalence is tested; MPS bitwise continuation was not established.

The final portable artifact is
[artifacts/smollm2-135m-lora](../../artifacts/smollm2-135m-lora/README.md), with a
model card, Apache-2.0 license, base revision, hashes, tokenizer and actual API
example. Base weights and optimizer states remain excluded.
[export_load_check.json](export_load_check.json) confirms a real CPU reload,
inference and final-checkpoint weight equality.
