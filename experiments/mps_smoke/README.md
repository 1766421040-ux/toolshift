# Real 135M MPS smoke experiment

This directory records an executed local LoRA run, separate from the optional
24GB CUDA QLoRA configuration. The base is the downloaded
`HuggingFaceTB/SmolLM2-135M-Instruct`; runtime manifests bind its weight hashes.

The fixed smoke configuration used 8 training records from 2 provenance groups,
fp32, rank 8 adapters on all linear layers, gradient accumulation 2, and 4
optimizer steps. Expanded acceptable answers contained 966–1098 tokens, of
which 18–53 were supervised assistant tokens. No example was truncated.

The run completed on MPS in 17.28 seconds with 2,442,240 trainable parameters.
The loss logs decreased from 36.93 to 25.95. These are training-process
observations; four optimizer steps on eight examples do not establish improved
task-solving ability.

`preflight.json` records data and masking checks. `summary.json` contains actual
timings, package versions, model hashes, memory counters and loss logs.
Adapters and full checkpoints remain under ignored `runs/mps_smoke/`.

The separate `inference_benchmark.json` compares candidate batch sizes 1, 2 and
4 on three public observations containing 14, 10 and 14 candidates. Median
times were 1.666, 1.796 and 1.783 seconds respectively. Larger batches did not
improve this small benchmark; default batch size remains one. Limiting LM-head
output to answer-prediction positions changed scores by at most 0.0000039 and
did not change the selected decision in the checked observation. This checks
numerical behavior and speed, not correctness.

Reproduction from the project root, using an isolated environment:

```bash
python experiments/mps_smoke/run_smoke.py --model-dir /path/to/local/SmolLM2-135M-Instruct
python experiments/mps_smoke/benchmark_inference.py --model-dir /path/to/local/SmolLM2-135M-Instruct
```

In restricted environments the process may not see Metal even though the host
supports MPS. The scripts require an actually available MPS device and refuse
to relabel a CPU execution as an MPS result.
