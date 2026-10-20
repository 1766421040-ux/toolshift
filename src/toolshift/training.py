"""Set-valued supervision, a stdlib CPU learner, and optional QLoRA training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from array import array
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from .model_policy import (
    FEATURE_VERSION, LinearPolicy, canonical_decision, candidate_features,
    candidate_list, decision_messages, encode_assistant_completion, softmax,
)


def atomic_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                   indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_records(source: str | Path | Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(source, (str, Path)):
        rows = []
        with Path(source).open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if line.strip():
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid JSONL at line {line_no}") from exc
        return rows
    return [dict(row) for row in source]


def validate_training_records(records: Sequence[Mapping[str, Any]]) -> None:
    """Reject mixed splits; the caller must deliberately supply training only."""
    if not records:
        raise ValueError("No training records")
    ids: set[str] = set()
    observation_targets: dict[str, set[str]] = {}
    for row in records:
        if row.get("split") != "train":
            raise ValueError(f"Training received a non-train row: {row.get('id')!r}")
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("Every record requires a nonempty string id")
        if row["id"] in ids:
            raise ValueError(f"Duplicate record id: {row['id']}")
        ids.add(row["id"])
        if not isinstance(row.get("group_id"), str) or not row["group_id"]:
            raise ValueError("Every record requires a provenance group_id")
        observation = row.get("observation")
        if not isinstance(observation, dict):
            raise ValueError("Every record requires a public observation object")
        menu = {canonical_decision(d) for d in candidate_list(observation)}
        labels = row.get("acceptable_decisions")
        if not isinstance(labels, list) or not labels:
            raise ValueError("Every training row needs a nonempty acceptable decision set")
        positive = {canonical_decision(d) for d in labels}
        if not positive.issubset(menu):
            raise ValueError(f"Oracle labels are not in public candidate menu: {row['id']}")
        visible = json.dumps(observation, sort_keys=True, ensure_ascii=False,
                             separators=(",", ":"), allow_nan=False)
        if visible in observation_targets:
            observation_targets[visible].intersection_update(positive)
            if not observation_targets[visible]:
                raise ValueError("Identical public observations have incompatible labels; "
                                 "check hidden suite/world information")
        else:
            observation_targets[visible] = set(positive)


def assert_group_disjoint(*collections: Sequence[Mapping[str, Any]]) -> None:
    """Validate groups across separate collections (train/dev/test), not within."""
    seen: set[str] = set()
    for records in collections:
        current = {str(row["group_id"]) for row in records}
        overlap = seen.intersection(current)
        if overlap:
            raise ValueError(f"Provenance groups leak across datasets: {sorted(overlap)[:5]}")
        seen.update(current)


def _fingerprint(records: Sequence[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in records:
        relevant = {key: row[key] for key in
                    ("id", "group_id", "split", "observation", "acceptable_decisions")}
        h.update(json.dumps(relevant, sort_keys=True, ensure_ascii=False,
                            separators=(",", ":"), allow_nan=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def set_softmax_loss_gradient(scores: Sequence[float],
                              acceptable_indices: Sequence[int]) -> tuple[float, list[float]]:
    """-log sum_{a in A*} p(a); gradient preserves multiple valid alternatives."""
    positive = sorted(set(acceptable_indices))
    if not positive or min(positive) < 0 or max(positive) >= len(scores):
        raise ValueError("A nonempty in-range acceptable set is required")
    probs = softmax(scores)
    conditional = softmax([scores[i] for i in positive])
    gradient = list(probs)
    for index, probability in zip(positive, conditional):
        gradient[index] -= probability
    peak = max(scores)
    good_peak = max(scores[i] for i in positive)
    log_all = peak + math.log(sum(math.exp(x - peak) for x in scores))
    log_good = good_peak + math.log(sum(math.exp(scores[i] - good_peak) for i in positive))
    return max(0.0, log_all - log_good), gradient


@dataclass(frozen=True)
class LinearConfig:
    dimensions: int = 32768
    learning_rate: float = 0.25
    epochs: int = 12
    seed: int = 17
    epsilon: float = 1e-8

    def validate(self) -> None:
        if self.dimensions < 32 or self.epochs < 1 or self.learning_rate <= 0 or self.epsilon <= 0:
            raise ValueError("Invalid linear training configuration")


def _tuple_tree(value: Any) -> Any:
    return tuple(_tuple_tree(x) for x in value) if isinstance(value, list) else value


def train_linear(records: str | Path | Iterable[Mapping[str, Any]],
                 output_dir: str | Path, config: LinearConfig | Mapping[str, Any] | None = None,
                 resume_from: str | Path | None = None) -> dict[str, Any]:
    """Train sparse AdaGrad with complete, deterministic epoch-boundary resume.

    ``epochs`` is the target total epoch count, including a resumed run.
    Held-out data is not accepted by this API and cannot affect early stopping.
    """
    config = config if isinstance(config, LinearConfig) else LinearConfig(**dict(config or {}))
    config.validate()
    rows = load_records(records)
    validate_training_records(rows)
    fingerprint = _fingerprint(rows)
    output = Path(output_dir)
    rng = random.Random(config.seed)
    weights = [0.0] * config.dimensions
    accumulator = [0.0] * config.dimensions
    completed = 0
    updates = 0
    history: list[dict[str, Any]] = []
    if resume_from is not None:
        source = Path(resume_from)
        if source.is_dir():
            source = source / "linear_checkpoint.json"
        saved = json.loads(source.read_text(encoding="utf-8"))
        if saved.get("dataset_sha256") != fingerprint or saved.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Resume dataset or feature fingerprint mismatch")
        previous = dict(saved["config"])
        current = asdict(config)
        previous.pop("epochs")
        current.pop("epochs")
        if previous != current:
            raise ValueError("Resume changes optimizer, seed or feature configuration")
        weights, accumulator = saved["weights"], saved["accumulator"]
        if len(weights) != config.dimensions or len(accumulator) != config.dimensions:
            raise ValueError("Invalid checkpoint dimensions")
        completed, updates = saved["completed_epochs"], saved["updates"]
        if completed > config.epochs:
            raise ValueError("Target epochs precede checkpoint")
        history = saved["history"]
        rng.setstate(_tuple_tree(saved["rng_state"]))
    # Cache features to avoid hashing on every epoch. Only train records enter.
    examples = []
    for row in rows:
        menu = candidate_list(row["observation"])
        good = {canonical_decision(d) for d in row["acceptable_decisions"]}
        examples.append((
            [(array("I", vector.keys()), array("d", vector.values()))
             for d in menu
             for vector in [candidate_features(row["observation"], d, config.dimensions)]],
            [i for i, d in enumerate(menu) if canonical_decision(d) in good],
        ))
    started = time.perf_counter()
    for epoch in range(completed, config.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)
        total_loss = 0.0
        hits = 0
        for index in order:
            features, good = examples[index]
            scores = [sum(weights[i] * x for i, x in zip(indices, values))
                      for indices, values in features]
            loss, coefficients = set_softmax_loss_gradient(scores, good)
            total_loss += loss
            hits += int(max(range(len(scores)), key=scores.__getitem__) in good)
            gradient: dict[int, float] = {}
            for coefficient, (indices, values) in zip(coefficients, features):
                for i, value in zip(indices, values):
                    gradient[i] = gradient.get(i, 0.0) + coefficient * value
            for i, g in gradient.items():
                accumulator[i] += g * g
                weights[i] -= config.learning_rate * g / (math.sqrt(accumulator[i]) + config.epsilon)
            updates += 1
        history.append({"epoch": epoch + 1, "train_set_loss": total_loss / len(rows),
                        "train_online_set_accuracy": hits / len(rows)})
        checkpoint = {
            "format_version": 1, "model_type": "hashed_linear_candidate_ranker",
            "feature_version": FEATURE_VERSION, "config": asdict(config),
            "dimensions": config.dimensions, "weights": weights,
            "accumulator": accumulator, "completed_epochs": epoch + 1, "updates": updates,
            "rng_state": rng.getstate(), "dataset_sha256": fingerprint, "history": history,
        }
        atomic_json(output / "linear_checkpoint.json", checkpoint)
    model_payload = {
        "format_version": 1, "model_type": "hashed_linear_candidate_ranker",
        "feature_version": FEATURE_VERSION, "dimensions": config.dimensions,
        "weights": weights, "dataset_sha256": fingerprint, "seed": config.seed,
    }
    atomic_json(output / "model.json", model_payload)
    atomic_json(output / "linear_model.json", model_payload)
    policy = LinearPolicy(weights, dimensions=config.dimensions)
    final_hits = sum(canonical_decision(policy.choose(row["observation"])) in
                     {canonical_decision(d) for d in row["acceptable_decisions"]} for row in rows)
    summary = {
        "status": "trained_cpu_linear", "is_language_model": False,
        "train_rows": len(rows), "train_groups": len({row["group_id"] for row in rows}),
        "dataset_sha256": fingerprint, "epochs": config.epochs, "updates": updates,
        "resumed_from_epoch": completed, "train_set_accuracy": final_hits / len(rows),
        "training_seconds_this_invocation": time.perf_counter() - started,
        "history": history, "config": asdict(config), "python": sys.version,
        "platform": platform.platform(), "model_path": str(output / "model.json"),
        "evaluation": "Training accuracy only; held-out rollout evaluation is separate.",
    }
    atomic_json(output / "training_summary.json", summary)
    return summary


@dataclass(frozen=True)
class HFTrainingConfig:
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    revision: str | None = None
    local_files_only: bool = True
    backend: str = "cuda_qlora"
    max_length: int = 2048
    max_alternatives: int = 8
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    max_steps: int = -1
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    seed: int = 17
    save_steps: int = 50
    logging_steps: int = 10
    precision: str = "bf16"
    completion_logits_only: bool = True
    mps_empty_cache_steps: int = 1


def _package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _base_model_identity(model_path: str, revision: str | None) -> dict[str, Any]:
    """Bind local weights, not just a mutable directory name, on resume."""
    root = Path(model_path)
    if not root.is_dir():
        return {"repository": model_path, "revision": revision}
    hashes = {}
    for path in sorted(root.iterdir()):
        if path.is_file() and (path.suffix in {".safetensors", ".bin"} or
                              path.name in {"config.json", "model.safetensors.index.json",
                                            "pytorch_model.bin.index.json"}):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            hashes[path.name] = digest.hexdigest()
    if "config.json" not in hashes or not any(
            key.endswith((".safetensors", ".bin")) for key in hashes):
        raise ValueError("Local base model must include config and weight files")
    return {"local_files_sha256": hashes}


def train_hf(records: str | Path | Iterable[Mapping[str, Any]],
                output_dir: str | Path, config: HFTrainingConfig | Mapping[str, Any] | None = None,
                resume_from: str | Path | None = None) -> dict[str, Any]:
    """Optional LoRA/QLoRA entry. Imports and downloads are never implicit.

    Multiple acceptable completions form one example. The objective maximizes
    their total sequence probability; no arbitrary single oracle answer is used.
    """
    config = config if isinstance(config, HFTrainingConfig) else HFTrainingConfig(**dict(config or {}))
    rows = load_records(records)
    validate_training_records(rows)
    if config.precision not in {"bf16", "fp16", "fp32"}:
        raise ValueError("precision must be bf16, fp16 or fp32")
    if config.backend not in {"cuda_qlora", "mps_lora", "cpu_lora"}:
        raise ValueError("backend must be cuda_qlora, mps_lora or cpu_lora")
    if config.backend in {"mps_lora", "cpu_lora"} and config.precision != "fp32":
        raise ValueError("The tested CPU/MPS path uses fp32; mixed precision is not silently enabled")
    if min(config.max_length, config.max_alternatives, config.micro_batch_size,
           config.gradient_accumulation_steps, config.lora_r, config.save_steps) < 1:
        raise ValueError("Invalid positive training configuration")
    output = Path(output_dir)
    fingerprint = _fingerprint(rows)
    packages = _package_versions()
    try:
        import torch
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
            TrainingArguments, TrainerCallback, set_seed,
        )
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("QLoRA requires optional HF dependencies; see docs/training.md") from exc
    # 4.46 introduced the Trainer APIs used here. Older/unknown versions fail
    # before model loading rather than silently changing loss scaling.
    version = tuple(int(p) for p in re_version(packages["transformers"])[:2])
    if version < (4, 46):
        raise RuntimeError("This optional entry requires Transformers >=4.46")
    if config.backend == "cuda_qlora" and not torch.cuda.is_available():
        raise RuntimeError("The QLoRA entry requires CUDA; the stdlib linear baseline runs on CPU")
    if config.backend == "mps_lora" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS backend requested but unavailable")
    if config.backend == "cuda_qlora" and config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is unsupported; select fp16 explicitly and re-profile")
    set_seed(config.seed)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.precision]
    kwargs = dict(revision=config.revision, local_files_only=config.local_files_only,
                  trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    encoded = []
    for row in rows:
        answers = sorted({canonical_decision(d) for d in row["acceptable_decisions"]})
        if len(answers) > config.max_alternatives:
            raise ValueError(f"{row['id']}: too many alternatives; do not silently drop valid labels")
        messages = decision_messages(row["observation"])
        encoded.append({"alternatives": [encode_assistant_completion(
            tokenizer, messages, answer, config.max_length) for answer in answers]})
    manifest = {
        "format_version": 1, "dataset_sha256": fingerprint, "config": asdict(config),
        "packages": packages, "train_rows": len(rows),
        "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest(),
        "tokenizer_vocabulary_sha256": hashlib.sha256(json.dumps(
            tokenizer.get_vocab(), sort_keys=True).encode()).hexdigest(),
        "base_model_identity": _base_model_identity(config.model_name_or_path, config.revision),
    }
    if resume_from:
        previous_path = Path(resume_from) / "toolshift_manifest.json"
        if not previous_path.is_file():
            raise ValueError("Resume requires a full Trainer checkpoint with ToolShift manifest")
        previous = json.loads(previous_path.read_text(encoding="utf-8"))
        if previous != manifest:
            raise ValueError("Resume manifest differs: data, configuration, package versions or template changed")
        for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"):
            if not (Path(resume_from) / name).is_file():
                raise ValueError(f"Incomplete resume checkpoint: missing {name}")
        if config.precision == "fp16" and not (Path(resume_from) / "scaler.pt").is_file():
            raise ValueError("Incomplete fp16 resume checkpoint: missing scaler.pt")
    atomic_json(output / "toolshift_manifest.json", manifest)
    model_kwargs: dict[str, Any] = dict(kwargs)
    # torch_dtype is supported on Transformers 4.x (dtype replaces it in 5.x).
    model_kwargs["dtype" if version >= (5, 0) else "torch_dtype"] = dtype
    if config.backend == "cuda_qlora":
        model_kwargs.update(
            device_map={"": torch.cuda.current_device()},
            quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype))
    model = AutoModelForCausalLM.from_pretrained(config.model_name_or_path, **model_kwargs)
    model.config.use_cache = False
    if config.backend == "cuda_qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM", r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout, target_modules=config.target_modules,
        bias="none"))
    base_forward = model.get_base_model().forward
    keep_answer_logits = (config.completion_logits_only and
                          "logits_to_keep" in inspect.signature(base_forward).parameters)

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        flat = [(group, alternative) for group, item in enumerate(batch)
                for alternative in item["alternatives"]]
        length = max(len(item["input_ids"]) for _, item in flat)
        ids, masks, labels, groups = [], [], [], []
        for group, item in flat:
            padding = length - len(item["input_ids"])
            ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding)
            masks.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
            groups.append(group)
        first_label = min(next(i for i, label in enumerate(item["labels"]) if label != -100)
                          for _, item in flat)
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(masks),
                "labels": torch.tensor(labels), "alternative_groups": torch.tensor(groups),
                "answer_logit_start": max(0, first_label - 1)}

    class SetValuedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            groups = inputs.pop("alternative_groups")
            labels = inputs.pop("labels")
            start = inputs.pop("answer_logit_start") if keep_answer_logits else 0
            inputs.pop("answer_logit_start", None)
            extra = {"logits_to_keep": labels.shape[1] - start} if keep_answer_logits else {}
            outputs = model(**inputs, **extra)
            logits = outputs.logits[:, :-1, :]
            target = labels[:, start + 1:]
            token_losses = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(),
                target.reshape(-1), ignore_index=-100, reduction="none").reshape(target.shape)
            sequence_log_probs = -token_losses.sum(dim=1)
            loss = torch.stack([
                -torch.logsumexp(sequence_log_probs[groups == group], dim=0)
                for group in torch.unique(groups)
            ]).mean()
            return (loss, outputs) if return_outputs else loss

    class ManifestCallback(TrainerCallback):
        def on_save(self, args, state, control, **unused):
            atomic_json(Path(args.output_dir) / f"checkpoint-{state.global_step}" /
                        "toolshift_manifest.json", manifest)
            return control

    step_measurements: list[dict[str, Any]] = []

    class TimingCallback(TrainerCallback):
        def on_step_begin(self, args, state, control, **unused):
            self.started = time.perf_counter()

        def on_step_end(self, args, state, control, **unused):
            if config.backend == "mps_lora":
                torch.mps.synchronize()
            elif config.backend == "cuda_qlora":
                torch.cuda.synchronize()
            entry = {"step": state.global_step,
                     "seconds": time.perf_counter() - self.started}
            if config.backend == "mps_lora":
                entry["mps_allocated_bytes_after_step"] = torch.mps.current_allocated_memory()
                entry["mps_driver_bytes_after_step"] = torch.mps.driver_allocated_memory()
                if (config.mps_empty_cache_steps > 0 and
                        state.global_step % config.mps_empty_cache_steps == 0):
                    torch.mps.empty_cache()
            step_measurements.append(entry)
            return control

    args = TrainingArguments(
        output_dir=str(output), per_device_train_batch_size=config.micro_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate, num_train_epochs=config.num_train_epochs,
        max_steps=config.max_steps, use_cpu=config.backend == "cpu_lora",
        bf16=config.precision == "bf16", fp16=config.precision == "fp16",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch", save_strategy="steps", save_steps=config.save_steps,
        save_total_limit=2, save_only_model=False, logging_steps=config.logging_steps,
        seed=config.seed, data_seed=config.seed, dataloader_num_workers=0,
        remove_unused_columns=False, report_to="none", push_to_hub=False,
    )
    trainer = SetValuedTrainer(
        model=model, args=args, train_dataset=encoded, data_collator=collate,
        processing_class=tokenizer, callbacks=[ManifestCallback(), TimingCallback()])
    # Our loss is a per-group microbatch mean, not an already accumulated mean.
    trainer.loss_is_scaled_for_ga = False
    trainer.model_accepts_loss_kwargs = False
    if config.backend == "cuda_qlora":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=str(resume_from) if resume_from else None)
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(str(output / "adapter"))
    atomic_json(output / "adapter" / "toolshift_manifest.json", manifest)
    trainer.save_state()
    summary = {
        "status": "trained_qlora" if config.backend == "cuda_qlora" else "trained_lora",
        "backend": config.backend, "model_type": "causal_lm_lora",
        "objective": "negative_log_total_probability_of_acceptable_completions",
        "training_seconds": time.perf_counter() - start,
        "peak_cuda_allocated_bytes": (torch.cuda.max_memory_allocated()
                                      if config.backend == "cuda_qlora" else None),
        "peak_cuda_reserved_bytes": (torch.cuda.max_memory_reserved()
                                     if config.backend == "cuda_qlora" else None),
        "metrics": result.metrics, "manifest": manifest,
        "step_measurements": step_measurements,
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_model_parameters": sum(p.numel() for p in model.parameters()),
        "completion_logits_only_applied": keep_answer_logits,
        "evaluation": "No held-out labels were used; rollout evaluation is separate.",
    }
    atomic_json(output / "training_summary.json", summary)
    return summary


def train_qlora(records: str | Path | Iterable[Mapping[str, Any]],
                output_dir: str | Path, config: HFTrainingConfig | Mapping[str, Any] | None = None,
                resume_from: str | Path | None = None) -> dict[str, Any]:
    """Compatibility entry; default backend is CUDA QLoRA."""
    return train_hf(records, output_dir, config, resume_from)


def re_version(value: str) -> list[str]:
    import re
    parts = re.findall(r"\d+", value)
    return parts if len(parts) >= 2 else ["0", "0"]
