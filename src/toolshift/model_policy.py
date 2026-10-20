"""Policies that use public observations only.

The dependency-free policy is a hashed linear candidate ranker, not a language
model. Hugging Face imports are deferred until HFCausalPolicy is instantiated.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import inspect
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence


FORBIDDEN_OBSERVATION_KEYS = frozenset({
    "oracle", "oracle_decision", "acceptable_decisions", "optimal_decisions",
    "correct_decisions", "hidden_world", "world_spec", "hidden_state",
})
FEATURE_VERSION = "public-candidate-cross-v2"


def canonical_decision(decision: Mapping[str, Any]) -> str:
    """Canonicalize executable fields; explanatory reasons are not actions."""
    kind = decision.get("type")
    if kind not in {"query", "act", "stop"}:
        raise ValueError(f"Unknown decision type: {kind!r}")
    normalized: dict[str, Any] = {"type": kind}
    if kind == "stop":
        normalized["status"] = decision.get("status", "pause")
    else:
        name = decision.get("name", decision.get("tool"))
        if not isinstance(name, str) or not name:
            raise ValueError("Query/action requires a nonempty name")
        arguments = decision.get("arguments", decision.get("args", {}))
        if not isinstance(arguments, dict):
            raise ValueError("Decision arguments must be an object")
        normalized.update(name=name, arguments=arguments)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False)


def assert_public_observation(value: Any) -> None:
    """Fail closed on accidentally supplied privileged label containers."""
    if isinstance(value, Mapping):
        illegal = FORBIDDEN_OBSERVATION_KEYS.intersection(value)
        if illegal:
            raise ValueError(f"Privileged fields in policy input: {sorted(illegal)}")
        for item in value.values():
            assert_public_observation(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            assert_public_observation(item)


def candidate_list(observation: Mapping[str, Any],
                   candidates: Sequence[Mapping[str, Any]] | None = None
                   ) -> list[dict[str, Any]]:
    assert_public_observation(observation)
    items = candidates if candidates is not None else observation.get("candidates")
    if not isinstance(items, (list, tuple)) or not items:
        raise ValueError("A nonempty public candidate menu is required")
    unique = {canonical_decision(d): json.loads(canonical_decision(d)) for d in items}
    # Tie breaking is independent of the producer's menu order.
    return [unique[key] for key in sorted(unique)]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+|[\u3400-\u9fff]", text.lower())[:64]


def _observation_features(value: Any, prefix: str = "") -> list[str]:
    result: list[str] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            # IDs are bookkeeping, not a task-solving signal. Candidates are
            # represented separately so menu size/order cannot label the task.
            if key in {"candidates", "id", "group_id", "episode_id", "scenario_id", "task_id"}:
                continue
            result.extend(_observation_features(value[key], f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        result.append(f"{prefix}.length={min(len(value), 12)}")
        for i, item in enumerate(value[-6:]):
            result.extend(_observation_features(item, f"{prefix}.recent{i-len(value[-6:])}"))
    else:
        for token in _tokens(str(value)):
            result.append(f"{prefix}={token}")
    return result


def candidate_features(observation: Mapping[str, Any],
                       decision: Mapping[str, Any],
                       dimensions: int = 32768) -> dict[int, float]:
    """Stable signed feature hashing with candidate/observation interactions."""
    if dimensions < 32:
        raise ValueError("dimensions must be >= 32")
    assert_public_observation(observation)
    # Reserve a separate feature budget for recent observations so long static
    # documentation cannot displace the evidence that was just queried.
    recent = {key: observation[key] for key in ("history", "remaining_budget")
              if key in observation}
    static = {key: value for key, value in observation.items()
              if key not in {"history", "remaining_budget"}}
    prior = static.pop("uncertainty_contract", None)
    if isinstance(static.get("context"), Mapping):
        static["context"] = dict(static["context"])
        nested_prior = static["context"].pop("uncertainty_contract", None)
        if nested_prior is not None:
            prior = nested_prior
    obs = sorted(set(_observation_features(static)))[:64]
    if prior is not None:
        obs += sorted(set(_observation_features(prior, "prior")))[:64]
    obs += sorted(set(_observation_features(recent)))[:96]
    normalized = json.loads(canonical_decision(decision))
    cand = sorted(set(_observation_features(normalized, "candidate")))[:24]
    # Hash atoms once, then mix interaction hashes. This keeps pure-Python
    # training practical without constructing and hashing thousands of strings.
    def hash_atom(feature: str) -> int:
        return int.from_bytes(hashlib.blake2b(feature.encode("utf-8"),
                                            digest_size=8).digest(), "little")
    c_hashes, o_hashes = [hash_atom(s) for s in cand], [hash_atom(s) for s in obs]
    hashes = [hash_atom("bias"), *c_hashes]
    hashes.extend(((a * 0x9E3779B185EBCA87) ^ b) & ((1 << 64) - 1)
                  for a in c_hashes for b in o_hashes)
    counts: Counter[int] = Counter()
    for value in hashes:
        index = (value >> 1) % dimensions
        counts[index] += 1 if value & 1 else -1
    norm = math.sqrt(sum(v * v for v in counts.values())) or 1.0
    return {index: value / norm for index, value in counts.items() if value}


def softmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        raise ValueError("Cannot normalize empty scores")
    peak = max(scores)
    values = [math.exp(s - peak) for s in scores]
    total = sum(values)
    return [value / total for value in values]


def decision_messages(observation: Mapping[str, Any],
                      candidates: Sequence[Mapping[str, Any]] | None = None
                      ) -> list[dict[str, str]]:
    """The identical public prompt format is used for training and inference."""
    menu = candidate_list(observation, candidates)
    public = dict(observation)
    public["candidates"] = menu
    return [
        {"role": "system", "content":
         "Choose one next decision from the public candidate menu. Use only "
         "the provided task, documentation and observations. Return exactly "
         "one JSON object with the candidate's executable fields."},
        {"role": "user", "content": json.dumps(public, ensure_ascii=False,
                                               sort_keys=True, separators=(",", ":"),
                                               allow_nan=False)},
    ]


class LinearPolicy:
    """A CPU linear ranker; no oracle, simulator, or labels are accessible."""

    def __init__(self, weights: Sequence[float] | None = None, *,
                 dimensions: int = 32768) -> None:
        self.dimensions = dimensions
        self.weights = list(weights) if weights is not None else [0.0] * dimensions
        if len(self.weights) != dimensions or not all(math.isfinite(w) for w in self.weights):
            raise ValueError("Invalid linear weights")

    def scores(self, observation: Mapping[str, Any],
               candidates: Sequence[Mapping[str, Any]] | None = None
               ) -> list[tuple[dict[str, Any], float]]:
        return [(decision, sum(self.weights[i] * value for i, value in
                               candidate_features(observation, decision, self.dimensions).items()))
                for decision in candidate_list(observation, candidates)]

    def choose(self, observation: Mapping[str, Any],
               candidates: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        return max(self.scores(observation, candidates), key=lambda pair: pair[1])[0]

    __call__ = choose

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        path = Path(path)
        if path.is_dir():
            path = path / "model.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("feature_version") != FEATURE_VERSION:
            raise ValueError("Unsupported feature version")
        if payload.get("model_type") != "hashed_linear_candidate_ranker":
            raise ValueError("This checkpoint is not a ToolShift linear ranker")
        return cls(payload["weights"], dimensions=payload["dimensions"])


def encode_assistant_completion(tokenizer: Any, messages: Sequence[Mapping[str, str]],
                                answer: str, max_length: int) -> dict[str, list[int]]:
    """Mask the entire prompt; reject overflow rather than silently truncating.

    Prefix consistency is checked because arbitrary chat templates need not
    preserve the generation prefix once an assistant response is appended.
    """
    if not messages or messages[-1]["role"] == "assistant":
        raise ValueError("Expected prompt messages ending before the assistant answer")
    prompt = tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": answer}],
        tokenize=True, add_generation_prompt=False)
    prompt, full = list(prompt), list(full)
    if not prompt or full[:len(prompt)] != prompt:
        raise ValueError("Chat template does not preserve a token-aligned assistant prefix")
    if len(full) > max_length:
        raise ValueError(f"Example has {len(full)} tokens, exceeds max_length={max_length}; "
                         "shorten public context or increase the explicit budget")
    if len(full) <= len(prompt):
        raise ValueError("No assistant tokens: refusing an all--100 training example")
    return {"input_ids": full, "attention_mask": [1] * len(full),
            "labels": [-100] * len(prompt) + full[len(prompt):]}


class HFCausalPolicy:
    """Optional local causal LM that scores each menu answer without an oracle.

    Scores are joint completion log probabilities (including template ending),
    consistent with the optional set-valued SFT objective. This costs one
    bounded candidate batches and is not advertised as free-form tool use.
    """

    def __init__(self, model_path: str, *, adapter_path: str | None = None,
                 revision: str | None = None, local_files_only: bool = True,
                 max_length: int = 2048, load_in_4bit: bool = True,
                 device: str | None = None, max_batch_size: int = 1,
                 completion_logits_only: bool = True) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Install ToolShift's optional HF dependencies in an isolated environment") from exc
        self.torch = torch
        self.max_length = max_length
        self.max_batch_size = max_batch_size
        kwargs: dict[str, Any] = dict(revision=revision, local_files_only=local_files_only,
                                      trust_remote_code=False)
        tokenizer_path = adapter_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, local_files_only=local_files_only, trust_remote_code=False,
            **({"revision": revision} if not adapter_path else {}))
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError("This 4-bit inference entry requires a CUDA GPU")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            kwargs.update(quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype),
                device_map={"": torch.cuda.current_device()}, dtype=dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        if adapter_path:
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise RuntimeError("Adapter inference requires optional peft") from exc
            self.model = PeftModel.from_pretrained(
                self.model, adapter_path, local_files_only=local_files_only)
        if not load_in_4bit:
            selected_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self.model.to(selected_device)
        self.model.eval()
        base = self.model.get_base_model() if hasattr(self.model, "get_base_model") else self.model
        self.completion_logits_only = (
            completion_logits_only and "logits_to_keep" in inspect.signature(base.forward).parameters)
        self.reset_usage()

    def reset_usage(self) -> None:
        self.usage = {
            "score_calls": 0, "candidate_completions": 0, "forward_passes": 0,
            "nonpadding_input_tokens": 0, "padded_forward_tokens": 0,
            "assistant_tokens_scored": 0, "scoring_wall_seconds": 0.0,
        }

    def scores(self, observation: Mapping[str, Any],
               candidates: Sequence[Mapping[str, Any]] | None = None
               ) -> list[tuple[dict[str, Any], float]]:
        menu = candidate_list(observation, candidates)
        messages = decision_messages(observation, menu)
        torch = self.torch
        device = self.model.get_input_embeddings().weight.device
        result: list[tuple[dict[str, Any], float]] = []
        encoded_items = [encode_assistant_completion(
            self.tokenizer, messages, canonical_decision(decision), self.max_length)
            for decision in menu]
        self.usage["score_calls"] += 1
        started = time.perf_counter()
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        if pad_id is None:
            raise ValueError("Candidate batching requires a padding or EOS token")
        with torch.inference_mode():
            for offset in range(0, len(menu), self.max_batch_size):
                batch = encoded_items[offset:offset + self.max_batch_size]
                width = max(len(item["input_ids"]) for item in batch)
                ids = torch.tensor([
                    item["input_ids"] + [pad_id] * (width - len(item["input_ids"]))
                    for item in batch], device=device)
                masks = torch.tensor([
                    item["attention_mask"] + [0] * (width - len(item["input_ids"]))
                    for item in batch], device=device)
                labels = torch.tensor([
                    item["labels"] + [-100] * (width - len(item["input_ids"]))
                    for item in batch], device=device)
                first_label = min(next(i for i, label in enumerate(item["labels"]) if label != -100)
                                  for item in batch)
                # Retain the logit immediately BEFORE the first answer token.
                # This optimization changes only which LM-head positions are
                # materialized, not attention context or answer probabilities.
                start = max(0, first_label - 1) if self.completion_logits_only else 0
                extra = {"logits_to_keep": width - start} if self.completion_logits_only else {}
                logits = self.model(input_ids=ids, attention_mask=masks,
                                    use_cache=False, **extra).logits[:, :-1]
                target = labels[:, start + 1:]
                token_losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]).float(),
                    target.reshape(-1), ignore_index=-100, reduction="none")
                scores = -token_losses.reshape(target.shape).sum(dim=1)
                result.extend(zip(menu[offset:offset + len(batch)], scores.tolist()))
                self.usage["candidate_completions"] += len(batch)
                self.usage["forward_passes"] += 1
                self.usage["padded_forward_tokens"] += width * len(batch)
                self.usage["nonpadding_input_tokens"] += sum(len(item["input_ids"]) for item in batch)
                self.usage["assistant_tokens_scored"] += sum(
                    sum(label != -100 for label in item["labels"]) for item in batch)
        self.usage["scoring_wall_seconds"] += time.perf_counter() - started
        return result

    def choose(self, observation: Mapping[str, Any],
               candidates: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
        return max(self.scores(observation, candidates), key=lambda pair: pair[1])[0]

    __call__ = choose
