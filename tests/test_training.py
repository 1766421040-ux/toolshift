"""Behavioral training tests require only Python's standard library."""
import json
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

from toolshift.model_policy import (
    LinearPolicy, canonical_decision, candidate_features, encode_assistant_completion,
)
from toolshift.training import (
    assert_group_disjoint, set_softmax_loss_gradient, train_linear,
    validate_training_records,
)


LEFT = {"type": "act", "name": "select", "arguments": {"side": "left"}}
RIGHT = {"type": "act", "name": "select", "arguments": {"side": "right"}}
PAUSE = {"type": "stop", "status": "pause"}


def examples():
    records = []
    for side, decision in (("left", LEFT), ("right", RIGHT)):
        for index in range(4):
            records.append({
                "id": f"{side}-{index}", "group_id": f"{side}-family-{index}",
                "split": "train", "observation": {
                    "task": {"instruction": f"Choose {side}"},
                    "history": [], "candidates": [LEFT, RIGHT, PAUSE]},
                "acceptable_decisions": [decision],
            })
    return records


class CharacterChatTokenizer:
    """Deterministic fixture exposes exact boundaries without external models."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        text = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return [ord(char) for char in text]


class TrainingTests(unittest.TestCase):
    def test_set_gradient_matches_numerical_derivative(self):
        scores = [0.3, -0.5, 0.9]
        loss, gradient = set_softmax_loss_gradient(scores, [0, 2])
        self.assertGreater(loss, 0)
        self.assertAlmostEqual(sum(gradient), 0, places=12)
        for i in range(3):
            plus, minus = scores.copy(), scores.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            numeric = (set_softmax_loss_gradient(plus, [0, 2])[0] -
                       set_softmax_loss_gradient(minus, [0, 2])[0]) / 2e-6
            self.assertAlmostEqual(gradient[i], numeric, places=7)

    def test_all_candidates_acceptable_has_zero_loss_and_gradient(self):
        loss, gradient = set_softmax_loss_gradient([1000, -1000, 3], [0, 1, 2])
        self.assertEqual(loss, 0)
        self.assertEqual(gradient, [0, 0, 0])

    def test_extreme_wrong_scores_stay_finite(self):
        loss, gradient = set_softmax_loss_gradient([1000, -1000], [1])
        self.assertEqual(loss, 2000)
        self.assertTrue(all(math.isfinite(value) for value in gradient))

    def test_rejects_heldout_rows_and_hidden_fields(self):
        records = examples()
        records[0]["split"] = "test"
        with self.assertRaisesRegex(ValueError, "non-train"):
            validate_training_records(records)
        records = examples()
        records[0]["observation"]["context"] = {"oracle": {"answer": "left"}}
        with self.assertRaisesRegex(ValueError, "Privileged"):
            validate_training_records(records)

    def test_rejects_labels_outside_menu(self):
        records = examples()
        records[0]["acceptable_decisions"] = [
            {"type": "query", "name": "secret", "arguments": {}}]
        with self.assertRaisesRegex(ValueError, "not in public"):
            validate_training_records(records)

    def test_detects_family_leak(self):
        with self.assertRaisesRegex(ValueError, "leak"):
            assert_group_disjoint([{"group_id": "g"}], [{"group_id": "g"}])

    def test_identical_public_states_cannot_have_disjoint_labels(self):
        records = examples()[:2]
        records[1]["acceptable_decisions"] = [RIGHT]
        with self.assertRaisesRegex(ValueError, "incompatible labels"):
            validate_training_records(records)

    def test_training_generalizes_to_new_bookkeeping_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            summary = train_linear(examples(), temp, {"epochs": 8, "dimensions": 1024})
            self.assertEqual(summary["train_set_accuracy"], 1)
            self.assertFalse(summary["is_language_model"])
            model = LinearPolicy.load(temp)
            unseen = examples()[0]["observation"]
            unseen["id"] = "unseen"
            self.assertEqual(model.choose(unseen), LEFT)
            unseen["candidates"].reverse()
            self.assertEqual(model.choose(unseen), LEFT)

    def test_resume_matches_uninterrupted_training(self):
        with tempfile.TemporaryDirectory() as temp:
            full, resumed = Path(temp) / "full", Path(temp) / "resumed"
            train_linear(examples(), full, {"epochs": 5, "dimensions": 512, "seed": 3})
            train_linear(examples(), resumed, {"epochs": 2, "dimensions": 512, "seed": 3})
            train_linear(examples(), resumed, {"epochs": 5, "dimensions": 512, "seed": 3},
                         resume_from=resumed)
            a = json.loads((full / "linear_checkpoint.json").read_text())
            b = json.loads((resumed / "linear_checkpoint.json").read_text())
            self.assertEqual(a, b)
            altered = examples()
            altered[0]["id"] = "changed-provenance-id"
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                train_linear(altered, resumed, {"epochs": 6, "dimensions": 512, "seed": 3},
                             resume_from=resumed)

    def test_feature_hash_is_deterministic_and_ignores_ids(self):
        obs = examples()[0]["observation"]
        before = candidate_features(obs, LEFT, 512)
        obs["group_id"] = "must-not-be-a-feature"
        self.assertEqual(before, candidate_features(obs, LEFT, 512))

    def test_long_documents_cannot_displace_public_semantic_prior(self):
        obs = {"context": {
            "legacy_docs": {f"section{i:04}": f"many words {i}" for i in range(500)},
            "uncertainty_contract": {"allowed": {"unit_factor": [1], "update_mode": ["append"]}},
        }, "history": [], "candidates": [LEFT, RIGHT]}
        before = candidate_features(obs, LEFT, 2048)
        obs["context"]["uncertainty_contract"]["allowed"]["unit_factor"] = [1, 1000]
        self.assertNotEqual(before, candidate_features(obs, LEFT, 2048))

    def test_assistant_only_mask_and_overflow(self):
        tokenizer = CharacterChatTokenizer()
        messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "question"}]
        encoded = encode_assistant_completion(tokenizer, messages, "answer", 200)
        prompt = tokenizer.apply_chat_template(messages, True, True)
        self.assertEqual(encoded["labels"][:len(prompt)], [-100] * len(prompt))
        self.assertEqual(encoded["labels"][len(prompt):],
                         encoded["input_ids"][len(prompt):])
        with self.assertRaisesRegex(ValueError, "exceeds"):
            encode_assistant_completion(tokenizer, messages, "answer", 10)

    def test_rejects_template_boundary_mismatch(self):
        class InconsistentTokenizer(CharacterChatTokenizer):
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                result = super().apply_chat_template(messages, tokenize, add_generation_prompt)
                return result + [99] if add_generation_prompt else result
        with self.assertRaisesRegex(ValueError, "token-aligned"):
            encode_assistant_completion(
                InconsistentTokenizer(), [{"role": "user", "content": "x"}], "a", 100)

    def test_canonical_decision_excludes_explanation(self):
        self.assertEqual(canonical_decision(LEFT),
                         canonical_decision(dict(LEFT, reason="not an executable field")))


@unittest.skipUnless(
    all(importlib.util.find_spec(name) is not None for name in
        ("torch", "transformers", "peft", "tokenizers")),
    "optional HF integration dependencies are not installed")
class OptionalHFIntegrationTests(unittest.TestCase):
    def test_random_tiny_model_trains_resumes_and_scores_without_download(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
        from safetensors.torch import load_file
        from toolshift.model_policy import HFCausalPolicy
        from toolshift.training import train_hf

        with tempfile.TemporaryDirectory() as temp:
            base, run, resumed = (Path(temp) / name for name in ("base", "run", "resumed"))
            tokens = ["[UNK]", "[PAD]", "[EOS]", "system", "user", "assistant",
                      "left", "right", "act", "type", "name", "arguments", "select", "side"]
            vocabulary = {token: i for i, token in enumerate(tokens)}
            raw = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
            raw.pre_tokenizer = Whitespace()
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=raw, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
            tokenizer.chat_template = (
                "{% for message in messages %}{{ message['role'] + ' ' + "
                "message['content'] + ' ' }}{% endfor %}"
                "{% if add_generation_prompt %}{{ 'assistant ' }}"
                "{% else %}{{ '[EOS]' }}{% endif %}")
            tokenizer.save_pretrained(base)
            torch.manual_seed(2)
            LlamaForCausalLM(LlamaConfig(
                vocab_size=len(vocabulary), hidden_size=32, intermediate_size=64,
                num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                max_position_embeddings=512)).save_pretrained(base)
            rows = examples()[:4]
            rows[1]["acceptable_decisions"] = [LEFT, RIGHT]
            config = dict(
                model_name_or_path=str(base), backend="cpu_lora", precision="fp32",
                max_length=512, max_steps=2, gradient_accumulation_steps=2,
                save_steps=1, logging_steps=1, lora_r=2, lora_alpha=4, lora_dropout=0.0)
            summary = train_hf(rows, run, config)
            self.assertEqual(summary["status"], "trained_lora")
            self.assertTrue(math.isfinite(summary["metrics"]["train_loss"]))
            unoptimized = Path(temp) / "full_logits"
            full_config = dict(config, completion_logits_only=False)
            train_hf(rows, unoptimized, full_config)
            optimized_weights = load_file(run / "adapter" / "adapter_model.safetensors")
            unoptimized_weights = load_file(unoptimized / "adapter" / "adapter_model.safetensors")
            for key in optimized_weights:
                torch.testing.assert_close(optimized_weights[key], unoptimized_weights[key],
                                           rtol=1e-4, atol=1e-7)
            train_hf(rows, resumed, config, resume_from=run / "checkpoint-1")
            full = load_file(run / "adapter" / "adapter_model.safetensors")
            restored = load_file(resumed / "adapter" / "adapter_model.safetensors")
            self.assertEqual(set(full), set(restored))
            for key in full:
                torch.testing.assert_close(full[key], restored[key], rtol=0, atol=0)
            policy = HFCausalPolicy(str(base), adapter_path=str(run / "adapter"),
                                    load_in_4bit=False, device="cpu", max_length=512)
            decision = policy.choose(rows[0]["observation"])
            self.assertIn(canonical_decision(decision),
                          {canonical_decision(d) for d in rows[0]["observation"]["candidates"]})
            policy.completion_logits_only = False
            single = policy.scores(rows[0]["observation"])
            policy.completion_logits_only = True
            policy.max_batch_size = 3
            policy.reset_usage()
            batched = policy.scores(rows[0]["observation"])
            self.assertEqual([a for a, _ in single], [a for a, _ in batched])
            for (_, a), (_, b) in zip(single, batched):
                self.assertAlmostEqual(a, b, places=4)
            self.assertEqual(policy.usage["score_calls"], 1)
            self.assertEqual(policy.usage["candidate_completions"], 3)
            self.assertEqual(policy.usage["forward_passes"], 1)
            self.assertGreater(policy.usage["nonpadding_input_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
