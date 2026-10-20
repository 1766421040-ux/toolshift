"""Publish only final adapter assets with portable identities and file hashes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

PROJECT = Path(__file__).resolve().parents[2]
BASE_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
BASE_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
ASSETS = (
    "adapter_model.safetensors", "adapter_config.json",
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "vocab.json", "merges.txt", "chat_template.jinja",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=PROJECT / "runs/lora_main/adapter")
    parser.add_argument("--destination", type=Path,
                        default=PROJECT / "artifacts/smollm2-135m-lora")
    parser.add_argument("--license-file", type=Path, required=True,
                        help="Unmodified Apache License 2.0 standard text")
    args = parser.parse_args()
    summary = json.loads((Path(__file__).parent / "training_summary.json").read_text())
    if summary["status"] != "trained_lora":
        raise ValueError("A completed training summary is required")
    license_text = args.license_file.read_text()
    if "Apache License" not in license_text or "Version 2.0, January 2004" not in license_text:
        raise ValueError("Expected the Apache License 2.0 text")
    for name in ASSETS:
        if not (args.source / name).is_file():
            raise FileNotFoundError(args.source / name)
    args.destination.mkdir(parents=True, exist_ok=True)
    for name in ASSETS:
        shutil.copyfile(args.source / name, args.destination / name)
    config_path = args.destination / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config.update(base_model_name_or_path=BASE_ID, revision=BASE_REVISION)
    write_json(config_path, config)
    tokenizer_config_path = args.destination / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text())
    if "name_or_path" in tokenizer_config:
        tokenizer_config["name_or_path"] = BASE_ID
    write_json(tokenizer_config_path, tokenizer_config)
    (args.destination / "LICENSE").write_text(license_text)
    (args.destination / "NOTICE").write_text(
        "ToolShift SmolLM2-135M LoRA adapter\n\n"
        "The tokenizer assets are from HuggingFaceTB/SmolLM2-135M-Instruct,\n"
        f"revision {BASE_REVISION}, licensed Apache-2.0.\n"
        "The adapter weights were trained for ToolShift and are released under\n"
        "Apache-2.0. The original base weights are not included.\n"
        "adapter_config.json was modified to replace a local filesystem path\n"
        "with the canonical base-model ID and pinned revision. Tokenizer metadata\n"
        "is normalized when it contains a local name_or_path field.\n"
        "The upstream model card is available at:\n"
        f"https://huggingface.co/{BASE_ID}/blob/{BASE_REVISION}/README.md\n"
    )
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "tokenizers",
                 "safetensors", "huggingface-hub", "numpy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    manifest = {
        "format_version": 1,
        "artifact_type": "peft_lora_adapter",
        "base_model": BASE_ID,
        "base_model_revision": BASE_REVISION,
        "base_model_files_sha256": summary["manifest"]["base_model_identity"]["local_files_sha256"],
        "license": "Apache-2.0",
        "license_source": f"https://huggingface.co/{BASE_ID}/blob/{BASE_REVISION}/README.md",
        "python_version": sys.version.split()[0],
        "platform": {"system": platform.system(), "release": platform.release(),
                     "machine": platform.machine()},
        "runtime_packages": packages,
        "training": {
            "records_available": summary["preflight"]["training_records"],
            "independent_train_groups": summary["preflight"]["training_groups"],
            "optimizer_steps": 200,
            "original_decision_exposures": 400,
            "seed": 17,
            "training_jsonl_sha256": summary["preflight"]["training_jsonl_sha256"],
            "training_summary_sha256": sha256(Path(__file__).parent / "training_summary.json"),
            "trainable_parameters": summary["trainable_parameters"],
            "objective": summary["objective"],
            "backend": summary["backend"],
            "config": {key: value for key, value in summary["manifest"]["config"].items()
                       if key not in ("model_name_or_path", "revision")},
        },
        "included_asset_sha256": {
            name: sha256(args.destination / name) for name in (*ASSETS, "LICENSE", "NOTICE")
        },
        "excluded": ["base-model weights", "optimizer", "scheduler", "RNG state",
                     "full checkpoints", "training_args.bin"],
    }
    write_json(args.destination / "artifact_manifest.json", manifest)
    # Manifest deliberately does not hash itself or the human-readable model
    # card. SHA256SUMS is regenerated after the final model card is written.
    included = sorted(path for path in args.destination.iterdir()
                      if path.is_file() and path.name != "SHA256SUMS")
    (args.destination / "SHA256SUMS").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in included))
    print(json.dumps({"artifact": str(args.destination.relative_to(PROJECT)),
                      "files": len(included) + 1,
                      "bytes": sum(path.stat().st_size for path in included)}))


if __name__ == "__main__":
    main()
