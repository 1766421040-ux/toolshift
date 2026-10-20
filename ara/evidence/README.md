# Evidence index

All entries describe AI-executed work; raw results remain in their original
directories and retain their manifests.

- [Reference implementation audit](../../docs/reference_audit.md):
  actual code/data structures inspected, license and scope distinctions.
- [Frozen CPU dataset and comparisons](../../experiments/cpu/):
  1,258 held-out episodes, 74 source groups; per-episode metrics.
- [Coupled diagnostic experiment](../../experiments/diagnostic_graph/):
  67 cases, 374 worlds, 1,870 SQL episodes, compiled trees and audits.
- [Actual LoRA training and rollout](../../experiments/lora_main/):
  locked protocol, OOM attempt, completed training, base/adapter traces.
- [Portable adapter](../../artifacts/smollm2-135m-lora/):
  weights, tokenizer, model card, license and provenance.
- [Public distribution verification](../../provenance/publication_validation.json):
  downloaded file hashes, tests, CLI and GitHub CI.
- [Development history](../../provenance/README.md):
  original local protocol commits; no independent notarization claim.
