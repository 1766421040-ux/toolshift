# Contributing

Use a separate environment, run the standard-library suite and explain changes
to assumptions, public information, data grouping and metrics. Fixing a benchmark
after observing results requires a new protocol or a clearly recorded revision.

Do not add hidden world IDs, suite IDs or evaluation labels to policy prompts.
Do not use an oracle to silently repair a learned policy's decision. Publish
negative results and preserve the original scoring definitions.

New task families need actual state transitions, independent goal/preservation
checks, a public support contract, a strong non-learning control, and evidence
that they add distinct mechanisms rather than renamed copies.

No API keys or production accounts are needed. Keep private artifacts,
pretrained weights and large checkpoints out of Git. Model and data licenses
are independent from this repository's code license.

