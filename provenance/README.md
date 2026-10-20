# Development protocol history

`protocol-history.bundle` preserves the original local Git history through
`24eb03cd82b073b9e8ea3a16324b3accb672f2d8`. It includes the dataset/protocol
construction, frozen LoRA configuration and evaluation IDs, the recorded OOM
repair, and the first complete CPU results. It does not include the later
adapter export or publication commit.

The GitHub repository was created in the authenticated browser and populated
through the connected GitHub Git Data API because command-line HTTPS/SSH
credentials were unavailable. Its publication commit therefore has a different
history. The bundle preserves the original commit references cited in experiment
manifests; it is not an independent timestamp/notarization service.

Verify or inspect it without executing project code:

```bash
git bundle verify provenance/protocol-history.bundle
git clone provenance/protocol-history.bundle /tmp/toolshift-protocol-history
git -C /tmp/toolshift-protocol-history log --all --oneline
```

The clone has a tag rather than a default branch; use
`git checkout local-protocol-evidence-2026-10-20` inside that clone if a checkout
is needed. The bundle was checked for private local paths and credential
patterns before publication. The final source, experiments and portable adapter
are ordinary files in the main repository.
