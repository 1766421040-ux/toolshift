"""Deployable non-learning controls, using only the public DTO."""
from __future__ import annotations

from .protocol import STOP


class StaticPolicy:
    """Handwritten action-sufficient diagnosis, deliberately a strong baseline.

    No oracle, environment, true world, or test labels are available here.
    Supports the current factorized migration notice, not arbitrary APIs.
    """

    def __init__(self, strategy: str = "static"):
        if strategy not in ("static", "refresh", "identify", "legacy", "schema"):
            raise ValueError("Unknown strategy")
        self.strategy = strategy

    def choose(self, observed: dict, candidates=None) -> dict:
        menu = candidates or observed["candidates"]
        ctx = observed["context"]
        history = observed["history"]
        docs = ctx["legacy_docs"]
        allowed = ctx["uncertainty_contract"]["allowed"]
        known = {key: choices[0] for key, choices in allowed.items() if len(choices) == 1}
        asked = set()
        for entry in history:
            if entry["kind"] != "query":
                continue
            name = entry["call"]["name"]
            asked.add(name)
            reply = entry["observation"]
            if reply.get("available") is False:
                return dict(STOP)
            reply = reply.get("contract", reply)
            for source, target in (("canonical_units_per_input", "unit_factor"),
                                   ("collection_update", "update_mode"),
                                   ("default_scope", "default_scope")):
                if source in reply:
                    known[target] = reply[source]
        mapping = {"unit_factor": "inspect_units", "update_mode": "inspect_update_mode",
                   "default_scope": "inspect_default_scope"}

        def query(name):
            return next((c for c in menu if c["type"] == "query" and c["name"] == name), dict(STOP))

        if self.strategy == "refresh" and "read_contract" not in asked:
            return query("read_contract")
        if self.strategy in ("legacy", "schema"):
            known = {"unit_factor": 1, "update_mode": docs["collection_update"],
                     "default_scope": docs["default_scope"]}
        else:
            needed = ["unit_factor", "update_mode"]
            if self.strategy == "identify":
                needed.append("default_scope")
            missing = [key for key in needed if key not in known]
            if missing:
                if known.get("observable") is False or len(missing) > observed["remaining_budget"]:
                    return dict(STOP)
                return query(mapping[missing[0]])

        remaining = [c for c in menu if c["type"] == "act"]
        if not remaining:
            return dict(STOP)
        numeric = [c for c in remaining if c["name"] == docs["numeric_tool"]]
        if numeric:
            # The public menu includes the canonical target and its supported
            # alternative encodings. Largest numeric value is canonical here.
            canonical_target = max(c["arguments"]["value"] for c in numeric)
            wanted = canonical_target // known["unit_factor"]
            selected = [c for c in numeric if c["arguments"]["value"] == wanted]
        else:
            selected = [c for c in remaining if c["name"] == docs["items_tool"]]
            lengths = [len(c["arguments"]["items"]) for c in selected]
            target_length = min(lengths) if known["update_mode"] == "append" else max(lengths)
            selected = [c for c in selected if len(c["arguments"]["items"]) == target_length]
        explicit = self.strategy != "legacy"
        return next((c for c in selected if ("scope" in c["arguments"]) == explicit), dict(STOP))


class RandomPolicy:
    def __init__(self, seed=17):
        import random
        self.rng = random.Random(seed)

    def choose(self, observed: dict, candidates=None) -> dict:
        return self.rng.choice(candidates or observed["candidates"])

