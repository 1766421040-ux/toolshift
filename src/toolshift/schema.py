"""Small, JSON-serializable contracts for the local ToolShift environment.

WorldSpec is privileged evaluation data.  A policy receives public_context()
and observation histories, never a WorldSpec or an oracle instance.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple
import json


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class Action:
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"tool": self.tool, "args": dict(self.args)}

    def decision(self) -> Dict[str, Any]:
        return {"type": "act", "name": self.tool, "arguments": dict(self.args)}

    @classmethod
    def from_dict(cls, value: Any) -> "Action":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("An action must be an object.")
        return cls(str(value.get("tool", value.get("name", ""))),
                   dict(value.get("args", value.get("arguments", {}))))

    def key(self) -> str:
        return canonical(self.to_dict())


@dataclass(frozen=True)
class Query:
    name: str
    args: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "args": dict(self.args)}

    def decision(self) -> Dict[str, Any]:
        return {"type": "query", "name": self.name, "arguments": dict(self.args)}

    @classmethod
    def from_dict(cls, value: Any) -> "Query":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("A query must be an object.")
        return cls(str(value.get("name", "")),
                   dict(value.get("args", value.get("arguments", {}))))

    def key(self) -> str:
        return canonical(self.to_dict())


@dataclass(frozen=True)
class WorldSpec:
    """Private semantics. IDs identify hypotheses, not public version labels."""

    world_id: str
    domain: str
    unit_factor: int = 1
    update_mode: str = "append"
    default_scope: str = "current"
    observable: bool = True

    def __post_init__(self) -> None:
        if self.domain not in ("travel", "catalog", "calendar"):
            raise ValueError("Unknown domain.")
        if type(self.unit_factor) is not int or self.unit_factor <= 0:
            raise ValueError("unit_factor must be a positive integer.")
        if self.update_mode not in ("append", "replace"):
            raise ValueError("Unknown update mode.")
        if self.default_scope not in ("current", "all"):
            raise ValueError("Unknown default scope.")

    def to_dict(self) -> Dict[str, Any]:
        """Private serialization; never place this object in model messages."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorldSpec":
        return cls(**dict(value))


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    domain: str
    instruction: str
    target_id: str
    target_value: int
    required_items: Tuple[str, ...]
    initial_items: Tuple[str, ...]
    initial_value: int
    protected_id: str
    protected_value: int
    protected_items: Tuple[str, ...]
    query_budget: int = 4
    seed: int = 0

    def __post_init__(self) -> None:
        if self.domain not in ("travel", "catalog", "calendar"):
            raise ValueError("Unknown domain.")
        if self.target_id == self.protected_id:
            raise ValueError("Target and protected records must differ.")
        if type(self.query_budget) is not int or self.query_budget < 0:
            raise ValueError("query_budget must be a nonnegative integer.")
        if not set(self.initial_items).issubset(self.required_items):
            raise ValueError("Goals must preserve initial items.")
        if len(set(self.required_items)) != len(self.required_items):
            raise ValueError("Goal items must be unique.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSpec":
        data = dict(value)
        for key in ("required_items", "initial_items", "protected_items"):
            data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class StepResult:
    """Tool output only. Correctness is intentionally absent from observations."""

    observation: Dict[str, Any]
    done: bool = False
    error: Optional[str] = None
    # Retained as explicit unavailable values for callers that expect these
    # attributes. Evaluation must call Sandbox.verify(), not inspect tool output.
    success: Optional[bool] = None
    invariant_ok: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        return {"observation": self.observation, "done": self.done,
                "error": self.error}


@dataclass(frozen=True)
class OracleDecision:
    """Privileged label. Never concatenate this report into policy inputs."""

    kind: str
    actions: Tuple[Action, ...] = ()
    queries: Tuple[Query, ...] = ()
    safe_actions: Tuple[Action, ...] = ()
    worst_cost: Optional[int] = None
    reason: str = ""
    states_examined: int = 0
    compatible_count: int = 0

    def decisions(self) -> list:
        """Label set after the explicit action-first secondary tie-break.

        Use all_optimal_decisions() for the full minimax-optimal action/query
        union. A query can have the same total future cost as acting first.
        """
        if self.kind == "act":
            return [a.decision() for a in self.actions]
        if self.kind == "query":
            return [q.decision() for q in self.queries]
        return [{"type": "stop", "status": self.kind, "reason": self.reason}]

    def all_optimal_decisions(self) -> list:
        if self.kind in ("act", "query"):
            return ([a.decision() for a in self.actions] +
                    [q.decision() for q in self.queries])
        return self.decisions()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "actions": [a.to_dict() for a in self.actions],
            "queries": [q.to_dict() for q in self.queries],
            "safe_actions": [a.to_dict() for a in self.safe_actions],
            "worst_cost": self.worst_cost,
            "reason": self.reason,
            "states_examined": self.states_examined,
            "compatible_count": self.compatible_count,
            "acceptable_decisions": self.decisions(),
            "all_optimal_decisions": self.all_optimal_decisions(),
            "tie_break": "act_before_query_at_equal_worst_case_query_cost",
        }
