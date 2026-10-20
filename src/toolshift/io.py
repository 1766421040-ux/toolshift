"""Small, deterministic storage helpers. No implicit network access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Iterable


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    temporary.replace(path)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hasher = hashlib.sha256()
    with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        for row in rows:
            line = (canonical(row) + "\n").encode("utf-8")
            handle.write(line)
            hasher.update(line)
    temporary.replace(path)
    return hasher.hexdigest()


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{number}: invalid JSON") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected an object")
                rows.append(row)
    return rows

