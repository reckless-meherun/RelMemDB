import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml


def read_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(read_text(path))


def write_json(path: str | Path, value: Any) -> None:
    serialized = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    write_text(path, f"{serialized}\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(read_text(path).splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {line_number} in {path}")
        record = json.loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"JSONL line {line_number} in {path} is not an object")
        records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    lines: list[str] = []
    for line_number, record in enumerate(records, start=1):
        if not isinstance(record, Mapping):
            raise TypeError(f"JSONL record {line_number} must be a mapping")
        lines.append(
            json.dumps(
                dict(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    write_text(path, "".join(f"{line}\n" for line in lines))


def read_yaml(path: str | Path) -> Any:
    return yaml.safe_load(read_text(path))


def write_yaml(path: str | Path, value: Any) -> None:
    serialized = yaml.safe_dump(
        value,
        allow_unicode=True,
        sort_keys=False,
    )
    write_text(path, serialized)
