"""File I/O and aggregation helpers for manuscript experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
from polars.type_aliases import SchemaDict

from scripts.result_schemas import CsvSchema, validate_loaded_rows, validate_row_keys


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(inner) for key, inner in value.items()}
    if isinstance(value, list | tuple):
        return [json_ready(inner) for inner in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pl.Series):
        return value.to_list()
    if isinstance(value, pl.DataFrame):
        return value.to_dicts()
    return value


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    schema: CsvSchema | None = None,
) -> None:
    ensure_directory(path.parent)
    if not rows:
        raise ValueError("rows must not be empty.")
    if schema is not None:
        validate_row_keys(rows, schema=schema)
    frame = pl.DataFrame(rows)
    frame.write_csv(path)


def read_csv(
    path: Path,
    *,
    schema: CsvSchema | None = None,
    dtypes: SchemaDict | None = None,
) -> list[dict[str, Any]]:
    frame = pl.read_csv(path, schema_overrides=dtypes)
    rows = frame.to_dicts()
    if schema is not None:
        validate_loaded_rows(rows, schema=schema, source_path=path)
    return rows


def read_csv_as_frame(
    path: Path,
    *,
    schema: CsvSchema | None = None,
    dtypes: SchemaDict | None = None,
) -> pl.DataFrame:
    frame = pl.read_csv(path, schema_overrides=dtypes)
    if schema is not None:
        rows = frame.to_dicts()
        validate_loaded_rows(rows, schema=schema, source_path=path)
    return frame


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        import json
        json.dump(json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def aggregate_mean(
    rows: list[dict[str, Any]],
    *,
    group_keys: tuple[str, ...],
    metric_keys: tuple[str, ...],
) -> list[dict[str, Any]]:
    buckets: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[name] for name in group_keys)
        bucket = buckets.setdefault(
            key,
            {
                **{name: row[name] for name in group_keys},
                "count": 0,
                **{name: 0.0 for name in metric_keys},
            },
        )
        bucket["count"] += 1
        for metric in metric_keys:
            bucket[metric] += float(row[metric])
    summary: list[dict[str, Any]] = []
    for bucket in buckets.values():
        count = int(bucket["count"])
        summary.append(
            {
                **{name: bucket[name] for name in group_keys},
                "count": count,
                **{metric: bucket[metric] / count for metric in metric_keys},
            }
        )
    return sorted(summary, key=lambda row: tuple(row[name] for name in group_keys))


def min_ess(row: dict[str, Any]) -> float:
    return min(float(row["source_ess"]), float(row["target_ess"]))
