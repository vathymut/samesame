"""Reusable harness for repeated manuscript experiments."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from scripts._io import aggregate_mean, write_csv, write_json
from scripts.result_schemas import CsvSchema

RepeatRunner = Callable[[int], list[dict[str, float | str]]]
MetadataValidator = Callable[[Path], list[str]]

SHARED_METRIC_KEYS: tuple[str, ...] = (
    "statistic",
    "pvalue",
    "reject",
    "source_ess",
    "target_ess",
    "source_max_weight",
    "target_max_weight",
)


def run_repeated_experiment(
    *,
    n_repeats: int,
    run_repeat: RepeatRunner,
    group_keys: tuple[str, ...],
    metric_keys: tuple[str, ...],
    detail_output: Path,
    summary_output: Path,
    metadata_output: Path,
    metadata: dict[str, Any],
    detail_schema: CsvSchema,
    summary_schema: CsvSchema,
    validate_metadata: MetadataValidator | None = None,
) -> None:
    detail_rows: list[dict[str, float | str]] = []
    for repeat in range(n_repeats):
        detail_rows.extend(run_repeat(repeat))

    summary_rows = aggregate_mean(
        detail_rows,
        group_keys=group_keys,
        metric_keys=metric_keys,
    )
    write_csv(detail_output, detail_rows, schema=detail_schema)
    write_csv(summary_output, summary_rows, schema=summary_schema)
    write_json(metadata_output, metadata)

    if validate_metadata is None:
        return
    errors = validate_metadata(metadata_output)
    if errors:
        listed = "\n".join(errors)
        raise ValueError(f"metadata validation failed for {metadata_output}:\n{listed}")
