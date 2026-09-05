"""CSV schema contracts for manuscript result files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CsvSchema:
    name: str
    required_columns: tuple[str, ...]


def validate_row_keys(rows: list[dict[str, Any]], *, schema: CsvSchema) -> None:
    if not rows:
        raise ValueError(f"{schema.name}: rows must not be empty")
    missing = sorted(set(schema.required_columns).difference(rows[0]))
    if missing:
        listed = ", ".join(missing)
        raise ValueError(f"{schema.name}: missing required column(s): {listed}")


def validate_loaded_rows(
    rows: list[dict[str, Any]],
    *,
    schema: CsvSchema,
    source_path: Path,
) -> None:
    if not rows:
        raise ValueError(f"{schema.name}: no rows found in {source_path}")
    missing = sorted(set(schema.required_columns).difference(rows[0]))
    if missing:
        listed = ", ".join(missing)
        raise ValueError(
            f"{schema.name}: {source_path} is missing required column(s): {listed}"
        )


SYNTHETIC_CALIBRATION_DETAIL_SCHEMA = CsvSchema(
    name="synthetic_calibration_detail",
    required_columns=("repeat", "severity", "mode", "reject"),
)

SYNTHETIC_CALIBRATION_SUMMARY_SCHEMA = CsvSchema(
    name="synthetic_calibration_summary",
    required_columns=("severity", "mode", "count", "reject"),
)

POWER_CURVE_DETAIL_SCHEMA = CsvSchema(
    name="power_curve_detail",
    required_columns=("repeat", "effect_size", "severity", "mode", "reject"),
)

POWER_CURVE_SUMMARY_SCHEMA = CsvSchema(
    name="power_curve_summary",
    required_columns=("effect_size", "mode", "count", "reject"),
)

MODE_COMPARISON_DETAIL_SCHEMA = CsvSchema(
    name="mode_comparison_detail",
    required_columns=("repeat", "scenario", "mode", "reject"),
)

MODE_COMPARISON_SUMMARY_SCHEMA = CsvSchema(
    name="mode_comparison_summary",
    required_columns=("scenario", "mode", "count", "reject"),
)

LAMBDA_SENSITIVITY_DETAIL_SCHEMA = CsvSchema(
    name="lambda_sensitivity_detail",
    required_columns=("repeat", "experiment", "mode", "lambda_value", "reject"),
)

LAMBDA_SENSITIVITY_SUMMARY_SCHEMA = CsvSchema(
    name="lambda_sensitivity_summary",
    required_columns=(
        "experiment",
        "mode",
        "lambda_value",
        "count",
        "reject",
        "source_ess",
        "target_ess",
    ),
)

DOMAIN_CLF_SENSITIVITY_DETAIL_SCHEMA = CsvSchema(
    name="domain_classifier_sensitivity_detail",
    required_columns=("repeat", "severity", "classifier", "mode", "reject"),
)

DOMAIN_CLF_SENSITIVITY_SUMMARY_SCHEMA = CsvSchema(
    name="domain_classifier_sensitivity_summary",
    required_columns=("severity", "classifier", "mode", "count", "reject"),
)

SECOND_DGP_CALIBRATION_DETAIL_SCHEMA = CsvSchema(
    name="second_dgp_calibration_detail",
    required_columns=("repeat", "overlap_severity", "mode", "reject"),
)

SECOND_DGP_CALIBRATION_SUMMARY_SCHEMA = CsvSchema(
    name="second_dgp_calibration_summary",
    required_columns=("overlap_severity", "mode", "count", "reject"),
)

SECOND_DGP_POWER_DETAIL_SCHEMA = CsvSchema(
    name="second_dgp_power_detail",
    required_columns=("repeat", "effect_size", "mode", "reject"),
)

SECOND_DGP_POWER_SUMMARY_SCHEMA = CsvSchema(
    name="second_dgp_power_summary",
    required_columns=("effect_size", "mode", "count", "reject"),
)

REAL_DATA_WORKFLOW_SUMMARY_SCHEMA = CsvSchema(
    name="real_data_workflow_summary",
    required_columns=(
        "task",
        "workflow",
        "mode",
        "pvalue",
        "source_ess",
        "target_ess",
    ),
)
