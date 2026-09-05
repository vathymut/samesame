"""Second synthetic DGP: regression-based harmful shift."""

from __future__ import annotations

from pathlib import Path

import typer

from scripts._dgp import draw_second_dgp
from scripts._repo import RESULTS_DIR, result_metadata
from scripts.common import MODES_WITH_BASELINES, run_harm_test
from scripts.experiment_harness import SHARED_METRIC_KEYS, run_repeated_experiment
from scripts.result_schemas import (
    SECOND_DGP_CALIBRATION_DETAIL_SCHEMA,
    SECOND_DGP_CALIBRATION_SUMMARY_SCHEMA,
    SECOND_DGP_POWER_DETAIL_SCHEMA,
    SECOND_DGP_POWER_SUMMARY_SCHEMA,
)
from scripts.verify_result_metadata import validate_metadata

app = typer.Typer()


@app.command()
def main(
    n_source: int = typer.Option(180, help="Number of source observations"),
    n_target: int = typer.Option(180, help="Number of target observations"),
    n_repeats: int = typer.Option(40, help="Number of repeated trials"),
    n_resamples: int = typer.Option(199, help="Permutation resamples"),
    lambda_value: float = typer.Option(0.5, help="RIW stabilisation parameter"),
    severity_grid: list[float] = typer.Option(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        help="Grid of overlap severity values",
    ),
    effect_grid: list[float] = typer.Option(
        [0.0, 0.15, 0.3, 0.45, 0.6],
        help="Grid of harmful-shift effect sizes",
    ),
    calibration_output: Path = typer.Option(
        RESULTS_DIR / "second_dgp_calibration_summary.csv",
        help="Calibration summary CSV output path",
    ),
    calibration_detail_output: Path = typer.Option(
        RESULTS_DIR / "second_dgp_calibration_detail.csv",
        help="Calibration detail CSV output path",
    ),
    power_output: Path = typer.Option(
        RESULTS_DIR / "second_dgp_power_summary.csv",
        help="Power summary CSV output path",
    ),
    power_detail_output: Path = typer.Option(
        RESULTS_DIR / "second_dgp_power_detail.csv",
        help="Power detail CSV output path",
    ),
    metadata_output: Path = typer.Option(
        RESULTS_DIR / "second_dgp_metadata.json",
        help="Metadata JSON output path",
    ),
) -> None:
    metadata = result_metadata(
        Path(__file__),
        {k: v for k, v in locals().items() if not k.startswith("_")},
        dgp="regression-based with 2D features",
    )

    def run_calibration_repeat(repeat: int) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for severity in severity_grid:
            dataset = draw_second_dgp(
                n_source=n_source,
                n_target=n_target,
                overlap_severity=severity,
                effect_size=0.0,
                seed=90_000 + repeat,
            )
            for mode in MODES_WITH_BASELINES:
                row = run_harm_test(
                    dataset["source_score"],
                    dataset["target_score"],
                    source_feature=dataset["source_feature"],
                    target_feature=dataset["target_feature"],
                    mode=mode,
                    lambda_value=lambda_value,
                    n_resamples=n_resamples,
                    seed=91_000 + repeat,
                )
                rows.append(
                    {"repeat": repeat, "overlap_severity": severity, **row}
                )
        return rows

    def run_power_repeat(repeat: int) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for effect in effect_grid:
            dataset = draw_second_dgp(
                n_source=n_source,
                n_target=n_target,
                overlap_severity=0.25,
                effect_size=effect,
                seed=92_000 + repeat,
            )
            for mode in MODES_WITH_BASELINES:
                row = run_harm_test(
                    dataset["source_score"],
                    dataset["target_score"],
                    source_feature=dataset["source_feature"],
                    target_feature=dataset["target_feature"],
                    mode=mode,
                    lambda_value=lambda_value,
                    n_resamples=n_resamples,
                    seed=93_000 + repeat,
                )
                rows.append(
                    {"repeat": repeat, "effect_size": effect, **row}
                )
        return rows

    run_repeated_experiment(
        n_repeats=n_repeats,
        run_repeat=run_calibration_repeat,
        group_keys=("overlap_severity", "mode"),
        metric_keys=SHARED_METRIC_KEYS,
        detail_output=calibration_detail_output,
        summary_output=calibration_output,
        metadata_output=metadata_output,
        metadata={**metadata, "experiment_type": "calibration"},
        detail_schema=SECOND_DGP_CALIBRATION_DETAIL_SCHEMA,
        summary_schema=SECOND_DGP_CALIBRATION_SUMMARY_SCHEMA,
        validate_metadata=validate_metadata,
    )

    run_repeated_experiment(
        n_repeats=n_repeats,
        run_repeat=run_power_repeat,
        group_keys=("effect_size", "mode"),
        metric_keys=SHARED_METRIC_KEYS,
        detail_output=power_detail_output,
        summary_output=power_output,
        metadata_output=metadata_output,
        metadata={**metadata, "experiment_type": "power"},
        detail_schema=SECOND_DGP_POWER_DETAIL_SCHEMA,
        summary_schema=SECOND_DGP_POWER_SUMMARY_SCHEMA,
        validate_metadata=validate_metadata,
    )


if __name__ == "__main__":
    app()
