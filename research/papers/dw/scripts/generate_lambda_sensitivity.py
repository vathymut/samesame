"""Measure how lambda changes doubly weighted behavior."""

from __future__ import annotations

from pathlib import Path

import typer

from scripts._dgp import draw_overlap_dataset
from scripts._repo import RESULTS_DIR, result_metadata
from scripts.common import run_harm_test
from scripts.experiment_harness import SHARED_METRIC_KEYS, run_repeated_experiment
from scripts.result_schemas import (
    LAMBDA_SENSITIVITY_DETAIL_SCHEMA,
    LAMBDA_SENSITIVITY_SUMMARY_SCHEMA,
)
from scripts.verify_result_metadata import validate_metadata

app = typer.Typer()

UNWEIGHTED_BASELINE = -1.0


@app.command()
def main(
    n_source: int = typer.Option(180, help="Number of source observations"),
    n_target: int = typer.Option(180, help="Number of target observations"),
    n_repeats: int = typer.Option(40, help="Number of repeated trials"),
    n_resamples: int = typer.Option(199, help="Permutation resamples"),
    severity: float = typer.Option(0.25, help="Fixed low-overlap severity"),
    effect_size: float = typer.Option(0.4, help="Harmful-shift effect size"),
    lambda_grid: list[float] = typer.Option(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        help="Grid of lambda values to test",
    ),
    output: Path = typer.Option(
        RESULTS_DIR / "lambda_sensitivity_summary.csv",
        help="Summary CSV output path",
    ),
    detail_output: Path = typer.Option(
        RESULTS_DIR / "lambda_sensitivity_detail.csv",
        help="Detail CSV output path",
    ),
    metadata_output: Path = typer.Option(
        RESULTS_DIR / "lambda_sensitivity_metadata.json",
        help="Metadata JSON output path",
    ),
) -> None:
    experiments = {
        "calibration": 0.0,
        "power": effect_size,
    }
    metadata = result_metadata(
        Path(__file__),
        {k: v for k, v in locals().items() if not k.startswith("_")},
        experiments=experiments,
        unweighted_baseline_lambda_value=UNWEIGHTED_BASELINE,
    )

    def run_repeat(repeat: int) -> list[dict[str, float | str]]:
        repeat_rows: list[dict[str, float | str]] = []
        for experiment_name, effect in experiments.items():
            dataset = draw_overlap_dataset(
                n_source=n_source,
                n_target=n_target,
                source_private_fraction=severity,
                target_private_fraction=severity,
                target_shared_shift=effect,
                seed=70_000 + repeat,
            )
            baseline = run_harm_test(
                dataset["source_score"],
                dataset["target_score"],
                source_feature=dataset["source_feature"],
                target_feature=dataset["target_feature"],
                mode="unweighted",
                lambda_value=0.5,
                n_resamples=n_resamples,
                seed=80_000 + repeat,
            )
            repeat_rows.append(
                {
                    "repeat": repeat,
                    "experiment": experiment_name,
                    "lambda_value": UNWEIGHTED_BASELINE,
                    **baseline,
                }
            )
            for lambda_value in lambda_grid:
                row = run_harm_test(
                    dataset["source_score"],
                    dataset["target_score"],
                    source_feature=dataset["source_feature"],
                    target_feature=dataset["target_feature"],
                    mode="both",
                    lambda_value=lambda_value,
                    n_resamples=n_resamples,
                    seed=90_000 + repeat,
                )
                repeat_rows.append(
                    {
                        "repeat": repeat,
                        "experiment": experiment_name,
                        "lambda_value": lambda_value,
                        **row,
                    }
                )
        return repeat_rows

    run_repeated_experiment(
        n_repeats=n_repeats,
        run_repeat=run_repeat,
        group_keys=("experiment", "mode", "lambda_value"),
        metric_keys=SHARED_METRIC_KEYS,
        detail_output=detail_output,
        summary_output=output,
        metadata_output=metadata_output,
        metadata=metadata,
        detail_schema=LAMBDA_SENSITIVITY_DETAIL_SCHEMA,
        summary_schema=LAMBDA_SENSITIVITY_SUMMARY_SCHEMA,
        validate_metadata=validate_metadata,
    )


if __name__ == "__main__":
    app()
