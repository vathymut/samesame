"""Generate power summaries for harmful shift on common support."""

from __future__ import annotations

from pathlib import Path

import typer

from scripts._dgp import draw_overlap_dataset
from scripts._repo import RESULTS_DIR, result_metadata
from scripts.common import MODES_WITH_BASELINES, run_harm_test
from scripts.experiment_harness import SHARED_METRIC_KEYS, run_repeated_experiment
from scripts.result_schemas import POWER_CURVE_DETAIL_SCHEMA, POWER_CURVE_SUMMARY_SCHEMA
from scripts.verify_result_metadata import validate_metadata

app = typer.Typer()


@app.command()
def main(
    n_source: int = typer.Option(180, help="Number of source observations"),
    n_target: int = typer.Option(180, help="Number of target observations"),
    n_repeats: int = typer.Option(40, help="Number of repeated trials"),
    n_resamples: int = typer.Option(199, help="Permutation resamples"),
    lambda_value: float = typer.Option(0.5, help="RIW stabilisation parameter"),
    severity: float = typer.Option(0.25, help="Fixed low-overlap severity"),
    effect_grid: list[float] = typer.Option(
        [0.0, 0.15, 0.3, 0.45, 0.6],
        help="Grid of harmful-shift effect sizes",
    ),
    output: Path = typer.Option(
        RESULTS_DIR / "power_curve_summary.csv",
        help="Summary CSV output path",
    ),
    detail_output: Path = typer.Option(
        RESULTS_DIR / "power_curve_detail.csv",
        help="Detail CSV output path",
    ),
    metadata_output: Path = typer.Option(
        RESULTS_DIR / "power_curve_metadata.json",
        help="Metadata JSON output path",
    ),
) -> None:
    metadata = result_metadata(
        Path(__file__),
        {k: v for k, v in locals().items() if not k.startswith("_")},
        alternative="Target scores shift upward on common support while overlap mismatch remains fixed.",
    )

    def run_repeat(repeat: int) -> list[dict[str, float | str]]:
        repeat_rows: list[dict[str, float | str]] = []
        for effect_size in effect_grid:
            dataset = draw_overlap_dataset(
                n_source=n_source,
                n_target=n_target,
                source_private_fraction=severity,
                target_private_fraction=severity,
                target_shared_shift=effect_size,
                seed=30_000 + repeat,
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
                    seed=40_000 + repeat,
                )
                repeat_rows.append(
                    {
                        "repeat": repeat,
                        "effect_size": effect_size,
                        "severity": severity,
                        **row,
                    }
                )
        return repeat_rows

    run_repeated_experiment(
        n_repeats=n_repeats,
        run_repeat=run_repeat,
        group_keys=("effect_size", "mode"),
        metric_keys=SHARED_METRIC_KEYS,
        detail_output=detail_output,
        summary_output=output,
        metadata_output=metadata_output,
        metadata=metadata,
        detail_schema=POWER_CURVE_DETAIL_SCHEMA,
        summary_schema=POWER_CURVE_SUMMARY_SCHEMA,
        validate_metadata=validate_metadata,
    )


if __name__ == "__main__":
    app()
