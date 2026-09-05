"""Compare weighting modes under asymmetric contamination patterns."""

from __future__ import annotations

from pathlib import Path

import typer

from scripts._dgp import draw_overlap_dataset
from scripts._repo import RESULTS_DIR, result_metadata
from scripts.common import MODES, run_harm_test
from scripts.experiment_harness import SHARED_METRIC_KEYS, run_repeated_experiment
from scripts.result_schemas import (
    MODE_COMPARISON_DETAIL_SCHEMA,
    MODE_COMPARISON_SUMMARY_SCHEMA,
)
from scripts.verify_result_metadata import validate_metadata

app = typer.Typer()

SCENARIOS: dict[str, tuple[float, float]] = {
    "source_only": (0.25, 0.0),
    "target_only": (0.0, 0.25),
    "both_sides": (0.25, 0.25),
}


@app.command()
def main(
    n_source: int = typer.Option(180, help="Number of source observations"),
    n_target: int = typer.Option(180, help="Number of target observations"),
    n_repeats: int = typer.Option(40, help="Number of repeated trials"),
    n_resamples: int = typer.Option(199, help="Permutation resamples"),
    lambda_value: float = typer.Option(0.5, help="RIW stabilisation parameter"),
    output: Path = typer.Option(
        RESULTS_DIR / "mode_comparison_summary.csv",
        help="Summary CSV output path",
    ),
    detail_output: Path = typer.Option(
        RESULTS_DIR / "mode_comparison_detail.csv",
        help="Detail CSV output path",
    ),
    metadata_output: Path = typer.Option(
        RESULTS_DIR / "mode_comparison_metadata.json",
        help="Metadata JSON output path",
    ),
) -> None:
    metadata = result_metadata(
        Path(__file__),
        {k: v for k, v in locals().items() if not k.startswith("_")},
        scenarios=SCENARIOS,
    )

    def run_repeat(repeat: int) -> list[dict[str, float | str]]:
        repeat_rows: list[dict[str, float | str]] = []
        for scenario, (source_fraction, target_fraction) in SCENARIOS.items():
            dataset = draw_overlap_dataset(
                n_source=n_source,
                n_target=n_target,
                source_private_fraction=source_fraction,
                target_private_fraction=target_fraction,
                target_shared_shift=0.0,
                seed=50_000 + repeat,
            )
            for mode in MODES:
                row = run_harm_test(
                    dataset["source_score"],
                    dataset["target_score"],
                    source_feature=dataset["source_feature"],
                    target_feature=dataset["target_feature"],
                    mode=mode,
                    lambda_value=lambda_value,
                    n_resamples=n_resamples,
                    seed=60_000 + repeat,
                )
                repeat_rows.append({"repeat": repeat, "scenario": scenario, **row})
        return repeat_rows

    run_repeated_experiment(
        n_repeats=n_repeats,
        run_repeat=run_repeat,
        group_keys=("scenario", "mode"),
        metric_keys=SHARED_METRIC_KEYS,
        detail_output=detail_output,
        summary_output=output,
        metadata_output=metadata_output,
        metadata=metadata,
        detail_schema=MODE_COMPARISON_DETAIL_SCHEMA,
        summary_schema=MODE_COMPARISON_SUMMARY_SCHEMA,
        validate_metadata=validate_metadata,
    )


if __name__ == "__main__":
    app()
