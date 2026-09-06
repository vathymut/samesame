"""Synthetic experiments: calibration, power, mode comparison, lambda sensitivity."""

from __future__ import annotations

from pathlib import Path

import typer

from scripts.dgp import draw_overlap_dataset
from scripts.experiments import run_harm_test, run_mode_grid
from scripts.utils import RESULTS_DIR, result_metadata, write_experiment_outputs

app = typer.Typer()

SCENARIOS: dict[str, tuple[float, float]] = {"source_only": (0.25, 0.0), "target_only": (0.0, 0.25), "both_sides": (0.25, 0.25)}


@app.command("calibration")
def calibration(
    n_source: int = typer.Option(180),
    n_target: int = typer.Option(180),
    n_repeats: int = typer.Option(40),
    n_resamples: int = typer.Option(199),
    lambda_value: float = typer.Option(0.5),
    severity_grid: list[float] = typer.Option([0.0, 0.1, 0.2, 0.3, 0.4]),
    output: Path = typer.Option(RESULTS_DIR / "synthetic_calibration_summary.csv"),
    detail_output: Path = typer.Option(RESULTS_DIR / "synthetic_calibration_detail.csv"),
    metadata_output: Path = typer.Option(RESULTS_DIR / "synthetic_calibration_metadata.json"),
) -> None:
    meta = result_metadata(Path(__file__), locals(), note="No harmful change on common support; mismatch only in low-overlap regions.")
    rows = run_mode_grid(
        n_repeats=n_repeats,
        values=severity_grid,
        draw=lambda sev, seed: draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=sev, target_private_fraction=sev, target_shared_shift=0.0, seed=seed),
        extra=lambda sev: {"severity": sev},
        lambda_value=lambda_value,
        n_resamples=n_resamples,
        draw_seed=10_000,
        test_seed=20_000,
    )
    write_experiment_outputs(detail=detail_output, summary=output, rows=rows, group_keys=("severity", "mode"), metadata=metadata_output, meta=meta)


@app.command("power")
def power(
    n_source: int = typer.Option(180),
    n_target: int = typer.Option(180),
    n_repeats: int = typer.Option(40),
    n_resamples: int = typer.Option(199),
    lambda_value: float = typer.Option(0.5),
    severity: float = typer.Option(0.25),
    effect_grid: list[float] = typer.Option([0.0, 0.15, 0.3, 0.45, 0.6]),
    output: Path = typer.Option(RESULTS_DIR / "power_curve_summary.csv"),
    detail_output: Path = typer.Option(RESULTS_DIR / "power_curve_detail.csv"),
    metadata_output: Path = typer.Option(RESULTS_DIR / "power_curve_metadata.json"),
) -> None:
    meta = result_metadata(Path(__file__), locals())
    rows = run_mode_grid(
        n_repeats=n_repeats,
        values=effect_grid,
        draw=lambda eff, seed: draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=severity, target_private_fraction=severity, target_shared_shift=eff, seed=seed),
        extra=lambda eff: {"effect_size": eff, "severity": severity},
        lambda_value=lambda_value,
        n_resamples=n_resamples,
        draw_seed=30_000,
        test_seed=40_000,
    )
    write_experiment_outputs(detail=detail_output, summary=output, rows=rows, group_keys=("effect_size", "mode"), metadata=metadata_output, meta=meta)


@app.command("mode-comparison")
def mode_comparison(
    n_source: int = typer.Option(180),
    n_target: int = typer.Option(180),
    n_repeats: int = typer.Option(40),
    n_resamples: int = typer.Option(199),
    lambda_value: float = typer.Option(0.5),
    output: Path = typer.Option(RESULTS_DIR / "mode_comparison_summary.csv"),
    detail_output: Path = typer.Option(RESULTS_DIR / "mode_comparison_detail.csv"),
    metadata_output: Path = typer.Option(RESULTS_DIR / "mode_comparison_metadata.json"),
) -> None:
    meta = result_metadata(Path(__file__), locals(), scenarios=SCENARIOS)
    rows = run_mode_grid(
        n_repeats=n_repeats,
        values=list(SCENARIOS.items()),
        draw=lambda item, seed: draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=item[1][0], target_private_fraction=item[1][1], target_shared_shift=0.0, seed=seed),
        extra=lambda item: {"scenario": item[0]},
        lambda_value=lambda_value,
        n_resamples=n_resamples,
        draw_seed=50_000,
        test_seed=60_000,
    )
    write_experiment_outputs(detail=detail_output, summary=output, rows=rows, group_keys=("scenario", "mode"), metadata=metadata_output, meta=meta)


@app.command("lambda")
def lambda_sensitivity(
    n_source: int = typer.Option(180),
    n_target: int = typer.Option(180),
    n_repeats: int = typer.Option(40),
    n_resamples: int = typer.Option(199),
    severity: float = typer.Option(0.25),
    effect_size: float = typer.Option(0.4),
    lambda_grid: list[float] = typer.Option([0.0, 0.25, 0.5, 0.75, 1.0]),
    output: Path = typer.Option(RESULTS_DIR / "lambda_sensitivity_summary.csv"),
    detail_output: Path = typer.Option(RESULTS_DIR / "lambda_sensitivity_detail.csv"),
    metadata_output: Path = typer.Option(RESULTS_DIR / "lambda_sensitivity_metadata.json"),
) -> None:
    meta = result_metadata(Path(__file__), locals(), experiments={"calibration": 0.0, "power": effect_size})
    rows: list[dict] = []
    for rep in range(n_repeats):
        for exp_name, eff in (("calibration", 0.0), ("power", effect_size)):
            ds = draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=severity, target_private_fraction=severity, target_shared_shift=eff, seed=70_000 + rep)
            base = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode="unweighted", lambda_value=0.5, n_resamples=n_resamples, seed=80_000 + rep)
            rows.append({"repeat": rep, "experiment": exp_name, "lambda_value": -1.0, **base})
            for lam in lambda_grid:
                r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode="both", lambda_value=lam, n_resamples=n_resamples, seed=90_000 + rep)
                rows.append({"repeat": rep, "experiment": exp_name, "lambda_value": lam, **r})
    write_experiment_outputs(detail=detail_output, summary=output, rows=rows, group_keys=("experiment", "mode", "lambda_value"), metadata=metadata_output, meta=meta)


if __name__ == "__main__":
    app()
