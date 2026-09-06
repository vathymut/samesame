"""Synthetic experiments: calibration, power, mode comparison, lambda sensitivity."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

from scripts.dgp import draw_overlap_dataset
from scripts.experiments import MODES, run_harm_test
from scripts.style import MODE_ORDER
from scripts.utils import RESULTS_DIR, result_metadata, write_csv, write_json

app = typer.Typer()

METRIC_KEYS = ("statistic", "pvalue", "reject", "source_ess", "target_ess", "source_max_weight", "target_max_weight")
SCENARIOS: dict[str, tuple[float, float]] = {"source_only": (0.25, 0.0), "target_only": (0.0, 0.25), "both_sides": (0.25, 0.25)}


def _summarize(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    df = pl.DataFrame(rows)
    agg = [pl.col(k).mean().alias(k) for k in METRIC_KEYS] + [pl.len().alias("count")]
    return df.group_by(group_keys).agg(agg).sort(group_keys).to_dicts()


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
    rows: list[dict] = []
    for rep in range(n_repeats):
        for sev in severity_grid:
            ds = draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=sev, target_private_fraction=sev, target_shared_shift=0.0, seed=10_000 + rep)
            for mode in MODE_ORDER:
                r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=20_000 + rep)
                rows.append({"repeat": rep, "severity": sev, **r})
    write_csv(detail_output, rows)
    write_csv(output, _summarize(rows, ("severity", "mode")))
    write_json(metadata_output, meta)


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
    rows: list[dict] = []
    for rep in range(n_repeats):
        for eff in effect_grid:
            ds = draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=severity, target_private_fraction=severity, target_shared_shift=eff, seed=30_000 + rep)
            for mode in MODE_ORDER:
                r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=40_000 + rep)
                rows.append({"repeat": rep, "effect_size": eff, "severity": severity, **r})
    write_csv(detail_output, rows)
    write_csv(output, _summarize(rows, ("effect_size", "mode")))
    write_json(metadata_output, meta)


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
    rows: list[dict] = []
    for rep in range(n_repeats):
        for scen, (sf, tf) in SCENARIOS.items():
            ds = draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=sf, target_private_fraction=tf, target_shared_shift=0.0, seed=50_000 + rep)
            for mode in MODES:
                r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=60_000 + rep)
                rows.append({"repeat": rep, "scenario": scen, **r})
    write_csv(detail_output, rows)
    write_csv(output, _summarize(rows, ("scenario", "mode")))
    write_json(metadata_output, meta)


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
        for exp_name, eff in [("calibration", 0.0), ("power", effect_size)]:
            ds = draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=severity, target_private_fraction=severity, target_shared_shift=eff, seed=70_000 + rep)
            base = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode="unweighted", lambda_value=0.5, n_resamples=n_resamples, seed=80_000 + rep)
            rows.append({"repeat": rep, "experiment": exp_name, "lambda_value": -1.0, **base})
            for lam in lambda_grid:
                r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode="both", lambda_value=lam, n_resamples=n_resamples, seed=90_000 + rep)
                rows.append({"repeat": rep, "experiment": exp_name, "lambda_value": lam, **r})
    write_csv(detail_output, rows)
    write_csv(output, _summarize(rows, ("experiment", "mode", "lambda_value")))
    write_json(metadata_output, meta)


if __name__ == "__main__":
    app()
