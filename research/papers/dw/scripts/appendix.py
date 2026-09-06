"""Appendix experiments: second DGP + domain-classifier sensitivity."""

from __future__ import annotations

from pathlib import Path

import typer
from sklearn.ensemble import RandomForestClassifier
from skrub import tabular_pipeline

from scripts.dgp import draw_overlap_dataset, draw_second_dgp
from scripts.experiments import (
    cross_fitted_domain_probs,
    estimate_domain_probabilities_hgb,
    run_harm_test,
    run_mode_grid,
)
from scripts.style import MODE_ORDER
from scripts.utils import RESULTS_DIR, result_metadata, write_experiment_outputs

app = typer.Typer()


def _rf_probs(source_feature, target_feature):
    n_s, n_t = len(source_feature), len(target_feature)
    return cross_fitted_domain_probs(
        source_feature,
        target_feature,
        make_estimator=lambda: tabular_pipeline(
            RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42)
        ),
        cv=min(5, n_s, n_t),
    )


@app.command("second-dgp")
def second_dgp(
    n_source: int = typer.Option(180),
    n_target: int = typer.Option(180),
    n_repeats: int = typer.Option(40),
    n_resamples: int = typer.Option(9999),
    lambda_value: float = typer.Option(0.5),
    severity_grid: list[float] = typer.Option([0.0, 0.1, 0.2, 0.3, 0.4]),
    effect_grid: list[float] = typer.Option([0.0, 0.15, 0.3, 0.45, 0.6]),
    calibration_output: Path = typer.Option(RESULTS_DIR / "second_dgp_calibration_summary.csv"),
    calibration_detail_output: Path = typer.Option(RESULTS_DIR / "second_dgp_calibration_detail.csv"),
    power_output: Path = typer.Option(RESULTS_DIR / "second_dgp_power_summary.csv"),
    power_detail_output: Path = typer.Option(RESULTS_DIR / "second_dgp_power_detail.csv"),
    metadata_output: Path = typer.Option(RESULTS_DIR / "second_dgp_metadata.json"),
) -> None:
    meta = result_metadata(Path(__file__), locals(), dgp="regression 2D")
    cal_rows = run_mode_grid(
        n_repeats=n_repeats,
        values=severity_grid,
        draw=lambda sev, seed: draw_second_dgp(n_source=n_source, n_target=n_target, overlap_severity=sev, effect_size=0.0, seed=seed),
        extra=lambda sev: {"overlap_severity": sev},
        lambda_value=lambda_value,
        n_resamples=n_resamples,
        draw_seed=90_000,
        test_seed=91_000,
    )
    pow_rows = run_mode_grid(
        n_repeats=n_repeats,
        values=effect_grid,
        draw=lambda eff, seed: draw_second_dgp(n_source=n_source, n_target=n_target, overlap_severity=0.25, effect_size=eff, seed=seed),
        extra=lambda eff: {"effect_size": eff},
        lambda_value=lambda_value,
        n_resamples=n_resamples,
        draw_seed=92_000,
        test_seed=93_000,
    )
    write_experiment_outputs(detail=calibration_detail_output, summary=calibration_output, rows=cal_rows, group_keys=("overlap_severity", "mode"), metadata=metadata_output, meta=meta)
    write_experiment_outputs(detail=power_detail_output, summary=power_output, rows=pow_rows, group_keys=("effect_size", "mode"))


@app.command("domain-clf")
def domain_clf(
    n_source: int = typer.Option(180),
    n_target: int = typer.Option(180),
    n_repeats: int = typer.Option(40),
    n_resamples: int = typer.Option(9999),
    lambda_value: float = typer.Option(0.5),
    severity_grid: list[float] = typer.Option([0.0, 0.1, 0.2, 0.3, 0.4]),
    output: Path = typer.Option(RESULTS_DIR / "domain_clf_sensitivity_summary.csv"),
    detail_output: Path = typer.Option(RESULTS_DIR / "domain_clf_sensitivity_detail.csv"),
    metadata_output: Path = typer.Option(RESULTS_DIR / "domain_clf_sensitivity_metadata.json"),
) -> None:
    classifiers = {"random_forest": _rf_probs, "hist_gbm": estimate_domain_probabilities_hgb}
    meta = result_metadata(Path(__file__), locals(), classifiers=list(classifiers))
    rows: list[dict] = []
    for rep in range(n_repeats):
        for sev in severity_grid:
            ds = draw_overlap_dataset(n_source=n_source, n_target=n_target, source_private_fraction=sev, target_private_fraction=sev, target_shared_shift=0.0, seed=70_000 + rep)
            for clf_name, est in classifiers.items():
                for mode in MODE_ORDER:
                    r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], estimator=est, mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=80_000 + rep)
                    rows.append({"repeat": rep, "severity": sev, "classifier": clf_name, **r})
    write_experiment_outputs(detail=detail_output, summary=output, rows=rows, group_keys=("severity", "classifier", "mode"), metadata=metadata_output, meta=meta)


if __name__ == "__main__":
    app()
