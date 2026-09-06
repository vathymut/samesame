"""Appendix experiments: second DGP + domain-classifier sensitivity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import typer
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from skrub import tabular_pipeline

from scripts.dgp import draw_overlap_dataset, draw_second_dgp
from scripts.experiments import clip_domain_probabilities, estimate_domain_probabilities_hgb, run_harm_test, run_harm_test_with_estimator
from scripts.style import MODE_ORDER
from scripts.utils import RESULTS_DIR, result_metadata, summarize_rows, write_csv, write_json

app = typer.Typer()


def _rf_probs(source_feature, target_feature):
    est = RandomForestClassifier(n_estimators=500, max_features="sqrt", random_state=42)
    s = np.asarray(source_feature, dtype=float)
    t = np.asarray(target_feature, dtype=float)
    if s.ndim == 1:
        s = s.reshape(-1, 1)
        t = t.reshape(-1, 1)
    X = np.vstack([s, t])
    y = np.concatenate([np.zeros(len(s)), np.ones(len(t))])
    folds = min(5, len(s), len(t))
    prob = cross_val_predict(tabular_pipeline(est), X, y, cv=folds, method="predict_proba")[:, 1]
    clipped = clip_domain_probabilities(prob)
    return clipped[: len(s)], clipped[len(s) :]


@app.command("second-dgp")
def second_dgp(
    n_source: int = typer.Option(180),
    n_target: int = typer.Option(180),
    n_repeats: int = typer.Option(40),
    n_resamples: int = typer.Option(199),
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
    # calibration
    cal_rows: list[dict] = []
    for rep in range(n_repeats):
        for sev in severity_grid:
            ds = draw_second_dgp(n_source=n_source, n_target=n_target, overlap_severity=sev, effect_size=0.0, seed=90_000 + rep)
            for mode in MODE_ORDER:
                r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=91_000 + rep)
                cal_rows.append({"repeat": rep, "overlap_severity": sev, **r})
    # power
    pow_rows: list[dict] = []
    for rep in range(n_repeats):
        for eff in effect_grid:
            ds = draw_second_dgp(n_source=n_source, n_target=n_target, overlap_severity=0.25, effect_size=eff, seed=92_000 + rep)
            for mode in MODE_ORDER:
                r = run_harm_test(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=93_000 + rep)
                pow_rows.append({"repeat": rep, "effect_size": eff, **r})
    write_csv(calibration_detail_output, cal_rows)
    write_csv(calibration_output, summarize_rows(cal_rows, ("overlap_severity", "mode")))
    write_csv(power_detail_output, pow_rows)
    write_csv(power_output, summarize_rows(pow_rows, ("effect_size", "mode")))
    write_json(metadata_output, meta)


@app.command("domain-clf")
def domain_clf(
    n_source: int = typer.Option(180),
    n_target: int = typer.Option(180),
    n_repeats: int = typer.Option(40),
    n_resamples: int = typer.Option(199),
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
                    r = run_harm_test_with_estimator(ds["source_score"], ds["target_score"], source_feature=ds["source_feature"], target_feature=ds["target_feature"], estimator=est, direction="higher", mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=80_000 + rep, alpha=0.05)
                    rows.append({"repeat": rep, "severity": sev, "classifier": clf_name, **r})
    write_csv(detail_output, rows)
    write_csv(output, summarize_rows(rows, ("severity", "classifier", "mode")))
    write_json(metadata_output, meta)


if __name__ == "__main__":
    app()
