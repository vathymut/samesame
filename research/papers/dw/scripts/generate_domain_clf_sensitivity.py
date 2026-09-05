"""Calibration comparison across domain classifier choices."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import typer
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from skrub import tabular_pipeline

from scripts._dgp import draw_overlap_dataset
from scripts._domain_clf import (
    DEFAULT_DOMAIN_CV,
    clip_domain_probabilities,
    estimate_domain_probabilities_hgb,
)
from scripts._repo import RESULTS_DIR, result_metadata
from scripts.common import MODES_WITH_BASELINES
from scripts.experiment_harness import SHARED_METRIC_KEYS, run_repeated_experiment
from scripts.result_schemas import (
    DOMAIN_CLF_SENSITIVITY_DETAIL_SCHEMA,
    DOMAIN_CLF_SENSITIVITY_SUMMARY_SCHEMA,
)
from scripts.verify_result_metadata import validate_metadata
from scripts.weighting import run_harm_test_with_estimator

app = typer.Typer()


def _rf_domain_probs_cv(
    source_feature: NDArray[np.float64],
    target_feature: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    estimator = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        random_state=42,
    )
    pipeline = tabular_pipeline(estimator)
    source_p = (
        pl.from_numpy(source_feature.reshape(-1, 1), schema=["feature"])
        if source_feature.ndim == 1
        else pl.from_numpy(source_feature)
    )
    target_p = (
        pl.from_numpy(target_feature.reshape(-1, 1), schema=["feature"])
        if target_feature.ndim == 1
        else pl.from_numpy(target_feature)
    )
    feature = pl.concat([source_p, target_p], how="vertical")
    group = np.concatenate(
        [np.zeros(len(source_feature)), np.ones(len(target_feature))]
    )
    folds = min(DEFAULT_DOMAIN_CV, len(source_feature), len(target_feature))
    probability = cross_val_predict(
        pipeline, feature.to_numpy(), group, cv=folds, method="predict_proba"
    )[:, 1]
    prob = clip_domain_probabilities(probability)
    return prob[: len(source_feature)], prob[len(source_feature) :]


CLASSIFIERS = {
    "random_forest": _rf_domain_probs_cv,
    "hist_gbm": estimate_domain_probabilities_hgb,
}


@app.command()
def main(
    n_source: int = typer.Option(180, help="Number of source observations"),
    n_target: int = typer.Option(180, help="Number of target observations"),
    n_repeats: int = typer.Option(40, help="Number of repeated trials"),
    n_resamples: int = typer.Option(199, help="Permutation resamples"),
    lambda_value: float = typer.Option(0.5, help="RIW stabilisation parameter"),
    severity_grid: list[float] = typer.Option(
        [0.0, 0.1, 0.2, 0.3, 0.4],
        help="Grid of low-overlap severity values",
    ),
    output: Path = typer.Option(
        RESULTS_DIR / "domain_clf_sensitivity_summary.csv",
        help="Summary CSV output path",
    ),
    detail_output: Path = typer.Option(
        RESULTS_DIR / "domain_clf_sensitivity_detail.csv",
        help="Detail CSV output path",
    ),
    metadata_output: Path = typer.Option(
        RESULTS_DIR / "domain_clf_sensitivity_metadata.json",
        help="Metadata JSON output path",
    ),
) -> None:
    metadata = result_metadata(
        Path(__file__),
        {k: v for k, v in locals().items() if not k.startswith("_")},
        classifiers=list(CLASSIFIERS),
    )

    def run_repeat(repeat: int) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for severity in severity_grid:
            dataset = draw_overlap_dataset(
                n_source=n_source,
                n_target=n_target,
                source_private_fraction=severity,
                target_private_fraction=severity,
                target_shared_shift=0.0,
                seed=70_000 + repeat,
            )
            for clf_name in CLASSIFIERS:
                for mode in MODES_WITH_BASELINES:
                    row = run_harm_test_with_estimator(
                        dataset["source_score"],
                        dataset["target_score"],
                        source_feature=dataset["source_feature"],
                        target_feature=dataset["target_feature"],
                        estimator=CLASSIFIERS[clf_name],
                        direction="higher",
                        mode=mode,
                        lambda_value=lambda_value,
                        n_resamples=n_resamples,
                        seed=80_000 + repeat,
                        alpha=0.05,
                    )
                    rows.append(
                        {
                            "repeat": repeat,
                            "severity": severity,
                            "classifier": clf_name,
                            **row,
                        }
                    )
        return rows

    run_repeated_experiment(
        n_repeats=n_repeats,
        run_repeat=run_repeat,
        group_keys=("severity", "classifier", "mode"),
        metric_keys=SHARED_METRIC_KEYS,
        detail_output=detail_output,
        summary_output=output,
        metadata_output=metadata_output,
        metadata=metadata,
        detail_schema=DOMAIN_CLF_SENSITIVITY_DETAIL_SCHEMA,
        summary_schema=DOMAIN_CLF_SENSITIVITY_SUMMARY_SCHEMA,
        validate_metadata=validate_metadata,
    )


if __name__ == "__main__":
    app()
