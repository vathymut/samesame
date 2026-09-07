"""OpenML-backed real-data workflow — heloc-anchored comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import typer
from numpy.typing import NDArray
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from skrub import tabular_pipeline

from scripts.datasets import load_task
from scripts.experiments import (
    DEFAULT_HGB_PARAMS,
    estimate_domain_probabilities_hgb,
    run_weighted_harm_test,
)
from scripts.style import MODE_ORDER
from scripts.utils import RESULTS_DIR, result_metadata, write_csv, write_json

app = typer.Typer()

WORKFLOW_ORDER = ("risk", "confidence", "error")
WORKFLOW_LABELS = {"risk": "Predicted risk", "confidence": "Model confidence", "error": "Prediction error"}
WORKFLOW_DIRECTIONS = {"risk": "higher", "confidence": "lower", "error": "higher"}

# Q18B/Q19: label scoring is cross-fitted OOS on the id pool plus a full
# refit for ood. Source (id) scores come from 90%-fold models; target (ood)
# scores come from the 100%-fit model. The capacity mismatch is expected
# and recorded in the manifest (LABEL_SCORING).
CV_N_SPLITS = 10
CV_SEED = 123_456


def cross_fitted_id_ood_probas(
    id_feature, id_label, ood_feature, *, seed: int = CV_SEED, n_splits: int = CV_N_SPLITS
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (id_oos_proba_1, ood_proba_1) for binary tasks.

    id side is 10-fold out-of-sample via ``cross_val_predict``;
    ood side is a final refit on 100% of id (Q18B).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    base = tabular_pipeline(HistGradientBoostingClassifier(random_state=seed, **DEFAULT_HGB_PARAMS))
    id_oos = np.asarray(
        cross_val_predict(base, id_feature, id_label, cv=skf, method="predict_proba"),
        dtype=np.float64,
    )
    final = tabular_pipeline(HistGradientBoostingClassifier(random_state=seed, **DEFAULT_HGB_PARAMS))
    final.fit(id_feature, id_label)
    ood_proba = np.asarray(final.predict_proba(ood_feature), dtype=np.float64)
    return id_oos[:, 1], ood_proba[:, 1]


def workflow_scores_from_probas(
    id_proba_1: NDArray[np.float64],
    ood_proba_1: NDArray[np.float64],
    id_label,
    ood_label,
) -> dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]]:
    id_label_f = np.asarray(id_label, dtype=np.float64)
    ood_label_f = np.asarray(ood_label, dtype=np.float64)
    id_stack = np.column_stack([1.0 - id_proba_1, id_proba_1])
    ood_stack = np.column_stack([1.0 - ood_proba_1, ood_proba_1])
    return {
        "risk": (id_proba_1, ood_proba_1),
        "confidence": (_conf_from_proba(id_stack), _conf_from_proba(ood_stack)),
        "error": (np.square(id_label_f - id_proba_1), np.square(ood_label_f - ood_proba_1)),
    }


@dataclass(frozen=True, slots=True)
class TaskSpec:
    label: str
    short_label: str
    domain: str


TASK_SPECS: dict[str, TaskSpec] = {
    "heloc": TaskSpec("FICO HELOC", "HELOC", "credit risk"),
    "diabetes_readmission": TaskSpec("Hospital readmission", "Readmission", "healthcare"),
    "acsincome": TaskSpec("ACS income", "Income", "socioeconomic screening"),
    "acspubcov": TaskSpec("ACS public coverage", "Coverage", "public benefits screening"),
    "nsw": TaskSpec("NSW employment program", "NSW", "employment policy"),
}
TASK_ORDER: tuple[str, ...] = ("heloc", "diabetes_readmission", "acsincome", "acspubcov")


def _logit_gap(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    m = np.max(logits, axis=1)
    return m - (np.sum(logits, axis=1) - m) / (logits.shape[1] - 1)


def _conf_from_proba(proba: NDArray[np.float64]) -> NDArray[np.float64]:
    return _logit_gap(logit(np.clip(proba, 1e-6, 1 - 1e-6)))


def _evaluate_task(task: str, *, max_train_rows: int, max_eval_rows: int, lambda_value: float, n_resamples: int, seed: int):
    spec = TASK_SPECS[task]
    td = load_task(task, max_train_rows=max_train_rows, max_eval_rows=max_eval_rows, seed=seed)
    s_p, t_p = estimate_domain_probabilities_hgb(td.source_feature, td.target_feature)
    if task == "nsw":
        wf_scores = {"outcome": (td.source_label.astype(float), td.target_label.astype(float))}
        wf_order, wf_dirs, train_n = ("outcome",), {"outcome": "lower"}, 0
    else:
        model = tabular_pipeline(HistGradientBoostingClassifier(random_state=seed, **DEFAULT_HGB_PARAMS))
        model.fit(td.train_feature, td.train_label)
        s_prob = np.asarray(model.predict_proba(td.source_feature), dtype=np.float64)
        t_prob = np.asarray(model.predict_proba(td.target_feature), dtype=np.float64)
        wf_scores = {
            "risk": (s_prob[:, 1], t_prob[:, 1]),
            "confidence": (_conf_from_proba(s_prob), _conf_from_proba(t_prob)),
            "error": (np.square(td.source_label - s_prob[:, 1]), np.square(td.target_label - t_prob[:, 1])),
        }
        wf_order, wf_dirs, train_n = WORKFLOW_ORDER, WORKFLOW_DIRECTIONS, len(td.train_label)
    rows: list[dict] = []
    for wi, wf in enumerate(wf_order):
        s_score, t_score = wf_scores[wf]
        direction = wf_dirs[wf]
        for mi, mode in enumerate(MODE_ORDER):
            r = run_weighted_harm_test(s_score, t_score, direction=direction, source_domain_prob=s_p, target_domain_prob=t_p, mode=mode, lambda_value=lambda_value, n_resamples=n_resamples, seed=seed + 100 * wi + mi, alpha=0.05)
            rows.append({"task": task, "task_label": spec.label, "task_short_label": spec.short_label, "task_domain": spec.domain, "workflow": wf, "direction": direction, "train_size": train_n, "source_size": int(len(td.source_label)), "target_size": int(len(td.target_label)), "source_mean": float(np.mean(s_score)), "target_mean": float(np.mean(t_score)), **r})
    return rows, {"task": task, **asdict(spec), "train_size": train_n, "source_size": int(len(td.source_label)), "target_size": int(len(td.target_label))}


@app.command()
def main(
    tasks: list[str] = typer.Option(list(TASK_ORDER)),
    n_resamples: int = typer.Option(9999),
    random_seed: int = typer.Option(123_456),
    lambda_value: float = typer.Option(0.5),
    max_train_rows: int = typer.Option(30_000),
    max_eval_rows: int = typer.Option(4_000),
    output: Path = typer.Option(RESULTS_DIR / "real_data_workflow_summary.csv"),
    metadata_output: Path = typer.Option(RESULTS_DIR / "real_data_workflow_metadata.json"),
) -> None:
    meta = result_metadata(Path(__file__), locals(), workflows=WORKFLOW_DIRECTIONS)
    tasks = list(dict.fromkeys(tasks))
    unknown = set(tasks) - set(TASK_SPECS)
    if unknown:
        raise ValueError(f"unknown tasks {unknown!r}; expected {sorted(TASK_SPECS)}")
    rows: list[dict] = []
    task_metas: list[dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        tr, tm = _evaluate_task(task, max_train_rows=max_train_rows, max_eval_rows=max_eval_rows, lambda_value=lambda_value, n_resamples=n_resamples, seed=random_seed + 1000 * idx)
        rows.extend(tr)
        task_metas.append(tm)
    write_csv(output, rows)
    write_json(metadata_output, {**meta, "tasks": tasks, "task_metadata": task_metas})


if __name__ == "__main__":
    app()
