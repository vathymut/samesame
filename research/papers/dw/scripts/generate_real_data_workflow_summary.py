"""Generate the OpenML-backed mirrored real-data workflow summary for Figure 5."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import typer
from numpy.typing import NDArray
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from skrub import tabular_pipeline

from scripts._domain_clf import DEFAULT_HGB_PARAMS, estimate_domain_probabilities_hgb
from scripts._io import write_csv, write_json
from scripts._repo import RESULTS_DIR, result_metadata
from scripts.manuscript_style import MODE_ORDER
from scripts.real_data_workflow_config import (
    INITIAL_TASK_ORDER,
    TASK_SPECS,
    WORKFLOW_DIRECTIONS,
    WORKFLOW_ORDER,
)
from scripts.real_data_workflow_sources import load_task
from scripts.result_schemas import REAL_DATA_WORKFLOW_SUMMARY_SCHEMA
from scripts.verify_result_metadata import validate_metadata
from scripts.weighting import run_weighted_harm_test

app = typer.Typer()


def unique_tasks(task_names: list[str]) -> list[str]:
    return list(dict.fromkeys(task_names))


def logit_gap(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    max_logits = np.max(logits, axis=1)
    mean_rest = (np.sum(logits, axis=1) - max_logits) / (logits.shape[1] - 1)
    return max_logits - mean_rest


def confidence_scores_from_probabilities(
    probabilities: NDArray[np.float64],
) -> NDArray[np.float64]:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1.0 - 1e-6)
    return logit_gap(logit(clipped))


def evaluate_task(
    task_name: str,
    *,
    max_train_rows: int,
    max_eval_rows: int,
    lambda_value: float,
    n_resamples: int,
    random_seed: int,
) -> tuple[list[dict[str, float | str]], dict[str, Any]]:
    task_spec = TASK_SPECS[task_name]
    task_data = load_task(
        task_name,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
        seed=random_seed,
    )

    source_domain_prob, target_domain_prob = estimate_domain_probabilities_hgb(
        task_data.source_feature,
        task_data.target_feature,
    )

    if task_name == "nsw":
        source_score = task_data.source_label.astype(np.float64)
        target_score = task_data.target_label.astype(np.float64)
        workflow_order = ("outcome",)
        workflow_directions = {"outcome": "lower"}
        workflow_scores = {"outcome": (source_score, target_score)}
        train_size = 0
    else:
        estimator = HistGradientBoostingClassifier(
            random_state=random_seed,
            **DEFAULT_HGB_PARAMS,
        )
        model = tabular_pipeline(estimator)
        model.fit(task_data.train_feature, task_data.train_label)

        source_probability_matrix = np.asarray(
            model.predict_proba(task_data.source_feature),
            dtype=np.float64,
        )
        target_probability_matrix = np.asarray(
            model.predict_proba(task_data.target_feature),
            dtype=np.float64,
        )

        workflow_scores = {
            "risk": (
                source_probability_matrix[:, 1],
                target_probability_matrix[:, 1],
            ),
            "confidence": (
                confidence_scores_from_probabilities(source_probability_matrix),
                confidence_scores_from_probabilities(target_probability_matrix),
            ),
            "error": (
                np.square(task_data.source_label - source_probability_matrix[:, 1]),
                np.square(task_data.target_label - target_probability_matrix[:, 1]),
            ),
        }
        workflow_order = WORKFLOW_ORDER
        workflow_directions = WORKFLOW_DIRECTIONS
        train_size = int(len(task_data.train_label))

    rows: list[dict[str, float | str]] = []
    for workflow_index, workflow in enumerate(workflow_order):
        source_score, target_score = workflow_scores[workflow]
        direction = workflow_directions[workflow]
        for mode_index, mode in enumerate(MODE_ORDER):
            test_summary = run_weighted_harm_test(
                source_score,
                target_score,
                direction=direction,
                source_domain_prob=source_domain_prob,
                target_domain_prob=target_domain_prob,
                mode=mode,
                lambda_value=lambda_value,
                n_resamples=n_resamples,
                seed=random_seed + 100 * workflow_index + mode_index,
                alpha=0.05,
            )
            rows.append(
                {
                    "task": task_name,
                    "task_label": task_spec.label,
                    "task_short_label": task_spec.short_label,
                    "task_domain": task_spec.domain,
                    "task_role": task_spec.narrative_role,
                    "data_source": task_spec.data_source,
                    "shift_variable": task_spec.shift_variable,
                    "source_definition": task_spec.source_definition,
                    "target_definition": task_spec.target_definition,
                    "label_definition": task_spec.label_definition,
                    "workflow": workflow,
                    "direction": direction,
                    "train_size": train_size,
                    "source_size": int(len(task_data.source_label)),
                    "target_size": int(len(task_data.target_label)),
                    "source_mean": float(np.mean(source_score)),
                    "target_mean": float(np.mean(target_score)),
                    "source_std": float(np.std(source_score)),
                    "target_std": float(np.std(target_score)),
                    **test_summary,
                }
            )

    metadata = {
        **asdict(task_spec),
        "train_size": train_size,
        "source_size": int(len(task_data.source_label)),
        "target_size": int(len(task_data.target_label)),
    }
    return rows, metadata


@app.command()
def main(
    ctx: typer.Context,
    tasks: list[str] = typer.Option(
        list(INITIAL_TASK_ORDER),
        help="OpenML-mirrored TableShift task identifiers to evaluate.",
    ),
    n_resamples: int = typer.Option(499, help="Permutation resamples"),
    random_seed: int = typer.Option(123_456, help="Base random seed"),
    lambda_value: float = typer.Option(0.5, help="RIW stabilisation parameter"),
    max_train_rows: int = typer.Option(30_000, help="Max training rows per task"),
    max_eval_rows: int = typer.Option(4_000, help="Max evaluation rows per task"),
    output: Path = typer.Option(
        RESULTS_DIR / "real_data_workflow_summary.csv",
        help="Summary CSV output path",
    ),
    metadata_output: Path = typer.Option(
        RESULTS_DIR / "real_data_workflow_metadata.json",
        help="Metadata JSON output path",
    ),
) -> None:
    run_metadata = result_metadata(
        Path(__file__),
        ctx.params,
        note="This OpenML-backed workflow keeps the HELOC package-doc split as the anchor, fetches pinned OpenML mirrors when they preserve the needed task ingredients, and recreates the TableShift split rules locally without relying on the upstream TableShift runtime.",
        workflows=WORKFLOW_DIRECTIONS,
    )
    task_names = unique_tasks(list(tasks))
    unknown = sorted(set(task_names).difference(TASK_SPECS))
    if unknown:
        listed = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"unknown task(s) {unknown!r}; expected one of: {listed}")

    rows: list[dict[str, float | str]] = []
    task_metadata: list[dict[str, Any]] = []
    for task_index, task_name in enumerate(task_names):
        task_rows, task_metadata_row = evaluate_task(
            task_name,
            max_train_rows=max_train_rows,
            max_eval_rows=max_eval_rows,
            lambda_value=lambda_value,
            n_resamples=n_resamples,
            random_seed=random_seed + 1_000 * task_index,
        )
        rows.extend(task_rows)
        task_metadata.append(task_metadata_row)

    write_csv(output, rows, schema=REAL_DATA_WORKFLOW_SUMMARY_SCHEMA)
    write_json(
        metadata_output,
        {
            **run_metadata,
            "tasks": task_names,
            "task_metadata": task_metadata,
        },
    )
    metadata_errors = validate_metadata(metadata_output)
    if metadata_errors:
        listed = "\n".join(metadata_errors)
        raise ValueError(f"metadata validation failed:\n{listed}")


if __name__ == "__main__":
    app()
