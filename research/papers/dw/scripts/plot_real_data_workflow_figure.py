"""Plot the OpenML-backed mirrored real-data workflow figure from summary CSV output."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from scripts._io import read_csv
from scripts._repo import MANUSCRIPT_DIR
from scripts.manuscript_style import MODE_COLORS, MODE_LABELS, MODE_ORDER
from scripts.real_data_workflow_config import (
    DEFAULT_SPOTLIGHT_TASK,
    INITIAL_TASK_ORDER,
    TASK_SPECS,
    WORKFLOW_LABELS,
    WORKFLOW_ORDER,
)
from scripts.result_schemas import REAL_DATA_WORKFLOW_SUMMARY_SCHEMA

app = typer.Typer()


def select_row(
    rows: list[dict[str, Any]],
    *,
    task: str,
    workflow: str,
    mode: str,
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if row["task"] == task and row["workflow"] == workflow and row["mode"] == mode
    )


def task_order(rows: list[dict[str, Any]]) -> list[str]:
    present = {row["task"] for row in rows}
    ordered = [task for task in INITIAL_TASK_ORDER if task in present]
    extras = sorted(present.difference(ordered))
    return ordered + extras


def plot_task_spotlight(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    spotlight_task: str,
) -> None:
    available_tasks = {row["task"] for row in rows}
    if spotlight_task not in available_tasks:
        listed = ", ".join(sorted(available_tasks))
        raise ValueError(
            f"spotlight task {spotlight_task!r} is not present in summary; "
            f"available tasks: {listed}"
        )
    positions = np.arange(len(WORKFLOW_ORDER))
    width = 0.18
    for index, mode in enumerate(MODE_ORDER):
        offset = (index - (len(MODE_ORDER) - 1) / 2.0) * width
        pvalues = [
            float(
                select_row(
                    rows,
                    task=spotlight_task,
                    workflow=workflow,
                    mode=mode,
                )["pvalue"]
            )
            for workflow in WORKFLOW_ORDER
        ]
        ax.bar(
            positions + offset,
            pvalues,
            width=width,
            color=MODE_COLORS[mode],
            label=MODE_LABELS[mode],
        )
    ax.axhline(0.05, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_xticks(positions, [WORKFLOW_LABELS[name] for name in WORKFLOW_ORDER])
    ax.set_ylabel("Harm-test p-value")
    ax.set_title(f"{TASK_SPECS[spotlight_task].short_label} spotlight")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper right")


def plot_comparison_summary(
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    spotlight_task: str | None = None,
) -> None:
    if spotlight_task is not None:
        tasks = [task for task in task_order(rows) if task != spotlight_task]
    else:
        tasks = task_order(rows)
    row_keys = [(task, workflow) for task in tasks for workflow in WORKFLOW_ORDER]
    positions = np.arange(len(row_keys))
    offset_scale = np.linspace(-0.24, 0.24, num=len(MODE_ORDER))

    for offset, mode in zip(offset_scale, MODE_ORDER, strict=True):
        x_values = []
        y_values = []
        for position, (task, workflow) in enumerate(row_keys):
            try:
                row = select_row(rows, task=task, workflow=workflow, mode=mode)
            except StopIteration:
                continue
            x_values.append(float(row["pvalue"]))
            y_values.append(position + offset)
        ax.scatter(
            x_values,
            y_values,
            color=MODE_COLORS[mode],
            s=28,
            label=MODE_LABELS[mode],
        )

    ax.axvline(0.05, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Harm-test p-value")
    ax.set_title("Cross-task workflow summary")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)

    labels = [
        f"{TASK_SPECS[task].short_label} / {WORKFLOW_LABELS[workflow]}"
        for task, workflow in row_keys
    ]
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()

    for boundary in range(len(WORKFLOW_ORDER), len(row_keys), len(WORKFLOW_ORDER)):
        ax.axhline(boundary - 0.5, color="#dddddd", linewidth=0.8)


def plot_ess_diagnostics(ax: plt.Axes, rows: list[dict[str, Any]]) -> None:
    tasks = task_order(rows)
    n_tasks = len(tasks)
    positions = np.arange(n_tasks)
    width = 0.30

    source_values = []
    target_values = []
    for task in tasks:
        mode_rows = [
            row for row in rows if row["task"] == task and row["mode"] == "both"
        ]
        source_values.append(min(float(row["source_ess"]) for row in mode_rows))
        target_values.append(min(float(row["target_ess"]) for row in mode_rows))

    ax.bar(
        positions - width / 2,
        source_values,
        width=width,
        color=MODE_COLORS["source"],
        label="Source ESS",
    )
    ax.bar(
        positions + width / 2,
        target_values,
        width=width,
        color=MODE_COLORS["target"],
        label="Target ESS",
    )

    ax.set_xticks(positions, [TASK_SPECS[task].short_label for task in tasks])
    ax.set_ylabel("Effective sample size")
    ax.set_title("Common-support retention (doubly weighted)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)


def plot_heloc_motivating_example(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    task: str = "heloc",
    workflow: str = "risk",
) -> None:
    modes = list(reversed(MODE_ORDER))
    pvalues = [
        float(select_row(rows, task=task, workflow=workflow, mode=mode)["pvalue"])
        for mode in modes
    ]
    colors = [MODE_COLORS[mode] for mode in modes]
    labels = [MODE_LABELS[mode] for mode in modes]

    figure, ax = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
    positions = np.arange(len(modes))
    ax.barh(positions, pvalues, color=colors, height=0.55, zorder=2)
    ax.axvline(0.05, color="#666666", linestyle="--", linewidth=1.2, zorder=3)
    ax.set_yticks(positions, labels)
    ax.set_xlabel("Harm-test p-value")
    ax.set_xlim(0.0, 1.0)
    ax.grid(axis="x", alpha=0.25, linewidth=0.8, zorder=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


def plot_real_data_workflow(
    rows: list[dict[str, Any]],
    *,
    output_path: Path,
    spotlight_task: str | None = None,
) -> None:
    # 2-panel layout: (a) cross-task scatter, (b) ESS diagnostic
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(7.0, 6.5),
        constrained_layout=True,
        height_ratios=(2.0, 1.0),
    )

    plot_comparison_summary(axes[0], rows, spotlight_task=spotlight_task)
    plot_ess_diagnostics(axes[1], rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)


@app.command()
def main(
    summary: Path = typer.Option(
        MANUSCRIPT_DIR / "results/real_data_workflow_summary.csv",
    ),
    output: Path = typer.Option(
        MANUSCRIPT_DIR / "figures/real_data_workflow_plot.pdf",
    ),
    spotlight_task: str = typer.Option(None),
    motivating_example_output: Path = typer.Option(None),
) -> None:
    rows = read_csv(summary, schema=REAL_DATA_WORKFLOW_SUMMARY_SCHEMA)
    plot_real_data_workflow(
        rows,
        output_path=output,
        spotlight_task=spotlight_task,
    )
    if motivating_example_output is not None:
        plot_heloc_motivating_example(
            rows,
            output_path=motivating_example_output,
        )



if __name__ == "__main__":
    app()
