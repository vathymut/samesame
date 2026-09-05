"""Generate follow-up manuscript figures from summary CSV outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from scripts._io import min_ess, read_csv
from scripts._repo import MANUSCRIPT_DIR
from scripts.manuscript_style import (
    EXPERIMENT_STYLE,
    MODE_ORDER,
    MODE_STYLE,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
)
from scripts.result_schemas import (
    LAMBDA_SENSITIVITY_SUMMARY_SCHEMA,
    MODE_COMPARISON_SUMMARY_SCHEMA,
)

app = typer.Typer()


def plot_mode_comparison(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    positions = np.arange(len(SCENARIO_ORDER))
    width = 0.18
    for index, mode in enumerate(MODE_ORDER):
        offset = (index - (len(MODE_ORDER) - 1) / 2.0) * width
        values = []
        for scenario in SCENARIO_ORDER:
            match = next(
                row
                for row in rows
                if row["scenario"] == scenario and row["mode"] == mode
            )
            values.append(100.0 * float(match["reject"]))
        ax.bar(
            positions + offset,
            values,
            width=width,
            color=MODE_STYLE[mode]["color"],
            label=MODE_STYLE[mode]["label"],
        )
    ax.axhline(5.0, color="#666666", linestyle=":", linewidth=1.2)
    ax.set_xticks(positions, [SCENARIO_LABELS[name] for name in SCENARIO_ORDER])
    ax.set_ylabel("Rejection rate (%)")
    ax.set_title("Weighting modes under asymmetric contamination")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_lambda_sensitivity(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    fig, (rate_ax, ess_ax) = plt.subplots(
        1,
        2,
        figsize=(8.2, 3.6),
        constrained_layout=True,
        sharex=True,
    )
    for experiment, style in EXPERIMENT_STYLE.items():
        weighted_rows = sorted(
            [
                row
                for row in rows
                if row["experiment"] == experiment and row["mode"] == "both"
            ],
            key=lambda row: float(row["lambda_value"]),
        )
        lambda_values = [float(row["lambda_value"]) for row in weighted_rows]
        reject_values = [100.0 * float(row["reject"]) for row in weighted_rows]
        ess_values = [min_ess(row) for row in weighted_rows]
        rate_ax.plot(
            lambda_values,
            reject_values,
            color=style["color"],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label=f"{style['label']}: doubly weighted",
        )
        ess_ax.plot(
            lambda_values,
            ess_values,
            color=style["color"],
            marker="o",
            linewidth=2.0,
            markersize=5.5,
            label=style["label"],
        )
        baseline = next(
            row
            for row in rows
            if row["experiment"] == experiment and row["mode"] == "unweighted"
        )
        rate_ax.axhline(
            100.0 * float(baseline["reject"]),
            color=style["color"],
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            label=f"{style['label']}: unweighted",
        )
    rate_ax.set_xlabel("Lambda")
    rate_ax.set_ylabel("Rejection rate (%)")
    rate_ax.set_title("Calibration and power")
    rate_ax.grid(alpha=0.25, linewidth=0.8)
    rate_ax.legend(frameon=False, fontsize=8)
    ess_ax.set_xlabel("Lambda")
    ess_ax.set_ylabel("Minimum effective sample size")
    ess_ax.set_title("Weight stability")
    ess_ax.grid(alpha=0.25, linewidth=0.8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


@app.command()
def main(
    mode_summary: Path = typer.Option(
        MANUSCRIPT_DIR / "results/mode_comparison_summary.csv",
    ),
    lambda_summary: Path = typer.Option(
        MANUSCRIPT_DIR / "results/lambda_sensitivity_summary.csv",
    ),
    mode_output: Path = typer.Option(
        MANUSCRIPT_DIR / "figures/mode_comparison_plot.pdf",
    ),
    lambda_output: Path = typer.Option(
        MANUSCRIPT_DIR / "figures/lambda_sensitivity_plot.pdf",
    ),
) -> None:
    plot_mode_comparison(
        read_csv(mode_summary, schema=MODE_COMPARISON_SUMMARY_SCHEMA),
        output_path=mode_output,
    )
    plot_lambda_sensitivity(
        read_csv(lambda_summary, schema=LAMBDA_SENSITIVITY_SUMMARY_SCHEMA),
        output_path=lambda_output,
    )


if __name__ == "__main__":
    app()
