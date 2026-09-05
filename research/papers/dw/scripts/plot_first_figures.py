"""Generate the first manuscript figures from summary CSV outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from scripts._io import read_csv
from scripts._plot_utils import group_by_mode, plot_lines_on_axis
from scripts._repo import MANUSCRIPT_DIR
from scripts.manuscript_style import MODE_ORDER, MODE_STYLE
from scripts.result_schemas import (
    POWER_CURVE_SUMMARY_SCHEMA,
    SYNTHETIC_CALIBRATION_SUMMARY_SCHEMA,
)

app = typer.Typer()


def plot_calibration_figure(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    detail_rows: list[dict[str, Any]] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    plot_lines_on_axis(
        ax, rows,
        modes=MODE_ORDER,
        x_key=x_key, y_key=y_key,
        xlabel=xlabel, ylabel=ylabel,
        show_legend=True,
        legend_ncol=2,
        detail_rows=detail_rows,
    )
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_power_figure(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
    detail_rows: list[dict[str, Any]] | None = None,
) -> None:
    riw_modes = ["unweighted", "source", "target", "both"]
    baseline_modes = ["unweighted", "crump", "overlap"]

    # 3-panel layout: (a) RIW family, (b) causal baselines, (c) MW statistic
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), constrained_layout=True)

    grouped = group_by_mode(rows, x_key, y_key)

    plot_lines_on_axis(
        axes[0], rows,
        modes=riw_modes, x_key=x_key, y_key=y_key,
        xlabel=xlabel, ylabel="Rejection rate",
        show_legend=False,
        detail_rows=detail_rows,
    )
    axes[0].set_title("(a) Density-ratio weighting")

    plot_lines_on_axis(
        axes[1], rows,
        modes=baseline_modes, x_key=x_key, y_key=y_key,
        xlabel=xlabel, ylabel="Rejection rate",
        show_legend=True,
        detail_rows=detail_rows,
    )
    axes[1].set_title("(b) Causal-inference baselines")

    # Panel (c): Mann-Whitney statistic (shared scale)
    stat_grouped = group_by_mode(rows, x_key, "statistic")
    for mode in MODE_ORDER:
        if mode not in stat_grouped:
            continue
        x_values, y_values = stat_grouped[mode]
        style = MODE_STYLE[mode]
        axes[2].plot(
            x_values, y_values,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=6.0,
            label=style["label"],
        )
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel("Mann-Whitney statistic")
    axes[2].set_title("(c) Test statistic (shared scale)")
    axes[2].grid(alpha=0.25, linewidth=0.8)
    axes[2].legend(frameon=False, ncol=2, fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


@app.command()
def main(
    calibration_summary: Path = typer.Option(
        MANUSCRIPT_DIR / "results/synthetic_calibration_summary.csv",
    ),
    power_summary: Path = typer.Option(
        MANUSCRIPT_DIR / "results/power_curve_summary.csv",
    ),
    calibration_detail: Path = typer.Option(None),
    power_detail: Path = typer.Option(None),
    calibration_output: Path = typer.Option(
        MANUSCRIPT_DIR / "figures/calibration_plot.pdf",
    ),
    power_output: Path = typer.Option(
        MANUSCRIPT_DIR / "figures/power_plot.pdf",
    ),
) -> None:
    calibration_rows = read_csv(
        calibration_summary,
        schema=SYNTHETIC_CALIBRATION_SUMMARY_SCHEMA,
    )
    power_rows = read_csv(power_summary, schema=POWER_CURVE_SUMMARY_SCHEMA)
    calibration_detail_rows = read_csv(calibration_detail) if calibration_detail else None
    power_detail_rows = read_csv(power_detail) if power_detail else None

    plot_calibration_figure(
        calibration_rows,
        x_key="severity",
        y_key="reject",
        xlabel="Low-overlap severity",
        ylabel="False positive rate",
        title="Calibration under support mismatch",
        output_path=calibration_output,
        detail_rows=calibration_detail_rows,
    )
    plot_power_figure(
        power_rows,
        x_key="effect_size",
        y_key="reject",
        xlabel="Effect size on common support",
        ylabel="Rejection rate",
        title="",
        output_path=power_output,
        detail_rows=power_detail_rows,
    )


if __name__ == "__main__":
    app()
