"""Generate figures for the second synthetic DGP (appendix)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer

from scripts._io import read_csv
from scripts._plot_utils import group_by_mode
from scripts._repo import MANUSCRIPT_DIR
from scripts.manuscript_style import MODE_STYLE

app = typer.Typer()


def plot_line_figure(
    rows: list[dict[str, Any]],
    *,
    x_key: str,
    y_key: str,
    xlabel: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    for mode, (x_values, y_values) in group_by_mode(rows, x_key, y_key).items():
        style = MODE_STYLE[mode]
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linewidth=2.0,
            markersize=6.0,
            label=style["label"],
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


@app.command()
def main(
    calibration_summary: Path = typer.Option(
        MANUSCRIPT_DIR / "results/second_dgp_calibration_summary.csv",
    ),
    power_summary: Path = typer.Option(
        MANUSCRIPT_DIR / "results/second_dgp_power_summary.csv",
    ),
    calibration_output: Path = typer.Option(
        MANUSCRIPT_DIR / "figures/second_dgp_calibration_plot.pdf",
    ),
    power_output: Path = typer.Option(
        MANUSCRIPT_DIR / "figures/second_dgp_power_plot.pdf",
    ),
) -> None:
    calibration_rows = read_csv(calibration_summary)
    power_rows = read_csv(power_summary)
    plot_line_figure(
        calibration_rows,
        x_key="overlap_severity",
        y_key="reject",
        xlabel="Low-overlap severity",
        ylabel="False positive rate",
        title="Second DGP: Calibration under support mismatch",
        output_path=calibration_output,
    )
    plot_line_figure(
        power_rows,
        x_key="effect_size",
        y_key="reject",
        xlabel="Harmful-shift effect size on common support",
        ylabel="Rejection rate",
        title="Second DGP: Power under harmful shift on common support",
        output_path=power_output,
    )


if __name__ == "__main__":
    app()
