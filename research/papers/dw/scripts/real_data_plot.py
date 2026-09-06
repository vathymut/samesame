"""Real-data workflow figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

from scripts.real_data import TASK_ORDER, TASK_SPECS, WORKFLOW_LABELS, WORKFLOW_ORDER
from scripts.style import MODE_COLORS, MODE_LABELS, MODE_ORDER
from scripts.utils import MANUSCRIPT_DIR, read_csv

app = typer.Typer()


def _select(rows, task: str, workflow: str, mode: str) -> dict[str, Any]:
    return next(r for r in rows if r["task"] == task and r["workflow"] == workflow and r["mode"] == mode)


def _task_order(rows) -> list[str]:
    present = {r["task"] for r in rows}
    ordered = [t for t in TASK_ORDER if t in present]
    return ordered + sorted(present - set(ordered))


def _plot_comparison(ax, rows, spotlight_task: str | None = None):
    tasks = [t for t in _task_order(rows) if t != spotlight_task] if spotlight_task else _task_order(rows)
    keys = [(t, w) for t in tasks for w in WORKFLOW_ORDER]
    pos = np.arange(len(keys))
    offsets = np.linspace(-0.24, 0.24, num=len(MODE_ORDER))
    for off, mode in zip(offsets, MODE_ORDER, strict=True):
        xs, ys = [], []
        for idx, (task, wf) in enumerate(keys):
            try:
                r = _select(rows, task, wf, mode)
            except StopIteration:
                continue
            xs.append(float(r["pvalue"]))
            ys.append(idx + off)
        ax.scatter(xs, ys, color=MODE_COLORS[mode], s=28, label=MODE_LABELS[mode])
    ax.axvline(0.05, color="#666666", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Harm-test p-value")
    ax.set_title("Cross-task workflow summary")
    ax.grid(axis="x", alpha=0.25, linewidth=0.8)
    ax.set_yticks(pos, [f"{TASK_SPECS[t].short_label} / {WORKFLOW_LABELS[w]}" for t, w in keys])
    ax.invert_yaxis()
    for b in range(len(WORKFLOW_ORDER), len(keys), len(WORKFLOW_ORDER)):
        ax.axhline(b - 0.5, color="#dddddd", linewidth=0.8)


def _plot_ess(ax, rows):
    tasks = _task_order(rows)
    pos = np.arange(len(tasks))
    w = 0.30
    src = [min(float(r["source_ess"]) for r in rows if r["task"] == t and r["mode"] == "both") for t in tasks]
    tgt = [min(float(r["target_ess"]) for r in rows if r["task"] == t and r["mode"] == "both") for t in tasks]
    ax.bar(pos - w / 2, src, width=w, color=MODE_COLORS["source"], label="Source ESS")
    ax.bar(pos + w / 2, tgt, width=w, color=MODE_COLORS["target"], label="Target ESS")
    ax.set_xticks(pos, [TASK_SPECS[t].short_label for t in tasks])
    ax.set_ylabel("Effective sample size")
    ax.set_title("Common-support retention (doubly weighted)")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)


@app.command()
def main(
    summary: Path = typer.Option(MANUSCRIPT_DIR / "results/real_data_workflow_summary.csv"),
    output: Path = typer.Option(MANUSCRIPT_DIR / "figures/real_data_workflow_plot.pdf"),
    spotlight_task: str = typer.Option(None),
    motivating_example_output: Path = typer.Option(None),
) -> None:
    rows = read_csv(summary)
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.5), constrained_layout=True, height_ratios=(2.0, 1.0))
    _plot_comparison(axes[0], rows, spotlight_task)
    _plot_ess(axes[1], rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    if motivating_example_output is not None:
        modes = list(reversed(MODE_ORDER))
        pvals = [float(_select(rows, "heloc", "risk", m)["pvalue"]) for m in modes]
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.2), constrained_layout=True)
        pos = np.arange(len(modes))
        ax2.barh(pos, pvals, color=[MODE_COLORS[m] for m in modes], height=0.55, zorder=2)
        ax2.axvline(0.05, color="#666666", linestyle="--", linewidth=1.2, zorder=3)
        ax2.set_yticks(pos, [MODE_LABELS[m] for m in modes])
        ax2.set_xlabel("Harm-test p-value")
        ax2.set_xlim(0.0, 1.0)
        ax2.grid(axis="x", alpha=0.25, linewidth=0.8, zorder=0)
        motivating_example_output.parent.mkdir(parents=True, exist_ok=True)
        fig2.savefig(motivating_example_output)
        plt.close(fig2)


if __name__ == "__main__":
    app()
