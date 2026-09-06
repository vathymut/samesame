"""All manuscript figures from summary CSVs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import typer

from scripts.style import (
    EXPERIMENT_STYLE,
    MODE_ORDER,
    MODE_STYLE,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
)
from scripts.utils import MANUSCRIPT_DIR, read_csv

app = typer.Typer()

# --- shared helpers ---

def _group_by_mode(rows: list[dict[str, Any]], x_key: str, y_key: str):
    grouped: dict[str, list[tuple[float, float]]] = {}
    for r in rows:
        grouped.setdefault(str(r["mode"]), []).append((float(r[x_key]), float(r[y_key])))
    return {m: ([x for x, _ in sorted(v)], [y for _, y in sorted(v)]) for m, v in grouped.items()}


def _bootstrap_ci(detail_rows, group_key, group_value, mode_value, metric="reject", n_bootstrap=1000):
    vals = [float(r[metric]) for r in detail_rows if r[group_key] == group_value and r["mode"] == mode_value]
    if not vals:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    boots = np.array([np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_bootstrap)])
    return (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def _plot_lines(ax, rows, modes, x_key, y_key, xlabel, ylabel, show_legend, legend_ncol=1, detail_rows=None):
    grouped = _group_by_mode(rows, x_key, y_key)
    for mode in modes:
        if mode not in grouped:
            continue
        x, y = grouped[mode]
        s = MODE_STYLE[mode]
        ax.plot(x, y, color=s["color"], marker=s["marker"], linewidth=2.0, markersize=6.0, label=s["label"])
        if detail_rows is not None:
            lo, hi = zip(*[_bootstrap_ci(detail_rows, x_key, xv, mode) for xv in x])
            ax.fill_between(x, lo, hi, alpha=0.15, color=s["color"])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.8)
    if show_legend:
        ax.legend(frameon=False, ncol=legend_ncol, fontsize=8)


def _save(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def _line_figure(rows, x_key, y_key, xlabel, ylabel, title, output, *, detail_rows=None):
    fig, ax = plt.subplots(figsize=(5.4, 3.6), constrained_layout=True)
    _plot_lines(ax, rows, MODE_ORDER, x_key, y_key, xlabel, ylabel, True, 2, detail_rows)
    ax.set_title(title)
    _save(fig, output)


# --- first figures ---

def _plot_power(rows, x_key, y_key, xlabel, output, detail_rows=None):
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.2), constrained_layout=True)
    _plot_lines(axes[0], rows, ["unweighted", "source", "target", "both"], x_key, y_key, xlabel, "Rejection rate", False, detail_rows=detail_rows)
    axes[0].set_title("(a) Density-ratio weighting")
    _plot_lines(axes[1], rows, ["unweighted", "crump", "overlap"], x_key, y_key, xlabel, "Rejection rate", True, detail_rows=detail_rows)
    axes[1].set_title("(b) Causal-inference baselines")
    _plot_lines(axes[2], rows, MODE_ORDER, x_key, "statistic", xlabel, "Mann-Whitney statistic", True, 2)
    axes[2].set_title("(c) Test statistic (shared scale)")
    _save(fig, output)


@app.command("first")
def first(
    calibration_summary: Path = typer.Option(MANUSCRIPT_DIR / "results/synthetic_calibration_summary.csv"),
    power_summary: Path = typer.Option(MANUSCRIPT_DIR / "results/power_curve_summary.csv"),
    calibration_detail: Path = typer.Option(None),
    power_detail: Path = typer.Option(None),
    calibration_output: Path = typer.Option(MANUSCRIPT_DIR / "figures/calibration_plot.pdf"),
    power_output: Path = typer.Option(MANUSCRIPT_DIR / "figures/power_plot.pdf"),
) -> None:
    cal_rows = read_csv(calibration_summary)
    pow_rows = read_csv(power_summary)
    cal_detail = read_csv(calibration_detail) if calibration_detail else None
    pow_detail = read_csv(power_detail) if power_detail else None
    _line_figure(cal_rows, "severity", "reject", "Low-overlap severity", "False positive rate", "Calibration under support mismatch", calibration_output, detail_rows=cal_detail)
    _plot_power(pow_rows, "effect_size", "reject", "Effect size on common support", power_output, detail_rows=pow_detail)


# --- follow-up ---

def _plot_mode_comparison(rows, output):
    fig, ax = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    pos = np.arange(len(SCENARIO_ORDER))
    w = 0.12
    for i, mode in enumerate(MODE_ORDER):
        off = (i - (len(MODE_ORDER) - 1) / 2) * w
        vals = [100 * float(next(r for r in rows if r["scenario"] == s and r["mode"] == mode)["reject"]) for s in SCENARIO_ORDER]
        ax.bar(pos + off, vals, width=w, color=MODE_STYLE[mode]["color"], label=MODE_STYLE[mode]["label"])
    ax.axhline(5.0, color="#666666", linestyle=":", linewidth=1.2)
    ax.set_xticks(pos, [SCENARIO_LABELS[n] for n in SCENARIO_ORDER])
    ax.set_ylabel("Rejection rate (%)")
    ax.set_title("Weighting modes under asymmetric contamination")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    _save(fig, output)


def _plot_lambda(rows, output):
    fig, (ra, ea) = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True, sharex=True)
    for exp, style in EXPERIMENT_STYLE.items():
        wr = sorted([r for r in rows if r["experiment"] == exp and r["mode"] == "both"], key=lambda r: float(r["lambda_value"]))
        xs = [float(r["lambda_value"]) for r in wr]
        ys = [100 * float(r["reject"]) for r in wr]
        es = [min(float(r["source_ess"]), float(r["target_ess"])) for r in wr]
        ra.plot(xs, ys, color=style["color"], marker="o", linewidth=2.0, label=f"{style['label']}: doubly weighted")
        ea.plot(xs, es, color=style["color"], marker="o", linewidth=2.0, label=style["label"])
        base = next(r for r in rows if r["experiment"] == exp and r["mode"] == "unweighted")
        ra.axhline(100 * float(base["reject"]), color=style["color"], linestyle="--", alpha=0.8, label=f"{style['label']}: unweighted")
    ra.set_xlabel("Lambda")
    ra.set_ylabel("Rejection rate (%)")
    ra.set_title("Calibration and power")
    ra.grid(alpha=0.25, linewidth=0.8)
    ra.legend(frameon=False, fontsize=8)
    ea.set_xlabel("Lambda")
    ea.set_ylabel("Minimum effective sample size")
    ea.set_title("Weight stability")
    ea.grid(alpha=0.25, linewidth=0.8)
    _save(fig, output)


@app.command("followup")
def followup(
    mode_summary: Path = typer.Option(MANUSCRIPT_DIR / "results/mode_comparison_summary.csv"),
    lambda_summary: Path = typer.Option(MANUSCRIPT_DIR / "results/lambda_sensitivity_summary.csv"),
    mode_output: Path = typer.Option(MANUSCRIPT_DIR / "figures/mode_comparison_plot.pdf"),
    lambda_output: Path = typer.Option(MANUSCRIPT_DIR / "figures/lambda_sensitivity_plot.pdf"),
) -> None:
    _plot_mode_comparison(read_csv(mode_summary), mode_output)
    _plot_lambda(read_csv(lambda_summary), lambda_output)


# --- second DGP ---

@app.command("second-dgp")
def second_dgp(
    calibration_summary: Path = typer.Option(MANUSCRIPT_DIR / "results/second_dgp_calibration_summary.csv"),
    power_summary: Path = typer.Option(MANUSCRIPT_DIR / "results/second_dgp_power_summary.csv"),
    calibration_output: Path = typer.Option(MANUSCRIPT_DIR / "figures/second_dgp_calibration_plot.pdf"),
    power_output: Path = typer.Option(MANUSCRIPT_DIR / "figures/second_dgp_power_plot.pdf"),
) -> None:
    _line_figure(read_csv(calibration_summary), "overlap_severity", "reject", "Low-overlap severity", "False positive rate", "Second DGP: Calibration under support mismatch", calibration_output)
    _line_figure(read_csv(power_summary), "effect_size", "reject", "Harmful-shift effect size on common support", "Rejection rate", "Second DGP: Power under harmful shift on common support", power_output)


if __name__ == "__main__":
    app()
