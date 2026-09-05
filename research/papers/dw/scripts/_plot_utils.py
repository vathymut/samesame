"""Shared plotting utilities for manuscript figures."""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts.manuscript_style import MODE_STYLE


def group_by_mode(
    rows: list[dict[str, Any]], x_key: str, y_key: str
) -> dict[str, tuple[list[float], list[float]]]:
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        mode = str(row["mode"])
        grouped.setdefault(mode, []).append((float(row[x_key]), float(row[y_key])))
    ordered: dict[str, tuple[list[float], list[float]]] = {}
    for mode, pairs in grouped.items():
        pairs.sort(key=lambda item: item[0])
        ordered[mode] = ([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    return ordered


def bootstrap_ci(
    detail_rows: list[dict[str, Any]],
    group_key: str,
    group_value: Any,
    mode_value: str,
    metric: str = "reject",
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    values = [
        float(row[metric])
        for row in detail_rows
        if row[group_key] == group_value and row["mode"] == mode_value
    ]
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    alpha = 1.0 - ci
    return (
        float(np.percentile(boot_means, 100 * alpha / 2)),
        float(np.percentile(boot_means, 100 * (1 - alpha / 2))),
    )


def plot_lines_on_axis(
    ax,
    rows,
    modes,
    x_key,
    y_key,
    xlabel,
    ylabel,
    show_legend,
    legend_ncol=1,
    detail_rows=None,
):
    grouped = group_by_mode(rows, x_key, y_key)
    for mode in modes:
        if mode not in grouped:
            continue
        x_values, y_values = grouped[mode]
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
        if detail_rows is not None:
            lower = []
            upper = []
            for xv in x_values:
                lo, hi = bootstrap_ci(detail_rows, x_key, xv, mode)
                lower.append(lo)
                upper.append(hi)
            ax.fill_between(
                x_values, lower, upper, alpha=0.15, color=style["color"]
            )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25, linewidth=0.8)
    if show_legend:
        ax.legend(frameon=False, ncol=legend_ncol, fontsize=8)
