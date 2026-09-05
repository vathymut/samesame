"""Render a LaTeX calibration table from the first-pass summary CSV.

Adds Crump-trimmed and overlap-weighted baselines plus 95% Clopper-Pearson
confidence intervals computed from the detail CSV.
"""

from __future__ import annotations

from pathlib import Path

import typer
from scipy.stats import beta

from scripts._io import read_csv
from scripts._repo import MANUSCRIPT_DIR
from scripts.manuscript_style import MODE_LABELS, MODE_ORDER
from scripts.result_schemas import SYNTHETIC_CALIBRATION_SUMMARY_SCHEMA

app = typer.Typer()

SEVERITY_ORDER = ["0.0", "0.1", "0.2", "0.3", "0.4"]


def clopper_pearson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    lower = beta.ppf(alpha / 2, k + 1, n - k) if k > 0 else 0.0
    upper = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lower, upper)


def render_table(
    summary_rows: list[dict],
    detail_rows: list[dict] | None,
    output_path: Path,
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{",
        r"  Empirical rejection rates (false positive rates) under support",
        r"  mismatch at $\alpha = 0.05$. Rows vary the low-overlap severity",
        r"  $s$ (symmetric). Parentheses show 95\% Clopper--Pearson confidence",
        r"  intervals. The doubly weighted test is the only condition that",
        r"  remains calibrated across all severity levels.}",
        r"\label{tab:calibration}",
        r"\begin{tabular}{l" + "c" * len(SEVERITY_ORDER) + "}",
        r"\toprule",
        r"Mode & "
        + " & ".join(
            rf"$s = {sev}$" + (r" \times 10^{-2}" if sev in ("0.3", "0.4") else "")
            for sev in SEVERITY_ORDER
        )
        + r" \\",
        r"\midrule",
    ]

    for mode in MODE_ORDER:
        label = MODE_LABELS[mode]
        row_cells = [label]
        for sev in SEVERITY_ORDER:
            matching = [
                r for r in summary_rows
                if str(r["mode"]) == mode and str(r["severity"]) == sev
            ]
            if not matching:
                row_cells.append("---")
                continue
            reject = float(matching[0]["reject"])
            count = int(matching[0]["count"])
            pct = 100.0 * reject
            if detail_rows is not None and count > 0:
                k = int(round(reject * count))
                lo, hi = clopper_pearson_ci(k, count)
                ci_str = rf"[{100*lo:.1f}, {100*hi:.1f}]"
                row_cells.append(rf"${pct:.1f}\,({ci_str})$")
            else:
                row_cells.append(rf"${pct:.1f}$")
        lines.append(" & ".join(row_cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


@app.command()
def main(
    input: Path = typer.Option(
        MANUSCRIPT_DIR / "results/synthetic_calibration_summary.csv",
    ),
    detail_input: Path = typer.Option(None),
    output: Path = typer.Option(
        MANUSCRIPT_DIR / "tables/calibration_rejection_table.tex",
    ),
) -> None:
    summary_rows = read_csv(input, schema=SYNTHETIC_CALIBRATION_SUMMARY_SCHEMA)
    detail_rows = read_csv(detail_input) if detail_input else None
    render_table(summary_rows, detail_rows, output)


if __name__ == "__main__":
    app()
