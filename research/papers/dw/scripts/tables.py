"""LaTeX tables."""

from __future__ import annotations

from pathlib import Path

import typer
from scipy.stats import beta

from scripts.style import MODE_LABELS, MODE_ORDER
from scripts.utils import MANUSCRIPT_DIR, read_csv

app = typer.Typer()
SEVERITY_ORDER = ["0.0", "0.1", "0.2", "0.3", "0.4"]


def _ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    lo = beta.ppf(alpha / 2, k + 1, n - k) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return (lo, hi)


def _render(summary_rows, detail_rows, output_path: Path) -> None:
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
        r"Mode & " + " & ".join(rf"$s = {sev}$" for sev in SEVERITY_ORDER) + r" \\",
        r"\midrule",
    ]
    for mode in MODE_ORDER:
        cells = [MODE_LABELS[mode]]
        for sev in SEVERITY_ORDER:
            m = [r for r in summary_rows if str(r["mode"]) == mode and str(r["severity"]) == sev]
            if not m:
                cells.append("---")
                continue
            rej, cnt = float(m[0]["reject"]), int(m[0]["count"])
            pct = 100 * rej
            if detail_rows is not None and cnt > 0:
                lo, hi = _ci(int(round(rej * cnt)), cnt)
                cells.append(rf"${pct:.1f}\,([{100*lo:.1f}, {100*hi:.1f}])$")
            else:
                cells.append(rf"${pct:.1f}$")
        lines.append(" & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


@app.command()
def main(
    input: Path = typer.Option(MANUSCRIPT_DIR / "results/synthetic_calibration_summary.csv"),
    detail_input: Path = typer.Option(None),
    output: Path = typer.Option(MANUSCRIPT_DIR / "tables/calibration_rejection_table.tex"),
) -> None:
    summary = read_csv(input)
    detail = read_csv(detail_input) if detail_input else None
    _render(summary, detail, output)


if __name__ == "__main__":
    app()
