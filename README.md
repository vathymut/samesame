# samesame-paper

Manuscript and experiments for **Testing Harmful Shift on Common Support** (CS-DSOS).

This is an orphan branch — it contains only the paper workspace and depends on the published `samesame` package from PyPI, not on the repository's `src/` tree.

## Setup

```bash
uv sync
# or with test deps
uv sync --extra test
```

The paper's experiment stack (`skrub`, `polars`, `typer`, `pandas`, `matplotlib`) is declared in `pyproject.toml` and installed automatically.

For the NSW employment experiment, restore the LaLonde CSV (ignored via `research/papers/dw/.gitignore`):

```bash
git show develop:research/papers/dw/data/nsw/lalonde.csv > research/papers/dw/data/nsw/lalonde.csv
# or from the backup branch
git show paper-dw-robust-adverse-shift-backup-603ea37:research/papers/dw/data/nsw/lalonde.csv > research/papers/dw/data/nsw/lalonde.csv
```

## Building the manuscript

```bash
cd research/papers/dw
make          # full PDF (pdflatex + bibtex, 3 passes)
# or directly
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Running experiments

```bash
# from research/papers/dw/
uv run python -m scripts.generate_synthetic_calibration --help
uv run python -m scripts.generate_real_data_workflow_summary --help
uv run --with matplotlib python -m scripts.plot_intro_figure --help
```

## Tests

```bash
uv run pytest
```

See `research/papers/dw/README.md` and `research/papers/dw/CONTEXT.md` for manuscript-specific guidance.

## History

- `develop` — package source (`samesame` 0.4.x, `src/samesame/`)
- `archive/paper-dw-robust-adverse-shift` — previous standalone workspace (git dep on `develop`)
- `paper-dw-robust-adverse-shift-backup-603ea37` — topic-branch snapshot before orphan conversion (includes review fixes)
