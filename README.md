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

For the NSW employment experiment, the loader fetches the Dehejia–Wahba files directly from the NBER mirror (CPS comparison group `cps3_controls.txt`, NSW treated `nswre74_treated.txt`), so no local data restore is needed. The OpenML tasks fetch by pinned dataset ID.

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
uv run python -m scripts.synthetic --help
uv run python -m scripts.real_data --help
uv run python -m scripts.intro --help
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
