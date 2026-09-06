# Paper Scripts

Keep every figure and table reproducible from code.

## Rules
- One script per artifact family.
- Prefer plain Python scripts over notebooks.
- Each script writes into `research/papers/dw/figures/` or `results/`.
- Record commit hash and seed in outputs.
- Do not hand-edit exported PDFs.
- Run as modules from `research/papers/dw/`, e.g. `uv run python -m scripts.synthetic calibration`.

## Scripts
- `dgp.py`: overlap + second DGP
- `experiments.py`: domain classifier + weighting + harm test (single seam)
- `datasets.py`: OpenML loaders (heloc, diabetes, acsincome, acspubcov, nsw) — explicit per-task code
- `synthetic.py`: calibration / power / mode-comparison / lambda (typer subcommands)
- `real_data.py`: HELOC-anchored real-data workflow
- `appendix.py`: second DGP + domain-clf sensitivity
- `plots.py`: calibration/power/mode/lambda/second-DGP figures (`first` / `followup` / `second-dgp`)
- `real_data_plot.py`: cross-task + ESS + HELOC motivating figure
- `intro.py`: conceptual reweighting figure
- `tables.py`: calibration LaTeX table
- `utils.py`: csv/json + paths + commit hash
- `style.py`: colors, markers, labels

## Make targets
- `make figures` — intro + synthetic plots
- `make real-data-figure` — real-data plots (requires OpenML fetch)
- `make tables` — LaTeX tables
- `make experiments-extended` — appendix corroboration
