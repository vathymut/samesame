# Paper Scripts

Keep every figure and table reproducible from code.

## Rules

- One script or notebook entrypoint per artifact family.
- Prefer plain Python scripts over interactive notebooks for final figure generation.
- Each script should write outputs into `research/papers/dw/figures/` or a dedicated results directory.
- Record the package commit hash and random seed in script outputs or logs.
- Do not hand-edit exported figure PDFs.
- Run scripts as modules from `research/papers/dw/`, for example
	`uv run python -m scripts.generate_synthetic_calibration`.

## Current scripts

- `common.py`: shared data-generation, weighting, aggregation, and metadata helpers
- `generate_synthetic_calibration.py`: support-mismatch calibration summaries
- `generate_power_curves.py`: power summaries for harmful shift on common support
- `generate_mode_comparison.py`: source-only vs target-only vs both-side contamination comparison
- `generate_lambda_sensitivity.py`: doubly weighted sensitivity across `lambda`
- `real_data_workflow_config.py`: shared task slate, pinned OpenML IDs, and mirrored TableShift split metadata for the preferred real-data figure
- `real_data_workflow_sources.py`: OpenML-backed loaders that recreate TableShift split rules locally for the executable mirrored task slate
- `generate_real_data_workflow_summary.py`: preferred HELOC-led real-data workflow summary across the executable OpenML-mirrored task slate
- `plot_first_figures.py`: first calibration and power figures from summary CSV outputs
- `plot_followup_figures.py`: mode-comparison and lambda-sensitivity figures from summary CSV outputs
- `plot_real_data_workflow_figure.py`: preferred HELOC spotlight plus OpenML-mirrored corroboration-task workflow figure from the summary CSV output
- `render_calibration_table.py`: first LaTeX table for calibration rejection rates
- `verify_result_metadata.py`: checks generated metadata hashes and schema

Keep CLI arguments stable so figures and tables can be regenerated for rebuttal or camera-ready revisions.
