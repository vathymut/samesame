PAPER := main
TEXBIN := /Library/TeX/texbin
PDFLATEX := $(if $(wildcard $(TEXBIN)/pdflatex),$(TEXBIN)/pdflatex,pdflatex)
BIBTEX := $(if $(wildcard $(TEXBIN)/bibtex),$(TEXBIN)/bibtex,bibtex)
PLOT_EXTRA_DEPS := --with matplotlib
REAL_DATA_EXTRA_DEPS := --with pandas --with matplotlib
COMMON_DEPS := scripts/__init__.py scripts/common.py scripts/_dgp.py scripts/_io.py scripts/_repo.py scripts/_domain_clf.py scripts/_plot_utils.py
REAL_DATA_TASKS ?=
REAL_DATA_SPOTLIGHT ?= heloc
CALIBRATION_SUMMARY := results/synthetic_calibration_summary.csv
POWER_SUMMARY := results/power_curve_summary.csv
MODE_SUMMARY := results/mode_comparison_summary.csv
LAMBDA_SUMMARY := results/lambda_sensitivity_summary.csv
DOMAIN_CLF_SUMMARY := results/domain_clf_sensitivity_summary.csv
SECOND_DGP_CALIBRATION_SUMMARY := results/second_dgp_calibration_summary.csv
SECOND_DGP_POWER_SUMMARY := results/second_dgp_power_summary.csv
REAL_DATA_SUMMARY := results/real_data_workflow_summary.csv
INTRO_FIGURE := figures/intro_reweighting_modes.pdf
CALIBRATION_FIGURE := figures/calibration_plot.pdf
POWER_FIGURE := figures/power_plot.pdf
MODE_FIGURE := figures/mode_comparison_plot.pdf
LAMBDA_FIGURE := figures/lambda_sensitivity_plot.pdf
REAL_DATA_FIGURE := figures/real_data_workflow_plot.pdf
HELOC_MOTIVATING_FIGURE := figures/heloc_motivating_example_plot.pdf
CALIBRATION_TABLE := tables/calibration_rejection_table.tex

SECOND_DGP_CALIBRATION_FIGURE := figures/second_dgp_calibration_plot.pdf
SECOND_DGP_POWER_FIGURE := figures/second_dgp_power_plot.pdf

all: $(PAPER).pdf

figures: $(INTRO_FIGURE) $(CALIBRATION_FIGURE) $(POWER_FIGURE) $(MODE_FIGURE) $(LAMBDA_FIGURE) $(SECOND_DGP_CALIBRATION_FIGURE) $(SECOND_DGP_POWER_FIGURE)

real-data-figure: $(REAL_DATA_FIGURE)

tables: $(CALIBRATION_TABLE)

verify-results: scripts/verify_result_metadata.py $(COMMON_DEPS)
	uv run python -m scripts.verify_result_metadata

$(CALIBRATION_SUMMARY) $(POWER_SUMMARY): scripts/generate_synthetic_calibration.py scripts/generate_power_curves.py $(COMMON_DEPS)
	uv run python -m scripts.generate_synthetic_calibration --n-repeats 100 --n-resamples 499 --n-source 180 --n-target 180 --severity-grid 0.0 --severity-grid 0.1 --severity-grid 0.2 --severity-grid 0.3 --severity-grid 0.4 --output $(CALIBRATION_SUMMARY) --detail-output results/synthetic_calibration_detail.csv --metadata-output results/synthetic_calibration_metadata.json
	uv run python -m scripts.generate_power_curves --n-repeats 100 --n-resamples 499 --n-source 180 --n-target 180 --severity 0.25 --effect-grid 0.0 --effect-grid 0.15 --effect-grid 0.3 --effect-grid 0.45 --effect-grid 0.6 --output $(POWER_SUMMARY) --detail-output results/power_curve_detail.csv --metadata-output results/power_curve_metadata.json

$(MODE_SUMMARY): scripts/generate_mode_comparison.py $(COMMON_DEPS)
	uv run python -m scripts.generate_mode_comparison --n-repeats 100 --n-resamples 499 --n-source 180 --n-target 180 --lambda-value 0.5 --output $(MODE_SUMMARY) --detail-output results/mode_comparison_detail.csv --metadata-output results/mode_comparison_metadata.json

$(LAMBDA_SUMMARY): scripts/generate_lambda_sensitivity.py $(COMMON_DEPS)
	uv run python -m scripts.generate_lambda_sensitivity --n-repeats 100 --n-resamples 499 --n-source 180 --n-target 180 --severity 0.25 --effect-size 0.4 --lambda-grid 0.0 --lambda-grid 0.25 --lambda-grid 0.5 --lambda-grid 0.75 --lambda-grid 1.0 --output $(LAMBDA_SUMMARY) --detail-output results/lambda_sensitivity_detail.csv --metadata-output results/lambda_sensitivity_metadata.json

$(REAL_DATA_SUMMARY): scripts/generate_real_data_workflow_summary.py scripts/real_data_workflow_sources.py scripts/real_data_workflow_config.py $(COMMON_DEPS)
	uv run $(REAL_DATA_EXTRA_DEPS) python -m scripts.generate_real_data_workflow_summary $(if $(REAL_DATA_TASKS),--tasks $(REAL_DATA_TASKS),) --output $(REAL_DATA_SUMMARY) --metadata-output results/real_data_workflow_metadata.json

$(INTRO_FIGURE): scripts/plot_intro_figure.py
	uv run $(PLOT_EXTRA_DEPS) --with scipy python -m scripts.plot_intro_figure --output $(INTRO_FIGURE)

CALIBRATION_DETAIL := results/synthetic_calibration_detail.csv
POWER_DETAIL := results/power_curve_detail.csv

$(CALIBRATION_FIGURE) $(POWER_FIGURE): $(CALIBRATION_SUMMARY) $(POWER_SUMMARY) $(CALIBRATION_DETAIL) $(POWER_DETAIL) scripts/plot_first_figures.py
	uv run $(PLOT_EXTRA_DEPS) python -m scripts.plot_first_figures --calibration-summary $(CALIBRATION_SUMMARY) --power-summary $(POWER_SUMMARY) --calibration-detail $(CALIBRATION_DETAIL) --power-detail $(POWER_DETAIL) --calibration-output $(CALIBRATION_FIGURE) --power-output $(POWER_FIGURE)

$(SECOND_DGP_CALIBRATION_FIGURE) $(SECOND_DGP_POWER_FIGURE): $(SECOND_DGP_CALIBRATION_SUMMARY) $(SECOND_DGP_POWER_SUMMARY) scripts/plot_second_dgp.py
	uv run $(PLOT_EXTRA_DEPS) python -m scripts.plot_second_dgp --calibration-summary $(SECOND_DGP_CALIBRATION_SUMMARY) --power-summary $(SECOND_DGP_POWER_SUMMARY) --calibration-output $(SECOND_DGP_CALIBRATION_FIGURE) --power-output $(SECOND_DGP_POWER_FIGURE)

$(MODE_FIGURE) $(LAMBDA_FIGURE): $(MODE_SUMMARY) $(LAMBDA_SUMMARY) scripts/plot_followup_figures.py
	uv run $(PLOT_EXTRA_DEPS) python -m scripts.plot_followup_figures --mode-summary $(MODE_SUMMARY) --lambda-summary $(LAMBDA_SUMMARY) --mode-output $(MODE_FIGURE) --lambda-output $(LAMBDA_FIGURE)

$(REAL_DATA_FIGURE) $(HELOC_MOTIVATING_FIGURE): $(REAL_DATA_SUMMARY) scripts/plot_real_data_workflow_figure.py scripts/real_data_workflow_config.py
	uv run $(REAL_DATA_EXTRA_DEPS) python -m scripts.plot_real_data_workflow_figure --summary $(REAL_DATA_SUMMARY) --output $(REAL_DATA_FIGURE) --spotlight-task $(REAL_DATA_SPOTLIGHT) --motivating-example-output $(HELOC_MOTIVATING_FIGURE)

$(CALIBRATION_TABLE): $(CALIBRATION_SUMMARY) scripts/render_calibration_table.py
	uv run python -m scripts.render_calibration_table --input $(CALIBRATION_SUMMARY) --output $(CALIBRATION_TABLE)

$(DOMAIN_CLF_SUMMARY): scripts/generate_domain_clf_sensitivity.py $(COMMON_DEPS)
	uv run python -m scripts.generate_domain_clf_sensitivity --n-repeats 100 --n-resamples 999 --n-source 180 --n-target 180 --output $(DOMAIN_CLF_SUMMARY) --detail-output results/domain_clf_sensitivity_detail.csv --metadata-output results/domain_clf_sensitivity_metadata.json

$(SECOND_DGP_CALIBRATION_SUMMARY) $(SECOND_DGP_POWER_SUMMARY): scripts/generate_second_dgp.py $(COMMON_DEPS)
	uv run python -m scripts.generate_second_dgp --n-repeats 40 --n-resamples 199 --n-source 180 --n-target 180 --calibration-output $(SECOND_DGP_CALIBRATION_SUMMARY) --calibration-detail-output results/second_dgp_calibration_detail.csv --power-output $(SECOND_DGP_POWER_SUMMARY) --power-detail-output results/second_dgp_power_detail.csv --metadata-output results/second_dgp_metadata.json

experiments-extended: $(DOMAIN_CLF_SUMMARY) $(SECOND_DGP_CALIBRATION_SUMMARY) $(SECOND_DGP_POWER_SUMMARY) $(SECOND_DGP_CALIBRATION_FIGURE) $(SECOND_DGP_POWER_FIGURE)
	@echo "Extended experiments generated: domain classifier sensitivity + second DGP"

.PHONY: experiments-extended

$(PAPER).pdf: $(PAPER).tex sections/*.tex references.bib aistats2026.sty $(INTRO_FIGURE) $(CALIBRATION_FIGURE) $(POWER_FIGURE) $(MODE_FIGURE) $(LAMBDA_FIGURE) $(REAL_DATA_FIGURE) $(HELOC_MOTIVATING_FIGURE) $(CALIBRATION_TABLE)
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex
	$(BIBTEX) $(PAPER)
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex
	$(PDFLATEX) -interaction=nonstopmode $(PAPER).tex

clean:
	rm -f $(PAPER).aux $(PAPER).bbl $(PAPER).blg $(PAPER).log $(PAPER).out $(PAPER).pdf $(PAPER).synctex.gz

.PHONY: all clean figures real-data-figure tables verify-results
