PAPER := main
TEXBIN := /Library/TeX/texbin
PDFLATEX := $(if $(wildcard $(TEXBIN)/pdflatex),$(TEXBIN)/pdflatex,pdflatex)
BIBTEX := $(if $(wildcard $(TEXBIN)/bibtex),$(TEXBIN)/bibtex,bibtex)
PLOT_DEPS := --with matplotlib
REAL_DATA_DEPS := --with pandas --with matplotlib
CORE_DEPS := scripts/__init__.py scripts/utils.py scripts/style.py scripts/dgp.py scripts/experiments.py scripts/datasets.py
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

$(CALIBRATION_SUMMARY): scripts/synthetic.py $(CORE_DEPS)
	uv run python -m scripts.synthetic calibration --n-repeats 100 --n-resamples 499 --n-source 180 --n-target 180 --severity-grid 0.0 --severity-grid 0.1 --severity-grid 0.2 --severity-grid 0.3 --severity-grid 0.4 --output $(CALIBRATION_SUMMARY) --detail-output results/synthetic_calibration_detail.csv --metadata-output results/synthetic_calibration_metadata.json

$(POWER_SUMMARY): scripts/synthetic.py $(CORE_DEPS)
	uv run python -m scripts.synthetic power --n-repeats 100 --n-resamples 499 --n-source 180 --n-target 180 --severity 0.25 --effect-grid 0.0 --effect-grid 0.15 --effect-grid 0.3 --effect-grid 0.45 --effect-grid 0.6 --output $(POWER_SUMMARY) --detail-output results/power_curve_detail.csv --metadata-output results/power_curve_metadata.json

$(MODE_SUMMARY): scripts/synthetic.py $(CORE_DEPS)
	uv run python -m scripts.synthetic mode-comparison --n-repeats 100 --n-resamples 499 --n-source 180 --n-target 180 --lambda-value 0.5 --output $(MODE_SUMMARY) --detail-output results/mode_comparison_detail.csv --metadata-output results/mode_comparison_metadata.json

$(LAMBDA_SUMMARY): scripts/synthetic.py $(CORE_DEPS)
	uv run python -m scripts.synthetic lambda --n-repeats 100 --n-resamples 499 --n-source 180 --n-target 180 --severity 0.25 --effect-size 0.4 --lambda-grid 0.0 --lambda-grid 0.25 --lambda-grid 0.5 --lambda-grid 0.75 --lambda-grid 1.0 --output $(LAMBDA_SUMMARY) --detail-output results/lambda_sensitivity_detail.csv --metadata-output results/lambda_sensitivity_metadata.json

$(REAL_DATA_SUMMARY): scripts/real_data.py $(CORE_DEPS)
	uv run $(REAL_DATA_DEPS) python -m scripts.real_data $(if $(REAL_DATA_TASKS),--tasks $(REAL_DATA_TASKS),) --output $(REAL_DATA_SUMMARY) --metadata-output results/real_data_workflow_metadata.json

$(INTRO_FIGURE): scripts/intro.py $(CORE_DEPS)
	uv run $(PLOT_DEPS) --with scipy python -m scripts.intro --output $(INTRO_FIGURE)

$(CALIBRATION_FIGURE) $(POWER_FIGURE): $(CALIBRATION_SUMMARY) $(POWER_SUMMARY) scripts/plots.py
	uv run $(PLOT_DEPS) python -m scripts.plots first --calibration-summary $(CALIBRATION_SUMMARY) --power-summary $(POWER_SUMMARY) --calibration-output $(CALIBRATION_FIGURE) --power-output $(POWER_FIGURE)

$(SECOND_DGP_CALIBRATION_FIGURE) $(SECOND_DGP_POWER_FIGURE): $(SECOND_DGP_CALIBRATION_SUMMARY) $(SECOND_DGP_POWER_SUMMARY) scripts/plots.py
	uv run $(PLOT_DEPS) python -m scripts.plots second-dgp --calibration-summary $(SECOND_DGP_CALIBRATION_SUMMARY) --power-summary $(SECOND_DGP_POWER_SUMMARY) --calibration-output $(SECOND_DGP_CALIBRATION_FIGURE) --power-output $(SECOND_DGP_POWER_FIGURE)

$(MODE_FIGURE) $(LAMBDA_FIGURE): $(MODE_SUMMARY) $(LAMBDA_SUMMARY) scripts/plots.py
	uv run $(PLOT_DEPS) python -m scripts.plots followup --mode-summary $(MODE_SUMMARY) --lambda-summary $(LAMBDA_SUMMARY) --mode-output $(MODE_FIGURE) --lambda-output $(LAMBDA_FIGURE)

$(REAL_DATA_FIGURE) $(HELOC_MOTIVATING_FIGURE): $(REAL_DATA_SUMMARY) scripts/real_data_plot.py
	uv run $(REAL_DATA_DEPS) python -m scripts.real_data_plot --summary $(REAL_DATA_SUMMARY) --output $(REAL_DATA_FIGURE) --spotlight-task $(REAL_DATA_SPOTLIGHT) --motivating-example-output $(HELOC_MOTIVATING_FIGURE)

$(CALIBRATION_TABLE): $(CALIBRATION_SUMMARY) scripts/tables.py
	uv run python -m scripts.tables --input $(CALIBRATION_SUMMARY) --output $(CALIBRATION_TABLE)

$(DOMAIN_CLF_SUMMARY): scripts/appendix.py $(CORE_DEPS)
	uv run python -m scripts.appendix domain-clf --n-repeats 100 --n-resamples 999 --n-source 180 --n-target 180 --output $(DOMAIN_CLF_SUMMARY) --detail-output results/domain_clf_sensitivity_detail.csv --metadata-output results/domain_clf_sensitivity_metadata.json

$(SECOND_DGP_CALIBRATION_SUMMARY) $(SECOND_DGP_POWER_SUMMARY): scripts/appendix.py $(CORE_DEPS)
	uv run python -m scripts.appendix second-dgp --n-repeats 40 --n-resamples 199 --n-source 180 --n-target 180 --calibration-output $(SECOND_DGP_CALIBRATION_SUMMARY) --calibration-detail-output results/second_dgp_calibration_detail.csv --power-output $(SECOND_DGP_POWER_SUMMARY) --power-detail-output results/second_dgp_power_detail.csv --metadata-output results/second_dgp_metadata.json

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

.PHONY: all clean figures real-data-figure tables
