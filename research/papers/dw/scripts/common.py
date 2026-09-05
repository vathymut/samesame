"""Backward-compatible re-export facade.

Prefer importing from the focused modules directly in new code:
``_dgp``, ``_io``, ``_repo``, ``_domain_clf``, ``weighting``.
"""

from __future__ import annotations

import warnings

from scripts._domain_clf import estimate_domain_probabilities_hgb
from scripts._io import (
    aggregate_mean,
    ensure_directory,
    json_ready,
    min_ess,
    read_csv,
    write_csv,
    write_json,
)
from scripts._repo import (
    MANUSCRIPT_DIR,
    PAPER_DIR,
    RESULTS_DIR,
    ROOT,
    file_sha256,
    repo_commit_hash,
    repo_has_uncommitted_changes,
    result_metadata,
)
from scripts.manuscript_style import MODE_ORDER
from scripts.weighting import run_harm_test_with_estimator

warnings.filterwarnings(
    "ignore",
    message="overflow encountered in scalar power",
    category=RuntimeWarning,
)

ALPHA = 0.05
MODES: tuple[str, ...] = MODE_ORDER
MODES_WITH_BASELINES: tuple[str, ...] = MODE_ORDER


def run_harm_test(
    source_score,
    target_score,
    *,
    source_feature,
    target_feature,
    mode: str,
    lambda_value: float,
    n_resamples: int,
    seed: int,
) -> dict:
    return run_harm_test_with_estimator(
        source_score,
        target_score,
        source_feature=source_feature,
        target_feature=target_feature,
        estimator=estimate_domain_probabilities_hgb,
        direction="higher",
        mode=mode,
        lambda_value=lambda_value,
        n_resamples=n_resamples,
        seed=seed,
        alpha=ALPHA,
    )


__all__ = [
    "ALPHA",
    "MODES",
    "MODES_WITH_BASELINES",
    "MANUSCRIPT_DIR",
    "PAPER_DIR",
    "RESULTS_DIR",
    "ROOT",
    "aggregate_mean",
    "ensure_directory",
    "file_sha256",
    "json_ready",
    "min_ess",
    "read_csv",
    "repo_commit_hash",
    "repo_has_uncommitted_changes",
    "result_metadata",
    "run_harm_test",
    "write_csv",
    "write_json",
]
