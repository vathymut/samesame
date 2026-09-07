from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest


def _scripts(monkeypatch: pytest.MonkeyPatch, name: str):
    paper_dir = Path(__file__).resolve().parents[1] / "research/papers/dw"
    monkeypatch.syspath_prepend(str(paper_dir))
    sys.modules.pop(f"scripts.{name}", None)
    return importlib.import_module(f"scripts.{name}")


def test_resolve_tasks_tiers_and_all(monkeypatch: pytest.MonkeyPatch) -> None:
    cache = _scripts(monkeypatch, "cache_tableshift")
    assert cache.resolve_tasks(["all"]) == list(cache.ALL_TASKS)
    assert len(cache.ALL_TASKS) == 15
    assert cache.resolve_tasks(["tier1"]) == list(cache.TIER_1)
    assert cache.resolve_tasks(["tier2", "tier1"])[: len(cache.TIER_2)] == list(cache.TIER_2)
    with pytest.raises(ValueError, match="unknown task"):
        cache.resolve_tasks(["not_a_task"])


def test_task_cache_paths_use_id_ood(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _scripts(monkeypatch, "cache_tableshift")
    id_path, ood_path = cache.task_cache_paths("acsincome", tmp_path)
    assert id_path == tmp_path / "acsincome" / "id.parquet"
    assert ood_path == tmp_path / "acsincome" / "ood.parquet"


def test_stage_task_records_missing_without_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cache = _scripts(monkeypatch, "cache_tableshift")

    def boom(task: str, id_path: Path, ood_path: Path) -> str:
        del id_path, ood_path
        raise RuntimeError("offline skeleton: no fetch")

    entry = cache.stage_task("anes", tmp_path, fetcher=boom)
    assert entry["provenance"] == "missing_manual"
    entry2 = cache.stage_task("acsincome", tmp_path, fetcher=boom)
    assert entry2["provenance"] == "pending_fetch"


def test_verify_task_files_missing_entry(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache = _scripts(monkeypatch, "cache_tableshift")
    problems = cache.verify_task_files("acsincome", tmp_path, {"tasks": {}})
    assert problems and "missing manifest entry" in problems[0]


def test_workflow_scores_from_probas_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    real_data = _scripts(monkeypatch, "real_data")
    id_p = np.array([0.1, 0.9, 0.4, 0.6])
    ood_p = np.array([0.2, 0.8, 0.5, 0.7])
    out = real_data.workflow_scores_from_probas(id_p, ood_p, [0, 1, 0, 1], [0, 1, 0, 1])
    assert set(out) == {"risk", "confidence", "error"}
    assert out["risk"][0].tolist() == id_p.tolist()
    assert len(out["error"][1]) == 4


def test_cross_fitted_probas_small_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    real_data = _scripts(monkeypatch, "real_data")
    monkeypatch.setattr(
        real_data, "DEFAULT_HGB_PARAMS", {"max_iter": 20, "max_depth": 2, "min_samples_leaf": 1}
    )
    rng = np.random.default_rng(0)
    n = 20
    id_feature = pl.DataFrame({"x": np.arange(n, dtype=float), "z": rng.normal(size=n)})
    id_label = np.array([0] * 10 + [1] * 10)
    ood_feature = pl.DataFrame({"x": np.arange(n, n + 6, dtype=float), "z": rng.normal(size=6)})
    id_p, ood_p = real_data.cross_fitted_id_ood_probas(
        id_feature, id_label, ood_feature, seed=123_456, n_splits=5
    )
    assert id_p.shape == (20,)
    assert ood_p.shape == (6,)
    assert bool(((id_p >= 0) & (id_p <= 1)).all())
    assert bool(((ood_p >= 0) & (ood_p <= 1)).all())
