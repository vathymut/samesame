from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _load_real_data_workflow_sources(monkeypatch: pytest.MonkeyPatch):
    paper_dir = Path(__file__).resolve().parents[1] / "research/papers/dw"
    monkeypatch.syspath_prepend(str(paper_dir))
    sys.modules.pop("scripts.real_data_workflow_sources", None)
    sys.modules.pop("scripts.real_data_workflow_config", None)
    return importlib.import_module("scripts.real_data_workflow_sources")


def _ready_task_frame(task_name: str) -> tuple[pd.DataFrame, pd.Series]:
    labels = pd.Series(["Bad", "Good", "Bad", "Good", "Bad", "Good", "Bad", "Good"])
    if task_name == "heloc":
        return (
            pd.DataFrame(
                {
                    "signal": range(8),
                    "ExternalRiskEstimate": [70, 71, 72, 73, 74, 75, 60, 61],
                }
            ),
            labels,
        )
    if task_name == "diabetes_readmission":
        return (
            pd.DataFrame(
                {
                    "signal": range(8),
                    "encounter_id": range(100, 108),
                    "patient_nbr": range(200, 208),
                    "admission_source_id": [1, 2, 3, 4, 5, 6, 7, "Emergency Room"],
                }
            ),
            pd.Series(["Yes", "No", "Yes", "No", "Yes", "No", "Yes", "No"]),
        )
    if task_name == "acsincome":
        return (
            pd.DataFrame(
                {
                    "signal": range(8),
                    "ST": ["06", "08", "10", "12", "17", "22", "09", "23"],
                }
            ),
            pd.Series([50_000, 60_000, 40_000, 70_000, 30_000, 80_000, 45_000, 65_000]),
        )
    if task_name == "acspubcov":
        return (
            pd.DataFrame(
                {
                    "signal": range(8),
                    "DIS": [0, 0, 0, 0, 0, 0, 1, "With a disability"],
                }
            ),
            pd.Series([0, 1, 0, 1, 0, 1, 1, 0]),
        )
    raise AssertionError(f"unexpected task {task_name!r}")


@pytest.mark.parametrize(
    ("task_name", "message_fragment"),
    [
        (
            "college_scorecard",
            "does not expose the CCBASIC Carnegie basic classification column",
        ),
        (
            "physionet",
            "fails checksum validation via scikit-learn/OpenML",
        ),
        (
            "mimic_extract_los_3",
            "exposes medication-route features instead of the los_3 target",
        ),
    ],
)
def test_blocked_openml_tasks_raise_clear_runtime_errors(
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
    message_fragment: str,
) -> None:
    sources = _load_real_data_workflow_sources(monkeypatch)

    with pytest.raises(RuntimeError, match=message_fragment):
        sources.load_task(task_name, max_train_rows=20, max_eval_rows=10, seed=0)


def test_fetch_openml_frame_retries_with_fresh_cache_on_md5_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _load_real_data_workflow_sources(monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_fetch_openml(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("md5 checksum of local file does not match description")
        return SimpleNamespace(
            data=pd.DataFrame({"feature": [1, 2], "all_missing": [None, None]}),
            target=pd.Series(["yes", "no"]),
        )

    monkeypatch.setattr(sources, "fetch_openml", fake_fetch_openml)

    feature, label = sources.fetch_openml_frame(123)

    assert len(calls) == 2
    assert calls[0] == {"data_id": 123, "as_frame": True, "parser": "auto"}
    assert calls[1]["data_id"] == 123
    assert calls[1]["as_frame"] is True
    assert calls[1]["parser"] == "auto"
    assert isinstance(calls[1]["data_home"], str)
    assert list(feature.columns) == ["feature"]
    assert label.tolist() == ["yes", "no"]


def test_fetch_openml_frame_reraises_non_checksum_value_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _load_real_data_workflow_sources(monkeypatch)

    def fake_fetch_openml(**kwargs):
        del kwargs
        raise ValueError("unexpected schema mismatch")

    monkeypatch.setattr(sources, "fetch_openml", fake_fetch_openml)

    with pytest.raises(ValueError, match="unexpected schema mismatch"):
        sources.fetch_openml_frame(123)


@pytest.mark.parametrize(
    "task_name",
    ["heloc", "diabetes_readmission", "acsincome", "acspubcov"],
)
def test_ready_openml_tasks_drop_split_drivers(
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
) -> None:
    sources = _load_real_data_workflow_sources(monkeypatch)

    def fake_fetch_openml_frame(data_id: int):
        assert data_id == sources.TASK_SPECS[task_name].openml_data_id
        return _ready_task_frame(task_name)

    monkeypatch.setattr(sources, "fetch_openml_frame", fake_fetch_openml_frame)

    task = sources.load_task(task_name, max_train_rows=4, max_eval_rows=2, seed=0)

    assert task.task == task_name
    assert list(task.train_feature.columns) == ["signal"]
    assert list(task.source_feature.columns) == ["signal"]
    assert list(task.target_feature.columns) == ["signal"]
    assert task.train_feature.shape == (4, 1)
    assert task.source_feature.shape == (2, 1)
    assert task.target_feature.shape == (2, 1)


def test_load_physionet_recreates_tableshift_split_without_runtime_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = _load_real_data_workflow_sources(monkeypatch)

    def fake_fetch_openml_frame(data_id: int):
        assert data_id == sources.TASK_SPECS["physionet"].openml_data_id
        return (
            pd.DataFrame(
                {
                    "signal": [10, 20, 30, 40, 50, 60, 70, 80],
                    "ICULOS": [10, 20, 30, 40, 41, 42, 48, 60],
                    "Hour": [1, 2, 3, 4, 5, 6, 7, 8],
                }
            ),
            pd.Series([0, 1, 0, 1, 0, 1, 1, 0]),
        )

    monkeypatch.setattr(sources, "fetch_openml_frame", fake_fetch_openml_frame)

    task = sources.load_recipe_task(
        "physionet", max_train_rows=4, max_eval_rows=2, seed=0
    )

    assert task.task == "physionet"
    assert task.train_feature.shape == (4, 1)
    assert task.source_feature.shape == (2, 1)
    assert task.target_feature.shape == (2, 1)
    assert list(task.train_feature.columns) == ["signal"]
    assert list(task.source_feature.columns) == ["signal"]
    assert list(task.target_feature.columns) == ["signal"]
    assert sorted(task.target_feature["signal"].tolist()) == [70, 80]
    assert sorted(
        pd.concat([task.train_feature, task.source_feature], ignore_index=True)[
            "signal"
        ].tolist()
    ) == [10, 20, 30, 40, 50, 60]
    assert task.train_label.dtype == int
    assert task.source_label.dtype == int
    assert task.target_label.dtype == int
