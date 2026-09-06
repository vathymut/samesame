from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def _datasets(monkeypatch: pytest.MonkeyPatch):
    paper_dir = Path(__file__).resolve().parents[1] / "research/papers/dw"
    monkeypatch.syspath_prepend(str(paper_dir))
    sys.modules.pop("scripts.datasets", None)
    return importlib.import_module("scripts.datasets")


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
                    "ST": ["06", "08", "10", "12", "17", "22", "09", "09"],
                }
            ),
            pd.Series([50_000, 60_000, 40_000, 70_000, 30_000, 80_000, 45_000, 65_000]),
        )
    if task_name == "acspubcov":
        return (
            pd.DataFrame(
                {
                    "signal": range(8),
                    "DIS": ["0", "0", "0", "0", "0", "0", "1", "1"],
                }
            ),
            pd.Series([0, 1, 0, 1, 0, 1, 1, 0]),
        )
    raise AssertionError(f"unexpected task {task_name!r}")


def test_unknown_openml_task_raises_clear_runtime_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = _datasets(monkeypatch)

    with pytest.raises(ValueError, match="unknown task 'college_scorecard'"):
        datasets.load_task("college_scorecard", max_train_rows=20, max_eval_rows=10, seed=0)


def test_fetch_openml_frame_retries_with_fresh_cache_on_md5_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = _datasets(monkeypatch)
    calls: list[dict[str, object]] = []

    def fake_fetch_openml(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise ValueError("md5 checksum of local file does not match description")
        return SimpleNamespace(
            data=pd.DataFrame({"feature": [1, 2], "all_missing": [None, None]}),
            target=pd.Series(["yes", "no"]),
        )

    monkeypatch.setattr(datasets, "fetch_openml", fake_fetch_openml)

    feature, label = datasets.fetch_openml_frame(123)

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
    datasets = _datasets(monkeypatch)

    def fake_fetch_openml(**kwargs):
        del kwargs
        raise ValueError("unexpected schema mismatch")

    monkeypatch.setattr(datasets, "fetch_openml", fake_fetch_openml)

    with pytest.raises(ValueError, match="unexpected schema mismatch"):
        datasets.fetch_openml_frame(123)


@pytest.mark.parametrize(
    "task_name",
    ["heloc", "diabetes_readmission", "acsincome", "acspubcov"],
)
def test_ready_openml_tasks_drop_split_drivers(
    monkeypatch: pytest.MonkeyPatch,
    task_name: str,
) -> None:
    datasets = _datasets(monkeypatch)

    def fake_fetch_openml_frame(data_id: int):
        assert data_id == datasets.TASK_IDS[task_name]
        return _ready_task_frame(task_name)

    monkeypatch.setattr(datasets, "fetch_openml_frame", fake_fetch_openml_frame)

    task = datasets.load_task(task_name, max_train_rows=4, max_eval_rows=2, seed=0)

    assert task.task == task_name
    assert list(task.train_feature.columns) == ["signal"]
    assert list(task.source_feature.columns) == ["signal"]
    assert list(task.target_feature.columns) == ["signal"]
    assert task.train_feature.shape == (4, 1)
    assert task.source_feature.shape == (2, 1)
    assert task.target_feature.shape == (2, 1)


def _nsw_frame(datasets) -> pd.DataFrame:
    cps = pd.DataFrame(
        {
            "treat": [0] * 8,
            "age": range(8),
            "educ": range(8),
            "black": [0, 1] * 4,
            "hisp": [1, 0] * 4,
            "married": [0, 0, 1, 1, 0, 0, 1, 1],
            "nodegr": [1, 1, 0, 0, 1, 1, 0, 0],
            "re74": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            "re75": [11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0, 81.0],
            "re78": [12.0, 22.0, 32.0, 42.0, 52.0, 62.0, 72.0, 82.0],
        }
    )
    trt = pd.DataFrame(
        {
            "treat": [1] * 4,
            "age": range(100, 104),
            "educ": range(10, 14),
            "black": [0, 1, 0, 1],
            "hisp": [1, 0, 1, 0],
            "married": [0, 1, 1, 0],
            "nodegr": [1, 0, 1, 0],
            "re74": [1.0, 2.0, 3.0, 4.0],
            "re75": [5.0, 6.0, 7.0, 8.0],
            "re78": [9.0, 10.0, 11.0, 12.0],
        }
    )
    return pd.concat([cps, trt], ignore_index=True)


def test_nsw_task_uses_continuous_earnings_and_comparison_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = _datasets(monkeypatch)
    frame = _nsw_frame(datasets)

    monkeypatch.setattr(datasets, "_fetch_nsw", lambda: frame.copy())

    task = datasets.load_task("nsw", max_train_rows=4, max_eval_rows=2, seed=0)

    assert task.task == "nsw"
    expected_cols = [c for c in datasets._NSW_COLS if c not in {"treat", "re78"}]
    assert list(task.source_feature.columns) == expected_cols
    assert task.train_label.dtype.kind == "f"
    assert task.source_label.dtype.kind == "f"
    assert task.target_label.dtype.kind == "f"
    assert set(task.source_label.tolist()).issubset(set(frame.loc[frame.treat == 0, "re78"].tolist()))
    assert set(task.target_label.tolist()).issubset(set(frame.loc[frame.treat == 1, "re78"].tolist()))


def test_nsw_fetch_recovers_lalonde_comparison_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    datasets = _datasets(monkeypatch)

    try:
        frame = datasets._fetch_nsw()
    except Exception as exc:
        pytest.skip(f"NBER fetch unavailable: {exc}")

    assert len(frame) == 614
    assert int((frame["treat"] == 0).sum()) == 429
    assert int((frame["treat"] == 1).sum()) == 185
    assert pd.api.types.is_float_dtype(frame["re78"])

    task = datasets.load_task("nsw", max_train_rows=30_000, max_eval_rows=4_000, seed=123_456)
    assert task.source_feature.height == 85
    assert task.target_feature.height == 185
    assert task.source_label.dtype.kind == "f"
    assert task.target_label.dtype.kind == "f"
