"""NSW employment dataset loader from local CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts._loaders import LoadedTask
from scripts._loaders._openml_utils import finalize_loaded_task


def nsw_label_from_re78(raw: pd.Series) -> pd.Series:
    return (raw > 5000.0).astype(int)


def load_nsw_task(
    task_name: str,
    *,
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
) -> LoadedTask:
    csv_path = (
        Path(__file__).resolve().parents[2] / "data" / "nsw" / "lalonde.csv"
    )
    frame = pd.read_csv(csv_path)
    feature = frame.drop(columns=["rownames", "treat", "re78"])
    raw_target = frame["re78"]
    split_values = frame["treat"]
    source_mask = split_values == 0
    label = nsw_label_from_re78(raw_target)
    return finalize_loaded_task(
        task_name,
        feature=feature,
        label=label,
        source_mask=source_mask,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
        seed=seed,
        empty_split_message="NSW split must produce non-empty treat pools",
    )
