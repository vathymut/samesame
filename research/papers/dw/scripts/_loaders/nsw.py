"""NSW employment dataset loader from the Python DoWhy/NBER source."""

from __future__ import annotations

import pandas as pd

from scripts._loaders import LoadedTask
from scripts._loaders._openml_utils import finalize_loaded_task

_NBER_CONTROL_URL = "https://www.nber.org/~rdehejia/data/nswre74_control.txt"
_NBER_TREATED_URL = "https://www.nber.org/~rdehejia/data/nswre74_treated.txt"
_NSW_COLUMNS = [
    "treat",
    "age",
    "educ",
    "black",
    "hisp",
    "married",
    "nodegr",
    "re74",
    "re75",
    "re78",
]


def _fetch_lalonde_dataframe() -> pd.DataFrame:
    """Return the 445-row Dehejia-Wahba NSW sample.

    Prefers ``dowhy.datasets.lalonde_dataset`` when available (same NBER
    source), otherwise fetches the two NBER text files directly. The
    result is the experimental NSW sample only — 260 controls + 185
    treated — with ``re78`` preserved as continuous earnings in 1978 USD.
    """

    try:
        from dowhy.datasets import lalonde_dataset  # type: ignore[import-not-found]

        frame = lalonde_dataset()
        available = [c for c in _NSW_COLUMNS if c in frame.columns]
        return frame[available].copy()
    except Exception:
        pass

    control = pd.read_csv(
        _NBER_CONTROL_URL, sep=r"\s+", header=None, names=_NSW_COLUMNS
    )
    treated = pd.read_csv(
        _NBER_TREATED_URL, sep=r"\s+", header=None, names=_NSW_COLUMNS
    )
    return pd.concat([control, treated], ignore_index=True)


def load_nsw_task(
    task_name: str,
    *,
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
) -> LoadedTask:
    frame = _fetch_lalonde_dataframe()
    feature = frame.drop(columns=["treat", "re78"])
    raw_target = pd.to_numeric(frame["re78"], errors="coerce")
    split_values = frame["treat"]
    source_mask = split_values == 0
    label = raw_target.astype(float)
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
