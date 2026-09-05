"""OpenML fetch utilities and shared orchestration for the mirrored data workflow."""

from __future__ import annotations

import tempfile

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from scripts._loaders import (
    LoadedTask,
    as_series,
    normalize_frame,
)


def fetch_openml_frame(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    try:
        dataset = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except ValueError as exc:
        if "md5 checksum of local file" not in str(exc):
            raise
        with tempfile.TemporaryDirectory(
            prefix=f"samesame-openml-{data_id}-"
        ) as temp_dir:
            dataset = fetch_openml(
                data_id=data_id,
                as_frame=True,
                parser="auto",
                data_home=temp_dir,
            )
    feature = normalize_frame(pd.DataFrame(dataset.data))
    label = as_series(dataset.target)
    return feature, label


def sample_split(
    feature: pd.DataFrame,
    label: pd.Series,
    *,
    max_rows: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if max_rows is None or len(feature) <= max_rows:
        return feature.reset_index(drop=True), label.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    index = np.sort(rng.choice(len(feature), size=max_rows, replace=False))
    return (
        feature.iloc[index].reset_index(drop=True),
        label.iloc[index].reset_index(drop=True),
    )


def _stratify_or_none(label: pd.Series) -> pd.Series | None:
    counts = label.value_counts()
    if len(counts) < 2 or int(counts.min()) < 2:
        return None
    return label


def split_source_pool(
    feature: pd.DataFrame,
    label: pd.Series,
    *,
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, NDArray[np.int_], NDArray[np.int_]]:
    if len(feature) < 4:
        raise ValueError("source pool must contain at least four rows")

    pooled_feature = feature
    pooled_label = label
    max_total_rows = max_train_rows + max_eval_rows
    if len(pooled_feature) > max_total_rows:
        pooled_feature, pooled_label = sample_split(
            pooled_feature,
            pooled_label,
            max_rows=max_total_rows,
            seed=seed,
        )

    if len(pooled_feature) >= max_total_rows:
        eval_rows = max_eval_rows
        train_rows = max_train_rows
    else:
        eval_rows = max(1, len(pooled_feature) // 5)
        train_rows = min(max_train_rows, len(pooled_feature) - eval_rows)
        eval_rows = len(pooled_feature) - train_rows
    if train_rows < 1 or eval_rows < 1:
        raise ValueError("source pool split produced an empty partition")

    train_feature, source_feature, train_label, source_label = train_test_split(
        pooled_feature,
        pooled_label,
        train_size=train_rows,
        test_size=eval_rows,
        stratify=_stratify_or_none(pooled_label),
        random_state=seed,
    )
    return (
        train_feature.reset_index(drop=True),
        source_feature.reset_index(drop=True),
        train_label.to_numpy(dtype=int),
        source_label.to_numpy(dtype=int),
    )


def finalize_loaded_task(
    task_name: str,
    *,
    feature: pd.DataFrame,
    label: pd.Series,
    source_mask: pd.Series,
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
    empty_split_message: str,
) -> LoadedTask:
    target_mask = ~source_mask
    if not source_mask.any() or not target_mask.any():
        raise ValueError(empty_split_message)

    source_pool_feature = feature.loc[source_mask].reset_index(drop=True)
    source_pool_label = label.loc[source_mask].reset_index(drop=True)
    target_feature = feature.loc[target_mask].reset_index(drop=True)
    target_label = label.loc[target_mask].reset_index(drop=True)
    target_feature, target_label = sample_split(
        target_feature,
        target_label,
        max_rows=max_eval_rows,
        seed=seed + 1,
    )

    train_feature, source_feature, train_label, source_label = split_source_pool(
        source_pool_feature,
        source_pool_label,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
        seed=seed,
    )
    return LoadedTask(
        task=task_name,
        train_feature=train_feature,
        source_feature=source_feature,
        target_feature=target_feature,
        train_label=train_label,
        source_label=source_label,
        target_label=target_label.to_numpy(dtype=int),
    )
