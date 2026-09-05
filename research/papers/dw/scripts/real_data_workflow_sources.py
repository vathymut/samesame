"""Backward-compatible re-export facade for the mirrored real-data workflow.

Prefer importing directly from ``scripts._loaders`` in new code.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_openml

from scripts._loaders import (
    LoadedTask,
    TaskRecipe,
    as_series,
    binary_from_membership,
    binary_from_numeric_indicator,
    binary_from_numeric_threshold,
    drop_combined_columns,
    drop_if_present,
    drop_located_columns,
    drop_named_columns,
    label_membership,
    label_numeric_indicator,
    label_numeric_threshold,
    locate_column,
    membership_mask,
    normalize_frame,
    normalize_label,
    normalize_name,
    normalize_state_code,
    normalize_token,
    source_greater_than,
    source_less_equal,
    source_not_in,
    split_column_values,
)
from scripts._loaders._openml_utils import (
    finalize_loaded_task,
    sample_split,
    split_source_pool,
)
from scripts._loaders.nsw import load_nsw_task, nsw_label_from_re78
from scripts._loaders.task_recipes import TASK_RECIPES
from scripts.real_data_workflow_config import TASK_SPECS


def fetch_openml_frame(
    data_id: int,
) -> tuple[pd.DataFrame, pd.Series]:
    try:
        result = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except ValueError as exc:
        if "md5" not in str(exc):
            raise
        import tempfile

        result = fetch_openml(
            data_id=data_id,
            as_frame=True,
            parser="auto",
            data_home=tempfile.mkdtemp(),
        )
    feature = result.data
    target = result.target
    drop_cols = [c for c in feature.columns if c.startswith("all_") or c == "idx"]
    feature = feature.drop(columns=drop_cols)
    return feature, target


Loader = callable  # type: ignore[typearg]


def load_recipe_task_from_frame(
    task_name: str,
    *,
    feature: pd.DataFrame,
    raw_target: pd.Series,
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
) -> LoadedTask:
    recipe = TASK_RECIPES[task_name]
    split_values = recipe.split_values(feature)
    valid = split_values.notna() & raw_target.notna()
    feature = feature.loc[valid].reset_index(drop=True)
    raw_target = raw_target.loc[valid].reset_index(drop=True)
    split_values = split_values.loc[valid].reset_index(drop=True)
    feature = drop_if_present(feature, *recipe.drop_columns(feature))
    label = recipe.label_values(raw_target)
    return finalize_loaded_task(
        task_name,
        feature=feature,
        label=label,
        source_mask=recipe.source_mask(split_values),
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
        seed=seed,
        empty_split_message=recipe.empty_split_message,
    )


def load_recipe_task(
    task_name: str,
    *,
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
) -> LoadedTask:
    feature, raw_target = fetch_openml_frame(TASK_SPECS[task_name].openml_data_id)
    return load_recipe_task_from_frame(
        task_name,
        feature=feature,
        raw_target=raw_target,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
        seed=seed,
    )


CUSTOM_LOADERS: dict[str, callable] = {
    "nsw": load_nsw_task,
}


def load_blocked_task(task_name: str) -> RuntimeError:
    task_spec = TASK_SPECS[task_name]
    reason = (
        task_spec.note
        or "the OpenML mirror does not yet preserve the TableShift task semantics"
    )
    return RuntimeError(
        f"task {task_name!r} is not currently executable from OpenML: {reason}"
    )


def load_task(
    task_name: str,
    *,
    max_train_rows: int,
    max_eval_rows: int,
    seed: int,
) -> LoadedTask:
    if task_name not in TASK_SPECS:
        listed = ", ".join(sorted(TASK_SPECS))
        raise ValueError(f"unknown task {task_name!r}; expected one of: {listed}")
    spec = TASK_SPECS[task_name]
    if spec.status != "ready":
        raise load_blocked_task(task_name)
    if task_name in CUSTOM_LOADERS:
        return CUSTOM_LOADERS[task_name](
            task_name,
            max_train_rows=max_train_rows,
            max_eval_rows=max_eval_rows,
            seed=seed,
        )
    if task_name not in TASK_RECIPES:
        raise RuntimeError(f"task {task_name!r} does not define a loader")
    return load_recipe_task(
        task_name,
        max_train_rows=max_train_rows,
        max_eval_rows=max_eval_rows,
        seed=seed,
    )


__all__ = [
    "CUSTOM_LOADERS",
    "TASK_RECIPES",
    "TASK_SPECS",
    "LoadedTask",
    "TaskRecipe",
    "as_series",
    "binary_from_membership",
    "binary_from_numeric_indicator",
    "binary_from_numeric_threshold",
    "drop_combined_columns",
    "drop_if_present",
    "drop_located_columns",
    "drop_named_columns",
    "fetch_openml",
    "fetch_openml_frame",
    "finalize_loaded_task",
    "label_membership",
    "label_numeric_indicator",
    "label_numeric_threshold",
    "load_blocked_task",
    "load_nsw_task",
    "load_recipe_task",
    "load_recipe_task_from_frame",
    "load_task",
    "locate_column",
    "membership_mask",
    "normalize_frame",
    "normalize_label",
    "normalize_name",
    "normalize_state_code",
    "normalize_token",
    "nsw_label_from_re78",
    "sample_split",
    "source_greater_than",
    "source_less_equal",
    "source_not_in",
    "split_column_values",
    "split_source_pool",
]
