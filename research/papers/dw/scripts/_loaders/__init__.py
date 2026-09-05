"""Dataset loaders for the mirrored real-data workflow study."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LoadedTask:
    task: str
    train_feature: pd.DataFrame
    source_feature: pd.DataFrame
    target_feature: pd.DataFrame
    train_label: NDArray[np.int_]
    source_label: NDArray[np.int_]
    target_label: NDArray[np.int_]


Loader = Callable[..., LoadedTask]


@dataclass(frozen=True, slots=True)
class TaskRecipe:
    split_values: Callable[[pd.DataFrame], pd.Series]
    label_values: Callable[[pd.Series], pd.Series]
    source_mask: Callable[[pd.Series], pd.Series]
    drop_columns: Callable[[pd.DataFrame], tuple[str, ...]]
    empty_split_message: str


STATE_CODE_TO_DIVISION = {
    "01": "06", "02": "09", "04": "08", "05": "07", "06": "09",
    "08": "08", "09": "01", "10": "05", "11": "05", "12": "05",
    "13": "05", "15": "09", "16": "08", "17": "03", "18": "03",
    "19": "04", "20": "04", "21": "06", "22": "07", "23": "01",
    "24": "05", "25": "01", "26": "03", "27": "04", "28": "06",
    "29": "04", "30": "08", "31": "04", "32": "08", "33": "01",
    "34": "02", "35": "08", "36": "02", "37": "05", "38": "04",
    "39": "03", "40": "07", "41": "09", "42": "02", "44": "01",
    "45": "05", "46": "04", "47": "06", "48": "07", "49": "08",
    "50": "01", "51": "05", "53": "09", "54": "05", "55": "03",
    "56": "08", "72": "00",
}


def as_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    if isinstance(values, pd.DataFrame):
        if values.shape[1] != 1:
            raise ValueError("label frame must contain exactly one column")
        return values.iloc[:, 0].reset_index(drop=True)
    return pd.Series(values).reset_index(drop=True)


def normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame(frame).replace("?", np.nan)
    normalized = normalized.dropna(axis="columns", how="all")
    return normalized.reset_index(drop=True)


def normalize_label(value: Any) -> str:
    return str(value).strip().lower()


def normalize_token(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text.lower()
    if not np.isfinite(number):
        return text.lower()
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def normalize_name(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def locate_column(frame: pd.DataFrame, *candidates: str) -> str:
    lookup = {normalize_name(column): column for column in frame.columns}
    for candidate in candidates:
        match = lookup.get(normalize_name(candidate))
        if match is not None:
            return match
    listed = ", ".join(frame.columns)
    raise KeyError(f"none of {candidates!r} found in columns: {listed}")


def drop_if_present(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    to_drop = [column for column in columns if column in frame.columns]
    if not to_drop:
        return frame
    return frame.drop(columns=to_drop)


def split_column_values(
    *candidates: str,
    numeric: bool = False,
) -> Callable[[pd.DataFrame], pd.Series]:
    def values(frame: pd.DataFrame) -> pd.Series:
        column = locate_column(frame, *candidates)
        series = frame[column]
        if numeric:
            return pd.to_numeric(series, errors="coerce")
        return series
    return values


def drop_located_columns(*candidates: str) -> Callable[[pd.DataFrame], tuple[str, ...]]:
    def columns(frame: pd.DataFrame) -> tuple[str, ...]:
        return (locate_column(frame, *candidates),)
    return columns


def drop_named_columns(*columns: str) -> Callable[[pd.DataFrame], tuple[str, ...]]:
    def existing(frame: pd.DataFrame) -> tuple[str, ...]:
        return tuple(column for column in columns if column in frame.columns)
    return existing


def drop_combined_columns(
    *getters: Callable[[pd.DataFrame], tuple[str, ...]],
) -> Callable[[pd.DataFrame], tuple[str, ...]]:
    def columns(frame: pd.DataFrame) -> tuple[str, ...]:
        ordered: list[str] = []
        for getter in getters:
            for column in getter(frame):
                if column not in ordered:
                    ordered.append(column)
        return tuple(ordered)
    return columns


def label_membership(*positives: Any) -> Callable[[pd.Series], pd.Series]:
    return lambda label: binary_from_membership(label, positives=positives)


def label_numeric_threshold(
    *, threshold: float, positive_when_leq: bool
) -> Callable[[pd.Series], pd.Series]:
    return lambda label: binary_from_numeric_threshold(
        label, threshold=threshold, positive_when_leq=positive_when_leq,
    )


def label_numeric_indicator(positive_value: int) -> Callable[[pd.Series], pd.Series]:
    return lambda label: binary_from_numeric_indicator(
        label, positive_value=positive_value,
    )


def source_not_in(*accepted: Any) -> Callable[[pd.Series], pd.Series]:
    return lambda values: ~membership_mask(values, accepted=accepted)


def source_greater_than(threshold: float) -> Callable[[pd.Series], pd.Series]:
    return lambda values: pd.to_numeric(values, errors="coerce") > threshold


def source_less_equal(threshold: float) -> Callable[[pd.Series], pd.Series]:
    return lambda values: pd.to_numeric(values, errors="coerce") <= threshold


def binary_from_membership(label: pd.Series, *, positives: tuple[Any, ...]) -> pd.Series:
    normalized_positives = {normalize_token(value) for value in positives}
    return label.map(
        lambda value: normalize_token(value) in normalized_positives
    ).astype(int)


def membership_mask(values: pd.Series, *, accepted: tuple[Any, ...]) -> pd.Series:
    normalized_accepted = {normalize_token(value) for value in accepted}
    return values.map(lambda value: normalize_token(value) in normalized_accepted)


def binary_from_numeric_threshold(
    label: pd.Series, *, threshold: float, positive_when_leq: bool
) -> pd.Series:
    numeric = pd.to_numeric(label, errors="raise")
    if positive_when_leq:
        return (numeric <= threshold).astype(int)
    return (numeric > threshold).astype(int)


def binary_from_numeric_indicator(label: pd.Series, *, positive_value: int) -> pd.Series:
    numeric = pd.to_numeric(label, errors="coerce")
    if numeric.notna().all():
        return (numeric == positive_value).astype(int)
    return binary_from_membership(label, positives=(positive_value,))


def normalize_state_code(value: Any) -> str:
    token = normalize_token(value)
    if not token:
        return ""
    if token.isdigit():
        return f"{int(token):02d}"
    return token.upper()


def derive_acs_division(feature: pd.DataFrame) -> pd.Series:
    state_column = locate_column(feature, "ST", "State", "State_postcode")
    return feature[state_column].map(
        lambda value: STATE_CODE_TO_DIVISION.get(normalize_state_code(value))
    )
