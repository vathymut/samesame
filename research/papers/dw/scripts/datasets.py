"""Dataset loaders for real-data workflow — explicit per-task logic."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

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

TASK_IDS: dict[str, int] = {
    "heloc": 46932,
    "diabetes_readmission": 46922,
    "acsincome": 43141,
    "acspubcov": 43140,
}


@dataclass(frozen=True, slots=True)
class LoadedTask:
    task: str
    train_feature: pd.DataFrame
    source_feature: pd.DataFrame
    target_feature: pd.DataFrame
    train_label: NDArray[np.int_] | NDArray[np.float64]
    source_label: NDArray[np.int_] | NDArray[np.float64]
    target_label: NDArray[np.int_] | NDArray[np.float64]


# --- small helpers ---

def _normalize_token(v: Any) -> str:
    if pd.isna(v):
        return ""
    t = str(v).strip()
    if not t:
        return ""
    try:
        n = float(t)
    except ValueError:
        return t.lower()
    if not np.isfinite(n):
        return t.lower()
    return str(int(n)) if n.is_integer() else f"{n:g}"


def _normalize_name(s: str) -> str:
    return "".join(c for c in s.lower() if c.isalnum())


def _locate(frame: pd.DataFrame, *candidates: str) -> str:
    lookup = {_normalize_name(c): c for c in frame.columns}
    for cand in candidates:
        m = lookup.get(_normalize_name(cand))
        if m is not None:
            return m
    raise KeyError(f"none of {candidates!r} found in {list(frame.columns)}")


def _as_series(v: Any) -> pd.Series:
    if isinstance(v, pd.Series):
        return v.reset_index(drop=True)
    if isinstance(v, pd.DataFrame):
        if v.shape[1] != 1:
            raise ValueError("label frame must have one column")
        return v.iloc[:, 0].reset_index(drop=True)
    return pd.Series(v).reset_index(drop=True)


def _normalize_frame(frame: pd.DataFrame) -> pd.DataFrame:
    n = pd.DataFrame(frame).replace("?", np.nan)
    n = n.dropna(axis="columns", how="all")
    return n.reset_index(drop=True)


def _state_division(feature: pd.DataFrame) -> pd.Series:
    col = _locate(feature, "ST", "State", "State_postcode")
    def _map(v: Any) -> str | None:
        tok = _normalize_token(v)
        if not tok:
            return None
        code = f"{int(tok):02d}" if tok.isdigit() else tok.upper()
        return STATE_CODE_TO_DIVISION.get(code)
    return feature[col].map(_map)


# --- OpenML fetch + splitting ---

def fetch_openml_frame(data_id: int) -> tuple[pd.DataFrame, pd.Series]:
    try:
        ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto")
    except ValueError as exc:
        if "md5 checksum" not in str(exc):
            raise
        with tempfile.TemporaryDirectory(prefix=f"samesame-openml-{data_id}-") as td:
            ds = fetch_openml(data_id=data_id, as_frame=True, parser="auto", data_home=td)
    return _normalize_frame(pd.DataFrame(ds.data)), _as_series(ds.target)


def _sample_split(feature: pd.DataFrame, label: pd.Series, *, max_rows: int | None, seed: int):
    if max_rows is None or len(feature) <= max_rows:
        return feature.reset_index(drop=True), label.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(feature), size=max_rows, replace=False))
    return feature.iloc[idx].reset_index(drop=True), label.iloc[idx].reset_index(drop=True)


def _split_source_pool(feature: pd.DataFrame, label: pd.Series, *, max_train_rows: int, max_eval_rows: int, seed: int):
    if len(feature) < 4:
        raise ValueError("source pool must contain at least four rows")
    max_total = max_train_rows + max_eval_rows
    pf, pl = feature, label
    if len(pf) > max_total:
        pf, pl = _sample_split(pf, pl, max_rows=max_total, seed=seed)
    if len(pf) >= max_total:
        eval_rows, train_rows = max_eval_rows, max_train_rows
    else:
        eval_rows = max(1, len(pf) // 5)
        train_rows = min(max_train_rows, len(pf) - eval_rows)
        eval_rows = len(pf) - train_rows
    strat = pl if pl.value_counts().min() >= 2 and len(pl.value_counts()) >= 2 else None
    tr_f, src_f, tr_l, src_l = train_test_split(pf, pl, train_size=train_rows, test_size=eval_rows, stratify=strat, random_state=seed)
    dtype = float if pd.api.types.is_float_dtype(pl) else int
    return tr_f.reset_index(drop=True), src_f.reset_index(drop=True), tr_l.to_numpy(dtype=dtype), src_l.to_numpy(dtype=dtype)


def _finalize(task_name: str, *, feature: pd.DataFrame, label: pd.Series, source_mask: pd.Series, max_train_rows: int, max_eval_rows: int, seed: int, msg: str) -> LoadedTask:
    if not source_mask.any() or not (~source_mask).any():
        raise ValueError(msg)
    src_pool_f = feature.loc[source_mask].reset_index(drop=True)
    src_pool_l = label.loc[source_mask].reset_index(drop=True)
    tgt_f = feature.loc[~source_mask].reset_index(drop=True)
    tgt_l = label.loc[~source_mask].reset_index(drop=True)
    tgt_f, tgt_l = _sample_split(tgt_f, tgt_l, max_rows=max_eval_rows, seed=seed + 1)
    tr_f, src_f, tr_l, src_l = _split_source_pool(src_pool_f, src_pool_l, max_train_rows=max_train_rows, max_eval_rows=max_eval_rows, seed=seed)
    dtype = float if pd.api.types.is_float_dtype(label) else int
    return LoadedTask(task=task_name, train_feature=tr_f, source_feature=src_f, target_feature=tgt_f, train_label=tr_l, source_label=src_l, target_label=tgt_l.to_numpy(dtype=dtype))


# --- per-task loaders ---

def _load_heloc(feature: pd.DataFrame, raw: pd.Series, **kw) -> LoadedTask:
    col = _locate(feature, "ExternalRiskEstimate", "External Risk Estimate")
    vals = pd.to_numeric(feature[col], errors="coerce")
    mask = vals > 63
    label = raw.map(lambda v: _normalize_token(v) == "bad").astype(int)
    feat = feature.drop(columns=[col])
    return _finalize("heloc", feature=feat, label=label, source_mask=mask, msg="HELOC split must produce non-empty pools", **kw)


def _load_diabetes(feature: pd.DataFrame, raw: pd.Series, **kw) -> LoadedTask:
    col = _locate(feature, "admission_source_id")
    vals = feature[col]
    # source = not in (7, Emergency Room)
    mask = ~vals.map(lambda v: _normalize_token(v) in {"7", "emergency room"})
    label = raw.map(lambda v: _normalize_token(v) == "yes").astype(int)
    feat = feature.drop(columns=[c for c in ["encounter_id", "patient_nbr", col] if c in feature.columns])
    return _finalize("diabetes_readmission", feature=feat, label=label, source_mask=mask, msg="readmission split must produce non-empty pools", **kw)


def _load_acsincome(feature: pd.DataFrame, raw: pd.Series, **kw) -> LoadedTask:
    div = _state_division(feature)
    mask = div != "01"
    label = (pd.to_numeric(raw, errors="raise") <= 56000).astype(int)
    feat = feature.drop(columns=[_locate(feature, "ST", "State", "State_postcode")])
    # need to align feature/label with valid div
    valid = div.notna() & raw.notna()
    feat = feat.loc[valid].reset_index(drop=True)
    label = label.loc[valid].reset_index(drop=True)
    mask = mask.loc[valid].reset_index(drop=True)
    return _finalize("acsincome", feature=feat, label=label, source_mask=mask, msg="ACS income split must produce non-empty pools", **kw)


def _load_acspubcov(feature: pd.DataFrame, raw: pd.Series, **kw) -> LoadedTask:
    col = _locate(feature, "DIS")
    mask = ~feature[col].map(lambda v: _normalize_token(v) in {"1", "1.0", "01", "with a disability"})
    # label PUBCOV == 1
    numeric = pd.to_numeric(raw, errors="coerce")
    if numeric.notna().all():
        label = (numeric == 1).astype(int)
    else:
        label = raw.map(lambda v: _normalize_token(v) == "1").astype(int)
    feat = feature.drop(columns=[col])
    return _finalize("acspubcov", feature=feat, label=label, source_mask=mask, msg="ACS public coverage split must produce non-empty pools", **kw)


_NSW_COLS = ["treat","age","educ","black","hisp","married","nodegr","re74","re75","re78"]
_NBER_CONTROL = "https://www.nber.org/~rdehejia/data/nswre74_control.txt"
_NBER_TREATED = "https://www.nber.org/~rdehejia/data/nswre74_treated.txt"

def _fetch_nsw() -> pd.DataFrame:
    try:
        from dowhy.datasets import lalonde_dataset  # type: ignore
        frame = lalonde_dataset()
        return frame[[c for c in _NSW_COLS if c in frame.columns]].copy()
    except Exception:
        pass
    ctrl = pd.read_csv(_NBER_CONTROL, sep=r"\s+", header=None, names=_NSW_COLS)
    trt = pd.read_csv(_NBER_TREATED, sep=r"\s+", header=None, names=_NSW_COLS)
    return pd.concat([ctrl, trt], ignore_index=True)


def load_nsw_task(task_name: str, *, max_train_rows: int, max_eval_rows: int, seed: int) -> LoadedTask:
    frame = _fetch_nsw()
    feat = frame.drop(columns=["treat","re78"])
    label = pd.to_numeric(frame["re78"], errors="coerce").astype(float)
    mask = frame["treat"] == 0
    return _finalize(task_name, feature=feat, label=label, source_mask=mask, msg="NSW split must produce non-empty treat pools", max_train_rows=max_train_rows, max_eval_rows=max_eval_rows, seed=seed)


# --- public entry ---

def load_task(task_name: str, *, max_train_rows: int, max_eval_rows: int, seed: int) -> LoadedTask:
    if task_name == "nsw":
        return load_nsw_task(task_name, max_train_rows=max_train_rows, max_eval_rows=max_eval_rows, seed=seed)
    if task_name not in TASK_IDS:
        raise ValueError(f"unknown task {task_name!r}; expected one of: {sorted(TASK_IDS)} + ['nsw']")
    feature, raw = fetch_openml_frame(TASK_IDS[task_name])
    # acsincome needs special valid handling before finalize; others generic
    common_kw = dict(max_train_rows=max_train_rows, max_eval_rows=max_eval_rows, seed=seed)
    # filter rows where split val or label is missing
    if task_name == "acsincome":
        return _load_acsincome(feature, raw, **common_kw)
    # generic valid filtering
    # derive split vals to filter na (for heloc/dis etc. we filter inside, but do here too)
    valid = raw.notna()
    feature, raw = feature.loc[valid].reset_index(drop=True), raw.loc[valid].reset_index(drop=True)
    if task_name == "heloc":
        return _load_heloc(feature, raw, **common_kw)
    if task_name == "diabetes_readmission":
        return _load_diabetes(feature, raw, **common_kw)
    if task_name == "acspubcov":
        return _load_acspubcov(feature, raw, **common_kw)
    raise RuntimeError(f"no loader for {task_name!r}")
