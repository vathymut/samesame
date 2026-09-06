"""Minimal I/O and repo helpers."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import polars as pl

MANUSCRIPT_DIR = Path(__file__).resolve().parents[1]
ROOT = MANUSCRIPT_DIR.parents[2]
PAPER_DIR = MANUSCRIPT_DIR
RESULTS_DIR = PAPER_DIR / "results"


def repo_commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return "unknown"
    return out.strip()


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("rows must not be empty")
    ensure_directory(path.parent)
    pl.DataFrame(rows).write_csv(path)


def read_csv(path: Path) -> list[dict[str, Any]]:
    return pl.read_csv(path).to_dicts()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        handle.write("\n")


def result_metadata(script_path: Path, arguments: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "script": script_path.name,
        "commit": repo_commit_hash(),
        "arguments": arguments,
        **extra,
    }


_EXPERIMENT_METRIC_KEYS: tuple[str, ...] = (
    "statistic",
    "pvalue",
    "reject",
    "source_ess",
    "target_ess",
    "source_max_weight",
    "target_max_weight",
)


def summarize_rows(rows: list[dict[str, Any]], group_keys: tuple[str, ...]) -> list[dict[str, Any]]:
    if not rows:
        return []
    df = pl.DataFrame(rows)
    agg = [pl.col(k).mean().alias(k) for k in _EXPERIMENT_METRIC_KEYS if k in df.columns] + [pl.len().alias("count")]
    return df.group_by(group_keys).agg(agg).sort(group_keys).to_dicts()
