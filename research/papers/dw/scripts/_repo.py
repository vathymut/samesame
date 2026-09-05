"""Git and filesystem metadata helpers for manuscript reproducibility."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

MANUSCRIPT_DIR = Path(__file__).resolve().parents[1]
ROOT = MANUSCRIPT_DIR.parents[2]
PAPER_DIR = MANUSCRIPT_DIR
RESULTS_DIR = PAPER_DIR / "results"


def repo_commit_hash() -> str:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknown"
    return output.strip()


def repo_has_uncommitted_changes() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return True
    return bool(output.strip())


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_path(path: Path) -> str:
    if path.is_relative_to(ROOT):
        return str(path.relative_to(ROOT))
    return str(path)


def result_metadata(
    script_path: Path,
    arguments: Mapping[str, Any] | Any,
    *,
    input_paths: Sequence[Path] = (),
    **extra: Any,
) -> dict[str, Any]:
    if isinstance(arguments, Mapping):
        serialized_arguments = dict(arguments)
    else:
        serialized_arguments = vars(arguments)
    return {
        "script": script_path.name,
        "commit": repo_commit_hash(),
        "dirty": repo_has_uncommitted_changes(),
        "command": list(sys.argv),
        "script_sha256": file_sha256(script_path),
        "input_sha256": {
            _relative_path(path): file_sha256(path) for path in input_paths
        },
        "arguments": serialized_arguments,
        **extra,
    }
