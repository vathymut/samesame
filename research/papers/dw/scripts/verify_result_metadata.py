"""Verify generated result metadata is structurally usable."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from scripts._repo import MANUSCRIPT_DIR, RESULTS_DIR, file_sha256

app = typer.Typer()

REQUIRED_FIELDS = {
    "script",
    "commit",
    "dirty",
    "command",
    "script_sha256",
    "input_sha256",
    "arguments",
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_metadata(path: Path) -> list[str]:
    payload = read_json(path)
    if payload.get("frozen_snapshot") is True:
        return []

    errors: list[str] = []
    missing = sorted(REQUIRED_FIELDS.difference(payload))
    if missing:
        errors.append(f"missing required field(s): {', '.join(missing)}")
        return errors

    script_path = MANUSCRIPT_DIR / "scripts" / str(payload["script"])
    if not script_path.exists():
        errors.append(f"script does not exist: {script_path}")
    elif file_sha256(script_path) != payload["script_sha256"]:
        errors.append(f"script hash is stale: {script_path}")

    input_hashes = payload["input_sha256"]
    if not isinstance(input_hashes, dict):
        errors.append("input_sha256 must be an object")
        return errors

    for relative_path, expected_hash in input_hashes.items():
        input_path = MANUSCRIPT_DIR.parents[2] / relative_path
        if not input_path.exists():
            errors.append(f"input does not exist: {relative_path}")
        elif file_sha256(input_path) != expected_hash:
            errors.append(f"input hash is stale: {relative_path}")
    return errors


@app.command()
def main(
    results_dir: Path = typer.Option(
        RESULTS_DIR,
        help="Directory containing *_metadata.json files.",
    ),
) -> None:
    metadata_paths = sorted(results_dir.glob("*_metadata.json"))
    if not metadata_paths:
        raise ValueError(f"no metadata files found in {results_dir}")

    failures: list[str] = []
    for path in metadata_paths:
        for error in validate_metadata(path):
            failures.append(f"{path}: {error}")
    if failures:
        joined = "\n".join(failures)
        raise SystemExit(f"result metadata verification failed:\n{joined}")


if __name__ == "__main__":
    app()
