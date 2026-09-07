"""One-time TableShift local-file stager — id/ood parquets, no subsampling.

Agreed spec (Q1-Q20 amendments):
- On-disk terminology is TableShift-native: ``id`` = train+validation+id_test,
  ``ood`` = ood_validation+ood_test. Mapping ``id -> source``,
  ``ood -> target`` happens at experiment time (two ``read_parquet`` calls).
- Local-vs-remote: a task is "cached" iff its parquets exist on disk.
  Missing files are fetched over network only on demand (lazy, per-file).
- This skeleton stages directories + manifest and verifies existing files.
  It never downloads in ``--verify-only`` mode and never requires network
  in tests (fetchers are injected callables).

Layout per task under ``<cache-dir>/<task>/``::

    id.parquet
    ood.parquet

Plus ``<cache-dir>/manifest.json`` with one entry per task.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import typer

app = typer.Typer()

TABLESHIFT_GIT_PIN = "mlfoundations/tableshift@main (pin a SHA before fetching)"

CACHE_ENV_VAR = "SAMESAME_DATA_DIR"
DEFAULT_CACHE_DIR = Path("data/tableshift")

ID_FILENAME = "id.parquet"
OOD_FILENAME = "ood.parquet"
MANIFEST_FILENAME = "manifest.json"

CV_SPEC: dict[str, Any] = {
    "splitter": "StratifiedKFold",
    "n_splits": 10,
    "shuffle": True,
    "seed": 123_456,
}
LABEL_SCORING = "cv-oos-source_plus_full-refit-target"

# All 15 TableShift benchmark tasks (string IDs for get_dataset).
ALL_TASKS: tuple[str, ...] = (
    "acsincome",
    "acspubcov",
    "acsfoodstamps",
    "acsunemployment",
    "brfss_diabetes",
    "brfss_blood_pressure",
    "nhanes_lead",
    "college_scorecard",
    "diabetes_readmission",
    "physionet",
    "assistments",
    "anes",
    "heloc",
    "mimic_extract_los_3",
    "mimic_extract_mort_hosp",
)

TIER_1: tuple[str, ...] = (
    "acsincome",
    "acspubcov",
    "diabetes_readmission",
    "heloc",
)
TIER_2: tuple[str, ...] = (
    "acsfoodstamps",
    "acsunemployment",
    "brfss_diabetes",
    "brfss_blood_pressure",
    "nhanes_lead",
    "college_scorecard",
    "physionet",
)
TIER_3: tuple[str, ...] = (
    "assistments",
    "anes",
    "mimic_extract_los_3",
    "mimic_extract_mort_hosp",
)

TASK_TIERS: dict[str, int] = (
    {t: 1 for t in TIER_1} | {t: 2 for t in TIER_2} | {t: 3 for t in TIER_3}
)

# Reference OOD conditions (DomainSplitter predicates, for manifest docs).
OOD_CONDITIONS: dict[str, str] = {
    "acsincome": "DIVISION=='01'",
    "acspubcov": "DIS==1.0",
    "acsfoodstamps": "DIVISION=='06'",
    "acsunemployment": "SCHL in 01-15 (no HS diploma)",
    "brfss_diabetes": "PRACE1 in {2,3,4,5,6} (ID=[1])",
    "brfss_blood_pressure": "BMI5CAT in {3.0,4.0}",
    "nhanes_lead": "INDFMPIRBelowCutoff==1.0",
    "college_scorecard": "CCBASIC in <8 listed types>",
    "diabetes_readmission": "admission_source_id==7",
    "physionet": "ICULOS>47.0",
    "assistments": "school_id in <10 held-out schools>",
    "anes": "VCF0112=='3.0'",
    "heloc": "ExternalRiskEstimateLow==0",
    "mimic_extract_los_3": "insurance==Medicare",
    "mimic_extract_mort_hosp": "insurance in {Medicare,Medicaid}",
}


def task_cache_paths(task: str, cache_dir: Path) -> tuple[Path, Path]:
    base = Path(cache_dir) / task
    return base / ID_FILENAME, base / OOD_FILENAME


def resolve_tasks(specs: list[str]) -> list[str]:
    """Expand CLI task specs (names, tier1/tier2/tier3, all) preserving order."""
    resolved: list[str] = []
    for spec in specs:
        key = spec.strip().lower()
        if key == "all":
            candidates = list(ALL_TASKS)
        elif key in {"tier1", "tier_1", "tier-1"}:
            candidates = list(TIER_1)
        elif key in {"tier2", "tier_2", "tier-2"}:
            candidates = list(TIER_2)
        elif key in {"tier3", "tier_3", "tier-3"}:
            candidates = list(TIER_3)
        else:
            candidates = [spec.strip()]
        for cand in candidates:
            if cand not in ALL_TASKS:
                raise ValueError(
                    f"unknown task {cand!r}; expected one of {list(ALL_TASKS)}"
                    " or tier1/tier2/tier3/all"
                )
            if cand not in resolved:
                resolved.append(cand)
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_entry(
    task: str,
    *,
    provenance: str,
    id_rows: int | None = None,
    ood_rows: int | None = None,
    id_sha256: str | None = None,
    ood_sha256: str | None = None,
    target_col: str | None = None,
    group_col: str | None = None,
    status: str = "pending_fetch",
) -> dict[str, Any]:
    return {
        "task": task,
        "tier": TASK_TIERS[task],
        "provenance": provenance,
        "status": status,
        "ood_condition": OOD_CONDITIONS[task],
        "target_col": target_col,
        "group_col": group_col,
        "id_rows": id_rows,
        "ood_rows": ood_rows,
        "id_sha256": id_sha256,
        "ood_sha256": ood_sha256,
        "subsampling": "none (full id/ood pools)",
        "cv": dict(CV_SPEC),
        "label_scoring": LABEL_SCORING,
        "tableshift_pin": TABLESHIFT_GIT_PIN,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def read_manifest(cache_dir: Path) -> dict[str, Any]:
    path = Path(cache_dir) / MANIFEST_FILENAME
    if not path.exists():
        return {"tasks": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(cache_dir: Path, entries: dict[str, dict[str, Any]]) -> Path:
    path = Path(cache_dir) / MANIFEST_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tasks": entries, "tableshift_pin": TABLESHIFT_GIT_PIN}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def verify_task_files(
    task: str, cache_dir: Path, manifest: dict[str, Any]
) -> list[str]:
    """Return a list of problems (empty == ok). Pure local check, no network."""
    problems: list[str] = []
    id_path, ood_path = task_cache_paths(task, cache_dir)
    entry = manifest.get("tasks", {}).get(task)
    if entry is None:
        return [f"{task}: missing manifest entry"]
    for label, path, key in (
        ("id", id_path, "id_sha256"),
        ("ood", ood_path, "ood_sha256"),
    ):
        if not path.exists():
            problems.append(f"{task}: missing {label} file {path}")
            continue
        expected = entry.get(key)
        if expected and sha256_file(path) != expected:
            problems.append(f"{task}: {label} sha256 mismatch")
    return problems


# Type for injected fetchers: (task, id_path, ood_path) -> provenance string.
Fetcher = Callable[[str, Path, Path], str]


def _default_fetcher(task: str, id_path: Path, ood_path: Path) -> str:
    del id_path, ood_path
    raise RuntimeError(
        f"no local files for {task!r} and no fetcher configured; "
        "run the isolated tableshift env or drop files in manually "
        f"(see OOD condition {OOD_CONDITIONS[task]!r})"
    )


def stage_task(
    task: str,
    cache_dir: Path,
    *,
    overwrite: bool = False,
    fetcher: Fetcher | None = None,
) -> dict[str, Any]:
    """Ensure local files exist or record pending status. No network by default."""
    id_path, ood_path = task_cache_paths(task, cache_dir)
    id_path.parent.mkdir(parents=True, exist_ok=True)
    fetch = fetcher or _default_fetcher
    if id_path.exists() and ood_path.exists() and not overwrite:
        return manifest_entry(task, provenance="local", status="cached")
    try:
        provenance = fetch(task, id_path, ood_path)
        status = "cached" if id_path.exists() and ood_path.exists() else "pending_fetch"
        return manifest_entry(task, provenance=provenance, status=status)
    except RuntimeError as exc:
        return manifest_entry(
            task,
            provenance="missing_manual" if TASK_TIERS[task] == 3 else "pending_fetch",
            status="missing" if TASK_TIERS[task] == 3 else "pending_fetch",
        ) | {"note": str(exc)}


@app.command()
def main(
    tasks: list[str] = typer.Option(["all"], help="task names or tier1/tier2/tier3/all"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, help="root cache directory"),
    overwrite: bool = typer.Option(False, help="re-fetch even if files exist"),
    verify_only: bool = typer.Option(False, help="verify local files, no fetching"),
) -> None:
    selected = resolve_tasks(list(tasks))
    manifest = read_manifest(cache_dir)
    entries: dict[str, dict[str, Any]] = dict(manifest.get("tasks", {}))
    problems: list[str] = []
    for task in selected:
        if verify_only:
            problems.extend(verify_task_files(task, cache_dir, manifest))
            continue
        entries[task] = stage_task(task, cache_dir, overwrite=overwrite)
    if verify_only:
        if problems:
            for line in problems:
                typer.echo(line)
            raise typer.Exit(code=1)
        typer.echo(f"verified {len(selected)} tasks under {cache_dir}")
        return
    out = write_manifest(cache_dir, entries)
    n_cached = sum(1 for t in selected if entries[t].get("status") == "cached")
    typer.echo(f"staged {len(selected)} tasks ({n_cached} cached) -> {out}")
    for task in selected:
        e = entries[task]
        typer.echo(f"  {task}: tier={e['tier']} status={e['status']} provenance={e['provenance']}")


if __name__ == "__main__":
    app()
