"""Inventário somente leitura para planejar retenção dos trabalhos locais."""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


JOB_ID_PATTERN = re.compile(r"[0-9a-f]{32}\Z")
UPLOAD_PATTERN = re.compile(r"[0-9a-f]{64}\.(?:wav|mp3)\Z")


def _bytes_in_directory(directory: Path) -> int:
    if not directory.is_dir() or directory.is_symlink():
        return 0
    total = 0
    for root, directories, files in os.walk(directory, followlinks=False):
        directories[:] = [
            name for name in directories if not (Path(root) / name).is_symlink()
        ]
        for name in files:
            path = Path(root) / name
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
    return total


def preview_local_retention(
    data_dir: str | Path,
    *,
    older_than: datetime,
) -> dict[str, Any]:
    """Lista candidatos antigos sem criar, alterar ou apagar arquivos."""
    if older_than.tzinfo is None:
        raise ValueError("A data de corte deve ter fuso horário.")
    root = Path(data_dir).resolve()
    database = root / "jobs.sqlite3"
    cutoff = older_than.astimezone(timezone.utc).isoformat()
    report: dict[str, Any] = {
        "cutoff_utc": cutoff,
        "eligible_jobs": [],
        "artifact_directories": [],
        "upload_files": [],
        "artifact_bytes": 0,
        "upload_bytes": 0,
        "invalid_records": 0,
        "uploads_not_evaluated": False,
        "database_present": database.is_file(),
    }
    if not database.is_file():
        return report

    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as connection:
        rows = connection.execute(
            "SELECT job_id, request_json, state, completed_at FROM jobs"
        ).fetchall()

    retained_uploads: set[Path] = set()
    eligible_uploads: set[Path] = set()
    uploads_root = (root / "uploads").resolve()
    for job_id, request_json, state, completed_at in rows:
        if not JOB_ID_PATTERN.fullmatch(job_id):
            report["invalid_records"] += 1
            continue
        try:
            input_path = Path(json.loads(request_json)["input_path"]).resolve()
        except (KeyError, TypeError, ValueError):
            report["invalid_records"] += 1
            continue
        eligible = False
        if state in {"succeeded", "failed"} and completed_at is not None:
            try:
                completed = datetime.fromisoformat(completed_at)
            except (TypeError, ValueError):
                report["invalid_records"] += 1
                continue
            if completed.tzinfo is None:
                report["invalid_records"] += 1
                continue
            eligible = completed.astimezone(timezone.utc) < older_than.astimezone(
                timezone.utc
            )
        if input_path.parent == uploads_root and UPLOAD_PATTERN.fullmatch(input_path.name):
            (eligible_uploads if eligible else retained_uploads).add(input_path)
        if not eligible:
            continue
        report["eligible_jobs"].append(job_id)
        artifact_dir = root / "artifacts" / job_id
        if artifact_dir.is_dir() and not artifact_dir.is_symlink():
            report["artifact_directories"].append(str(artifact_dir))
            report["artifact_bytes"] += _bytes_in_directory(artifact_dir)

    if report["invalid_records"]:
        report["uploads_not_evaluated"] = True
        return report
    for path in sorted(eligible_uploads - retained_uploads):
        if path.is_file() and not path.is_symlink():
            report["upload_files"].append(str(path))
            report["upload_bytes"] += path.stat().st_size
    report["eligible_jobs"].sort()
    report["artifact_directories"].sort()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prévia somente leitura dos arquivos antigos da fila local."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "outputs" / "jobs",
    )
    parser.add_argument("--older-than-days", type=float, required=True)
    args = parser.parse_args()
    if args.older_than_days < 0:
        parser.error("--older-than-days deve ser não negativo")
    cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than_days)
    print(json.dumps(preview_local_retention(args.data_dir, older_than=cutoff), indent=2))


if __name__ == "__main__":
    main()
