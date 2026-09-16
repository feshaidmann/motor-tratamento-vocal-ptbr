"""Contrato v1 e fila SQLite para desenvolvimento local com um worker."""

from __future__ import annotations

import json
import hashlib
import importlib.metadata
import platform
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

from motor_vocal.pipeline import ProcessingResult, process_audio_file
from motor_vocal.processing import DSP_PRESETS


SCHEMA_VERSION = 1
PILOT_TIMEZONE = ZoneInfo("America/Sao_Paulo")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def engine_identity() -> dict[str, Any]:
    """Identifica fontes e runtime usados por um trabalho local."""
    digest = hashlib.sha256()
    project_root = Path(__file__).resolve().parent.parent
    for relative_path in (
        "motor_vocal/jobs.py",
        "motor_vocal/pipeline.py",
        "motor_vocal/processing.py",
        "motor_vocal/quality.py",
        "motor_vocal/separation.py",
        "requirements.txt",
    ):
        path = project_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(path.read_bytes())
    packages: dict[str, str | None] = {}
    for distribution in ("demucs", "librosa", "numpy", "pedalboard", "soundfile", "torch"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "source_sha256": digest.hexdigest(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
    }


@dataclass(frozen=True)
class JobRequest:
    input_path: str
    preset_dsp: str = "Balanceado"
    correct_nasality: bool = True
    correct_stridency: bool = True
    correct_sibilance: bool = True
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("Versão de contrato não suportada.")
        if not self.input_path:
            raise ValueError("O caminho de entrada é obrigatório.")
        if self.preset_dsp not in DSP_PRESETS:
            raise ValueError(f"Preset DSP desconhecido: {self.preset_dsp}")


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    idempotency_key: str
    request: JobRequest
    engine_identity: dict[str, Any]
    state: str
    attempts: int
    created_at: str
    started_at: str | None
    completed_at: str | None
    result: dict[str, Any] | None
    error: str | None


def _record(row: sqlite3.Row) -> JobRecord:
    return JobRecord(
        job_id=row["job_id"],
        idempotency_key=row["idempotency_key"],
        request=JobRequest(**json.loads(row["request_json"])),
        engine_identity=json.loads(row["engine_json"]),
        state=row["state"],
        attempts=row["attempts"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        result=json.loads(row["result_json"]) if row["result_json"] else None,
        error=row["error"],
    )


class LocalJobStore:
    """Persistência transacional local; cada operação usa sua própria conexão."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    request_json TEXT NOT NULL,
                    engine_json TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (
                        state IN ('queued', 'running', 'succeeded', 'failed')
                    ),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    result_json TEXT,
                    error TEXT
                )"""
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def enqueue(
        self,
        request: JobRequest,
        idempotency_key: str,
        *,
        max_jobs_per_day: int | None = None,
    ) -> JobRecord:
        if not idempotency_key.strip():
            raise ValueError("A chave de idempotência é obrigatória.")
        if max_jobs_per_day is not None and max_jobs_per_day < 1:
            raise ValueError("O limite diário deve ser positivo.")
        payload = json.dumps(asdict(request), sort_keys=True, separators=(",", ":"))
        identity_payload = json.dumps(engine_identity(), sort_keys=True)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing:
                if existing["request_json"] != payload:
                    raise ValueError("Chave de idempotência usada com outra entrada.")
                if existing["engine_json"] != identity_payload:
                    raise ValueError("Chave de idempotência usada com outra build.")
                return _record(existing)
            if max_jobs_per_day is not None:
                local_day = datetime.now(PILOT_TIMEZONE).date()
                day_start = datetime.combine(
                    local_day, datetime.min.time(), PILOT_TIMEZONE
                ).astimezone(timezone.utc)
                day_end = (day_start.astimezone(PILOT_TIMEZONE) + timedelta(days=1)).astimezone(timezone.utc)
                accepted = connection.execute(
                    "SELECT COUNT(*) FROM jobs WHERE created_at >= ? AND created_at < ?",
                    (day_start.isoformat(), day_end.isoformat()),
                ).fetchone()[0]
                if accepted >= max_jobs_per_day:
                    raise ValueError("Limite diário de processamentos atingido.")
            job_id = uuid.uuid4().hex
            connection.execute(
                """INSERT INTO jobs (
                    job_id, idempotency_key, request_json, engine_json, state, created_at
                ) VALUES (?, ?, ?, ?, 'queued', ?)""",
                (job_id, idempotency_key, payload, identity_payload, _now()),
            )
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return _record(row)

    def get(self, job_id: str) -> JobRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            return _record(row) if row else None

    def claim_next(self) -> JobRecord | None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """SELECT * FROM jobs WHERE state = 'queued'
                ORDER BY created_at, rowid LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """UPDATE jobs SET state = 'running', attempts = attempts + 1,
                    started_at = ? WHERE job_id = ?""",
                (_now(), row["job_id"]),
            )
            claimed = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?", (row["job_id"],)
            ).fetchone()
            return _record(claimed)

    def recover_stale(self, *, max_age_seconds: int = 900) -> int:
        """Recoloca em fila apenas trabalhos sem heartbeat há tempo suficiente."""
        if max_age_seconds < 1:
            raise ValueError("A idade mínima deve ser positiva.")
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=max_age_seconds)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET state = 'queued', started_at = NULL
                    WHERE state = 'running' AND started_at < ?""",
                (cutoff,),
            )
            return cursor.rowcount

    def heartbeat(self, job_id: str, attempt: int) -> bool:
        """Renova a atividade da tentativa ainda em execução."""
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET started_at = ?
                    WHERE job_id = ? AND state = 'running' AND attempts = ?""",
                (_now(), job_id, attempt),
            )
            return cursor.rowcount == 1

    def complete(self, job_id: str, attempt: int, result: ProcessingResult) -> None:
        payload = json.dumps(
            {
                "mixed_path": str(result.mixed_path),
                "vocal_path": str(result.vocal_path),
                "processed_vocal_path": str(result.processed_vocal_path),
                "instrumental_path": str(result.instrumental_path),
                "analysis": result.analysis,
            },
            ensure_ascii=False,
        )
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET state = 'succeeded', completed_at = ?,
                    result_json = ?, error = NULL
                    WHERE job_id = ? AND state = 'running' AND attempts = ?""",
                (_now(), payload, job_id, attempt),
            )
            if cursor.rowcount != 1:
                raise ValueError("Trabalho não encontrado ou tentativa substituída.")

    def fail(self, job_id: str, attempt: int, error: str) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """UPDATE jobs SET state = 'failed', completed_at = ?, error = ?
                    WHERE job_id = ? AND state = 'running' AND attempts = ?""",
                (_now(), error[:2_000], job_id, attempt),
            )
            if cursor.rowcount != 1:
                raise ValueError("Trabalho não encontrado ou tentativa substituída.")


class LocalWorker:
    """Executa um trabalho por chamada; o controle de loop fica no invocador."""

    def __init__(self, store: LocalJobStore, output_root: str | Path) -> None:
        self.store = store
        self.output_root = Path(output_root)

    def run_once(self) -> JobRecord | None:
        job = self.store.claim_next()
        if job is None:
            return None
        heartbeat_stop = threading.Event()

        def renew_lease() -> None:
            while not heartbeat_stop.wait(5.0):
                try:
                    if not self.store.heartbeat(job.job_id, job.attempts):
                        break
                except sqlite3.Error:
                    continue

        heartbeat_thread = threading.Thread(
            target=renew_lease,
            name=f"motor-vocal-heartbeat-{job.job_id[:8]}",
            daemon=True,
        )
        heartbeat_thread.start()
        try:
            if job.engine_identity != engine_identity():
                raise RuntimeError("A build mudou desde a criação do trabalho.")
            request = job.request
            result = process_audio_file(
                request.input_path,
                self.output_root / job.job_id / f"attempt_{job.attempts}",
                preset_dsp=request.preset_dsp,
                correct_nasality=request.correct_nasality,
                correct_stridency=request.correct_stridency,
                correct_sibilance=request.correct_sibilance,
            )
            self.store.complete(job.job_id, job.attempts, result)
        except Exception as exc:
            try:
                self.store.fail(
                    job.job_id, job.attempts, f"{type(exc).__name__}: {exc}"
                )
            except ValueError:
                # Um claim mais novo já assumiu o trabalho recuperado.
                pass
        finally:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1.0)
        return self.store.get(job.job_id)
