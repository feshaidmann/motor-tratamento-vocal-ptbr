"""Testes de contrato das portas de estado de job e de executor de áudio.

O objetivo é provar que o ciclo do worker funciona contra um segundo adaptador,
sem SQLite e sem Demucs. Sem este teste as portas parecem código morto e tendem
a ser removidas; ver ``docs/ARCHITECTURE_DECOUPLING.md``.
"""

from __future__ import annotations

import inspect
import tempfile
import unittest
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from motor_vocal.jobs import (
    JobRecord,
    JobRequest,
    LocalJobStore,
    LocalWorker,
    engine_identity,
)
from motor_vocal.pipeline import ProcessingResult, process_audio_file
from motor_vocal.ports import AudioExecutorPort, JobStatePort


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class InMemoryJobStore:
    """Adaptador de estado sem SQLite, para exercitar o contrato."""

    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self.pending: list[str] = []
        self.heartbeats: list[tuple[str, int]] = []

    def enqueue(
        self,
        request: JobRequest,
        idempotency_key: str,
        *,
        max_jobs_per_day: int | None = None,
    ) -> JobRecord:
        for job in self.jobs.values():
            if job.idempotency_key == idempotency_key:
                if job.request != request:
                    raise ValueError("Chave de idempotência reutilizada.")
                return job
        job_id = f"job-{len(self.jobs) + 1}"
        record = JobRecord(
            job_id=job_id,
            idempotency_key=idempotency_key,
            request=request,
            engine_identity=engine_identity(),
            state="queued",
            attempts=0,
            created_at=_now(),
            started_at=None,
            completed_at=None,
            result=None,
            error=None,
        )
        self.jobs[job_id] = record
        self.pending.append(job_id)
        return record

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def claim_next(self) -> JobRecord | None:
        if not self.pending:
            return None
        job_id = self.pending.pop(0)
        record = replace(
            self.jobs[job_id],
            state="running",
            attempts=self.jobs[job_id].attempts + 1,
            started_at=_now(),
        )
        self.jobs[job_id] = record
        return record

    def heartbeat(self, job_id: str, attempt: int) -> bool:
        self.heartbeats.append((job_id, attempt))
        record = self.jobs.get(job_id)
        return bool(record and record.attempts == attempt and record.state == "running")

    def complete(self, job_id: str, attempt: int, result: ProcessingResult) -> None:
        record = self.jobs[job_id]
        if record.attempts != attempt:
            raise ValueError("Tentativa expirada não pode concluir.")
        self.jobs[job_id] = replace(
            record,
            state="succeeded",
            completed_at=_now(),
            result={"mixed_path": str(result.mixed_path)},
        )

    def fail(self, job_id: str, attempt: int, error: str) -> None:
        record = self.jobs[job_id]
        if record.attempts != attempt:
            raise ValueError("Tentativa expirada não pode falhar.")
        self.jobs[job_id] = replace(
            record, state="failed", completed_at=_now(), error=error
        )


class RecordingExecutor:
    """Adaptador de executor que não roda Demucs."""

    def __init__(self, failure: Exception | None = None) -> None:
        self.calls: list[tuple[str, Path]] = []
        self.failure = failure

    def __call__(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        *,
        preset_dsp: str = "Balanceado",
        correct_nasality: bool = True,
        correct_stridency: bool = True,
        correct_sibilance: bool = True,
    ) -> ProcessingResult:
        self.calls.append((str(audio_path), Path(output_dir)))
        if self.failure is not None:
            raise self.failure
        base = Path(output_dir)
        return ProcessingResult(
            mixed_path=base / "mixed.wav",
            vocal_path=base / "vocal.wav",
            processed_vocal_path=base / "vocal_corrigido.wav",
            instrumental_path=base / "instrumental.wav",
            analysis={"preset": preset_dsp},
        )


class JobStatePortContractTests(unittest.TestCase):
    """O mesmo ciclo deve valer para qualquer adaptador de estado."""

    def test_worker_completes_job_against_in_memory_adapter(self) -> None:
        store = InMemoryJobStore()
        executor = RecordingExecutor()
        store.enqueue(JobRequest(input_path="entrada.wav"), "pedido-1")
        with tempfile.TemporaryDirectory() as temp_dir:
            worker = LocalWorker(store, temp_dir, executor=executor)
            finished = worker.run_once()
        self.assertIsNotNone(finished)
        assert finished is not None
        self.assertEqual(finished.state, "succeeded")
        self.assertEqual(finished.attempts, 1)
        self.assertEqual(len(executor.calls), 1)
        self.assertEqual(executor.calls[0][0], "entrada.wav")

    def test_worker_records_failure_without_losing_attempt(self) -> None:
        store = InMemoryJobStore()
        executor = RecordingExecutor(failure=RuntimeError("falha simulada"))
        store.enqueue(JobRequest(input_path="entrada.wav"), "pedido-1")
        with tempfile.TemporaryDirectory() as temp_dir:
            finished = LocalWorker(store, temp_dir, executor=executor).run_once()
        self.assertIsNotNone(finished)
        assert finished is not None
        self.assertEqual(finished.state, "failed")
        self.assertEqual(finished.attempts, 1)
        self.assertIn("falha simulada", finished.error or "")

    def test_worker_returns_none_when_queue_is_empty(self) -> None:
        store = InMemoryJobStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(
                LocalWorker(store, temp_dir, executor=RecordingExecutor()).run_once()
            )

    def test_stale_attempt_cannot_complete(self) -> None:
        store = InMemoryJobStore()
        record = store.enqueue(JobRequest(input_path="entrada.wav"), "pedido-1")
        store.claim_next()
        with self.assertRaises(ValueError):
            store.complete(
                record.job_id,
                0,
                ProcessingResult(
                    mixed_path=Path("m.wav"),
                    vocal_path=Path("v.wav"),
                    processed_vocal_path=Path("p.wav"),
                    instrumental_path=Path("i.wav"),
                    analysis={},
                ),
            )


class PortConformanceTests(unittest.TestCase):
    """As implementações reais devem continuar satisfazendo as portas."""

    def test_local_job_store_matches_job_state_port(self) -> None:
        for name in ("enqueue", "get", "claim_next", "heartbeat", "complete", "fail"):
            with self.subTest(metodo=name):
                self.assertTrue(hasattr(LocalJobStore, name))
                self.assertEqual(
                    str(inspect.signature(getattr(LocalJobStore, name))),
                    str(inspect.signature(getattr(JobStatePort, name))),
                )

    def test_in_memory_store_matches_job_state_port(self) -> None:
        for name in ("enqueue", "get", "claim_next", "heartbeat", "complete", "fail"):
            with self.subTest(metodo=name):
                self.assertEqual(
                    str(inspect.signature(getattr(InMemoryJobStore, name))),
                    str(inspect.signature(getattr(JobStatePort, name))),
                )

    def test_process_audio_file_matches_audio_executor_port(self) -> None:
        port = inspect.signature(AudioExecutorPort.__call__).parameters
        real = inspect.signature(process_audio_file).parameters
        for name, expected in port.items():
            if name == "self":
                continue
            with self.subTest(parametro=name):
                self.assertIn(name, real)
                self.assertEqual(real[name].kind, expected.kind)
                self.assertEqual(real[name].default, expected.default)


if __name__ == "__main__":
    unittest.main()
