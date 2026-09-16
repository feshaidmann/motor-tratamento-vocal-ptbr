"""Estados, idempotência e reclamação concorrente da fila local."""

from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from motor_vocal.jobs import JobRequest, LocalJobStore, LocalWorker
from motor_vocal.pipeline import ProcessingResult


class LocalJobsTests(unittest.TestCase):
    def test_idempotency_returns_same_job_and_rejects_changed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalJobStore(Path(temp_dir) / "jobs.sqlite3")
            request = JobRequest(input_path="entrada.wav")
            first = store.enqueue(request, "pedido-1")
            second = store.enqueue(request, "pedido-1")
            self.assertEqual(first.job_id, second.job_id)
            self.assertEqual(first.state, "queued")
            with self.assertRaises(ValueError):
                store.enqueue(JobRequest(input_path="outra.wav"), "pedido-1")

    def test_concurrent_claims_assign_each_job_once(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalJobStore(Path(temp_dir) / "jobs.sqlite3")
            with ThreadPoolExecutor(max_workers=5) as executor:
                enqueued = list(
                    executor.map(
                        lambda index: store.enqueue(
                            JobRequest(input_path=f"entrada_{index}.wav"),
                            f"pedido-{index}",
                        ),
                        range(10),
                    )
                )
                claimed = list(executor.map(lambda _: store.claim_next(), range(10)))
            self.assertEqual(len({job.job_id for job in enqueued}), 10)
            self.assertEqual(len({job.job_id for job in claimed}), 10)
            self.assertTrue(all(job.state == "running" for job in claimed))
            self.assertTrue(all(job.attempts == 1 for job in claimed))
            self.assertIsNone(store.claim_next())

    def test_worker_persists_success_and_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = LocalJobStore(root / "jobs.sqlite3")
            worker = LocalWorker(store, root / "outputs")
            success = store.enqueue(JobRequest(input_path="entrada.wav"), "sucesso")
            expected = ProcessingResult(
                mixed_path=root / "mix.wav",
                vocal_path=root / "vocal.wav",
                processed_vocal_path=root / "corrigido.wav",
                instrumental_path=root / "instrumental.wav",
                analysis={"controle_qualidade_saida": {"status": "aprovado"}},
            )
            with patch("motor_vocal.jobs.process_audio_file", return_value=expected):
                completed = worker.run_once()
            self.assertEqual(completed.job_id, success.job_id)
            self.assertEqual(completed.state, "succeeded")
            self.assertEqual(completed.result["mixed_path"], str(expected.mixed_path))

            failed = store.enqueue(JobRequest(input_path="inexistente.wav"), "falha")
            with patch("motor_vocal.jobs.process_audio_file", side_effect=FileNotFoundError("sem áudio")):
                failure = worker.run_once()
            self.assertEqual(failure.job_id, failed.job_id)
            self.assertEqual(failure.state, "failed")
            self.assertIn("FileNotFoundError", failure.error)
            self.assertIsNone(worker.run_once())

    def test_daily_limit_counts_new_jobs_but_allows_idempotent_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalJobStore(Path(temp_dir) / "jobs.sqlite3")
            first = store.enqueue(
                JobRequest(input_path="um.wav"), "um", max_jobs_per_day=2
            )
            store.enqueue(JobRequest(input_path="dois.wav"), "dois", max_jobs_per_day=2)
            self.assertEqual(
                store.enqueue(
                    JobRequest(input_path="um.wav"), "um", max_jobs_per_day=2
                ).job_id,
                first.job_id,
            )
            with self.assertRaisesRegex(ValueError, "Limite diário"):
                store.enqueue(
                    JobRequest(input_path="tres.wav"), "tres", max_jobs_per_day=2
                )

    def test_stale_running_job_returns_to_queue_for_next_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalJobStore(Path(temp_dir) / "jobs.sqlite3")
            original = store.enqueue(JobRequest(input_path="um.wav"), "um")
            store.claim_next()
            old = (datetime.now(timezone.utc) - timedelta(minutes=20)).isoformat()
            with store._connect() as connection:
                connection.execute(
                    "UPDATE jobs SET started_at = ? WHERE job_id = ?",
                    (old, original.job_id),
                )
            self.assertTrue(store.heartbeat(original.job_id, 1))
            self.assertEqual(store.recover_stale(max_age_seconds=900), 0)
            with store._connect() as connection:
                connection.execute(
                    "UPDATE jobs SET started_at = ? WHERE job_id = ?",
                    (old, original.job_id),
                )
            self.assertEqual(store.recover_stale(max_age_seconds=900), 1)
            reclaimed = store.claim_next()
            self.assertEqual(reclaimed.job_id, original.job_id)
            self.assertEqual(reclaimed.attempts, 2)
            stale_result = ProcessingResult(
                mixed_path=Path(temp_dir) / "mix.wav",
                vocal_path=Path(temp_dir) / "vocal.wav",
                processed_vocal_path=Path(temp_dir) / "corrigido.wav",
                instrumental_path=Path(temp_dir) / "instrumental.wav",
                analysis={},
            )
            with self.assertRaisesRegex(ValueError, "tentativa substituída"):
                store.complete(original.job_id, 1, stale_result)
            self.assertEqual(store.get(original.job_id).state, "running")

    def test_daily_limit_is_atomic_under_concurrent_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = LocalJobStore(Path(temp_dir) / "jobs.sqlite3")

            def submit(index: int) -> bool:
                try:
                    store.enqueue(
                        JobRequest(input_path=f"entrada_{index}.wav"),
                        f"pedido-{index}",
                        max_jobs_per_day=5,
                    )
                    return True
                except ValueError:
                    return False

            with ThreadPoolExecutor(max_workers=10) as executor:
                accepted = list(executor.map(submit, range(10)))
            self.assertEqual(sum(accepted), 5)


if __name__ == "__main__":
    unittest.main()
