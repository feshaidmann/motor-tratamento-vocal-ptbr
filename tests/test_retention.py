"""Inventário de retenção sem exclusão ou alteração de dados."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from motor_vocal.jobs import JobRequest, LocalJobStore
from motor_vocal.retention import preview_local_retention


class RetentionPreviewTests(unittest.TestCase):
    def test_missing_database_is_not_created(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "jobs"
            report = preview_local_retention(
                root, older_than=datetime.now(timezone.utc)
            )
            self.assertFalse(report["database_present"])
            self.assertFalse(root.exists())

    def test_shared_upload_is_preserved_while_a_job_still_needs_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            upload = root / "uploads" / ("a" * 64 + ".wav")
            upload.parent.mkdir()
            upload.write_bytes(b"audio")
            store = LocalJobStore(root / "jobs.sqlite3")
            first = store.enqueue(JobRequest(input_path=str(upload)), "primeiro")
            second = store.enqueue(JobRequest(input_path=str(upload)), "segundo")
            claimed = store.claim_next()
            store.fail(claimed.job_id, claimed.attempts, "falha controlada")
            old = (datetime.now(timezone.utc) - timedelta(days=10)).isoformat()
            with store._connect() as connection:
                connection.execute(
                    "UPDATE jobs SET completed_at = ? WHERE job_id = ?",
                    (old, first.job_id),
                )
            artifact_dir = root / "artifacts" / first.job_id / "attempt_1"
            artifact_dir.mkdir(parents=True)
            (artifact_dir / "mix.wav").write_bytes(b"123456")
            cutoff = datetime.now(timezone.utc) - timedelta(days=7)
            preview = preview_local_retention(root, older_than=cutoff)
            self.assertEqual(preview["eligible_jobs"], [first.job_id])
            self.assertEqual(preview["artifact_bytes"], 6)
            self.assertEqual(preview["upload_files"], [])
            self.assertTrue(upload.is_file())
            self.assertTrue((artifact_dir / "mix.wav").is_file())

            claimed = store.claim_next()
            self.assertEqual(claimed.job_id, second.job_id)
            store.fail(claimed.job_id, claimed.attempts, "falha controlada")
            with store._connect() as connection:
                connection.execute(
                    "UPDATE jobs SET completed_at = ? WHERE job_id = ?",
                    (old, second.job_id),
                )
            preview = preview_local_retention(root, older_than=cutoff)
            self.assertEqual(len(preview["eligible_jobs"]), 2)
            self.assertEqual(preview["upload_files"], [str(upload.resolve())])
            self.assertEqual(preview["upload_bytes"], 5)
            with store._connect() as connection:
                connection.execute(
                    "UPDATE jobs SET request_json = '{}' WHERE job_id = ?",
                    (second.job_id,),
                )
            preview = preview_local_retention(root, older_than=cutoff)
            self.assertEqual(preview["invalid_records"], 1)
            self.assertTrue(preview["uploads_not_evaluated"])
            self.assertEqual(preview["upload_files"], [])


if __name__ == "__main__":
    unittest.main()
