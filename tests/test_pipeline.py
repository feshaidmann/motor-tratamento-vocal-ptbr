"""Integração local entre a interface e o executor sem servidor web."""

from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf
import gradio as gr

import app
from motor_vocal import pipeline
from motor_vocal.pipeline import ProcessingResult


class PipelineTests(unittest.TestCase):
    def test_ui_uses_headless_pipeline_and_exports_four_float_wavs(self) -> None:
        sample_rate = 44_100
        samples = np.arange(sample_rate, dtype=np.float32)
        mono = 0.1 * np.sin(2.0 * np.pi * 220.0 * samples / sample_rate)
        original = np.column_stack((mono, mono)).astype(np.float32)
        vocals = original * 0.6
        instrumental = original * 0.4
        progress_events: list[float] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_path = root / "entrada.wav"
            sf.write(input_path, original, sample_rate, subtype="FLOAT")
            with patch.object(app, "OUTPUT_DIR", root / "resultados"), patch.object(
                pipeline,
                "separate_stems",
                return_value=(vocals, instrumental, sample_rate),
            ):
                outputs = app.process_audio(
                    str(input_path),
                    correct_nasality=False,
                    correct_stridency=False,
                    correct_sibilance=False,
                    progress=lambda value, **_: progress_events.append(value),
                )

            *paths, analysis = outputs
            self.assertEqual(len(paths), 4)
            self.assertEqual(progress_events[-1], 1.0)
            self.assertEqual(analysis["controle_qualidade_saida"]["status"], "aprovado")
            self.assertTrue(analysis["qualidade_separacao"]["confiavel"])
            for path in paths:
                info = sf.info(path)
                self.assertEqual(info.samplerate, sample_rate)
                self.assertEqual(info.frames, sample_rate)
                self.assertEqual(info.channels, 2)
                self.assertEqual(info.subtype, "FLOAT")
            rendered, _ = sf.read(paths[0], dtype="float32", always_2d=True)
            np.testing.assert_allclose(rendered, original, atol=1e-6)

    def test_existing_destination_is_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(FileExistsError):
                pipeline.process_audio_file("inexistente.wav", temp_dir)

    def test_submission_enforces_duration_and_persists_upload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            short = root / "curto.wav"
            boundary = root / "limite.wav"
            long = root / "acima_do_limite.wav"
            sf.write(short, np.zeros((8_000, 1), dtype=np.float32), 16_000)
            sf.write(boundary, np.zeros((300 * 8_000, 1), dtype=np.float32), 8_000)
            sf.write(long, np.zeros((301 * 8_000, 1), dtype=np.float32), 8_000)
            with patch.object(app, "JOB_DATA_DIR", root / "jobs"):
                first = app.enqueue_audio_job(str(short))
                repeated = app.enqueue_audio_job(str(short))
                self.assertEqual(first.job_id, repeated.job_id)
                submitted = app.submit_audio_job(
                    str(short), "Balanceado", True, True, True
                )
                self.assertEqual(len(submitted), 13)
                self.assertEqual(submitted[0], first.job_id)
                self.assertTrue(submitted[2]["active"])
                self.assertTrue(app.resume_audio_job(first.job_id)[1]["active"])
                short.unlink()
                self.assertTrue(Path(first.request.input_path).is_file())
                accepted_boundary = app.enqueue_audio_job(str(boundary))
                self.assertEqual(accepted_boundary.state, "queued")
                with self.assertRaisesRegex(ValueError, "até 5 minutos"):
                    app.enqueue_audio_job(str(long))

    def test_submission_enforces_file_size_before_staging(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "entrada.wav"
            sf.write(source, np.zeros((8_000, 1), dtype=np.float32), 16_000)
            size = source.stat().st_size
            with patch.object(app, "JOB_DATA_DIR", root / "jobs"), patch.object(
                app, "PILOT_MAX_FILE_SIZE_BYTES", size - 1
            ):
                with self.assertRaisesRegex(ValueError, "até 150 MB"):
                    app.enqueue_audio_job(str(source))
                self.assertFalse((root / "jobs" / "uploads").exists())
            with patch.object(app, "JOB_DATA_DIR", root / "jobs"), patch.object(
                app, "PILOT_MAX_FILE_SIZE_BYTES", size
            ):
                self.assertEqual(app.enqueue_audio_job(str(source)).state, "queued")

    def test_web_lifespan_starts_worker_and_status_can_be_polled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "entrada.wav"
            sf.write(source, np.zeros((8_000, 1), dtype=np.float32), 16_000)
            expected = ProcessingResult(
                mixed_path=root / "mix.wav",
                vocal_path=root / "vocal.wav",
                processed_vocal_path=root / "corrigido.wav",
                instrumental_path=root / "instrumental.wav",
                analysis={"controle_qualidade_saida": {"status": "aprovado"}},
            )
            with patch.object(app, "JOB_DATA_DIR", root / "jobs"), patch.object(
                app, "AUDITION_DB_PATH", root / "auditions.sqlite3"
            ), patch("motor_vocal.jobs.process_audio_file", return_value=expected), patch.object(
                app,
                "prepare_loudness_comparison",
                return_value=("original.wav", "processado.wav", {"metodo": "teste"}),
            ), patch.object(
                app, "gradio_interface", return_value=object()
            ), patch.object(
                app.gr, "mount_gradio_app", side_effect=lambda web_app, *_args, **_kwargs: web_app
            ):
                job = app.enqueue_audio_job(str(source))
                web_app = app.create_web_app()
                async def run_worker() -> None:
                    nonlocal current
                    async with web_app.router.lifespan_context(web_app):
                        deadline = time.monotonic() + 5
                        while time.monotonic() < deadline:
                            current = app._job_store().get(job.job_id)
                            if current.state == "succeeded":
                                break
                            await asyncio.sleep(0.05)

                current = None
                asyncio.run(run_worker())
                self.assertEqual(current.state, "succeeded")
                status = app.poll_audio_job(job.job_id)
                self.assertEqual(len(status), 10)
                self.assertIn("concluído", status[0])
                self.assertEqual(status[2], str(expected.mixed_path))
                self.assertFalse(status[1]["active"])
                with patch.object(
                    app, "create_audition_link", return_value=("url", "ok")
                ) as publish:
                    published = app.create_audition_link_for_job(
                        job.job_id,
                        str(source),
                        str(expected.mixed_path),
                        "Balanceado",
                        True, True, True,
                        "piloto", False, None,
                    )
                    self.assertEqual(published, ("url", "ok"))
                    self.assertEqual(
                        publish.call_args.args[0], job.request.input_path
                    )
                    with self.assertRaises(gr.Error):
                        app.create_audition_link_for_job(
                            job.job_id,
                            str(source),
                            "resultado-de-outro-trabalho.wav",
                            "Balanceado",
                            True, True, True,
                            "piloto", False, None,
                        )


if __name__ == "__main__":
    unittest.main()
