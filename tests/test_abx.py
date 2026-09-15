"""Testes da preparação e coleta de audições cegas A/B/ABX."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

import app


class ABXAuditionTests(unittest.TestCase):
    sample_rate = 16_000

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.previous_db_path = app.AUDITION_DB_PATH
        self.previous_log_path = app.AUDITION_LOG_PATH
        self.previous_stimulus_dir = app.AUDITION_STIMULUS_DIR
        data_dir = Path(self.temp_dir.name)
        app.AUDITION_DB_PATH = data_dir / "auditions.sqlite3"
        app.AUDITION_LOG_PATH = data_dir / "abx_votes.jsonl"
        app.AUDITION_STIMULUS_DIR = data_dir / "stimuli"

    def tearDown(self) -> None:
        app.AUDITION_DB_PATH = self.previous_db_path
        app.AUDITION_LOG_PATH = self.previous_log_path
        app.AUDITION_STIMULUS_DIR = self.previous_stimulus_dir
        self.temp_dir.cleanup()

    def _tone(self, frequency: float, amplitude: float) -> np.ndarray:
        time_axis = np.arange(self.sample_rate, dtype=np.float32) / self.sample_rate
        mono = amplitude * np.sin(2.0 * np.pi * frequency * time_axis)
        return np.column_stack((mono, mono)).astype(np.float32)

    def test_loudness_matching_removes_bs1770_advantage(self) -> None:
        original = self._tone(440.0, 0.10)
        processed = self._tone(880.0, 0.20)

        matched_original, matched_processed, report = app._level_match_abx_pair(
            original,
            processed,
            self.sample_rate,
        )

        original_lufs = app._integrated_loudness_lufs(
            matched_original, self.sample_rate
        )
        processed_lufs = app._integrated_loudness_lufs(
            matched_processed, self.sample_rate
        )
        self.assertIsNotNone(original_lufs)
        self.assertIsNotNone(processed_lufs)
        self.assertAlmostEqual(original_lufs, processed_lufs, places=4)
        self.assertAlmostEqual(report["delta_residual_lu"], 0.0, places=4)
        self.assertEqual(report["metodo"], "ITU-R BS.1770-4 · loudness integrado")
        target_peak = 10.0 ** (app.TRUE_PEAK_TARGET_DBTP / 20.0)
        self.assertLessEqual(
            app._approximate_true_peak_linear(matched_original),
            target_peak + 1e-5,
        )
        self.assertLessEqual(
            app._approximate_true_peak_linear(matched_processed),
            target_peak + 1e-5,
        )

    def test_loudness_matching_handles_silence_without_amplification(self) -> None:
        silence = np.zeros((self.sample_rate, 2), dtype=np.float32)
        processed = self._tone(880.0, 0.20)

        matched_silence, matched_processed, report = app._level_match_abx_pair(
            silence,
            processed,
            self.sample_rate,
        )

        np.testing.assert_array_equal(matched_silence, silence)
        np.testing.assert_array_equal(matched_processed, processed)
        self.assertEqual(report["metodo"], "fallback RMS · loudness abaixo do gate")
        self.assertTrue(np.isfinite(matched_processed).all())

    def test_randomization_balances_each_four_round_block(self) -> None:
        with tempfile.TemporaryDirectory() as audio_dir:
            original_path = Path(audio_dir) / "original.wav"
            processed_path = Path(audio_dir) / "processed.wav"
            sf.write(original_path, self._tone(440.0, 0.10), self.sample_rate)
            sf.write(processed_path, self._tone(880.0, 0.12), self.sample_rate)
            _, _, _, state, _ = app.prepare_abx_session(
                str(original_path),
                str(processed_path),
                "Balanceado",
                experiment_id="bloco-balanceado",
            )

        states = [state]
        for _ in range(3):
            _, _, _, state, _ = app.prepare_next_abx_session(state)
            states.append(state)
        cells = {(item["a_condicao"], item["x_resposta"]) for item in states}
        self.assertEqual(len(cells), 4)

    def test_abx_session_x_is_exact_copy_of_declared_stimulus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = Path(temp_dir) / "original.wav"
            processed_path = Path(temp_dir) / "processed.wav"
            sf.write(
                original_path,
                self._tone(440.0, 0.10),
                self.sample_rate,
                subtype="FLOAT",
            )
            sf.write(
                processed_path,
                self._tone(880.0, 0.12),
                self.sample_rate,
                subtype="FLOAT",
            )

            a_path, b_path, x_path, state, _ = app.prepare_abx_session(
                str(original_path),
                str(processed_path),
                "Balanceado",
            )

            a_audio, _ = sf.read(a_path, dtype="float32", always_2d=True)
            b_audio, _ = sf.read(b_path, dtype="float32", always_2d=True)
            x_audio, _ = sf.read(x_path, dtype="float32", always_2d=True)
            expected = a_audio if state["x_resposta"] == "A" else b_audio
            np.testing.assert_array_equal(x_audio, expected)
            self.assertEqual(
                {state["a_condicao"], state["b_condicao"]},
                {"original", "processado"},
            )

            next_a, next_b, next_x, next_state, _ = app.prepare_next_abx_session(
                state
            )
            self.assertEqual({next_a, next_b}, {a_path, b_path})
            next_x_audio, _ = sf.read(next_x, dtype="float32", always_2d=True)
            next_expected_path = (
                next_a if next_state["x_resposta"] == "A" else next_b
            )
            next_expected, _ = sf.read(
                next_expected_path, dtype="float32", always_2d=True
            )
            np.testing.assert_array_equal(next_x_audio, next_expected)
            self.assertNotEqual(next_state["session_id"], state["session_id"])

    def test_vote_is_append_only_and_cannot_be_submitted_twice(self) -> None:
        with tempfile.TemporaryDirectory() as audio_dir:
            original_path = Path(audio_dir) / "original.wav"
            processed_path = Path(audio_dir) / "processed.wav"
            sf.write(original_path, self._tone(440.0, 0.10), self.sample_rate)
            sf.write(processed_path, self._tone(880.0, 0.12), self.sample_rate)
            _, _, _, state, _ = app.prepare_abx_session(
                str(original_path),
                str(processed_path),
                "Balanceado",
                {"modulos_ativos": {"nasalidade": True}},
            )

        updated, _ = app.save_abx_vote(
            state,
            "ouvinte-01",
            state["x_resposta"],
            "A",
            4,
            "diferença sutil",
        )
        self.assertTrue(updated["voto_registrado"])

        # Repete deliberadamente o estado antigo, que antes burlava a proteção.
        with self.assertRaises(gr.Error):
            app.save_abx_vote(
                state,
                "ouvinte-01",
                state["x_resposta"],
                "A",
                4,
                "",
            )

        export_path = app.export_audition_votes_jsonl()
        record = json.loads(Path(export_path).read_text(encoding="utf-8"))
        self.assertTrue(record["resposta_x_correta"])
        self.assertEqual(record["preferencia_condicao"], state["a_condicao"])
        self.assertTrue(
            record["diagnostico_motor"]["modulos_ativos"]["nasalidade"]
        )
        self.assertFalse(any(key.startswith("_") for key in record))

    def test_participant_page_has_no_condition_identity_or_reveal(self) -> None:
        page = app._participant_page("token-seguro")

        self.assertNotIn("original", page.lower())
        self.assertNotIn("processado", page.lower())
        self.assertNotIn("revelar", page.lower())
        self.assertIn("AudioContext", page)
        self.assertIn("linearRampToValueAtTime", page)

    def test_official_mode_rejects_dirty_build(self) -> None:
        with patch.object(
            app,
            "_build_identity",
            return_value={"git_dirty": True},
        ):
            with self.assertRaises(gr.Error):
                app.prepare_abx_session(
                    "original.wav",
                    "processed.wav",
                    "Balanceado",
                    official_mode=True,
                )

    def test_concurrent_votes_commit_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as audio_dir:
            original_path = Path(audio_dir) / "original.wav"
            processed_path = Path(audio_dir) / "processed.wav"
            sf.write(original_path, self._tone(440.0, 0.10), self.sample_rate)
            sf.write(processed_path, self._tone(880.0, 0.12), self.sample_rate)
            _, _, _, state, _ = app.prepare_abx_session(
                str(original_path), str(processed_path), "Balanceado"
            )

        vote = app.AuditionVote(
            participante_id="concorrente",
            resposta_x="A",
            preferencia="Sem diferença",
            confianca=3,
        )

        def submit() -> str:
            try:
                app._save_vote_by_token(state["session_id"], vote)
                return "gravado"
            except app.DuplicateVoteError:
                return "duplicado"

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: submit(), range(2)))

        self.assertCountEqual(results, ["gravado", "duplicado"])

    def test_public_routes_hide_ground_truth_and_reject_second_vote(self) -> None:
        with tempfile.TemporaryDirectory() as audio_dir:
            original_path = Path(audio_dir) / "original.wav"
            processed_path = Path(audio_dir) / "processed.wav"
            sf.write(original_path, self._tone(440.0, 0.10), self.sample_rate)
            sf.write(processed_path, self._tone(880.0, 0.12), self.sample_rate)
            _, _, _, state, _ = app.prepare_abx_session(
                str(original_path), str(processed_path), "Balanceado"
            )

            async def exercise_routes() -> None:
                token = state["session_id"]
                transport = httpx.ASGITransport(app=app.create_web_app())
                async with httpx.AsyncClient(
                    transport=transport,
                    base_url="http://testserver",
                ) as client:
                    page = await client.get(f"/audicao/{token}")
                    self.assertEqual(page.status_code, 200)
                    self.assertNotIn("processado", page.text.lower())
                    manifest = await client.get(f"/api/audicao/{token}/manifesto")
                    self.assertEqual(manifest.status_code, 200)
                    self.assertNotIn("x_resposta", manifest.json())
                    audio = await client.get(f"/api/audicao/{token}/audio/A")
                    self.assertEqual(audio.status_code, 200)
                    self.assertEqual(audio.headers["content-type"], "audio/wav")

                    payload = {
                        "participante_id": "rota-publica",
                        "resposta_x": "A",
                        "preferencia": "Sem diferença",
                        "confianca": 3,
                        "observacoes": "",
                    }
                    first = await client.post(
                        f"/api/audicao/{token}/voto", json=payload
                    )
                    second = await client.post(
                        f"/api/audicao/{token}/voto", json=payload
                    )
                    self.assertEqual(first.status_code, 200)
                    self.assertEqual(second.status_code, 409)

            asyncio.run(exercise_routes())


if __name__ == "__main__":
    unittest.main()
