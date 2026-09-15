"""Testes do validador independente de loudness."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from benchmarks.validate_loudness import (
    evaluate_pair,
    parse_ebur128_summary,
    validate_pair,
)


class LoudnessValidationTests(unittest.TestCase):
    def test_parse_ebur128_uses_final_summary(self) -> None:
        output = """
        I:         -70.0 LUFS
        Summary:
        Integrated loudness:
          I:         -20.1 LUFS
        True peak:
          Peak:       -1.2 dBFS
        """

        measurement = parse_ebur128_summary(output)

        self.assertEqual(measurement["loudness_integrado_lufs"], -20.1)
        self.assertEqual(measurement["true_peak_dbtp"], -1.2)

    def test_evaluate_pair_approves_aligned_measurements(self) -> None:
        original = self._measurement(-20.0, -20.05, -1.1)
        processed = self._measurement(-20.1, -20.02, -1.0)

        result = evaluate_pair(original, processed)

        self.assertTrue(result["aprovado"])
        self.assertEqual(result["delta_loudness_ffmpeg_lu"], 0.1)

    def test_evaluate_pair_rejects_loudness_or_alignment_failure(self) -> None:
        original = self._measurement(-20.0, -20.0, -1.1)
        processed = self._measurement(-19.6, -19.6, -0.5)
        processed["frames"] += 1

        result = evaluate_pair(original, processed)

        self.assertFalse(result["aprovado"])
        self.assertFalse(result["verificacoes"]["frames_identicos"])
        self.assertFalse(result["verificacoes"]["residual_loudness_dentro_do_limite"])
        self.assertFalse(result["verificacoes"]["true_peak_processado_dentro_do_limite"])

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg não disponível")
    def test_validate_pair_with_real_ffmpeg_measurement(self) -> None:
        sample_rate = 48_000
        time_axis = np.arange(sample_rate, dtype=np.float64) / sample_rate
        mono = 0.02 * np.sin(2.0 * np.pi * 440.0 * time_axis)
        stereo = np.column_stack((mono, mono))

        with tempfile.TemporaryDirectory() as temp:
            original = Path(temp) / "original.wav"
            processed = Path(temp) / "processado.wav"
            sf.write(original, stereo, sample_rate, subtype="FLOAT")
            sf.write(processed, stereo, sample_rate, subtype="FLOAT")

            result = validate_pair(original, processed)

        self.assertTrue(result["aprovado"])
        self.assertEqual(result["delta_loudness_ffmpeg_lu"], 0.0)

    @staticmethod
    def _measurement(ffmpeg_lufs: float, pyloudnorm_lufs: float, peak: float) -> dict:
        return {
            "sample_rate_hz": 48_000,
            "canais": 2,
            "frames": 48_000,
            "ffmpeg": {
                "loudness_integrado_lufs": ffmpeg_lufs,
                "true_peak_dbtp": peak,
            },
            "pyloudnorm_lufs": pyloudnorm_lufs,
        }


if __name__ == "__main__":
    unittest.main()
