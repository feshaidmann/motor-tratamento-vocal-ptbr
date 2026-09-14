"""Testes de regressão do processamento DSP da PoC."""

from __future__ import annotations

import unittest

import numpy as np
import soundfile as sf

import app


class DSPCorrectionTests(unittest.TestCase):
    sample_rate = 44_100

    def _test_audio(self) -> np.ndarray:
        time_axis = np.arange(self.sample_rate, dtype=np.float32) / self.sample_rate
        signal = 0.20 * np.sin(2 * np.pi * 800 * time_axis)
        signal += 0.10 * np.sin(2 * np.pi * 6_000 * time_axis)
        return np.column_stack((signal, signal)).astype(np.float32)

    def _analysis(self, with_regions: bool) -> dict:
        regions = (
            [{"inicio_s": 0.25, "fim_s": 0.55, "intensidade_relativa": 0.5}]
            if with_regions
            else []
        )
        return {
            "preset_dsp": "Balanceado",
            "nasalidade_ao_o": {
                "frequencia_alvo_hz": 800.0,
                "regioes_sustentadas": regions,
            },
            "sibilancia_s_x": {
                "frequencia_alvo_hz": 6_000.0,
                "regioes_sustentadas": regions,
            },
        }

    def test_dsp_is_transparent_without_detected_regions(self) -> None:
        audio = self._test_audio()
        processed = app.apply_dsp_correction(
            audio,
            self.sample_rate,
            self._analysis(with_regions=False),
        )

        np.testing.assert_array_equal(processed, audio)

    def test_dsp_changes_only_near_detected_regions(self) -> None:
        audio = self._test_audio()
        processed = app.apply_dsp_correction(
            audio,
            self.sample_rate,
            self._analysis(with_regions=True),
        )

        before_event = slice(0, int(0.10 * self.sample_rate))
        inside_event = slice(int(0.30 * self.sample_rate), int(0.50 * self.sample_rate))
        np.testing.assert_array_equal(processed[before_event], audio[before_event])
        self.assertGreater(
            float(np.mean(np.abs(processed[inside_event] - audio[inside_event]))),
            1e-4,
        )

    def test_mixdown_exports_float_wav_with_peak_protection(self) -> None:
        stem = np.full((1_024, 2), 0.75, dtype=np.float32)
        output_path = app.mixdown_and_export(
            processed_vocals=stem,
            instrumental=stem,
            sr=self.sample_rate,
        )

        info = sf.info(output_path)
        rendered, rendered_sr = sf.read(output_path, dtype="float32", always_2d=True)
        self.assertEqual(info.format, "WAV")
        self.assertEqual(info.subtype, "FLOAT")
        self.assertEqual(rendered_sr, self.sample_rate)
        self.assertLessEqual(float(np.max(np.abs(rendered))), 0.990001)


if __name__ == "__main__":
    unittest.main()

