"""Testes unitários do núcleo de separação de fontes."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from motor_vocal import separation
from motor_vocal.separation import select_available_device


class DeviceSelectionTests(unittest.TestCase):
    def test_cuda_has_priority_over_mps(self) -> None:
        selected = select_available_device(
            cuda_available=True,
            mps_built=True,
            mps_available=True,
        )
        self.assertEqual(selected, "cuda")

    def test_mps_is_used_when_cuda_is_unavailable(self) -> None:
        selected = select_available_device(
            cuda_available=False,
            mps_built=True,
            mps_available=True,
        )
        self.assertEqual(selected, "mps")

    def test_cpu_is_the_safe_fallback(self) -> None:
        selected = select_available_device(
            cuda_available=False,
            mps_built=False,
            mps_available=False,
        )
        self.assertEqual(selected, "cpu")

    @unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg não disponível")
    def test_mp3_uses_aligned_decoded_input_for_demucs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.wav"
            mp3 = Path(temp) / "source.mp3"
            samples = np.arange(44_100, dtype=np.float32) / 44_100
            tone = (0.1 * np.sin(2 * np.pi * 440 * samples)).astype(np.float32)
            sf.write(source, np.column_stack((tone, tone)), 44_100, subtype="FLOAT")
            subprocess.run(
                ["ffmpeg", "-v", "error", "-i", str(source), str(mp3)],
                check=True,
                capture_output=True,
            )
            expected, sample_rate = sf.read(mp3, dtype="float32", always_2d=True)

            def fake_demucs(audio_path: Path, output_dir: Path, device: str) -> None:
                self.assertEqual(audio_path.suffix, ".wav")
                self.assertEqual(device, "cpu")
                decoded, decoded_rate = sf.read(
                    audio_path, dtype="float32", always_2d=True
                )
                self.assertEqual(decoded_rate, sample_rate)
                np.testing.assert_array_equal(decoded, expected)
                stems_dir = output_dir / "fake"
                stems_dir.mkdir()
                sf.write(stems_dir / "vocals.wav", decoded, decoded_rate, subtype="FLOAT")
                sf.write(
                    stems_dir / "no_vocals.wav",
                    np.zeros_like(decoded),
                    decoded_rate,
                    subtype="FLOAT",
                )

            with mock.patch.object(
                separation, "preferred_demucs_device", return_value="cpu"
            ), mock.patch.object(separation, "run_demucs", side_effect=fake_demucs):
                vocals, instrumental, result_rate = separation.separate_stems(str(mp3))

            self.assertEqual(result_rate, sample_rate)
            np.testing.assert_array_equal(vocals, expected)
            np.testing.assert_array_equal(instrumental, np.zeros_like(expected))


if __name__ == "__main__":
    unittest.main()
