"""Testes unitários do harness de benchmark, sem executar o Demucs."""

from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.benchmark_backends import (
    build_demucs_command,
    parse_process_snapshot,
    summarize_runs,
)


class BenchmarkHarnessTests(unittest.TestCase):
    def test_build_command_selects_model_device_and_float_output(self) -> None:
        command = build_demucs_command(Path("entrada.wav"), Path("saida"), "mps")

        self.assertIn("htdemucs", command)
        self.assertIn("mps", command)
        self.assertIn("--float32", command)
        self.assertEqual(command[-1], Path("entrada.wav").as_posix())

    def test_process_snapshot_ignores_malformed_lines(self) -> None:
        parsed = parse_process_snapshot("10 1 2048\ntexto inválido\n11 10 1024\n")

        self.assertEqual(parsed, {10: (1, 2048), 11: (10, 1024)})

    def test_summary_uses_only_successful_runs(self) -> None:
        summary = summarize_runs(
            [
                {
                    "status": "ok",
                    "tempo_segundos": 10.0,
                    "rtf": 0.5,
                    "pico_rss_mb": 500.0,
                },
                {
                    "status": "falha",
                    "tempo_segundos": 1.0,
                    "rtf": 0.1,
                    "pico_rss_mb": 100.0,
                },
                {
                    "status": "ok",
                    "tempo_segundos": 14.0,
                    "rtf": 0.7,
                    "pico_rss_mb": 650.0,
                },
            ]
        )

        self.assertEqual(summary["execucoes_validas"], 2)
        self.assertEqual(summary["tempo_medio_segundos"], 12.0)
        self.assertEqual(summary["rtf_medio"], 0.6)
        self.assertEqual(summary["pico_rss_maximo_mb"], 650.0)
