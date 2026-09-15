"""Testes do construtor do corpus piloto."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmarks.build_pilot_corpus import (
    discover_sessions,
    select_window_from_energies,
    write_annotation_sheet,
)


class CorpusBuilderTests(unittest.TestCase):
    def test_select_window_uses_highest_contiguous_energy(self) -> None:
        start = select_window_from_energies([0.1, 0.2, 2.0, 3.0, 0.1], 2)

        self.assertEqual(start, 2)

    def test_select_window_clamps_width_to_available_blocks(self) -> None:
        start = select_window_from_energies([0.1, 0.2], 30)

        self.assertEqual(start, 0)

    def test_discover_sessions_requires_complete_artifact_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            complete = root / "complete"
            incomplete = root / "incomplete"
            complete.mkdir()
            incomplete.mkdir()
            for name in (
                "vocal_antes.wav",
                "instrumental.wav",
                "mix_processado.wav",
                "relatorio.json",
            ):
                (complete / name).touch()
            (incomplete / "vocal_antes.wav").touch()

            sessions = discover_sessions(root)

        self.assertEqual(sessions, [complete])

    def test_annotation_sheet_marks_rights_as_unconfirmed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = write_annotation_sheet(
                Path(temp),
                [{"id": "item_001", "origem_local": "sessao_local"}],
            )
            contents = path.read_text(encoding="utf-8")

        self.assertIn("direitos_confirmados", contents)
        self.assertIn("item_001,sessao_local", contents)
        self.assertIn("não", contents)


if __name__ == "__main__":
    unittest.main()
