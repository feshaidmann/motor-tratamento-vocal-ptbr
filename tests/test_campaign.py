"""Testes da campanha ABX interna e do relatório exploratório."""

from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

import app
import httpx
from motor_vocal.campaign import campaign_progress, campaign_report, create_campaign, load_corpus


class CampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.old_db = app.AUDITION_DB_PATH
        self.old_stimulus = app.AUDITION_STIMULUS_DIR
        app.AUDITION_DB_PATH = root / "auditions.sqlite3"
        app.AUDITION_STIMULUS_DIR = root / "stimuli"
        app.initialize_audition_database()

    def tearDown(self) -> None:
        app.AUDITION_DB_PATH = self.old_db
        app.AUDITION_STIMULUS_DIR = self.old_stimulus
        self.temp_dir.cleanup()

    def _corpus_paths(self) -> tuple[Path, Path]:
        root = Path(__file__).resolve().parents[1]
        return (
            root / "outputs/corpus_ptbr_seed_v1/manifest.json",
            root / "outputs/corpus_ptbr_seed_v1/anotacoes.csv",
        )

    def test_campaign_creates_balanced_sequence_and_enforces_order(self) -> None:
        manifest, annotations = self._corpus_paths()
        with app._audition_db() as connection:
            tokens = create_campaign(
                connection,
                manifest_path=manifest,
                annotations_path=annotations,
                stimulus_dir=app.AUDITION_STIMULUS_DIR,
                campaign_id="teste-campanha",
                participant_count=2,
                build_identity={"git_dirty": False, "git_commit": "teste"},
            )
            first = connection.execute(
                "SELECT session_token FROM campaign_rounds WHERE participant_token = ? ORDER BY position",
                (tokens[0],),
            ).fetchall()
            second = connection.execute(
                "SELECT session_token FROM campaign_rounds WHERE participant_token = ? ORDER BY position",
                (tokens[1],),
            ).fetchall()
            self.assertEqual(len(first), 5)
            self.assertEqual(len(second), 5)
            first_audit = connection.execute(
                "SELECT audit_json FROM audition_sessions WHERE token = ?", (first[0]["session_token"],)
            ).fetchone()
            audit = json.loads(first_audit["audit_json"])
            self.assertIn("campaign_seed", audit)
            self.assertEqual(audit["item_order"], [row["item_id"] for row in connection.execute(
                "SELECT item_id FROM campaign_rounds WHERE participant_token = ? ORDER BY position",
                (tokens[0],),
            )])
            self.assertEqual(
                {
                    (row["a_condition"], row["x_answer"])
                    for row in connection.execute(
                        "SELECT a_condition, x_answer FROM audition_sessions WHERE experiment_id = ?",
                        ("teste-campanha",),
                    )
                },
                {("original", "A"), ("original", "B"), ("processado", "A"), ("processado", "B")},
            )
            first_session = first[0]["session_token"]
            later_session = first[1]["session_token"]

        with self.assertRaises(ValueError):
            app._save_vote_by_token(
                later_session,
                app.AuditionVote(resposta_x="A", preferencia="A", confianca=3),
            )
        session = app._get_audition_session(first_session)
        app._save_vote_by_token(
            first_session,
            app.AuditionVote(
                resposta_x=session["x_answer"], preferencia="A", confianca=4
            ),
        )
        with app._audition_db() as connection:
            progress = campaign_progress(connection, tokens[0])
        self.assertEqual(progress["completed"], 1)
        self.assertEqual(progress["next_session_token"], later_session)

    def test_report_counts_only_complete_participants(self) -> None:
        manifest, annotations = self._corpus_paths()
        with app._audition_db() as connection:
            tokens = create_campaign(
                connection,
                manifest_path=manifest,
                annotations_path=annotations,
                stimulus_dir=app.AUDITION_STIMULUS_DIR,
                campaign_id="teste-relatorio",
                participant_count=1,
                build_identity={"git_dirty": False, "git_commit": "teste"},
            )
            sessions = connection.execute(
                "SELECT token AS session_token, x_answer FROM audition_sessions WHERE experiment_id = ? ORDER BY rowid",
                ("teste-relatorio",),
            ).fetchall()
        for row in sessions:
            app._save_vote_by_token(
                row["session_token"],
                app.AuditionVote(
                    resposta_x=row["x_answer"], preferencia="A", confianca=4
                ),
            )
        with app._audition_db() as connection:
            report = campaign_report(connection, "teste-relatorio")
        self.assertEqual(report["participantes_completos"], 1)
        self.assertEqual(report["resumo"]["votos"], 5)
        self.assertEqual(report["resumo"]["taxa_acerto"], 1.0)
        self.assertEqual(report["resumo_por_participante"]["participantes"], 1)
        self.assertEqual(report["resumo_por_participante"]["media_acerto"], 1.0)
        self.assertEqual(report["participantes_excluidos_incompletos"], 0)

    def test_campaign_route_exposes_progress_without_condition_identity(self) -> None:
        manifest, annotations = self._corpus_paths()
        with app._audition_db() as connection:
            tokens = create_campaign(
                connection,
                manifest_path=manifest,
                annotations_path=annotations,
                stimulus_dir=app.AUDITION_STIMULUS_DIR,
                campaign_id="teste-rota-campanha",
                participant_count=1,
                build_identity={"git_dirty": False, "git_commit": "teste"},
            )
            first_session = connection.execute(
                "SELECT session_token, x_answer FROM campaign_rounds r "
                "JOIN audition_sessions s ON s.token = r.session_token "
                "WHERE participant_token = ? ORDER BY position LIMIT 1",
                (tokens[0],),
            ).fetchone()

        async def fetch() -> httpx.Response:
            transport = httpx.ASGITransport(app=app.create_web_app())
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                response = await client.get(f"/campanha/{tokens[0]}")
                self.assertEqual(response.status_code, 200)
                vote = await client.post(
                    f"/api/audicao/{first_session['session_token']}/voto",
                    json={
                        "participante_id": "",
                        "resposta_x": first_session["x_answer"],
                        "preferencia": "Sem diferença",
                        "confianca": 3,
                        "observacoes": "",
                    },
                )
                self.assertEqual(vote.status_code, 200)
                return await client.get(f"/campanha/{tokens[0]}")

        response = asyncio.run(fetch())
        self.assertEqual(response.status_code, 200)
        self.assertIn('"position": 2', response.text)
        self.assertIn('"total": 5', response.text)
        self.assertNotIn("original", response.text.lower())
        self.assertNotIn("processado", response.text.lower())
        self.assertNotIn("__CAMPAIGN_JSON__", response.text)

    def test_corpus_rights_gate_rejects_unconfirmed_annotation(self) -> None:
        manifest, annotations = self._corpus_paths()
        with tempfile.TemporaryDirectory() as temp:
            replacement = Path(temp) / "anotacoes.csv"
            text = annotations.read_text(encoding="utf-8")
            replacement.write_text(text.replace("sim (", "não ("), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Direitos"):
                load_corpus(manifest, replacement)

    def test_structured_rights_column_is_supported(self) -> None:
        manifest, annotations = self._corpus_paths()
        with tempfile.TemporaryDirectory() as temp:
            replacement = Path(temp) / "anotacoes.csv"
            lines = annotations.read_text(encoding="utf-8").splitlines()
            header = lines[0] + ",uso_interno_autorizado"
            body = [line + ",true" for line in lines[1:]]
            replacement.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")
            self.assertEqual(len(load_corpus(manifest, replacement)), 5)


if __name__ == "__main__":
    unittest.main()
