#!/usr/bin/env python3
"""Cria campanhas ABX internas e exporta relatórios sem divulgar estímulos."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
from motor_vocal.campaign import campaign_report, create_campaign


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="Cria uma campanha local de uso interno.")
    create.add_argument("--campaign-id", required=True)
    create.add_argument("--participants", type=int, required=True)
    create.add_argument(
        "--manifest", type=Path,
        default=PROJECT_ROOT / "outputs/corpus_ptbr_seed_v1/manifest.json",
    )
    create.add_argument("--annotations", type=Path,
                        default=PROJECT_ROOT / "outputs/corpus_ptbr_seed_v1/anotacoes.csv")
    create.add_argument("--output", type=Path, help="JSON privado com links locais.")
    create.add_argument(
        "--base-url",
        default=os.environ.get("MOTOR_VOCAL_BASE_URL", "http://127.0.0.1:7860"),
        help="URL pública/base da aplicação (ou MOTOR_VOCAL_BASE_URL).",
    )
    report = commands.add_parser("report", help="Relatório de votos completos.")
    report.add_argument("--campaign-id", required=True)
    report.add_argument("--output", type=Path, help="Caminho do relatório JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app.initialize_audition_database()
    if args.command == "create":
        with app._audition_db() as connection:
            tokens = create_campaign(
                connection,
                manifest_path=args.manifest,
                annotations_path=args.annotations,
                stimulus_dir=app.AUDITION_STIMULUS_DIR,
                campaign_id=args.campaign_id,
                participant_count=args.participants,
                build_identity=app._build_identity(),
            )
        result = {
            "schema": "motor-clareza-campaign-links/v1",
            "campaign_id": args.campaign_id,
            "scope": "interno; não compartilhar publicamente",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_url": args.base_url.rstrip("/"),
            "links": [f"{args.base_url.rstrip('/')}/campanha/{token}" for token in tokens],
        }
        output = args.output or app.AUDITION_DATA_DIR / f"campaign_{args.campaign_id}_links.json"
    else:
        with app._audition_db() as connection:
            result = campaign_report(connection, args.campaign_id)
        output = args.output or app.AUDITION_DATA_DIR / f"campaign_{args.campaign_id}_report.json"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.command == "create":
        output.chmod(0o600)
        print(f"Campanha interna criada: {len(tokens)} participante(s).")
        print("Os links são credenciais privadas; mantenha o JSON apenas localmente.")
    else:
        print(f"Relatório: {result['participantes_completos']} participante(s) completo(s).")
    print(f"Arquivo: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
