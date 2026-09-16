"""Campanhas ABX locais sobre um corpus auditado; sem publicação externa."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import secrets
import shutil
import sqlite3
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf


def initialize_campaign_tables(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS campaigns (
            id TEXT PRIMARY KEY,
            created_at_utc TEXT NOT NULL,
            corpus_sha256 TEXT NOT NULL,
            item_count INTEGER NOT NULL CHECK (item_count > 0),
            scope TEXT NOT NULL CHECK (scope = 'interno')
        );
        CREATE TABLE IF NOT EXISTS campaign_participants (
            token TEXT PRIMARY KEY,
            campaign_id TEXT NOT NULL REFERENCES campaigns(id),
            ordinal INTEGER NOT NULL,
            UNIQUE(campaign_id, ordinal)
        );
        CREATE TABLE IF NOT EXISTS campaign_rounds (
            participant_token TEXT NOT NULL REFERENCES campaign_participants(token),
            position INTEGER NOT NULL,
            item_id TEXT NOT NULL,
            session_token TEXT NOT NULL UNIQUE REFERENCES audition_sessions(token),
            PRIMARY KEY(participant_token, position),
            UNIQUE(participant_token, item_id)
        );
        """
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_corpus(manifest_path: Path, annotations_path: Path) -> list[dict[str, Any]]:
    """Valida direitos internos, integridade, alinhamento e arquivos do corpus."""
    manifest_path = manifest_path.resolve(strict=True)
    annotations_path = annotations_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "motor-clareza-corpus/v1":
        raise ValueError("Schema do corpus não reconhecido.")
    items = manifest.get("itens")
    if not isinstance(items, list) or not items or not manifest.get("todos_aprovados"):
        raise ValueError("O corpus precisa conter itens tecnicamente aprovados.")
    if manifest.get("total_itens") != len(items):
        raise ValueError("A contagem do manifesto não corresponde aos itens.")
    with annotations_path.open(encoding="utf-8-sig", newline="") as stream:
        annotations = list(csv.DictReader(stream))
    rights = {row["item_id"]: row for row in annotations}
    if len(rights) != len(annotations):
        raise ValueError("Há IDs duplicados na tabela de anotações.")

    root = manifest_path.parent
    prepared: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        item_id = str(item.get("id", ""))
        if not item_id or item_id in seen:
            raise ValueError("ID de item ausente ou duplicado.")
        seen.add(item_id)
        rights_row = rights.get(item_id, {})
        structured_rights = str(rights_row.get("uso_interno_autorizado", "")).strip().casefold()
        legacy_rights = str(rights_row.get("direitos_confirmados", "")).strip().casefold()
        rights_confirmed = (
            structured_rights in {"1", "true", "sim", "yes"}
            if structured_rights
            else legacy_rights.split("(", 1)[0].strip() in {"sim", "true", "1", "yes"}
        )
        if not rights_confirmed:
            raise ValueError(f"Direitos para uso interno não confirmados: {item_id}.")
        if not item.get("validacao_independente", {}).get("aprovado"):
            raise ValueError(f"Validação técnica reprovada: {item_id}.")
        paths: dict[str, Path] = {}
        for label, measurement in (
            ("original", item["validacao_independente"]["original"]),
            ("processado", item["validacao_independente"]["processado"]),
        ):
            relative_name = "estimulo_1" if label == "original" else "estimulo_2"
            path = (root / item["arquivos"][relative_name]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError(f"Caminho fora do corpus: {item_id}.")
            if _sha256(path) != measurement["sha256"]:
                raise ValueError(f"Hash do estímulo alterado: {item_id}/{label}.")
            paths[label] = path
        first, second = sf.info(paths["original"]), sf.info(paths["processado"])
        if (first.samplerate, first.channels, first.frames) != (
            second.samplerate, second.channels, second.frames
        ):
            raise ValueError(f"Estímulos desalinhados: {item_id}.")
        prepared.append({"id": item_id, "paths": paths, "analysis": item.get("analise_motor", {}),
                         "sample_rate_hz": first.samplerate, "canais": first.channels,
                         "duracao_s": round(first.duration, 3)})
    return prepared


def create_campaign(
    connection: sqlite3.Connection,
    *,
    manifest_path: Path,
    annotations_path: Path,
    stimulus_dir: Path,
    campaign_id: str,
    participant_count: int,
    build_identity: dict[str, Any],
) -> list[str]:
    """Cria sequências balanceadas e tokens opacos para um piloto interno."""
    if not campaign_id.strip() or len(campaign_id) > 120:
        raise ValueError("Informe um ID de campanha de até 120 caracteres.")
    if not 1 <= participant_count <= 100:
        raise ValueError("A campanha deve ter entre 1 e 100 participantes.")
    items = load_corpus(manifest_path, annotations_path)
    initialize_campaign_tables(connection)
    stimulus_dir.mkdir(parents=True, exist_ok=True)
    created_x: list[Path] = []
    participant_tokens: list[str] = []
    campaign_seed = secrets.randbits(64)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO campaigns VALUES (?, ?, ?, ?, 'interno')",
            (campaign_id.strip(), datetime.now(timezone.utc).isoformat(),
             _sha256(manifest_path.resolve()), len(items)),
        )
        for participant_index in range(participant_count):
            participant_token = secrets.token_urlsafe(24)
            participant_tokens.append(participant_token)
            participant_seed = int.from_bytes(
                hashlib.sha256(f"{campaign_seed}:{participant_index}".encode()).digest()[:8],
                "big",
            )
            rng = random.Random(participant_seed)
            connection.execute(
                "INSERT INTO campaign_participants VALUES (?, ?, ?)",
                (participant_token, campaign_id.strip(), participant_index + 1),
            )
            order = list(items)
            rng.shuffle(order)
            for position, item in enumerate(order, start=1):
                # Cada faixa percorre as quatro células A/B × X ao longo dos ouvintes.
                item_index = items.index(item)
                cell = (participant_index + item_index) % 4
                a_condition = "original" if cell < 2 else "processado"
                b_condition = "processado" if cell < 2 else "original"
                x_answer = "A" if cell % 2 == 0 else "B"
                x_condition = a_condition if x_answer == "A" else b_condition
                session_token = secrets.token_urlsafe(24)
                x_path = stimulus_dir / f"stimulus_{secrets.token_hex(16)}.wav"
                shutil.copyfile(item["paths"][x_condition], x_path)
                created_x.append(x_path)
                audit = {
                    "schema_version": 1, "session_id": session_token,
                    "experiment_id": campaign_id.strip(), "campaign_id": campaign_id.strip(),
                    "item_id": item["id"], "round_position": position,
                    "campaign_seed": campaign_seed, "participant_seed": participant_seed,
                    "item_order": [entry["id"] for entry in order],
                    "criada_em_utc": datetime.now(timezone.utc).isoformat(),
                    "modo_oficial": not bool(build_identity.get("git_dirty")),
                    "identidade_build": build_identity,
                    "a_condicao": a_condition, "b_condicao": b_condition,
                    "x_resposta": x_answer,
                    "cell_index": cell,
                    "hash_a_sha256": _sha256(item["paths"][a_condition]),
                    "hash_b_sha256": _sha256(item["paths"][b_condition]),
                    "hash_x_sha256": _sha256(x_path),
                    "sample_rate_hz": item["sample_rate_hz"],
                    "canais": item["canais"], "duracao_s": item["duracao_s"],
                    "diagnostico_motor": item["analysis"],
                    "escopo": "piloto interno",
                }
                connection.execute(
                    """INSERT INTO audition_sessions
                    (token, created_at_utc, experiment_id, stimulus_a_path,
                     stimulus_b_path, stimulus_x_path, a_condition, b_condition,
                     x_answer, audit_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (session_token, audit["criada_em_utc"], campaign_id.strip(),
                     str(item["paths"][a_condition]), str(item["paths"][b_condition]),
                     str(x_path), a_condition, b_condition, x_answer,
                     json.dumps(audit, ensure_ascii=False, sort_keys=True)),
                )
                connection.execute(
                    "INSERT INTO campaign_rounds VALUES (?, ?, ?, ?)",
                    (participant_token, position, item["id"], session_token),
                )
        connection.commit()
    except Exception:
        connection.rollback()
        for path in created_x:
            path.unlink(missing_ok=True)
        raise
    return participant_tokens


def campaign_progress(connection: sqlite3.Connection, token: str) -> dict[str, Any] | None:
    participant = connection.execute(
        "SELECT campaign_id FROM campaign_participants WHERE token = ?", (token,)
    ).fetchone()
    if participant is None:
        return None
    rows = connection.execute(
        """SELECT r.position, r.session_token, v.id AS vote_id
        FROM campaign_rounds r LEFT JOIN audition_votes v ON v.session_token = r.session_token
        WHERE r.participant_token = ? ORDER BY r.position""", (token,)
    ).fetchall()
    completed = sum(row["vote_id"] is not None for row in rows)
    next_round = next((row for row in rows if row["vote_id"] is None), None)
    return {"campaign_id": participant["campaign_id"], "completed": completed,
            "total": len(rows), "next_session_token": next_round["session_token"] if next_round else None,
            "next_position": next_round["position"] if next_round else None}


def campaign_round_context(connection: sqlite3.Connection, session_token: str) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT participant_token FROM campaign_rounds WHERE session_token = ?",
        (session_token,),
    ).fetchone()
    if row is None:
        return None
    progress = campaign_progress(connection, row["participant_token"])
    return {**progress, "participant_token": row["participant_token"]}


def _binomial_two_sided(successes: int, total: int) -> float | None:
    if total == 0:
        return None
    tail = sum(math.comb(total, k) for k in range(0, min(successes, total - successes) + 1))
    return min(1.0, 2.0 * tail / (2 ** total))


def _wilson_interval(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = statistics.NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - radius), 4), round(min(1.0, centre + radius), 4)]


def _aggregate(rows: list[sqlite3.Row]) -> dict[str, Any]:
    total = len(rows)
    correct = sum(int(row["x_answer_correct"]) for row in rows)
    preferences = {"original": 0, "processado": 0, "Sem diferença": 0}
    for row in rows:
        choice = row["preference_condition"] or "Sem diferença"
        preferences[choice] += 1
    return {"votos": total, "acertos_abx": correct,
            "taxa_acerto": round(correct / total, 4) if total else None,
            "ic95_wilson": _wilson_interval(correct, total),
            "p_binomial_bilateral": _binomial_two_sided(correct, total),
            "preferencias": preferences}


def _participant_aggregate(rows_by_participant: list[list[sqlite3.Row]]) -> dict[str, Any]:
    """Resume primeiro por ouvinte, preservando a unidade experimental correta."""
    if not rows_by_participant:
        return {"participantes": 0, "media_acerto": None, "desvio_padrao": None,
                "ic95_media": None, "media_preferencia_processado": None}
    accuracies = [
        sum(int(row["x_answer_correct"]) for row in rows) / len(rows)
        for rows in rows_by_participant if rows
    ]
    preference_rates = []
    for rows in rows_by_participant:
        answered = [row for row in rows if row["preference_condition"] in {"original", "processado"}]
        if answered:
            preference_rates.append(sum(row["preference_condition"] == "processado" for row in answered) / len(answered))
    mean_accuracy = statistics.fmean(accuracies)
    standard_deviation = statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0
    standard_error = standard_deviation / math.sqrt(len(accuracies))
    margin = 1.96 * standard_error
    return {
        "participantes": len(accuracies),
        "media_acerto": round(mean_accuracy, 4),
        "desvio_padrao": round(standard_deviation, 4),
        "ic95_media": [round(max(0.0, mean_accuracy - margin), 4),
                       round(min(1.0, mean_accuracy + margin), 4)],
        "media_preferencia_processado": round(statistics.fmean(preference_rates), 4)
        if preference_rates else None,
        "unidade": "participante",
    }


def campaign_report(connection: sqlite3.Connection, campaign_id: str) -> dict[str, Any]:
    campaign = connection.execute("SELECT * FROM campaigns WHERE id = ?", (campaign_id,)).fetchone()
    if campaign is None:
        raise KeyError("Campanha não encontrada.")
    participants = connection.execute(
        "SELECT token FROM campaign_participants WHERE campaign_id = ?", (campaign_id,)
    ).fetchall()
    valid_tokens: list[str] = []
    excluded: list[str] = []
    for participant in participants:
        progress = campaign_progress(connection, participant["token"])
        if progress["total"] == campaign["item_count"] and progress["completed"] == progress["total"]:
            valid_tokens.append(participant["token"])
        else:
            excluded.append(participant["token"])
    rows: list[sqlite3.Row] = []
    rows_by_participant: list[list[sqlite3.Row]] = []
    for token in valid_tokens:
        participant_rows = connection.execute(
            """SELECT r.item_id, s.audit_json, v.x_answer_correct, v.preference_condition
            FROM campaign_rounds r JOIN audition_sessions s ON s.token = r.session_token
            JOIN audition_votes v ON v.session_token = r.session_token
            WHERE r.participant_token = ? ORDER BY r.position""", (token,)
        ).fetchall()
        rows.extend(participant_rows)
        rows_by_participant.append(participant_rows)
    by_item: dict[str, list[sqlite3.Row]] = {}
    by_module: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_item.setdefault(row["item_id"], []).append(row)
        audit = json.loads(row["audit_json"])
        modules = audit.get("diagnostico_motor", {}).get("modulos_ativos") or {}
        for module, active in modules.items():
            if active:
                by_module.setdefault(module, []).append(row)
    return {"schema": "motor-clareza-campaign-report/v1", "campaign_id": campaign_id,
            "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
            "participantes_criados": len(participants), "participantes_completos": len(valid_tokens),
            "participantes_excluidos_incompletos": len(excluded),
            "resumo": _aggregate(rows),
            "resumo_por_participante": _participant_aggregate(rows_by_participant),
            "por_item": {key: _aggregate(value) for key, value in sorted(by_item.items())},
            "por_modulo_ativo": {key: _aggregate(value) for key, value in sorted(by_module.items())},
            "nota_inferencial": "Votos do mesmo ouvinte não são independentes; p binomial e IC por voto são exploratórios."}
