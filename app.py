"""PoC do Motor de Clareza Vocal PT-BR.

Arquitetura híbrida com separação Demucs, diagnóstico MIR auditável, decisão
por confiança, DSP regionalizado e controle de qualidade antes da exportação.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import secrets
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr
import librosa
import numpy as np
import pyloudnorm as pyln
import soundfile as sf
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from motor_vocal.campaign import (
    campaign_progress,
    campaign_round_context,
    initialize_campaign_tables,
)
from motor_vocal.quality import evaluate_stem_reconstruction
from motor_vocal.jobs import JobRecord, JobRequest, LocalJobStore, LocalWorker
from motor_vocal.processing import (
    DSP_PRESETS,
    MINIMUM_VOCAL_RMS_DBFS,
    TRUE_PEAK_TARGET_DBTP,
    _analyze_band,
    _approximate_true_peak_linear,
    _authorized_regions,
    _blend_regionally,
    _find_sustained_regions,
    _linear_to_dbfs,
    _match_rms_limited,
    _module_decision,
    _normalized_similarity,
    _run_pedalboard,
    _safe_target_frequency,
    _spectral_centroid_excerpt,
    _stereo_correlation,
    _temporal_envelope,
    analyze_ptbr_artifacts,
    apply_dsp_correction,
    evaluate_output_quality,
    mixdown_stems,
)
from motor_vocal.pipeline import process_audio_file
from motor_vocal.separation import (
    DemucsExecutionError,
    adapt_channel_count as _adapt_channel_count,
    align_stem_to_source as _align_stem_to_source,
    load_demucs_stem as _load_demucs_stem,
    load_source_audio as _load_source_audio,
    preferred_demucs_device as _preferred_demucs_device,
    run_demucs as _run_demucs,
    separate_stems,
    source_audio_metadata as _source_audio_metadata,
)


APP_TITLE = "Motor de Clareza Vocal PT-BR"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "motor_vocal_ptbr"
JOB_DATA_DIR = Path(__file__).resolve().parent / "outputs" / "jobs"
PILOT_MAX_JOBS_PER_DAY = 50
PILOT_MAX_DURATION_SECONDS = 300.0
PILOT_MAX_FILE_SIZE_BYTES = 150_000_000
AUDITION_DATA_DIR = Path(__file__).resolve().parent / "outputs" / "auditions"
AUDITION_DB_PATH = AUDITION_DATA_DIR / "auditions.sqlite3"
AUDITION_LOG_PATH = AUDITION_DATA_DIR / "abx_votes.jsonl"
AUDITION_STIMULUS_DIR = AUDITION_DATA_DIR / "stimuli"
ADMIN_USERNAME = os.environ.get("MOTOR_VOCAL_ADMIN_USER", "admin")
_CONFIGURED_ADMIN_PASSWORD = os.environ.get("MOTOR_VOCAL_ADMIN_PASSWORD")
ADMIN_PASSWORD = _CONFIGURED_ADMIN_PASSWORD or secrets.token_urlsafe(12)


class AuditionVote(BaseModel):
    """Payload público aceito pela página de audição."""

    participante_id: str = Field(default="", max_length=120)
    resposta_x: str
    preferencia: str
    confianca: int = Field(ge=1, le=5)
    observacoes: str = Field(default="", max_length=2_000)


class DuplicateVoteError(RuntimeError):
    """Indica que uma sessão já recebeu seu voto único."""

# O CSS complementa o tema Monochrome com uma estética escura de estúdio.
STUDIO_CSS = """
:root, body, .gradio-container {
    background: #09090b !important;
    color: #f4f4f5 !important;
}

.gradio-container {
    --body-background-fill: #09090b;
    --background-fill-primary: #111113;
    --background-fill-secondary: #18181b;
    --block-background-fill: #111113;
    --block-border-color: #2d2d31;
    --border-color-primary: #2d2d31;
    --body-text-color: #f4f4f5;
    --body-text-color-subdued: #a1a1aa;
    max-width: 1180px !important;
}

.studio-header {
    border-bottom: 1px solid #2d2d31;
    margin-bottom: 1.25rem;
    padding: 1.25rem 0 1rem;
}

.studio-header h1 {
    font-size: clamp(1.8rem, 4vw, 3rem);
    letter-spacing: -0.04em;
    margin-bottom: 0.35rem;
}

.studio-header p {
    color: #a1a1aa;
    font-size: 1rem;
    max-width: 760px;
}

.studio-panel {
    background: linear-gradient(145deg, #151518, #0f0f11) !important;
    border: 1px solid #2d2d31 !important;
    border-radius: 12px !important;
    padding: 0.65rem !important;
}

.mock-badge {
    background: #27272a;
    border: 1px solid #3f3f46;
    border-radius: 999px;
    color: #d4d4d8;
    display: inline-block;
    font-family: monospace;
    font-size: 0.78rem;
    letter-spacing: 0.05em;
    padding: 0.3rem 0.65rem;
    text-transform: uppercase;
}

.blind-panel {
    border-color: #52525b !important;
    margin-top: 1.25rem;
}

.blind-panel audio {
    width: 100%;
}

.blind-note {
    color: #a1a1aa;
    font-size: 0.9rem;
}

footer { display: none !important; }
"""


def _log(message: str) -> None:
    """Emite logs imediatamente no terminal que executa a aplicação."""
    print(f"[Motor Vocal PT-BR] {message}", flush=True)


def mixdown_and_export(
    processed_vocals: np.ndarray,
    instrumental: np.ndarray,
    sr: int,
) -> str:
    """Soma os stems e exporta um WAV em ponto flutuante de 32 bits."""
    _log("Etapa 4/4 — Remontando stems e exportando WAV 32-bit float...")

    mixed, gain_db = mixdown_stems(processed_vocals, instrumental)
    if gain_db < 0.0:
        _log(f"Proteção de true peak estimado aplicada: {gain_db:.2f} dB.")

    output_path = _export_audio(mixed, sr, "mix_processado")
    _log(f"Processamento concluído: {output_path}")
    return output_path


def _export_audio(audio: np.ndarray, sr: int, label: str) -> str:
    """Exporta um artefato de áudio temporário como WAV 32-bit float."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{label}_{uuid.uuid4().hex[:10]}.wav"
    sf.write(output_path, audio, sr, format="WAV", subtype="FLOAT")
    return str(output_path)


def _export_audition_stimulus(audio: np.ndarray, sr: int) -> str:
    """Persiste um estímulo com nome neutro junto ao banco de audições."""
    AUDITION_STIMULUS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AUDITION_STIMULUS_DIR / f"stimulus_{secrets.token_hex(16)}.wav"
    sf.write(output_path, audio, sr, format="WAV", subtype="FLOAT")
    return str(output_path)


def _sha256_file(path: str | Path) -> str:
    """Identifica um estímulo sem registrar seu nome original no resultado."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(*arguments: str) -> str | None:
    """Executa uma consulta Git curta sem tornar o Git obrigatório."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output or None


def _git_revision() -> str | None:
    """Retorna a revisão Git curta, quando disponível."""
    return _run_git("rev-parse", "--short=12", "HEAD")


def _build_identity() -> dict[str, Any]:
    """Identifica exatamente o código, configuração e runtime da audição."""
    project_dir = Path(__file__).resolve().parent
    status = _run_git("status", "--porcelain")
    dependency_path = project_dir / "requirements.txt"
    packages: dict[str, str | None] = {}
    for distribution in (
        "demucs",
        "gradio",
        "librosa",
        "numpy",
        "pedalboard",
        "pyloudnorm",
        "torch",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    preset_payload = json.dumps(
        DSP_PRESETS,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {
        "git_commit": _git_revision(),
        "git_dirty": bool(status),
        "git_status_sha256": (
            hashlib.sha256(status.encode("utf-8")).hexdigest() if status else None
        ),
        "app_sha256": _sha256_file(Path(__file__).resolve()),
        "requirements_sha256": (
            _sha256_file(dependency_path) if dependency_path.is_file() else None
        ),
        "configuracao_dsp_sha256": hashlib.sha256(preset_payload).hexdigest(),
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "pacotes": packages,
    }


def _load_aligned_pair(
    original_path: str,
    processed_path: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Carrega e alinha os dois estímulos antes da randomização cega."""
    original, original_sr = sf.read(original_path, dtype="float32", always_2d=True)
    processed, processed_sr = sf.read(
        processed_path,
        dtype="float32",
        always_2d=True,
    )
    original = np.asarray(original, dtype=np.float32)
    processed = np.asarray(processed, dtype=np.float32)
    if int(processed_sr) != int(original_sr):
        processed = librosa.resample(
            processed.T,
            orig_sr=int(processed_sr),
            target_sr=int(original_sr),
            axis=-1,
            res_type="soxr_hq",
        ).T
    processed = _adapt_channel_count(processed, original.shape[1])
    sample_count = min(original.shape[0], processed.shape[0])
    if sample_count <= 0:
        raise ValueError("Os estímulos da audição não contêm áudio válido.")
    return (
        np.ascontiguousarray(original[:sample_count], dtype=np.float32),
        np.ascontiguousarray(processed[:sample_count], dtype=np.float32),
        int(original_sr),
    )


def _integrated_loudness_lufs(audio: np.ndarray, sr: int) -> float | None:
    """Mede loudness integrado BS.1770, com proteção para silêncio e clipes curtos."""
    if audio.size == 0 or sr <= 0:
        return None
    measurement = np.asarray(audio, dtype=np.float64)
    if measurement.ndim == 2 and measurement.shape[1] > 5:
        measurement = measurement[:, :5]
    minimum_samples = int(np.ceil(0.4 * sr))
    if measurement.shape[0] < minimum_samples:
        padding = [(0, minimum_samples - measurement.shape[0])]
        if measurement.ndim == 2:
            padding.append((0, 0))
        measurement = np.pad(measurement, padding)
    try:
        loudness = float(pyln.Meter(sr).integrated_loudness(measurement))
    except (ValueError, IndexError, FloatingPointError):
        return None
    return loudness if np.isfinite(loudness) else None


def _level_match_abx_pair(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Iguala loudness BS.1770 por atenuação e preserva margem de true peak."""
    epsilon = np.finfo(np.float32).eps
    matched_original = original.copy()
    matched_processed = processed.copy()
    original_lufs = _integrated_loudness_lufs(original, sr)
    processed_lufs = _integrated_loudness_lufs(processed, sr)
    method = "ITU-R BS.1770-4 · loudness integrado"

    if original_lufs is not None and processed_lufs is not None:
        target_lufs = min(original_lufs, processed_lufs)
        original_gain_db = target_lufs - original_lufs
        processed_gain_db = target_lufs - processed_lufs
    else:
        # Um sinal abaixo do gate absoluto não possui LUFS integrado útil. O
        # fallback conserva o comportamento seguro sem tentar amplificar silêncio.
        method = "fallback RMS · loudness abaixo do gate"
        original_rms = float(np.sqrt(np.mean(np.square(original), dtype=np.float64)))
        processed_rms = float(np.sqrt(np.mean(np.square(processed), dtype=np.float64)))
        if original_rms <= epsilon or processed_rms <= epsilon:
            target_lufs = None
            original_gain_db = 0.0
            processed_gain_db = 0.0
        else:
            delta_db = float(20.0 * np.log10(processed_rms / original_rms))
            target_lufs = None
            original_gain_db = min(delta_db, 0.0)
            processed_gain_db = min(-delta_db, 0.0)

    matched_original *= float(10.0 ** (original_gain_db / 20.0))
    matched_processed *= float(10.0 ** (processed_gain_db / 20.0))

    # O mesmo ganho de segurança é aplicado aos dois lados para preservar a
    # igualdade perceptual conquistada acima.
    maximum_peak = max(
        _approximate_true_peak_linear(matched_original),
        _approximate_true_peak_linear(matched_processed),
    )
    target_linear = float(10.0 ** (TRUE_PEAK_TARGET_DBTP / 20.0))
    safety_gain_db = 0.0
    if maximum_peak > target_linear:
        safety_gain = target_linear / maximum_peak
        matched_original *= safety_gain
        matched_processed *= safety_gain
        safety_gain_db = float(20.0 * np.log10(safety_gain))

    original_after = _integrated_loudness_lufs(matched_original, sr)
    processed_after = _integrated_loudness_lufs(matched_processed, sr)
    residual = (
        processed_after - original_after
        if original_after is not None and processed_after is not None
        else None
    )
    rounded = lambda value: round(float(value), 4) if value is not None else None
    report = {
        "metodo": method,
        "original_lufs_antes": rounded(original_lufs),
        "processado_lufs_antes": rounded(processed_lufs),
        "delta_processado_menos_original_lu_antes": rounded(
            processed_lufs - original_lufs
            if original_lufs is not None and processed_lufs is not None
            else None
        ),
        "alvo_lufs_antes_protecao_pico": rounded(target_lufs),
        "ganho_original_db": rounded(original_gain_db + safety_gain_db),
        "ganho_processado_db": rounded(processed_gain_db + safety_gain_db),
        "ganho_comum_protecao_true_peak_db": rounded(safety_gain_db),
        "original_lufs_depois": rounded(original_after),
        "processado_lufs_depois": rounded(processed_after),
        "delta_residual_lu": rounded(residual),
        "alvo_true_peak_dbtp": TRUE_PEAK_TARGET_DBTP,
    }

    return matched_original, matched_processed, report


@contextmanager
def _audition_db() -> Iterator[sqlite3.Connection]:
    """Abre o banco local com integridade referencial e espera concorrente."""
    AUDITION_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(AUDITION_DB_PATH, timeout=15.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 15000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_audition_database() -> None:
    """Cria o armazenamento transacional e habilita concorrência WAL."""
    with _audition_db() as connection:
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS audition_sessions (
                token TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                experiment_id TEXT NOT NULL,
                stimulus_a_path TEXT NOT NULL,
                stimulus_b_path TEXT NOT NULL,
                stimulus_x_path TEXT NOT NULL,
                a_condition TEXT NOT NULL,
                b_condition TEXT NOT NULL,
                x_answer TEXT NOT NULL CHECK (x_answer IN ('A', 'B')),
                audit_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audition_votes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_token TEXT NOT NULL UNIQUE,
                submitted_at_utc TEXT NOT NULL,
                participant_id TEXT NOT NULL,
                x_answer TEXT NOT NULL CHECK (x_answer IN ('A', 'B')),
                x_answer_correct INTEGER NOT NULL CHECK (x_answer_correct IN (0, 1)),
                preference TEXT NOT NULL CHECK (preference IN ('A', 'B', 'Sem diferença')),
                preference_condition TEXT,
                confidence INTEGER NOT NULL CHECK (confidence BETWEEN 1 AND 5),
                notes TEXT,
                FOREIGN KEY (session_token) REFERENCES audition_sessions(token)
            );
            """
        )
        initialize_campaign_tables(connection)


def _store_audition_session(state: dict[str, Any]) -> None:
    """Persiste a rodada e sua verdade-terreno antes de publicar o link."""
    initialize_audition_database()
    paths = state["_condition_paths"]
    a_condition = state["a_condicao"]
    b_condition = state["b_condicao"]
    audit = {
        key: value
        for key, value in state.items()
        if not key.startswith("_") and key != "voto_registrado"
    }
    with _audition_db() as connection:
        connection.execute(
            """
            INSERT INTO audition_sessions (
                token, created_at_utc, experiment_id,
                stimulus_a_path, stimulus_b_path, stimulus_x_path,
                a_condition, b_condition, x_answer, audit_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                state["session_id"],
                state["criada_em_utc"],
                state["experiment_id"],
                paths[a_condition],
                paths[b_condition],
                state["_x_path"],
                a_condition,
                b_condition,
                state["x_resposta"],
                json.dumps(audit, ensure_ascii=False, sort_keys=True),
            ),
        )


def _get_audition_session(token: str) -> sqlite3.Row | None:
    """Busca uma rodada publicada por seu token opaco."""
    initialize_audition_database()
    with _audition_db() as connection:
        return connection.execute(
            """
            SELECT s.*,
                   EXISTS(
                       SELECT 1 FROM audition_votes v
                       WHERE v.session_token = s.token
                   ) AS has_vote
            FROM audition_sessions s
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()


def _save_vote_by_token(token: str, vote: AuditionVote) -> dict[str, Any]:
    """Grava exatamente um voto por sessão usando unicidade no SQLite."""
    if vote.resposta_x not in {"A", "B"}:
        raise ValueError("Informe se X corresponde a A ou B.")
    if vote.preferencia not in {"A", "B", "Sem diferença"}:
        raise ValueError("Informe sua preferência entre A, B ou sem diferença.")

    initialize_audition_database()
    clean_participant = vote.participante_id.strip() or "anonimo"
    clean_notes = vote.observacoes.strip() or None
    submitted_at = datetime.now(timezone.utc).isoformat()
    with _audition_db() as connection:
        connection.execute("BEGIN IMMEDIATE")
        session = connection.execute(
            """
            SELECT a_condition, b_condition, x_answer
            FROM audition_sessions WHERE token = ?
            """,
            (token,),
        ).fetchone()
        if session is None:
            raise KeyError("Sessão de audição não encontrada.")
        campaign_context = campaign_round_context(connection, token)
        if campaign_context is not None:
            if campaign_context["next_session_token"] != token:
                raise ValueError("Esta rodada não é a próxima da campanha.")
            clean_participant = "campanha:" + campaign_context["participant_token"][:12]
        preference_condition = (
            None
            if vote.preferencia == "Sem diferença"
            else session[f"{vote.preferencia.lower()}_condition"]
        )
        try:
            connection.execute(
                """
                INSERT INTO audition_votes (
                    session_token, submitted_at_utc, participant_id,
                    x_answer, x_answer_correct, preference,
                    preference_condition, confidence, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    token,
                    submitted_at,
                    clean_participant,
                    vote.resposta_x,
                    int(vote.resposta_x == session["x_answer"]),
                    vote.preferencia,
                    preference_condition,
                    vote.confianca,
                    clean_notes,
                ),
            )
        except sqlite3.IntegrityError as exc:
            if "audition_votes.session_token" in str(exc):
                raise DuplicateVoteError("Esta rodada já recebeu um voto.") from exc
            raise

    _log(f"Voto ABX registrado para a rodada {token[:8]} ({clean_participant}).")
    result = {
        "session_id": token,
        "voto_em_utc": submitted_at,
        "mensagem": "Voto registrado. Obrigado por participar.",
    }
    if campaign_context is not None:
        result["proxima_url"] = f"/campanha/{campaign_context['participant_token']}"
    return result


def export_audition_votes_jsonl() -> str:
    """Exporta do SQLite um JSONL reproduzível para análise externa."""
    initialize_audition_database()
    AUDITION_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _audition_db() as connection, AUDITION_LOG_PATH.open(
        "w", encoding="utf-8"
    ) as output:
        rows = connection.execute(
            """
            SELECT s.audit_json, v.submitted_at_utc, v.participant_id,
                   v.x_answer, v.x_answer_correct, v.preference,
                   v.preference_condition, v.confidence, v.notes
            FROM audition_votes v
            JOIN audition_sessions s ON s.token = v.session_token
            ORDER BY v.id
            """
        )
        for row in rows:
            record = json.loads(row["audit_json"])
            record.update(
                {
                    "voto_em_utc": row["submitted_at_utc"],
                    "participante_id": row["participant_id"],
                    "resposta_x": row["x_answer"],
                    "resposta_x_correta": bool(row["x_answer_correct"]),
                    "preferencia": row["preference"],
                    "preferencia_condicao": row["preference_condition"],
                    "confianca_1_a_5": row["confidence"],
                    "observacoes": row["notes"],
                }
            )
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return str(AUDITION_LOG_PATH)


def _balanced_abx_assignment(experiment_id: str, seed: int) -> tuple[bool, bool]:
    """Escolhe a célula menos usada do bloco A/B × X para reduzir viés de ordem."""
    initialize_audition_database()
    cells = [(True, True), (True, False), (False, True), (False, False)]
    counts = {cell: 0 for cell in cells}
    with _audition_db() as connection:
        rows = connection.execute(
            """
            SELECT a_condition, x_answer, COUNT(*) AS total
            FROM audition_sessions
            WHERE experiment_id = ?
            GROUP BY a_condition, x_answer
            """,
            (experiment_id,),
        )
        for row in rows:
            cell = (row["a_condition"] == "original", row["x_answer"] == "A")
            counts[cell] = int(row["total"])
    minimum = min(counts.values())
    candidates = [cell for cell, count in counts.items() if count == minimum]
    rng = np.random.default_rng(seed)
    return candidates[int(rng.integers(0, len(candidates)))]


def _randomize_abx_round(
    common_state: dict[str, Any],
) -> tuple[str, str, str, dict[str, Any], str]:
    """Sorteia A/B/X reutilizando os mesmos estímulos equalizados."""
    condition_paths = common_state["_condition_paths"]
    condition_hashes = common_state["_condition_hashes"]
    seed = secrets.randbits(64)
    a_is_original, x_is_a = _balanced_abx_assignment(
        common_state["experiment_id"],
        seed,
    )
    a_condition = "original" if a_is_original else "processado"
    b_condition = "processado" if a_is_original else "original"
    x_answer = "A" if x_is_a else "B"
    x_condition = a_condition if x_is_a else b_condition
    a_path = condition_paths[a_condition]
    b_path = condition_paths[b_condition]
    x_path = str(AUDITION_STIMULUS_DIR / f"stimulus_{secrets.token_hex(16)}.wav")
    AUDITION_STIMULUS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(condition_paths[x_condition], x_path)
    session_id = secrets.token_urlsafe(24)
    state = {
        **common_state,
        "session_id": session_id,
        "criada_em_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "a_condicao": a_condition,
        "b_condicao": b_condition,
        "x_resposta": x_answer,
        "hash_a_sha256": condition_hashes[a_condition],
        "hash_b_sha256": condition_hashes[b_condition],
        "hash_x_sha256": condition_hashes[x_condition],
        "_x_path": x_path,
        "voto_registrado": False,
    }
    _store_audition_session(state)
    status = (
        f"**Rodada pronta:** `{session_id[:8]}` · estímulos alinhados em "
        f"{state['sample_rate_hz']} Hz · loudness equalizado por BS.1770. Use os "
        "botões A/B/X para trocar sem perder a posição."
    )
    return a_path, b_path, x_path, state, status


def _audition_analysis_snapshot(
    analysis_data: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Seleciona as decisões técnicas necessárias para analisar os votos."""
    if not analysis_data:
        return None
    bands: dict[str, Any] = {}
    for output_name, analysis_name in (
        ("nasalidade", "nasalidade_ao_o"),
        ("estridencia", "estridencia_2_4khz"),
        ("sibilancia", "sibilancia_s_x"),
    ):
        band = analysis_data.get(analysis_name, {})
        bands[output_name] = {
            "confianca": band.get("confianca"),
            "intervencao_autorizada": band.get("intervencao_autorizada"),
            "motivo_abstencao": band.get("motivo_abstencao"),
            "eventos_sustentados": len(band.get("regioes_sustentadas", [])),
        }
    return {
        "modulos_ativos": analysis_data.get("modulos_ativos"),
        "bandas": bands,
        "correcao_dsp": analysis_data.get("correcao_dsp"),
        "qualidade_separacao": analysis_data.get("qualidade_separacao"),
        "controle_qualidade_saida": analysis_data.get("controle_qualidade_saida"),
    }


def prepare_abx_session(
    original_path: str | None,
    processed_path: str | None,
    preset_dsp: str,
    analysis_data: dict[str, Any] | None = None,
    experiment_id: str = "piloto",
    official_mode: bool = False,
) -> tuple[str, str, str, dict[str, Any], str]:
    """Gera uma rodada ABX cega, alinhada e reproduzível."""
    if not original_path or not processed_path:
        raise gr.Error("Processe um áudio antes de iniciar a audição cega.")

    try:
        build_identity = _build_identity()
        if official_mode and build_identity["git_dirty"]:
            raise gr.Error(
                "O modo oficial exige uma árvore Git limpa. Faça commit ou "
                "desative o modo oficial para testes exploratórios."
            )
        original, processed, sample_rate = _load_aligned_pair(
            original_path,
            processed_path,
        )
        matched_original, matched_processed, loudness_report = _level_match_abx_pair(
            original,
            processed,
            sample_rate,
        )
        original_stimulus_path = _export_audition_stimulus(
            matched_original,
            sample_rate,
        )
        processed_stimulus_path = _export_audition_stimulus(
            matched_processed,
            sample_rate,
        )
        common_state: dict[str, Any] = {
            "schema_version": 1,
            "experiment_id": str(experiment_id or "piloto").strip() or "piloto",
            "modo_oficial": bool(official_mode),
            "identidade_build": build_identity,
            "preset_dsp": str(preset_dsp),
            "diagnostico_motor": _audition_analysis_snapshot(analysis_data),
            "sample_rate_hz": sample_rate,
            "duracao_s": round(matched_original.shape[0] / sample_rate, 3),
            "canais": int(matched_original.shape[1]),
            "equalizacao_loudness": loudness_report,
            "_condition_paths": {
                "original": original_stimulus_path,
                "processado": processed_stimulus_path,
            },
            "_condition_hashes": {
                "original": _sha256_file(original_stimulus_path),
                "processado": _sha256_file(processed_stimulus_path),
            },
        }
        return _randomize_abx_round(common_state)
    except gr.Error:
        raise
    except Exception as exc:
        _log(f"Falha ao preparar rodada ABX: {exc}")
        raise gr.Error(f"Não foi possível preparar a rodada ABX: {exc}") from exc


def prepare_next_abx_session(
    previous_state: dict[str, Any] | None,
) -> tuple[str, str, str, dict[str, Any], str]:
    """Cria outra rodada imediatamente, sem recarregar ou reexportar o áudio."""
    if not previous_state:
        raise gr.Error("Processe um áudio antes de criar uma nova rodada.")
    condition_paths = previous_state.get("_condition_paths")
    condition_hashes = previous_state.get("_condition_hashes")
    if not isinstance(condition_paths, dict) or not isinstance(condition_hashes, dict):
        raise gr.Error("Os estímulos da sessão não estão mais disponíveis.")
    if not all(Path(path).is_file() for path in condition_paths.values()):
        raise gr.Error("Os arquivos temporários da sessão expiraram; reprocesse a faixa.")

    round_keys = {
        "session_id",
        "criada_em_utc",
        "seed",
        "a_condicao",
        "b_condicao",
        "x_resposta",
        "hash_a_sha256",
        "hash_b_sha256",
        "hash_x_sha256",
        "_x_path",
        "voto_registrado",
    }
    common_state = {
        key: value for key, value in previous_state.items() if key not in round_keys
    }
    return _randomize_abx_round(common_state)


def save_abx_vote(
    session_state: dict[str, Any] | None,
    participant_id: str,
    x_answer: str | None,
    preference: str | None,
    confidence: float,
    notes: str,
) -> tuple[dict[str, Any], str]:
    """Compatibilidade interna para registrar votos pelo estado da sessão."""
    if not session_state or not session_state.get("session_id"):
        raise gr.Error("Crie uma rodada ABX antes de registrar o voto.")
    try:
        vote = AuditionVote(
            participante_id=str(participant_id or ""),
            resposta_x=str(x_answer or ""),
            preferencia=str(preference or ""),
            confianca=int(round(float(confidence))),
            observacoes=str(notes or ""),
        )
        _save_vote_by_token(session_state["session_id"], vote)
    except DuplicateVoteError as exc:
        raise gr.Error(str(exc)) from exc
    except (KeyError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    updated_state = {**session_state, "voto_registrado": True}
    return (
        updated_state,
        "✅ Voto registrado sem revelar a resposta.",
    )


def create_audition_link(
    original_path: str | None,
    processed_path: str | None,
    preset_dsp: str,
    analysis_data: dict[str, Any] | None,
    experiment_id: str,
    official_mode: bool,
    request: gr.Request,
) -> tuple[str, str]:
    """Publica uma rodada no banco e retorna somente seu link participante."""
    _, _, _, state, _ = prepare_abx_session(
        original_path,
        processed_path,
        preset_dsp,
        analysis_data,
        experiment_id,
        official_mode,
    )
    headers = request.headers if request is not None else {}
    host = headers.get("x-forwarded-host") or headers.get("host") or "127.0.0.1:7860"
    protocol = headers.get("x-forwarded-proto") or "http"
    url = f"{protocol}://{host}/audicao/{state['session_id']}"
    build = state["identidade_build"]
    build_label = build["git_commit"] or build["app_sha256"][:12]
    status = (
        f"✅ **Rodada publicada** · experimento `{state['experiment_id']}` · "
        f"build `{build_label}` · modo "
        f"{'oficial' if official_mode else 'exploratório'}."
    )
    return url, status


def create_audition_link_for_job(
    job_id: str,
    original_path: str | None,
    processed_path: str | None,
    preset_dsp: str,
    correct_nasality: bool,
    correct_stridency: bool,
    correct_sibilance: bool,
    experiment_id: str,
    official_mode: bool,
    request: gr.Request,
) -> tuple[str, str]:
    """Impede publicar um par que não corresponde ao trabalho exibido."""
    job = _job_store().get(job_id.strip()) if job_id else None
    if job is None or job.state != "succeeded" or job.result is None:
        raise gr.Error("Conclua um trabalho antes de criar a audição.")
    options_match = (
        job.request.preset_dsp == preset_dsp
        and job.request.correct_nasality == bool(correct_nasality)
        and job.request.correct_stridency == bool(correct_stridency)
        and job.request.correct_sibilance == bool(correct_sibilance)
    )
    if not options_match or processed_path != job.result["mixed_path"]:
        raise gr.Error("A configuração ou o resultado exibido mudou; processe novamente.")
    try:
        source_matches = bool(original_path) and (
            _sha256_file(original_path) == _sha256_file(job.request.input_path)
        )
    except OSError as exc:
        raise gr.Error("O áudio original não está mais disponível.") from exc
    if not source_matches:
        raise gr.Error("O áudio original mudou; processe novamente.")
    return create_audition_link(
        job.request.input_path,
        job.result["mixed_path"],
        job.request.preset_dsp,
        job.result["analysis"],
        experiment_id,
        official_mode,
        request,
    )


PARTICIPANT_PAGE = r"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Audição cega A/B/ABX</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, system-ui, sans-serif; }
    * { box-sizing: border-box; }
    body { margin: 0; background: #09090b; color: #f4f4f5; }
    main { width: min(880px, calc(100% - 32px)); margin: 40px auto; }
    .badge { display: inline-block; padding: 6px 10px; border: 1px solid #3f3f46;
      border-radius: 999px; color: #d4d4d8; font: 12px ui-monospace, monospace; }
    h1 { margin: 18px 0 8px; font-size: clamp(28px, 5vw, 46px); }
    p { color: #a1a1aa; line-height: 1.55; }
    .card { margin-top: 24px; padding: 24px; border: 1px solid #2d2d31;
      border-radius: 16px; background: linear-gradient(145deg, #151518, #0f0f11); }
    .players { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
    button { min-height: 52px; border: 1px solid #52525b; border-radius: 10px;
      background: #27272a; color: #fafafa; font-weight: 700; cursor: pointer; }
    button:hover, button.active { border-color: #fafafa; background: #3f3f46; }
    button.primary { background: #fafafa; color: #09090b; }
    button:disabled { opacity: .5; cursor: not-allowed; }
    .transport { display: grid; grid-template-columns: 90px 1fr 110px; gap: 12px;
      align-items: center; margin: 20px 0; }
    input[type=range] { width: 100%; }
    fieldset { border: 0; padding: 0; margin: 20px 0; }
    legend, label.field { display: block; margin-bottom: 8px; font-weight: 700; }
    .options { display: flex; flex-wrap: wrap; gap: 16px; }
    input[type=text], textarea, select { width: 100%; padding: 12px; color: #fafafa;
      background: #18181b; border: 1px solid #3f3f46; border-radius: 8px; }
    textarea { min-height: 88px; resize: vertical; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
    #status { min-height: 24px; color: #d4d4d8; }
    .success { color: #86efac !important; }
    .error { color: #fca5a5 !important; }
    @media (max-width: 640px) { .grid { grid-template-columns: 1fr; }
      .transport { grid-template-columns: 72px 1fr; } #time { grid-column: 1 / -1; }
    }
  </style>
</head>
<body>
<main>
  <span class="badge">TESTE CEGO · ABX</span>
  <h1>Audição A/B/ABX</h1>
  <p>Compare os três estímulos. X é idêntico a A ou B. Use os botões ou as
    teclas 1, 2 e 3; a reprodução permanece no mesmo ponto durante a troca.</p>
  <p id="campaignProgress" hidden></p>
  <section class="card">
    <div class="players">
      <button type="button" data-listen="A" disabled>1 · Ouvir A</button>
      <button type="button" data-listen="B" disabled>2 · Ouvir B</button>
      <button type="button" data-listen="X" disabled>3 · Ouvir X</button>
    </div>
    <div class="transport">
      <button type="button" id="playPause" disabled>Reproduzir</button>
      <input id="position" type="range" min="0" max="1" step="0.01" value="0"
        aria-label="Posição da reprodução">
      <output id="time">00:00 / 00:00</output>
    </div>
    <div id="status" role="status">Carregando estímulos…</div>
  </section>
  <form id="voteForm" class="card">
    <div class="grid">
      <fieldset><legend>X corresponde a</legend><div class="options">
        <label><input required type="radio" name="resposta_x" value="A"> A</label>
        <label><input required type="radio" name="resposta_x" value="B"> B</label>
      </div></fieldset>
      <fieldset><legend>Qual você prefere?</legend><div class="options">
        <label><input required type="radio" name="preferencia" value="A"> A</label>
        <label><input required type="radio" name="preferencia" value="B"> B</label>
        <label><input required type="radio" name="preferencia" value="Sem diferença">
          Sem diferença</label>
      </div></fieldset>
    </div>
    <div class="grid">
      <label class="field">Participante (código ou pseudônimo)
        <input name="participante_id" type="text" maxlength="120">
      </label>
      <label class="field">Confiança
        <select name="confianca"><option>1</option><option>2</option>
          <option selected>3</option><option>4</option><option>5</option></select>
      </label>
    </div>
    <label class="field">Observações
      <textarea name="observacoes" maxlength="2000"></textarea>
    </label>
    <button class="primary" id="submitVote" type="submit">Registrar voto</button>
  </form>
  <p><a id="nextRound" href="#" hidden>Continuar para o próximo item</a></p>
</main>
<script>
const token = __TOKEN_JSON__;
const campaignInfo = __CAMPAIGN_JSON__;
const labels = ['A', 'B', 'X'];
const buttons = Object.fromEntries(labels.map(label => [label,
  document.querySelector(`[data-listen="${label}"]`)]));
const statusBox = document.querySelector('#status');
const slider = document.querySelector('#position');
const timeBox = document.querySelector('#time');
const playPause = document.querySelector('#playPause');
let context, buffers = {}, sources = {}, gains = {};
let selected = 'A', playing = false, offset = 0, startedAt = 0, duration = 0;
if (campaignInfo) {
  const progress = document.querySelector('#campaignProgress');
  progress.hidden = false;
  progress.textContent = `Item ${campaignInfo.position} de ${campaignInfo.total}`;
  document.querySelector('input[name="participante_id"]').closest('label').hidden = true;
}

const formatTime = value => {
  const seconds = Math.max(0, Math.floor(value || 0));
  return `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(seconds % 60).padStart(2, '0')}`;
};
const positionNow = () => playing
  ? Math.min(duration, offset + Math.max(0, context.currentTime - startedAt)) : offset;
function stopSources(remember = true) {
  if (remember) offset = positionNow();
  Object.values(sources).forEach(source => { try { source.stop(); } catch (_) {} });
  sources = {}; gains = {}; playing = false; playPause.textContent = 'Reproduzir';
}
async function startAll(label) {
  await context.resume();
  if (offset >= duration - 0.01) offset = 0;
  const startAt = context.currentTime + 0.025;
  labels.forEach(name => {
    const source = context.createBufferSource();
    const gain = context.createGain();
    source.buffer = buffers[name]; gain.gain.value = name === label ? 1 : 0;
    source.connect(gain).connect(context.destination);
    source.start(startAt, offset); sources[name] = source; gains[name] = gain;
  });
  selected = label; startedAt = startAt; playing = true;
  playPause.textContent = 'Pausar'; updateButtons();
}
function updateButtons() {
  labels.forEach(label => buttons[label].classList.toggle('active', label === selected));
}
async function select(label) {
  if (!playing) return startAll(label);
  const now = context.currentTime;
  labels.forEach(name => {
    const parameter = gains[name].gain;
    parameter.cancelScheduledValues(now);
    parameter.setValueAtTime(parameter.value, now);
    parameter.linearRampToValueAtTime(name === label ? 1 : 0, now + 0.005);
  });
  selected = label; updateButtons();
}
function animate() {
  const current = positionNow();
  slider.value = String(current); timeBox.textContent = `${formatTime(current)} / ${formatTime(duration)}`;
  if (playing && current >= duration - 0.02) { stopSources(false); offset = 0; }
  requestAnimationFrame(animate);
}
async function load() {
  const manifestResponse = await fetch(`/api/audicao/${token}/manifesto`, {cache: 'no-store'});
  if (!manifestResponse.ok) throw new Error('Rodada indisponível.');
  const manifest = await manifestResponse.json();
  if (manifest.voto_registrado) throw new Error('Esta rodada já recebeu um voto.');
  context = new AudioContext();
  await Promise.all(labels.map(async label => {
    const response = await fetch(`/api/audicao/${token}/audio/${label}`, {cache: 'no-store'});
    if (!response.ok) throw new Error(`Falha ao carregar ${label}.`);
    buffers[label] = await context.decodeAudioData(await response.arrayBuffer());
  }));
  duration = Math.min(...labels.map(label => buffers[label].duration));
  slider.max = String(duration); Object.values(buttons).forEach(button => button.disabled = false);
  playPause.disabled = false; statusBox.textContent = 'Estímulos prontos.'; updateButtons();
}
Object.entries(buttons).forEach(([label, button]) => button.addEventListener('click', () => select(label)));
playPause.addEventListener('click', () => playing ? stopSources() : startAll(selected));
slider.addEventListener('input', async () => {
  const resume = playing; if (playing) stopSources(false); offset = Number(slider.value);
  if (resume) await startAll(selected);
});
document.addEventListener('keydown', event => {
  if (event.target.matches('input, textarea, select')) return;
  if (event.key === '1') select('A'); if (event.key === '2') select('B');
  if (event.key === '3') select('X'); if (event.code === 'Space') {
    event.preventDefault(); playing ? stopSources() : startAll(selected);
  }
});
document.querySelector('#voteForm').addEventListener('submit', async event => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  const payload = Object.fromEntries(form.entries()); payload.confianca = Number(payload.confianca);
  const submit = document.querySelector('#submitVote'); submit.disabled = true;
  const response = await fetch(`/api/audicao/${token}/voto`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)
  });
  const result = await response.json();
  if (!response.ok) { statusBox.textContent = result.detail || 'Não foi possível registrar.';
    statusBox.className = 'error'; submit.disabled = false; return; }
  stopSources(); statusBox.textContent = result.mensagem; statusBox.className = 'success';
  event.currentTarget.querySelectorAll('input, textarea, select, button').forEach(element => element.disabled = true);
  if (result.proxima_url) {
    const next = document.querySelector('#nextRound');
    next.href = result.proxima_url;
    next.textContent = campaignInfo && campaignInfo.position === campaignInfo.total
      ? 'Concluir campanha' : 'Continuar para o próximo item';
    next.hidden = false;
  }
});
load().catch(error => { statusBox.textContent = error.message; statusBox.className = 'error'; });
animate();
</script>
</body></html>"""


def _participant_page(token: str, campaign_info: dict[str, int] | None = None) -> str:
    """Renderiza a superfície cega sem incluir mapa ou nomes de condições."""
    safe_token_json = json.dumps(token).replace("<", "\\u003c")
    safe_campaign_json = json.dumps(campaign_info).replace("<", "\\u003c")
    return (PARTICIPANT_PAGE.replace("__TOKEN_JSON__", safe_token_json)
            .replace("__CAMPAIGN_JSON__", safe_campaign_json))


def _job_store() -> LocalJobStore:
    return LocalJobStore(JOB_DATA_DIR / "jobs.sqlite3")


def _stage_job_input(audio_path: str) -> tuple[Path, str]:
    """Valida e copia o upload para um caminho que sobreviva ao evento Gradio."""
    source = Path(audio_path).resolve()
    if not source.is_file() or source.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError("Envie um arquivo WAV ou MP3 válido.")
    if source.stat().st_size > PILOT_MAX_FILE_SIZE_BYTES:
        raise ValueError("O arquivo deve ter até 150 MB.")
    info = sf.info(source)
    if info.duration <= 0 or info.duration > PILOT_MAX_DURATION_SECONDS:
        raise ValueError("O áudio deve ter duração maior que zero e até 5 minutos.")

    staging_dir = JOB_DATA_DIR / "uploads"
    staging_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(dir=staging_dir, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with source.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1_048_576), b""):
                    digest.update(chunk)
                    temporary.write(chunk)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
    staged = staging_dir / f"{digest.hexdigest()}{source.suffix.lower()}"
    try:
        if not staged.is_file():
            os.replace(temporary_path, staged)
    finally:
        temporary_path.unlink(missing_ok=True)
    upload_identity = hashlib.sha256(
        f"{source}|{digest.hexdigest()}".encode()
    ).hexdigest()
    return staged, upload_identity


def enqueue_audio_job(
    audio_path: str | None,
    preset_dsp: str = "Balanceado",
    correct_nasality: bool = True,
    correct_stridency: bool = True,
    correct_sibilance: bool = True,
) -> JobRecord:
    """Admite um upload local com limites do piloto e idempotência por envio."""
    if not audio_path:
        raise ValueError("Envie um arquivo WAV ou MP3 antes de iniciar.")
    staged, upload_identity = _stage_job_input(audio_path)
    request = JobRequest(
        input_path=str(staged),
        preset_dsp=preset_dsp,
        correct_nasality=bool(correct_nasality),
        correct_stridency=bool(correct_stridency),
        correct_sibilance=bool(correct_sibilance),
    )
    key_payload = json.dumps(
        {"upload": upload_identity, "request": request.__dict__}, sort_keys=True
    )
    idempotency_key = hashlib.sha256(key_payload.encode()).hexdigest()
    return _job_store().enqueue(
        request, idempotency_key, max_jobs_per_day=PILOT_MAX_JOBS_PER_DAY
    )


def submit_audio_job(
    audio_path: str | None,
    preset_dsp: str,
    correct_nasality: bool,
    correct_stridency: bool,
    correct_sibilance: bool,
) -> tuple[Any, ...]:
    """Envia o trabalho e limpa saídas anteriores no painel técnico."""
    try:
        job = enqueue_audio_job(
            audio_path,
            preset_dsp,
            correct_nasality,
            correct_stridency,
            correct_sibilance,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc
    return (
        job.job_id,
        f"Trabalho `{job.job_id}`: {job.state}.",
        gr.update(active=job.state in {"queued", "running"}),
        None, None, None, None, None, None, None, None, "", "",
    )


def resume_audio_job(job_id: str) -> tuple[str, dict[str, Any]]:
    """Reanexa a consulta a um trabalho após recarregar o painel."""
    job = _job_store().get(job_id.strip()) if job_id else None
    if job is None:
        raise gr.Error("Trabalho não encontrado.")
    return f"Consultando trabalho `{job.job_id}`.", gr.update(active=True)


def poll_audio_job(job_id: str | None) -> tuple[Any, ...]:
    """Consulta o trabalho pelo ID e entrega os artefatos após a conclusão."""
    if not job_id:
        return ("", gr.update(active=False), *(gr.skip() for _ in range(8)))
    job = _job_store().get(job_id)
    if job is None:
        return ("Trabalho não encontrado.", gr.update(active=False), *(gr.skip() for _ in range(8)))
    if job.state in {"queued", "running"}:
        label = "Na fila" if job.state == "queued" else "Processando"
        return (
            f"Trabalho `{job.job_id}`: {label}.",
            gr.update(active=True),
            *(gr.skip() for _ in range(8)),
        )
    if job.state == "failed":
        return (
            f"Trabalho `{job.job_id}`: falhou. {job.error or ''}",
            gr.update(active=False),
            *(gr.skip() for _ in range(8)),
        )
    assert job.result is not None
    result = job.result
    try:
        comparison_original, comparison_processed, loudness_report = (
            prepare_loudness_comparison(
                job.request.input_path, result["mixed_path"]
            )
        )
        status = f"Trabalho `{job.job_id}`: concluído."
    except Exception as exc:
        comparison_original = comparison_processed = loudness_report = None
        status = f"Trabalho `{job.job_id}`: concluído; comparação indisponível: {exc}"
    return (
        status,
        gr.update(active=False),
        result["mixed_path"],
        result["vocal_path"],
        result["processed_vocal_path"],
        result["instrumental_path"],
        result["analysis"],
        comparison_original,
        comparison_processed,
        loudness_report,
    )


def _local_worker_loop(stop: threading.Event, store: LocalJobStore) -> None:
    """Mantém exatamente um consumidor no processo web local."""
    worker = LocalWorker(store, JOB_DATA_DIR / "artifacts")
    last_recovery = 0.0
    while not stop.is_set():
        try:
            if time.monotonic() - last_recovery >= 60.0:
                store.recover_stale(max_age_seconds=900)
                last_recovery = time.monotonic()
            if worker.run_once() is None:
                stop.wait(0.5)
        except Exception as exc:
            _log(f"Worker local indisponível: {type(exc).__name__}.")
            stop.wait(1.0)


def process_audio(
    audio_path: str | None,
    preset_dsp: str = "Balanceado",
    correct_nasality: bool = True,
    correct_stridency: bool = True,
    correct_sibilance: bool = True,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, dict[str, Any]]:
    """Orquestra o pipeline e atualiza o progresso visual do Gradio."""
    if not audio_path:
        raise gr.Error("Envie um arquivo WAV ou MP3 antes de iniciar.")

    try:
        progress(0.05, desc="Preparando o arquivo de áudio...")
        result = process_audio_file(
            audio_path,
            OUTPUT_DIR / uuid.uuid4().hex,
            preset_dsp=preset_dsp,
            correct_nasality=correct_nasality,
            correct_stridency=correct_stridency,
            correct_sibilance=correct_sibilance,
            progress=lambda value, description: progress(value, desc=description),
        )
        return (
            str(result.mixed_path),
            str(result.vocal_path),
            str(result.processed_vocal_path),
            str(result.instrumental_path),
            result.analysis,
        )
    except gr.Error:
        raise
    except Exception as exc:
        _log(f"Falha no pipeline: {exc}")
        traceback.print_exc()
        raise gr.Error(f"Não foi possível processar o áudio: {exc}") from exc


def prepare_loudness_comparison(
    original_path: str | None,
    processed_path: str | None,
) -> tuple[str, str, dict[str, Any]]:
    """Gera o par técnico nivelado usado como referência pela audição cega."""
    if not original_path or not processed_path:
        raise gr.Error("Processe um áudio antes de preparar a comparação nivelada.")
    try:
        original, processed, sample_rate = _load_aligned_pair(
            original_path,
            processed_path,
        )
        matched_original, matched_processed, report = _level_match_abx_pair(
            original,
            processed,
            sample_rate,
        )
        original_output = _export_audio(
            matched_original,
            sample_rate,
            "comparacao_original_lufs",
        )
        processed_output = _export_audio(
            matched_processed,
            sample_rate,
            "comparacao_processado_lufs",
        )
        return original_output, processed_output, report
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Não foi possível equalizar o loudness: {exc}") from exc


def gradio_interface() -> gr.Blocks:
    """Constrói o painel técnico, separado da página dos participantes."""
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(
            """
            <section class="studio-header">
                <span class="mock-badge">PoC · Demucs + DSP ativos</span>
                <h1>Motor de Clareza Vocal PT-BR</h1>
                <p>
                    Separação vocal, análise espectral e correção de nasalidade,
                    estridência e sibilância regionalizada para voz cantada brasileira.
                </p>
            </section>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1, elem_classes="studio-panel"):
                gr.Markdown("## A · Áudio original")
                input_audio = gr.Audio(
                    label="Envie um arquivo WAV ou MP3",
                    sources=["upload"],
                    type="filepath",
                )
                dsp_preset = gr.Radio(
                    choices=list(DSP_PRESETS),
                    value="Balanceado",
                    label="Intensidade da correção",
                )
                gr.Markdown("#### Módulos ativos")
                with gr.Row():
                    correct_nasality = gr.Checkbox(
                        value=True,
                        label="Nasalidade",
                    )
                    correct_stridency = gr.Checkbox(
                        value=True,
                        label="Estridência",
                    )
                    correct_sibilance = gr.Checkbox(
                        value=True,
                        label="Sibilância",
                    )
                process_button = gr.Button(
                    "Processar áudio",
                    variant="primary",
                    size="lg",
                )
                job_id = gr.Textbox(
                    label="ID do trabalho",
                    interactive=True,
                    buttons=["copy"],
                )
                resume_button = gr.Button("Consultar trabalho")
                job_status = gr.Markdown("Aguardando envio.")
                poll_timer = gr.Timer(0.5, active=False)
                gr.Markdown(
                    "Na primeira execução, o Demucs baixará o modelo htdemucs. "
                    "Use o preset **Suave** para material já masterizado e "
                    "compare o resultado com o original antes de aumentar a "
                    "intensidade."
                )

            with gr.Column(scale=1, elem_classes="studio-panel"):
                gr.Markdown("## Arquivo processado")
                output_audio = gr.Audio(
                    label="Resultado bruto para download e auditoria",
                    type="filepath",
                    interactive=False,
                )
                with gr.Accordion("Auditoria dos stems", open=False):
                    vocal_output = gr.Audio(
                        label="Vocal isolado · antes do DSP",
                        type="filepath",
                        interactive=False,
                    )
                    processed_vocal_output = gr.Audio(
                        label="Vocal isolado · depois do DSP",
                        type="filepath",
                        interactive=False,
                    )
                    instrumental_output = gr.Audio(
                        label="Instrumental · no_vocals",
                        type="filepath",
                        interactive=False,
                    )
                analysis_output = gr.JSON(
                    label="Diagnóstico espectral preliminar",
                    open=False,
                )

        with gr.Column(elem_classes=["studio-panel", "blind-panel"]):
            gr.Markdown(
                "## Comparação técnica nivelada\n"
                "Use estes dois players para a análise auditiva. Ambos recebem "
                "equalização de loudness integrado BS.1770 e a mesma proteção "
                "de true peak aplicada aos estímulos cegos."
            )
            with gr.Row():
                comparison_original = gr.Audio(
                    label="Original · loudness nivelado",
                    type="filepath",
                    interactive=False,
                    editable=False,
                    buttons=[],
                )
                comparison_processed = gr.Audio(
                    label="Processado · loudness nivelado",
                    type="filepath",
                    interactive=False,
                    editable=False,
                    buttons=[],
                )
            loudness_report = gr.JSON(
                label="Auditoria de loudness",
                open=False,
            )

        with gr.Column(elem_classes=["studio-panel", "blind-panel"]):
            gr.Markdown(
                "## Publicar audição cega\n"
                "Crie um link isolado para o participante. A página pública não "
                "exibe os players técnicos, o diagnóstico nem o mapa A/B/X."
            )
            with gr.Row():
                experiment_id = gr.Textbox(
                    value=f"piloto-{datetime.now().strftime('%Y%m%d')}",
                    label="ID do experimento",
                )
                official_mode = gr.Checkbox(
                    value=False,
                    label="Modo oficial · exigir Git limpo",
                )
            with gr.Row():
                publish_button = gr.Button(
                    "Criar link de audição",
                    variant="primary",
                )
                export_votes_button = gr.Button("Exportar votos JSONL")
            participant_link = gr.Textbox(
                label="Link do participante",
                interactive=False,
                buttons=["copy"],
            )
            publish_status = gr.Markdown(elem_classes="blind-note")
            exported_votes = gr.File(label="Exportação dos votos", interactive=False)

        process_button.click(
            fn=submit_audio_job,
            inputs=[
                input_audio,
                dsp_preset,
                correct_nasality,
                correct_stridency,
                correct_sibilance,
            ],
            outputs=[
                job_id,
                job_status,
                poll_timer,
                output_audio,
                vocal_output,
                processed_vocal_output,
                instrumental_output,
                analysis_output,
                comparison_original,
                comparison_processed,
                loudness_report,
                participant_link,
                publish_status,
            ],
            show_progress="minimal",
            concurrency_limit=5,
        )
        poll_timer.tick(
            fn=poll_audio_job,
            inputs=[job_id],
            outputs=[
                job_status,
                poll_timer,
                output_audio,
                vocal_output,
                processed_vocal_output,
                instrumental_output,
                analysis_output,
                comparison_original,
                comparison_processed,
                loudness_report,
            ],
            queue=False,
        )
        resume_button.click(
            fn=resume_audio_job,
            inputs=[job_id],
            outputs=[job_status, poll_timer],
            show_progress="minimal",
        )
        publish_button.click(
            fn=create_audition_link_for_job,
            inputs=[
                job_id,
                input_audio,
                output_audio,
                dsp_preset,
                correct_nasality,
                correct_stridency,
                correct_sibilance,
                experiment_id,
                official_mode,
            ],
            outputs=[participant_link, publish_status],
            show_progress="minimal",
        )
        export_votes_button.click(
            fn=export_audition_votes_jsonl,
            outputs=[exported_votes],
            show_progress="minimal",
        )

    return demo


def create_web_app() -> FastAPI:
    """Monta painel técnico e audição cega em superfícies HTTP separadas."""
    initialize_audition_database()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        stop = threading.Event()
        worker = threading.Thread(
            target=_local_worker_loop,
            args=(stop, _job_store()),
            name="motor-vocal-local-worker",
            daemon=True,
        )
        worker.start()
        try:
            yield
        finally:
            stop.set()
            worker.join(timeout=5.0)

    web_app = FastAPI(title=APP_TITLE, lifespan=lifespan)

    @web_app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/admin/")

    @web_app.get("/audicao/{token}", response_class=HTMLResponse)
    async def participant(token: str) -> HTMLResponse:
        session = _get_audition_session(token)
        if session is None:
            raise HTTPException(status_code=404, detail="Rodada não encontrada.")
        return HTMLResponse(
            _participant_page(token),
            headers={"Cache-Control": "no-store"},
        )

    @web_app.get("/campanha/{token}", response_class=HTMLResponse)
    async def campaign_participant(token: str) -> HTMLResponse:
        with _audition_db() as connection:
            progress = campaign_progress(connection, token)
        if progress is None:
            raise HTTPException(status_code=404, detail="Campanha não encontrada.")
        if progress["next_session_token"] is None:
            return HTMLResponse(
                "<!doctype html><html lang='pt-BR'><meta charset='utf-8'>"
                "<title>Campanha concluída</title><body><h1>Campanha concluída</h1>"
                "<p>Obrigado por participar.</p></body></html>",
                headers={"Cache-Control": "no-store"},
            )
        return HTMLResponse(
            _participant_page(progress["next_session_token"], {
                "position": progress["next_position"], "total": progress["total"]
            }),
            headers={"Cache-Control": "no-store"},
        )

    @web_app.get("/api/audicao/{token}/manifesto")
    async def audition_manifest(token: str) -> dict[str, Any]:
        session = _get_audition_session(token)
        if session is None:
            raise HTTPException(status_code=404, detail="Rodada não encontrada.")
        audit = json.loads(session["audit_json"])
        return {
            "session_id": token,
            "experimento": session["experiment_id"],
            "duracao_s": audit["duracao_s"],
            "sample_rate_hz": audit["sample_rate_hz"],
            "canais": audit["canais"],
            "voto_registrado": bool(session["has_vote"]),
        }

    @web_app.get("/api/audicao/{token}/audio/{label}")
    async def audition_audio(token: str, label: str) -> FileResponse:
        session = _get_audition_session(token)
        if session is None:
            raise HTTPException(status_code=404, detail="Rodada não encontrada.")
        normalized_label = label.upper()
        column_by_label = {
            "A": "stimulus_a_path",
            "B": "stimulus_b_path",
            "X": "stimulus_x_path",
        }
        column = column_by_label.get(normalized_label)
        if column is None:
            raise HTTPException(status_code=404, detail="Estímulo não encontrado.")
        path = Path(session[column])
        if not path.is_file():
            raise HTTPException(status_code=410, detail="O estímulo expirou.")
        return FileResponse(
            path,
            media_type="audio/wav",
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
        )

    @web_app.post("/api/audicao/{token}/voto")
    async def audition_vote(token: str, vote: AuditionVote) -> dict[str, Any]:
        try:
            return _save_vote_by_token(token, vote)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except DuplicateVoteError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    admin = gradio_interface()
    return gr.mount_gradio_app(
        web_app,
        admin,
        path="/admin",
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Monochrome(),
        css=STUDIO_CSS,
        auth=(ADMIN_USERNAME, ADMIN_PASSWORD),
        auth_message="Painel reservado ao operador da audição.",
        allowed_paths=[
            str(OUTPUT_DIR),
            str(JOB_DATA_DIR / "artifacts"),
            str(AUDITION_DATA_DIR),
        ],
    )


if __name__ == "__main__":
    _log("Painel técnico: http://127.0.0.1:7860/admin/")
    if _CONFIGURED_ADMIN_PASSWORD:
        _log("Credencial administrativa configurada pelo ambiente.")
    else:
        _log(f"Credencial temporária local: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    _log("Audições cegas: links gerados no painel técnico")
    uvicorn.run(create_web_app(), host="127.0.0.1", port=7860)
