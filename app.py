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
import sys
import tempfile
import traceback
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
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
from pedalboard import Compressor, PeakFilter, Pedalboard
from pydantic import BaseModel, Field
from scipy.signal import resample_poly


APP_TITLE = "Motor de Clareza Vocal PT-BR"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "motor_vocal_ptbr"
AUDITION_DATA_DIR = Path(__file__).resolve().parent / "outputs" / "auditions"
AUDITION_DB_PATH = AUDITION_DATA_DIR / "auditions.sqlite3"
AUDITION_LOG_PATH = AUDITION_DATA_DIR / "abx_votes.jsonl"
AUDITION_STIMULUS_DIR = AUDITION_DATA_DIR / "stimuli"
ADMIN_USERNAME = os.environ.get("MOTOR_VOCAL_ADMIN_USER", "admin")
ADMIN_PASSWORD = os.environ.get("MOTOR_VOCAL_ADMIN_PASSWORD") or secrets.token_urlsafe(12)
MINIMUM_VOCAL_RMS_DBFS = -55.0
TRUE_PEAK_TARGET_DBTP = -1.0
DSP_PRESETS: dict[str, dict[str, float]] = {
    "Suave": {
        "nasality_gain_db": -1.5,
        "nasality_q": 1.3,
        "stridency_gain_db": -1.0,
        "stridency_q": 1.2,
        "deesser_gain_db": -2.5,
        "deesser_q": 1.7,
        "compressor_threshold_db": -14.0,
        "compressor_ratio": 1.5,
        "wet_mix": 0.70,
    },
    "Balanceado": {
        "nasality_gain_db": -3.0,
        "nasality_q": 1.6,
        "stridency_gain_db": -2.0,
        "stridency_q": 1.5,
        "deesser_gain_db": -4.5,
        "deesser_q": 2.0,
        "compressor_threshold_db": -18.0,
        "compressor_ratio": 2.5,
        "wet_mix": 0.90,
    },
    "Intenso": {
        "nasality_gain_db": -5.0,
        "nasality_q": 2.0,
        "stridency_gain_db": -3.5,
        "stridency_q": 1.8,
        "deesser_gain_db": -7.0,
        "deesser_q": 2.4,
        "compressor_threshold_db": -22.0,
        "compressor_ratio": 4.0,
        "wet_mix": 1.0,
    },
}


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


class DemucsExecutionError(RuntimeError):
    """Representa uma falha retornada pelo processo de separação do Demucs."""


def _preferred_demucs_device() -> str:
    """Escolhe MPS quando disponível e usa CPU como opção segura."""
    try:
        import torch

        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass
    return "cpu"


def _run_demucs(audio_path: Path, output_dir: Path, device: str) -> None:
    """Executa a CLI do Demucs e retransmite sua saída para o terminal."""
    command = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "--name",
        "htdemucs",
        "--device",
        device,
        "--float32",
        "--clip-mode",
        "none",
        "--out",
        str(output_dir),
        str(audio_path),
    ]

    _log(f"Executando Demucs no dispositivo '{device}'...")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    output_lines: list[str] = []
    if process.stdout is not None:
        for line in process.stdout:
            clean_line = line.rstrip()
            if clean_line:
                output_lines.append(clean_line)
                _log(f"Demucs · {clean_line}")

    return_code = process.wait()
    if return_code != 0:
        diagnostic = "\n".join(output_lines[-12:])
        raise DemucsExecutionError(
            f"Demucs encerrou com código {return_code} usando {device}."
            + (f"\n{diagnostic}" if diagnostic else "")
        )


def _load_demucs_stem(stem_path: Path) -> tuple[np.ndarray, int]:
    """Carrega um stem WAV como float32 no formato (amostras, canais)."""
    audio, sample_rate = sf.read(stem_path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def _source_audio_metadata(source_path: Path) -> tuple[int, int, int]:
    """Retorna sample rate, número de amostras e canais do arquivo original."""
    try:
        info = sf.info(source_path)
    except RuntimeError as exc:
        raise ValueError(f"Não foi possível ler os metadados de {source_path.name}.") from exc

    if info.samplerate <= 0 or info.frames <= 0 or info.channels <= 0:
        raise ValueError("O arquivo de entrada não contém áudio válido.")
    return int(info.samplerate), int(info.frames), int(info.channels)


def _adapt_channel_count(audio: np.ndarray, target_channels: int) -> np.ndarray:
    """Converte stems mono/estéreo para a quantidade de canais da fonte."""
    current_channels = audio.shape[1]
    if current_channels == target_channels:
        return audio
    if target_channels == 1:
        return np.mean(audio, axis=1, keepdims=True, dtype=np.float32)
    if current_channels == 1:
        return np.repeat(audio, target_channels, axis=1)
    if current_channels > target_channels:
        return audio[:, :target_channels]

    repetitions = int(np.ceil(target_channels / current_channels))
    return np.tile(audio, (1, repetitions))[:, :target_channels]


def _align_stem_to_source(
    audio: np.ndarray,
    stem_sr: int,
    source_sr: int,
    source_samples: int,
    source_channels: int,
) -> np.ndarray:
    """Restaura sample rate, duração e canais do arquivo enviado."""
    if stem_sr != source_sr:
        _log(f"Reamostrando stem de {stem_sr} Hz para {source_sr} Hz...")
        audio = librosa.resample(
            audio.T,
            orig_sr=stem_sr,
            target_sr=source_sr,
            axis=-1,
            res_type="soxr_hq",
        ).T

    audio = _adapt_channel_count(np.asarray(audio, dtype=np.float32), source_channels)
    if audio.shape[0] > source_samples:
        audio = audio[:source_samples]
    elif audio.shape[0] < source_samples:
        audio = np.pad(audio, ((0, source_samples - audio.shape[0]), (0, 0)))

    return np.ascontiguousarray(audio, dtype=np.float32)


def _load_source_audio(source_path: str, target_sr: int) -> np.ndarray:
    """Carrega o áudio original para métricas de reconstrução e saída."""
    audio, source_sr = sf.read(source_path, dtype="float32", always_2d=True)
    if int(source_sr) != target_sr:
        audio = librosa.resample(
            audio.T,
            orig_sr=int(source_sr),
            target_sr=target_sr,
            axis=-1,
            res_type="soxr_hq",
        ).T
    return np.ascontiguousarray(audio, dtype=np.float32)


def _normalized_similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    """Calcula similaridade normalizada entre sinais com proteção para silêncio."""
    dot_product = 0.0
    reference_energy = 0.0
    candidate_energy = 0.0
    chunk_size = 262_144
    for start in range(0, reference.shape[0], chunk_size):
        reference_chunk = reference[start : start + chunk_size]
        candidate_chunk = candidate[start : start + chunk_size]
        dot_product += float(np.sum(reference_chunk * candidate_chunk, dtype=np.float64))
        reference_energy += float(
            np.sum(np.square(reference_chunk), dtype=np.float64)
        )
        candidate_energy += float(
            np.sum(np.square(candidate_chunk), dtype=np.float64)
        )
    denominator = float(np.sqrt(reference_energy * candidate_energy))
    if denominator <= np.finfo(np.float64).eps:
        return 1.0 if np.allclose(reference, candidate) else 0.0
    return dot_product / denominator


def evaluate_stem_reconstruction(
    original: np.ndarray,
    vocals: np.ndarray,
    instrumental: np.ndarray,
) -> dict[str, Any]:
    """Mede quanto a soma dos stems preserva a entrada antes do DSP."""
    sample_count = min(original.shape[0], vocals.shape[0], instrumental.shape[0])
    channel_count = min(original.shape[1], vocals.shape[1], instrumental.shape[1])
    reference = original[:sample_count, :channel_count]
    epsilon = np.finfo(np.float32).eps
    reference_energy = 0.0
    error_energy = 0.0
    dot_product = 0.0
    reconstruction_energy = 0.0
    value_count = sample_count * channel_count
    chunk_size = 262_144
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        reference_chunk = reference[start:end]
        reconstruction_chunk = vocals[start:end, :channel_count] + instrumental[
            start:end, :channel_count
        ]
        error_chunk = reference_chunk - reconstruction_chunk
        reference_energy += float(
            np.sum(np.square(reference_chunk), dtype=np.float64)
        )
        error_energy += float(np.sum(np.square(error_chunk), dtype=np.float64))
        reconstruction_energy += float(
            np.sum(np.square(reconstruction_chunk), dtype=np.float64)
        )
        dot_product += float(
            np.sum(reference_chunk * reconstruction_chunk, dtype=np.float64)
        )

    reference_rms = float(np.sqrt(reference_energy / max(value_count, 1)))
    error_rms = float(np.sqrt(error_energy / max(value_count, 1)))
    relative_error_db = float(
        20.0 * np.log10((error_rms + epsilon) / (reference_rms + epsilon))
    )
    similarity_denominator = float(
        np.sqrt(reference_energy * reconstruction_energy)
    )
    similarity = (
        dot_product / similarity_denominator
        if similarity_denominator > np.finfo(np.float64).eps
        else 0.0
    )
    reliable = relative_error_db <= -15.0 and similarity >= 0.95
    return {
        "confiavel": reliable,
        "erro_rms_relativo_db": round(relative_error_db, 2),
        "similaridade_normalizada": round(similarity, 5),
        "criterio": "erro <= -15 dB e similaridade >= 0,95",
    }


def separate_stems(audio_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Separa ``vocals`` e ``no_vocals`` com o modelo htdemucs.

    A função tenta usar o backend MPS/Metal. Se essa execução falhar, refaz a
    separação em CPU. Os WAVs intermediários vivem somente durante esta chamada
    e são removidos automaticamente depois de carregados em memória.
    """
    _log("Etapa 1/4 — Separando voz e instrumental com Demucs/htdemucs...")
    source_path = Path(audio_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {source_path}")
    if source_path.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError("Formato não suportado. Envie um arquivo WAV ou MP3.")

    source_sr, source_samples, source_channels = _source_audio_metadata(source_path)

    preferred_device = _preferred_demucs_device()

    with tempfile.TemporaryDirectory(prefix="motor_vocal_demucs_") as temp_dir:
        demucs_output = Path(temp_dir)
        try:
            _run_demucs(source_path, demucs_output, preferred_device)
        except DemucsExecutionError:
            if preferred_device == "cpu":
                raise
            _log("A execução MPS falhou; repetindo a separação em CPU...")
            _run_demucs(source_path, demucs_output, "cpu")

        vocal_candidates = list(demucs_output.rglob("vocals.wav"))
        instrumental_candidates = list(demucs_output.rglob("no_vocals.wav"))
        if len(vocal_candidates) != 1 or len(instrumental_candidates) != 1:
            raise FileNotFoundError(
                "O Demucs não produziu exatamente um par vocals/no_vocals."
            )

        vocals, vocal_sr = _load_demucs_stem(vocal_candidates[0])
        no_vocals, instrumental_sr = _load_demucs_stem(
            instrumental_candidates[0]
        )

    if vocal_sr != instrumental_sr:
        raise ValueError(
            "Os stems do Demucs possuem taxas de amostragem incompatíveis: "
            f"{vocal_sr} Hz e {instrumental_sr} Hz."
        )
    vocals = _align_stem_to_source(
        vocals,
        vocal_sr,
        source_sr,
        source_samples,
        source_channels,
    )
    no_vocals = _align_stem_to_source(
        no_vocals,
        instrumental_sr,
        source_sr,
        source_samples,
        source_channels,
    )

    _log(
        "Separação Demucs concluída "
        f"({source_channels} canal(is), {source_sr} Hz, "
        f"{source_samples / source_sr:.2f} s)."
    )
    return vocals, no_vocals, source_sr


def _find_sustained_regions(
    band_ratio: np.ndarray,
    frame_times: np.ndarray,
    minimum_frames: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Localiza sequências sustentadas acima de um limiar adaptativo."""
    if band_ratio.size == 0 or not np.any(band_ratio > 0):
        return np.zeros_like(band_ratio, dtype=bool), []

    # A suavização reduz a influência de transientes isolados. Em sinais muito
    # estáveis (por exemplo, uma vogal longa), a banda inteira é considerada.
    smoothing_frames = min(5, band_ratio.size)
    smoothing_kernel = np.ones(smoothing_frames, dtype=np.float32)
    smoothing_kernel /= smoothing_frames
    smoothed = np.convolve(band_ratio, smoothing_kernel, mode="same")
    median = float(np.median(smoothed))
    deviation = float(np.std(smoothed))
    if deviation <= max(median * 0.03, 1e-8):
        active = smoothed >= median * 0.95
    else:
        threshold = max(
            float(np.percentile(smoothed, 68)),
            median + 0.30 * deviation,
        )
        threshold = min(threshold, float(np.percentile(smoothed, 90)))
        active = smoothed >= threshold

    # Une eventos separados por até dois frames, evitando que pequenas quedas
    # internas quebrem uma mesma vogal ou consoante fricativa em vários trechos.
    inactive_padded = np.pad((~active).astype(np.int8), (1, 1))
    inactive_transitions = np.diff(inactive_padded)
    gap_starts = np.flatnonzero(inactive_transitions == 1)
    gap_ends = np.flatnonzero(inactive_transitions == -1)
    for gap_start, gap_end in zip(gap_starts, gap_ends):
        if gap_start > 0 and gap_end < active.size and gap_end - gap_start <= 2:
            active[gap_start:gap_end] = True

    sustained = np.zeros_like(active, dtype=bool)
    regions: list[dict[str, float]] = []

    padded = np.pad(active.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)

    for start, end in zip(starts, ends):
        if end - start < minimum_frames:
            continue
        sustained[start:end] = True
        end_index = min(end - 1, frame_times.size - 1)
        regions.append(
            {
                "inicio_s": round(float(frame_times[start]), 3),
                "fim_s": round(float(frame_times[end_index]), 3),
                "intensidade_relativa": round(float(np.mean(band_ratio[start:end])), 4),
            }
        )

    return sustained, regions


def _analyze_band(
    power_spectrogram: np.ndarray,
    frequencies: np.ndarray,
    total_power: np.ndarray,
    frame_times: np.ndarray,
    low_hz: float,
    high_hz: float,
    minimum_frames: int,
    minimum_relative_energy: float,
    minimum_confidence: float = 0.45,
) -> dict[str, Any]:
    """Resume energia, frequência dominante e regiões sustentadas de uma banda."""
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band_mask):
        return {
            "faixa_hz": [low_hz, high_hz],
            "frequencia_alvo_hz": None,
            "confianca": 0.0,
            "intervencao_autorizada": False,
            "regioes_candidatas": [],
            "regioes_sustentadas": [],
        }

    band_power = power_spectrogram[band_mask]
    energy_per_frame = np.sum(band_power, axis=0)
    band_ratio = energy_per_frame / np.maximum(total_power, np.finfo(np.float32).eps)
    sustained, regions = _find_sustained_regions(
        band_ratio,
        frame_times,
        minimum_frames,
    )

    # Calcula a frequência mais energética nos frames detectados. Se nenhum
    # trecho for sustentado, usa todos os frames como estimativa preliminar.
    selected = band_power[:, sustained] if np.any(sustained) else band_power
    aggregate_energy = np.sum(selected, axis=1)
    band_frequencies = frequencies[band_mask]
    target_hz = float(band_frequencies[int(np.argmax(aggregate_energy))])

    epsilon = np.finfo(np.float32).eps
    mean_ratio = float(np.mean(band_ratio))
    peak_ratio = float(np.percentile(band_ratio, 95))
    median_ratio = float(np.median(band_ratio))
    contrast_db = float(
        10.0 * np.log10((peak_ratio + epsilon) / (median_ratio + epsilon))
    )
    coverage = float(np.mean(sustained))
    energy_score = float(
        np.clip(peak_ratio / max(minimum_relative_energy * 2.0, epsilon), 0.0, 1.0)
    )
    contrast_score = float(np.clip(contrast_db / 9.0, 0.0, 1.0))
    duration_score = float(np.clip(coverage / 0.03, 0.0, 1.0))
    confidence = 0.50 * energy_score + 0.30 * contrast_score + 0.20 * duration_score
    intervention_authorized = bool(regions) and confidence >= minimum_confidence
    if not regions:
        abstention_reason: str | None = "nenhuma região sustentada"
    elif confidence < minimum_confidence:
        abstention_reason = "confiança abaixo do limiar"
    else:
        abstention_reason = None

    return {
        "faixa_hz": [low_hz, high_hz],
        "frequencia_alvo_hz": round(target_hz, 1),
        "energia_relativa_media": round(mean_ratio, 5),
        "energia_relativa_pico_p95": round(peak_ratio, 5),
        "contraste_db": round(contrast_db, 2),
        "cobertura_temporal": round(coverage, 4),
        "confianca": round(confidence, 3),
        "limiar_confianca": minimum_confidence,
        "intervencao_autorizada": intervention_authorized,
        "motivo_abstencao": abstention_reason,
        "regioes_candidatas": regions,
        "regioes_sustentadas": regions,
    }


def analyze_ptbr_artifacts(vocal_array: np.ndarray, sr: int) -> dict[str, Any]:
    """Analisa regiões associadas a nasalidade, estridência e sibilância.

    A detecção é uma heurística MIR para a PoC, não um classificador fonético.
    Ela usa STFT, proporção de energia por banda e centroide espectral.
    """
    _log("Etapa 2/4 — Analisando artefatos vocais com Librosa...")

    mono_vocal = np.mean(vocal_array, axis=1, dtype=np.float32)
    if mono_vocal.size == 0:
        raise ValueError("O arquivo de áudio está vazio.")
    vocal_rms = float(np.sqrt(np.mean(np.square(mono_vocal), dtype=np.float64)))
    vocal_rms_dbfs = float(20.0 * np.log10(max(vocal_rms, np.finfo(np.float32).eps)))

    n_fft = 2048
    hop_length = 512
    stft = librosa.stft(mono_vocal, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft).astype(np.float32)
    power = np.square(magnitude)
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_times = librosa.frames_to_time(
        np.arange(power.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )
    total_power = np.sum(power, axis=0)

    nasality = _analyze_band(
        power,
        frequencies,
        total_power,
        frame_times,
        low_hz=600.0,
        high_hz=1_200.0,
        minimum_frames=max(3, int(np.ceil(0.10 * sr / hop_length))),
        minimum_relative_energy=0.06,
    )
    stridency = _analyze_band(
        power,
        frequencies,
        total_power,
        frame_times,
        low_hz=2_000.0,
        high_hz=4_000.0,
        minimum_frames=max(2, int(np.ceil(0.060 * sr / hop_length))),
        minimum_relative_energy=0.035,
    )
    sibilance = _analyze_band(
        power,
        frequencies,
        total_power,
        frame_times,
        low_hz=4_000.0,
        high_hz=9_000.0,
        minimum_frames=max(2, int(np.ceil(0.035 * sr / hop_length))),
        minimum_relative_energy=0.012,
    )

    if vocal_rms_dbfs < MINIMUM_VOCAL_RMS_DBFS:
        for band_analysis in (nasality, stridency, sibilance):
            band_analysis["intervencao_autorizada"] = False
            band_analysis["motivo_abstencao"] = "stem vocal abaixo do nível mínimo"

    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    mean_centroid = float(np.mean(centroid)) if centroid.size else 0.0

    analysis_data: dict[str, Any] = {
        "nasalidade_ao_o": nasality,
        "estridencia_2_4khz": stridency,
        "sibilancia_s_x": sibilance,
        "nivel_vocal_rms_dbfs": round(vocal_rms_dbfs, 2),
        "centroide_espectral_medio_hz": round(mean_centroid, 1),
        "observacao": (
            "Diagnóstico heurístico da PoC; valide o resultado por audição A/B."
        ),
    }

    _log(
        "Análise concluída — alvos estimados: "
        f"nasalidade={nasality['frequencia_alvo_hz']} Hz, "
        f"estridência={stridency['frequencia_alvo_hz']} Hz, "
        f"sibilância={sibilance['frequencia_alvo_hz']} Hz."
    )
    return analysis_data


def _temporal_envelope(
    sample_count: int,
    sr: int,
    regions: list[dict[str, float]],
    attack_ms: float,
    release_ms: float,
    maximum_level: float,
) -> np.ndarray:
    """Cria um envelope suave para aplicar DSP somente nas regiões detectadas."""
    envelope = np.zeros(sample_count, dtype=np.float32)
    attack_samples = max(1, int(sr * attack_ms / 1_000))
    release_samples = max(1, int(sr * release_ms / 1_000))

    for region in regions:
        start = int(float(region["inicio_s"]) * sr)
        end = int(float(region["fim_s"]) * sr) + 1
        start = min(max(start, 0), sample_count)
        end = min(max(end, start + 1), sample_count)
        if start >= sample_count:
            continue

        attack_start = max(0, start - attack_samples)
        release_end = min(sample_count, end + release_samples)

        if start > attack_start:
            attack = np.linspace(
                0.0,
                maximum_level,
                start - attack_start,
                endpoint=False,
                dtype=np.float32,
            )
            envelope[attack_start:start] = np.maximum(
                envelope[attack_start:start],
                attack,
            )

        envelope[start:end] = np.maximum(envelope[start:end], maximum_level)

        if release_end > end:
            release = np.linspace(
                maximum_level,
                0.0,
                release_end - end,
                endpoint=True,
                dtype=np.float32,
            )
            envelope[end:release_end] = np.maximum(
                envelope[end:release_end],
                release,
            )

    return envelope


def _run_pedalboard(
    board: Pedalboard,
    audio: np.ndarray,
    sr: int,
) -> np.ndarray:
    """Executa plugins no layout esperado e restaura (amostras, canais)."""
    channels_samples = np.ascontiguousarray(audio.T, dtype=np.float32)
    processed = board(channels_samples, sr)
    return np.asarray(processed, dtype=np.float32).T


def _blend_regionally(
    dry_audio: np.ndarray,
    wet_audio: np.ndarray,
    envelope: np.ndarray,
) -> np.ndarray:
    """Faz crossfade sample a sample entre o sinal seco e o processado."""
    wet_amount = envelope[:, np.newaxis]
    return np.asarray(
        dry_audio + (wet_audio - dry_audio) * wet_amount,
        dtype=np.float32,
    )


def _safe_target_frequency(value: Any, fallback: float, sr: int) -> float:
    """Mantém a frequência-alvo dentro da faixa processável do áudio."""
    target = fallback if value is None else float(value)
    return float(np.clip(target, 40.0, sr * 0.48))


def _match_rms_limited(
    reference: np.ndarray,
    processed: np.ndarray,
    maximum_adjustment_db: float = 1.5,
    envelope: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Aproxima o volume A/B sem compensar integralmente a correção espectral."""
    epsilon = np.finfo(np.float32).eps
    reference_rms = float(np.sqrt(np.mean(np.square(reference), dtype=np.float64)))
    processed_rms = float(np.sqrt(np.mean(np.square(processed), dtype=np.float64)))
    if reference_rms <= epsilon or processed_rms <= epsilon:
        return processed, 0.0

    adjustment_db = 20.0 * np.log10(reference_rms / processed_rms)
    adjustment_db = float(
        np.clip(adjustment_db, -maximum_adjustment_db, maximum_adjustment_db)
    )
    gain = float(10.0 ** (adjustment_db / 20.0))
    if envelope is None:
        gain_curve: float | np.ndarray = gain
    else:
        gain_curve = 1.0 + (gain - 1.0) * envelope[:, np.newaxis]
    return np.asarray(processed * gain_curve, dtype=np.float32), adjustment_db


def _authorized_regions(
    band_analysis: dict[str, Any],
    module_enabled: bool,
    separation_reliable: bool,
) -> list[dict[str, float]]:
    """Libera eventos somente quando módulo, banda e separação são confiáveis."""
    if not module_enabled or not separation_reliable:
        return []
    if not bool(band_analysis.get("intervencao_autorizada", True)):
        return []
    return list(band_analysis.get("regioes_sustentadas", []))


def _module_decision(
    band_analysis: dict[str, Any],
    module_enabled: bool,
    separation_reliable: bool,
    applied_regions: list[dict[str, float]],
) -> dict[str, Any]:
    """Explica de forma auditável por que cada módulo agiu ou se absteve."""
    if not module_enabled:
        status = "desativado_pelo_usuario"
        reason = "módulo desativado na interface"
    elif not separation_reliable:
        status = "abstencao_separacao"
        reason = "reconstrução dos stems abaixo do critério mínimo"
    elif applied_regions:
        status = "aplicado"
        reason = None
    else:
        status = "abstencao_local"
        reason = band_analysis.get("motivo_abstencao") or "nenhum evento autorizado"

    return {
        "status": status,
        "motivo": reason,
        "confianca": band_analysis.get("confianca"),
        "limiar_confianca": band_analysis.get("limiar_confianca"),
        "eventos_aplicados": len(applied_regions),
    }


def apply_dsp_correction(
    vocal_array: np.ndarray,
    sr: int,
    analysis_data: dict[str, Any],
) -> np.ndarray:
    """Aplica EQ dinâmica e De-Esser nas regiões detectadas pelo Librosa."""
    _log("Etapa 3/4 — Aplicando EQ dinâmica e De-Esser com Pedalboard...")

    preset_name = str(analysis_data.get("preset_dsp", "Balanceado"))
    preset = DSP_PRESETS.get(preset_name, DSP_PRESETS["Balanceado"])
    if preset_name not in DSP_PRESETS:
        preset_name = "Balanceado"

    nasality = analysis_data["nasalidade_ao_o"]
    stridency = analysis_data.get(
        "estridencia_2_4khz",
        {"frequencia_alvo_hz": 3_000.0, "regioes_sustentadas": []},
    )
    sibilance = analysis_data["sibilancia_s_x"]
    modules = analysis_data.get(
        "modulos_ativos",
        {"nasalidade": True, "estridencia": True, "sibilancia": True},
    )
    separation_reliable = bool(
        analysis_data.get("qualidade_separacao", {}).get("confiavel", True)
    )
    nasality_regions = _authorized_regions(
        nasality,
        bool(modules.get("nasalidade", True)),
        separation_reliable,
    )
    stridency_regions = _authorized_regions(
        stridency,
        bool(modules.get("estridencia", True)),
        separation_reliable,
    )
    sibilance_regions = _authorized_regions(
        sibilance,
        bool(modules.get("sibilancia", True)),
        separation_reliable,
    )
    nasality_target = _safe_target_frequency(
        nasality.get("frequencia_alvo_hz"),
        fallback=900.0,
        sr=sr,
    )
    sibilance_target = _safe_target_frequency(
        sibilance.get("frequencia_alvo_hz"),
        fallback=6_500.0,
        sr=sr,
    )
    stridency_target = _safe_target_frequency(
        stridency.get("frequencia_alvo_hz"),
        fallback=3_000.0,
        sr=sr,
    )

    # A EQ de nasalidade cria uma variante atenuada do vocal. O envelope MIR
    # faz o crossfade apenas sobre vogais sustentadas detectadas na banda.
    nasality_board = Pedalboard(
        [
            PeakFilter(
                cutoff_frequency_hz=nasality_target,
                gain_db=preset["nasality_gain_db"],
                q=preset["nasality_q"],
            )
        ]
    )
    nasal_wet = _run_pedalboard(nasality_board, vocal_array, sr)
    nasal_envelope = _temporal_envelope(
        vocal_array.shape[0],
        sr,
        nasality_regions,
        attack_ms=25.0,
        release_ms=110.0,
        maximum_level=preset["wet_mix"],
    )
    processed = _blend_regionally(vocal_array, nasal_wet, nasal_envelope)

    # A estridência é tratada em uma etapa independente para que o usuário
    # possa desligá-la sem alterar a decisão de nasalidade ou sibilância.
    stridency_board = Pedalboard(
        [
            PeakFilter(
                cutoff_frequency_hz=stridency_target,
                gain_db=preset["stridency_gain_db"],
                q=preset["stridency_q"],
            )
        ]
    )
    stridency_wet = _run_pedalboard(stridency_board, processed, sr)
    stridency_envelope = _temporal_envelope(
        processed.shape[0],
        sr,
        stridency_regions,
        attack_ms=12.0,
        release_ms=90.0,
        maximum_level=preset["wet_mix"],
    )
    processed = _blend_regionally(processed, stridency_wet, stridency_envelope)

    # O De-Esser combina um notch na frequência dominante com compressão. Como
    # o resultado é misturado só durante fricativas, o restante do vocal mantém
    # brilho e dinâmica originais.
    deesser_board = Pedalboard(
        [
            PeakFilter(
                cutoff_frequency_hz=sibilance_target,
                gain_db=preset["deesser_gain_db"],
                q=preset["deesser_q"],
            ),
            Compressor(
                threshold_db=preset["compressor_threshold_db"],
                ratio=preset["compressor_ratio"],
                attack_ms=2.0,
                release_ms=70.0,
            ),
        ]
    )
    deesser_wet = _run_pedalboard(deesser_board, processed, sr)
    deesser_envelope = _temporal_envelope(
        processed.shape[0],
        sr,
        sibilance_regions,
        attack_ms=3.0,
        release_ms=75.0,
        maximum_level=preset["wet_mix"],
    )
    processed = _blend_regionally(processed, deesser_wet, deesser_envelope)
    processed = np.nan_to_num(
        processed,
        nan=0.0,
        posinf=1.0,
        neginf=-1.0,
    ).astype(np.float32, copy=False)

    applied = bool(nasality_regions or stridency_regions or sibilance_regions)
    makeup_gain_db = 0.0
    if applied:
        combined_envelope = np.maximum.reduce(
            (nasal_envelope, stridency_envelope, deesser_envelope)
        )
        processed, makeup_gain_db = _match_rms_limited(
            vocal_array,
            processed,
            envelope=combined_envelope,
        )

    if applied:
        correction_status = "aplicada"
    elif not separation_reliable:
        correction_status = "abstencao_separacao"
    elif not any(bool(modules.get(name, True)) for name in modules):
        correction_status = "desativada_pelo_usuario"
    else:
        correction_status = "abstencao_local"

    analysis_data["correcao_dsp"] = {
        "status": correction_status,
        "preset": preset_name,
        "separacao_confiavel": separation_reliable,
        "ajuste_ab_rms_db": round(makeup_gain_db, 3),
        "nasalidade": {
            **_module_decision(
                nasality,
                bool(modules.get("nasalidade", True)),
                separation_reliable,
                nasality_regions,
            ),
            "eventos": len(nasality_regions),
            "frequencia_hz": round(nasality_target, 1),
            "atenuacao_maxima_db": preset["nasality_gain_db"],
        },
        "estridencia": {
            **_module_decision(
                stridency,
                bool(modules.get("estridencia", True)),
                separation_reliable,
                stridency_regions,
            ),
            "eventos": len(stridency_regions),
            "frequencia_hz": round(stridency_target, 1),
            "atenuacao_maxima_db": preset["stridency_gain_db"],
        },
        "de_esser": {
            **_module_decision(
                sibilance,
                bool(modules.get("sibilancia", True)),
                separation_reliable,
                sibilance_regions,
            ),
            "eventos": len(sibilance_regions),
            "frequencia_hz": round(sibilance_target, 1),
            "atenuacao_eq_maxima_db": preset["deesser_gain_db"],
            "compressor_threshold_db": preset["compressor_threshold_db"],
            "compressor_ratio": preset["compressor_ratio"],
        },
    }

    _log(
        f"DSP concluído com preset {preset_name}: "
        f"{len(nasality_regions)} evento(s) de nasalidade e "
        f"{len(stridency_regions)} de estridência e "
        f"{len(sibilance_regions)} evento(s) de sibilância."
    )
    return processed


def _linear_to_dbfs(value: float) -> float:
    """Converte amplitude linear em dBFS com piso numérico estável."""
    return float(20.0 * np.log10(max(value, np.finfo(np.float32).eps)))


def _approximate_true_peak_linear(audio: np.ndarray, oversampling: int = 4) -> float:
    """Estima true peak por oversampling em blocos para limitar uso de memória."""
    if audio.size == 0:
        return 0.0

    maximum = float(np.max(np.abs(audio)))
    chunk_size = 65_536
    overlap = 128
    for start in range(0, audio.shape[0], chunk_size):
        end = min(start + chunk_size, audio.shape[0])
        extended_start = max(0, start - overlap)
        extended_end = min(audio.shape[0], end + overlap)
        oversampled = resample_poly(
            audio[extended_start:extended_end],
            up=oversampling,
            down=1,
            axis=0,
        )
        crop_start = (start - extended_start) * oversampling
        crop_end = crop_start + (end - start) * oversampling
        central = oversampled[crop_start:crop_end]
        if central.size:
            maximum = max(maximum, float(np.max(np.abs(central))))
    return maximum


def _stereo_correlation(audio: np.ndarray) -> float | None:
    """Calcula correlação L/R; retorna None para arquivos mono ou silenciosos."""
    if audio.shape[1] < 2:
        return None
    left = audio[:, 0]
    right = audio[:, 1]
    left_centered = left - float(np.mean(left))
    right_centered = right - float(np.mean(right))
    denominator = float(
        np.sqrt(
            np.sum(np.square(left_centered), dtype=np.float64)
            * np.sum(np.square(right_centered), dtype=np.float64)
        )
    )
    if denominator <= np.finfo(np.float64).eps:
        return None
    numerator = float(np.sum(left_centered * right_centered, dtype=np.float64))
    return numerator / denominator


def _spectral_centroid_excerpt(audio: np.ndarray, sr: int) -> float:
    """Calcula centroide em até 30 segundos centrais para manter o QC leve."""
    maximum_samples = sr * 30
    if audio.shape[0] > maximum_samples:
        start = (audio.shape[0] - maximum_samples) // 2
        audio = audio[start : start + maximum_samples]
    mono = np.mean(audio, axis=1, dtype=np.float32)
    centroid = librosa.feature.spectral_centroid(y=mono, sr=sr)
    return float(np.mean(centroid)) if centroid.size else 0.0


def evaluate_output_quality(
    original: np.ndarray,
    processed: np.ndarray,
    sr: int,
) -> dict[str, Any]:
    """Produz métricas objetivas básicas para auditoria da mix processada."""
    sample_count = min(original.shape[0], processed.shape[0])
    channel_count = min(original.shape[1], processed.shape[1])
    reference = original[:sample_count, :channel_count]
    output = processed[:sample_count, :channel_count]
    output_peak = float(np.max(np.abs(output))) if output.size else 0.0
    true_peak = _approximate_true_peak_linear(output)
    output_energy = 0.0
    reference_energy = 0.0
    delta_energy = 0.0
    chunk_size = 262_144
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        reference_chunk = reference[start:end]
        output_chunk = output[start:end]
        delta_chunk = output_chunk - reference_chunk
        output_energy += float(np.sum(np.square(output_chunk), dtype=np.float64))
        reference_energy += float(
            np.sum(np.square(reference_chunk), dtype=np.float64)
        )
        delta_energy += float(np.sum(np.square(delta_chunk), dtype=np.float64))
    value_count = max(sample_count * channel_count, 1)
    output_rms = float(np.sqrt(output_energy / value_count))
    reference_rms = float(np.sqrt(reference_energy / value_count))
    delta_rms = float(np.sqrt(delta_energy / value_count))
    relative_delta_db = _linear_to_dbfs(
        delta_rms / max(reference_rms, np.finfo(np.float32).eps)
    )
    original_centroid = _spectral_centroid_excerpt(reference, sr)
    output_centroid = _spectral_centroid_excerpt(output, sr)
    output_correlation = _stereo_correlation(output)
    finite = bool(np.isfinite(output).all())
    passed = finite and _linear_to_dbfs(true_peak) <= TRUE_PEAK_TARGET_DBTP + 0.1

    return {
        "status": "aprovado" if passed else "revisar",
        "amostras_finitas": finite,
        "sample_peak_dbfs": round(_linear_to_dbfs(output_peak), 2),
        "true_peak_estimado_dbtp": round(_linear_to_dbfs(true_peak), 2),
        "alvo_true_peak_dbtp": TRUE_PEAK_TARGET_DBTP,
        "nivel_rms_dbfs": round(_linear_to_dbfs(output_rms), 2),
        "delta_rms_relativo_db": round(relative_delta_db, 2),
        "correlacao_estereo": (
            round(output_correlation, 5) if output_correlation is not None else None
        ),
        "centroide_original_hz": round(original_centroid, 1),
        "centroide_processado_hz": round(output_centroid, 1),
        "delta_centroide_hz": round(output_centroid - original_centroid, 1),
        "nota": "True peak estimado por oversampling 4x; não é medidor certificado.",
    }


def mixdown_and_export(
    processed_vocals: np.ndarray,
    instrumental: np.ndarray,
    sr: int,
) -> str:
    """Soma os stems e exporta um WAV em ponto flutuante de 32 bits."""
    _log("Etapa 4/4 — Remontando stems e exportando WAV 32-bit float...")

    if processed_vocals.ndim != 2 or instrumental.ndim != 2:
        raise ValueError("Os stems devem usar o formato (amostras, canais).")

    sample_count = min(processed_vocals.shape[0], instrumental.shape[0])
    channel_count = min(processed_vocals.shape[1], instrumental.shape[1])
    if sample_count == 0 or channel_count == 0:
        raise ValueError("Não há amostras suficientes para realizar o mixdown.")

    mixed = (
        processed_vocals[:sample_count, :channel_count]
        + instrumental[:sample_count, :channel_count]
    )
    mixed = np.nan_to_num(mixed, nan=0.0, posinf=1.0, neginf=-1.0).astype(
        np.float32,
        copy=False,
    )

    true_peak = _approximate_true_peak_linear(mixed)
    target_linear = float(10.0 ** (TRUE_PEAK_TARGET_DBTP / 20.0))
    if true_peak > target_linear:
        protection_gain = target_linear / true_peak
        mixed *= protection_gain
        gain_db = 20.0 * np.log10(protection_gain)
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
    return {
        "session_id": token,
        "voto_em_utc": submitted_at,
        "mensagem": "Voto registrado. Obrigado por participar.",
    }


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
</main>
<script>
const token = __TOKEN_JSON__;
const labels = ['A', 'B', 'X'];
const buttons = Object.fromEntries(labels.map(label => [label,
  document.querySelector(`[data-listen="${label}"]`)]));
const statusBox = document.querySelector('#status');
const slider = document.querySelector('#position');
const timeBox = document.querySelector('#time');
const playPause = document.querySelector('#playPause');
let context, buffers = {}, sources = {}, gains = {};
let selected = 'A', playing = false, offset = 0, startedAt = 0, duration = 0;

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
});
load().catch(error => { statusBox.textContent = error.message; statusBox.className = 'error'; });
animate();
</script>
</body></html>"""


def _participant_page(token: str) -> str:
    """Renderiza a superfície cega sem incluir mapa ou nomes de condições."""
    safe_token_json = json.dumps(token)
    return PARTICIPANT_PAGE.replace("__TOKEN_JSON__", safe_token_json)


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

        progress(0.15, desc="Separando voz e instrumental com Demucs...")
        vocals, instrumental, sample_rate = separate_stems(audio_path)
        original = _load_source_audio(audio_path, sample_rate)
        original = _align_stem_to_source(
            original,
            sample_rate,
            sample_rate,
            vocals.shape[0],
            vocals.shape[1],
        )
        reconstruction_quality = evaluate_stem_reconstruction(
            original,
            vocals,
            instrumental,
        )

        progress(0.45, desc="Analisando nasalidade, estridência e sibilância...")
        analysis_data = analyze_ptbr_artifacts(vocals, sample_rate)
        analysis_data["preset_dsp"] = preset_dsp
        analysis_data["modulos_ativos"] = {
            "nasalidade": bool(correct_nasality),
            "estridencia": bool(correct_stridency),
            "sibilancia": bool(correct_sibilance),
        }
        analysis_data["qualidade_separacao"] = reconstruction_quality

        progress(0.70, desc="Aplicando EQ dinâmica e De-Esser...")
        processed_vocals = apply_dsp_correction(
            vocals,
            sample_rate,
            analysis_data,
        )

        progress(0.90, desc="Remontando e exportando o áudio...")
        output_path = mixdown_and_export(
            processed_vocals,
            instrumental,
            sample_rate,
        )
        rendered_mix, rendered_sr = sf.read(
            output_path,
            dtype="float32",
            always_2d=True,
        )
        analysis_data["controle_qualidade_saida"] = evaluate_output_quality(
            original,
            np.asarray(rendered_mix, dtype=np.float32),
            int(rendered_sr),
        )
        vocal_path = _export_audio(vocals, sample_rate, "vocal_isolado")
        processed_vocal_path = _export_audio(
            processed_vocals,
            sample_rate,
            "vocal_corrigido",
        )
        instrumental_path = _export_audio(
            instrumental,
            sample_rate,
            "instrumental",
        )

        progress(1.0, desc="Processamento concluído.")
        return (
            output_path,
            vocal_path,
            processed_vocal_path,
            instrumental_path,
            analysis_data,
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

        process_event = process_button.click(
            fn=process_audio,
            inputs=[
                input_audio,
                dsp_preset,
                correct_nasality,
                correct_stridency,
                correct_sibilance,
            ],
            outputs=[
                output_audio,
                vocal_output,
                processed_vocal_output,
                instrumental_output,
                analysis_output,
            ],
            show_progress="full",
        )
        process_event.success(
            fn=prepare_loudness_comparison,
            inputs=[input_audio, output_audio],
            outputs=[comparison_original, comparison_processed, loudness_report],
            show_progress="minimal",
        )
        publish_button.click(
            fn=create_audition_link,
            inputs=[
                input_audio,
                output_audio,
                dsp_preset,
                analysis_output,
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
    web_app = FastAPI(title=APP_TITLE)

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
        allowed_paths=[str(OUTPUT_DIR), str(AUDITION_DATA_DIR)],
    )


if __name__ == "__main__":
    _log("Painel técnico: http://127.0.0.1:7860/admin/")
    _log(f"Credencial do painel: {ADMIN_USERNAME} / {ADMIN_PASSWORD}")
    _log("Audições cegas: links gerados no painel técnico")
    uvicorn.run(create_web_app(), host="127.0.0.1", port=7860)
