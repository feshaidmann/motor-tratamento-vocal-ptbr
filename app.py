"""PoC do Motor de Tratamento Vocal PT-BR com IA.

Terceira iteração: separação real com Demucs/htdemucs, aceleração MPS com
fallback para CPU e correção DSP regionalizada com Pedalboard.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Compressor, PeakFilter, Pedalboard


APP_TITLE = "Motor de Tratamento Vocal PT-BR com IA"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "motor_vocal_ptbr"
DSP_PRESETS: dict[str, dict[str, float]] = {
    "Suave": {
        "nasality_gain_db": -1.5,
        "nasality_q": 1.3,
        "deesser_gain_db": -2.5,
        "deesser_q": 1.7,
        "compressor_threshold_db": -14.0,
        "compressor_ratio": 1.5,
        "wet_mix": 0.70,
    },
    "Balanceado": {
        "nasality_gain_db": -3.0,
        "nasality_q": 1.6,
        "deesser_gain_db": -4.5,
        "deesser_q": 2.0,
        "compressor_threshold_db": -18.0,
        "compressor_ratio": 2.5,
        "wet_mix": 0.90,
    },
    "Intenso": {
        "nasality_gain_db": -5.0,
        "nasality_q": 2.0,
        "deesser_gain_db": -7.0,
        "deesser_q": 2.4,
        "compressor_threshold_db": -22.0,
        "compressor_ratio": 4.0,
        "wet_mix": 1.0,
    },
}

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
    if vocals.shape[1] != no_vocals.shape[1]:
        raise ValueError(
            "Os stems do Demucs possuem quantidades de canais incompatíveis."
        )

    # Diferenças residuais de comprimento são truncadas para manter sample sync.
    sample_count = min(vocals.shape[0], no_vocals.shape[0])
    vocals = vocals[:sample_count]
    no_vocals = no_vocals[:sample_count]

    _log(
        "Separação Demucs concluída "
        f"({vocals.shape[1]} canal(is), {vocal_sr} Hz, "
        f"{sample_count / vocal_sr:.2f} s)."
    )
    return vocals, no_vocals, vocal_sr


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
) -> dict[str, Any]:
    """Resume energia, frequência dominante e regiões sustentadas de uma banda."""
    band_mask = (frequencies >= low_hz) & (frequencies <= high_hz)
    if not np.any(band_mask):
        return {
            "faixa_hz": [low_hz, high_hz],
            "frequencia_alvo_hz": None,
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

    return {
        "faixa_hz": [low_hz, high_hz],
        "frequencia_alvo_hz": round(target_hz, 1),
        "energia_relativa_media": round(float(np.mean(band_ratio)), 5),
        "energia_relativa_pico": round(float(np.max(band_ratio)), 5),
        "regioes_sustentadas": regions,
    }


def analyze_ptbr_artifacts(vocal_array: np.ndarray, sr: int) -> dict[str, Any]:
    """Analisa regiões associadas a nasalidade e sibilância no vocal.

    A detecção é uma heurística MIR para a PoC, não um classificador fonético.
    Ela usa STFT, proporção de energia por banda e centroide espectral.
    """
    _log("Etapa 2/4 — Analisando artefatos vocais com Librosa...")

    mono_vocal = np.mean(vocal_array, axis=1, dtype=np.float32)
    if mono_vocal.size == 0:
        raise ValueError("O arquivo de áudio está vazio.")

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
    )
    sibilance = _analyze_band(
        power,
        frequencies,
        total_power,
        frame_times,
        low_hz=4_000.0,
        high_hz=9_000.0,
        minimum_frames=max(2, int(np.ceil(0.035 * sr / hop_length))),
    )

    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    mean_centroid = float(np.mean(centroid)) if centroid.size else 0.0

    analysis_data: dict[str, Any] = {
        "nasalidade_ao_o": nasality,
        "sibilancia_s_x": sibilance,
        "centroide_espectral_medio_hz": round(mean_centroid, 1),
        "observacao": (
            "Diagnóstico heurístico da PoC; valide o resultado por audição A/B."
        ),
    }

    _log(
        "Análise concluída — alvos estimados: "
        f"nasalidade={nasality['frequencia_alvo_hz']} Hz, "
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
    sibilance = analysis_data["sibilancia_s_x"]
    nasality_regions = nasality["regioes_sustentadas"]
    sibilance_regions = sibilance["regioes_sustentadas"]
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

    applied = bool(nasality_regions or sibilance_regions)
    analysis_data["correcao_dsp"] = {
        "status": "aplicada" if applied else "nenhum_evento_detectado",
        "preset": preset_name,
        "nasalidade": {
            "eventos": len(nasality_regions),
            "frequencia_hz": round(nasality_target, 1),
            "atenuacao_maxima_db": preset["nasality_gain_db"],
        },
        "de_esser": {
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
        f"{len(sibilance_regions)} evento(s) de sibilância."
    )
    return processed


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

    peak = float(np.max(np.abs(mixed)))
    if peak > 0.99:
        protection_gain = 0.99 / peak
        mixed *= protection_gain
        gain_db = 20.0 * np.log10(protection_gain)
        _log(f"Proteção de pico aplicada ao mixdown: {gain_db:.2f} dB.")

    output_path = _export_audio(mixed, sr, "mix_processado")
    _log(f"Processamento concluído: {output_path}")
    return output_path


def _export_audio(audio: np.ndarray, sr: int, label: str) -> str:
    """Exporta um artefato de áudio temporário como WAV 32-bit float."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{label}_{uuid.uuid4().hex[:10]}.wav"
    sf.write(output_path, audio, sr, format="WAV", subtype="FLOAT")
    return str(output_path)


def process_audio(
    audio_path: str | None,
    preset_dsp: str = "Balanceado",
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, dict[str, Any]]:
    """Orquestra o pipeline e atualiza o progresso visual do Gradio."""
    if not audio_path:
        raise gr.Error("Envie um arquivo WAV ou MP3 antes de iniciar.")

    try:
        progress(0.05, desc="Preparando o arquivo de áudio...")

        progress(0.15, desc="Separando voz e instrumental com Demucs...")
        vocals, instrumental, sample_rate = separate_stems(audio_path)

        progress(0.45, desc="Analisando nasalidade e sibilância...")
        analysis_data = analyze_ptbr_artifacts(vocals, sample_rate)
        analysis_data["preset_dsp"] = preset_dsp

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


def gradio_interface() -> gr.Blocks:
    """Constrói a interface Gradio em duas colunas para comparação A/B."""
    with gr.Blocks(title=APP_TITLE) as demo:
        gr.HTML(
            """
            <section class="studio-header">
                <span class="mock-badge">PoC · Demucs + DSP ativos</span>
                <h1>Motor de Tratamento Vocal PT-BR</h1>
                <p>
                    Separação vocal, análise espectral e correção de nasalidade
                    e sibilância regionalizada para voz cantada brasileira.
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
                gr.Markdown("## B · Áudio processado")
                output_audio = gr.Audio(
                    label="Resultado para comparação A/B",
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

        process_button.click(
            fn=process_audio,
            inputs=[input_audio, dsp_preset],
            outputs=[
                output_audio,
                vocal_output,
                processed_vocal_output,
                instrumental_output,
                analysis_output,
            ],
            show_progress="full",
        )

    return demo


if __name__ == "__main__":
    _log("Iniciando interface local em http://127.0.0.1:7860")
    app = gradio_interface()
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        theme=gr.themes.Monochrome(),
        css=STUDIO_CSS,
    )
