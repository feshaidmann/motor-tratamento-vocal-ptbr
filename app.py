"""PoC do Motor de Tratamento Vocal PT-BR com IA.

Primeira iteração: a interface e o pipeline estão completos, mas a separação de
fontes e a correção DSP são mocks intencionais para permitir a validação rápida
do fluxo sem carregar o Demucs/PyTorch nem alterar o áudio enviado.
"""

from __future__ import annotations

import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
from pedalboard import Pedalboard


APP_TITLE = "Motor de Tratamento Vocal PT-BR com IA"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "motor_vocal_ptbr"
MOCK_DELAY_SECONDS = 1.0

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


def _as_samples_channels(audio: np.ndarray) -> np.ndarray:
    """Normaliza áudio do Librosa para o formato (amostras, canais)."""
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 1:
        return audio[:, np.newaxis]
    if audio.ndim == 2:
        return audio.T
    raise ValueError(f"Formato de áudio não suportado: shape={audio.shape}")


def separate_stems(audio_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Simula a separação htdemucs e retorna vocals, no_vocals e sample rate.

    Nesta iteração, o stem ``vocals`` recebe uma cópia do áudio original e o
    stem ``no_vocals`` recebe silêncio. Portanto, a mixagem final reconstrói o
    sinal de entrada sem duplicar o ganho.
    """
    _log("Etapa 1/4 — Simulando separação com Demucs/htdemucs...")
    time.sleep(MOCK_DELAY_SECONDS)

    # mono=False preserva os canais; sr=None preserva a taxa de amostragem.
    audio, sample_rate = librosa.load(audio_path, sr=None, mono=False)
    audio = _as_samples_channels(audio)

    vocals = audio.copy()
    no_vocals = np.zeros_like(audio)

    _log(
        "Mock Demucs concluído "
        f"({audio.shape[1]} canal(is), {sample_rate} Hz, "
        f"{audio.shape[0] / sample_rate:.2f} s)."
    )
    return vocals, no_vocals, int(sample_rate)


def _find_sustained_regions(
    band_ratio: np.ndarray,
    frame_times: np.ndarray,
    minimum_frames: int,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Localiza sequências sustentadas acima de um limiar adaptativo."""
    if band_ratio.size == 0 or not np.any(band_ratio > 0):
        return np.zeros_like(band_ratio, dtype=bool), []

    threshold = max(
        float(np.percentile(band_ratio, 75)),
        float(np.median(band_ratio) + 0.5 * np.std(band_ratio)),
    )
    active = band_ratio >= threshold
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

    # Aproximadamente 150 ms: reduz falsos positivos causados por transientes.
    minimum_frames = max(2, int(np.ceil(0.15 * sr / hop_length)))

    nasality = _analyze_band(
        power,
        frequencies,
        total_power,
        frame_times,
        low_hz=600.0,
        high_hz=1_200.0,
        minimum_frames=minimum_frames,
    )
    sibilance = _analyze_band(
        power,
        frequencies,
        total_power,
        frame_times,
        low_hz=4_000.0,
        high_hz=9_000.0,
        minimum_frames=minimum_frames,
    )

    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)
    mean_centroid = float(np.mean(centroid)) if centroid.size else 0.0

    analysis_data: dict[str, Any] = {
        "nasalidade_ao_o": nasality,
        "sibilancia_s_x": sibilance,
        "centroide_espectral_medio_hz": round(mean_centroid, 1),
        "observacao": "Diagnóstico heurístico da PoC; DSP ainda em modo mock.",
    }

    _log(
        "Análise concluída — alvos estimados: "
        f"nasalidade={nasality['frequencia_alvo_hz']} Hz, "
        f"sibilância={sibilance['frequencia_alvo_hz']} Hz."
    )
    return analysis_data


def apply_dsp_correction(
    vocal_array: np.ndarray,
    sr: int,
    analysis_data: dict[str, Any],
) -> np.ndarray:
    """Simula a correção DSP com uma cadeia Pedalboard transparente.

    Na próxima iteração, a cadeia vazia será substituída por EQ dinâmico e
    de-esser orientados pelas frequências e regiões de ``analysis_data``.
    """
    _log("Etapa 3/4 — Simulando EQ dinâmica e De-Esser com Pedalboard...")
    time.sleep(MOCK_DELAY_SECONDS)

    # O Pedalboard trabalha no formato (canais, amostras). Uma cadeia vazia é
    # intencionalmente transparente e mantém o mock integrado à API real.
    transparent_board = Pedalboard([])
    channels_samples = np.ascontiguousarray(vocal_array.T, dtype=np.float32)
    processed = transparent_board(channels_samples, sr)

    _log(
        "Mock DSP concluído sem alterações no sinal "
        f"({len(analysis_data['nasalidade_ao_o']['regioes_sustentadas'])} "
        "região(ões) de nasalidade sinalizada(s))."
    )
    return np.asarray(processed, dtype=np.float32).T


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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"vocal_ptbr_processado_{uuid.uuid4().hex[:10]}.wav"
    sf.write(output_path, mixed, sr, format="WAV", subtype="FLOAT")

    _log(f"Processamento concluído: {output_path}")
    return str(output_path)


def process_audio(
    audio_path: str | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, dict[str, Any]]:
    """Orquestra o pipeline e atualiza o progresso visual do Gradio."""
    if not audio_path:
        raise gr.Error("Envie um arquivo WAV ou MP3 antes de iniciar.")

    try:
        progress(0.05, desc="Preparando o arquivo de áudio...")

        progress(0.15, desc="Separando voz e instrumental (mock)...")
        vocals, instrumental, sample_rate = separate_stems(audio_path)

        progress(0.45, desc="Analisando nasalidade e sibilância...")
        analysis_data = analyze_ptbr_artifacts(vocals, sample_rate)

        progress(0.70, desc="Aplicando EQ dinâmica e De-Esser (mock)...")
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

        progress(1.0, desc="Processamento concluído.")
        return output_path, analysis_data
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
                <span class="mock-badge">PoC · Pipeline Mock</span>
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
                process_button = gr.Button(
                    "Processar áudio",
                    variant="primary",
                    size="lg",
                )
                gr.Markdown(
                    "A primeira execução pode levar alguns segundos. Nesta "
                    "iteração, Demucs e DSP estão em modo de simulação."
                )

            with gr.Column(scale=1, elem_classes="studio-panel"):
                gr.Markdown("## B · Áudio processado")
                output_audio = gr.Audio(
                    label="Resultado para comparação A/B",
                    type="filepath",
                    interactive=False,
                )
                analysis_output = gr.JSON(
                    label="Diagnóstico espectral preliminar",
                    open=False,
                )

        process_button.click(
            fn=process_audio,
            inputs=input_audio,
            outputs=[output_audio, analysis_output],
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
