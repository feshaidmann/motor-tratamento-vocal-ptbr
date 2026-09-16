"""Análise acústica, DSP e métricas de saída independentes da interface."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from pedalboard import Compressor, PeakFilter, Pedalboard
from scipy.signal import resample_poly


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


def _log(message: str) -> None:
    """Emite logs imediatamente no terminal que executa a aplicação."""
    print(f"[Motor Vocal PT-BR] {message}", flush=True)


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


def mixdown_stems(
    processed_vocals: np.ndarray,
    instrumental: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Remonta os stems e limita o true peak; devolve o ganho aplicado em dB."""
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
        return mixed, float(20.0 * np.log10(protection_gain))
    return mixed, 0.0
