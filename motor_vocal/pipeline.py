"""Executor local do processamento, independente da interface web."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from motor_vocal.processing import (
    DSP_PRESETS,
    analyze_ptbr_artifacts,
    apply_dsp_correction,
    evaluate_output_quality,
    mixdown_stems,
)
from motor_vocal.quality import evaluate_stem_reconstruction
from motor_vocal.separation import (
    align_stem_to_source,
    load_source_audio,
    separate_stems,
)


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True)
class ProcessingResult:
    mixed_path: Path
    vocal_path: Path
    processed_vocal_path: Path
    instrumental_path: Path
    analysis: dict[str, Any]


def _write_float_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    sf.write(path, audio, sample_rate, format="WAV", subtype="FLOAT")


def process_audio_file(
    audio_path: str | Path,
    output_dir: str | Path,
    *,
    preset_dsp: str = "Balanceado",
    correct_nasality: bool = True,
    correct_stridency: bool = True,
    correct_sibilance: bool = True,
    progress: ProgressCallback | None = None,
) -> ProcessingResult:
    """Executa uma entrada e grava os quatro WAVs em um diretório exclusivo."""
    if preset_dsp not in DSP_PRESETS:
        raise ValueError(f"Preset DSP desconhecido: {preset_dsp}")
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Diretório de saída já existe: {destination}")

    def report(fraction: float, description: str) -> None:
        if progress is not None:
            progress(fraction, description)

    report(0.15, "Separando voz e instrumental com Demucs...")
    vocals, instrumental, sample_rate = separate_stems(str(audio_path))
    original = load_source_audio(str(audio_path), sample_rate)
    original = align_stem_to_source(
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

    report(0.45, "Analisando nasalidade, estridência e sibilância...")
    analysis = analyze_ptbr_artifacts(vocals, sample_rate)
    analysis["preset_dsp"] = preset_dsp
    analysis["modulos_ativos"] = {
        "nasalidade": bool(correct_nasality),
        "estridencia": bool(correct_stridency),
        "sibilancia": bool(correct_sibilance),
    }
    analysis["qualidade_separacao"] = reconstruction_quality

    report(0.70, "Aplicando EQ dinâmica e De-Esser...")
    processed_vocals = apply_dsp_correction(vocals, sample_rate, analysis)

    report(0.90, "Remontando e exportando o áudio...")
    mixed, _protection_gain_db = mixdown_stems(processed_vocals, instrumental)
    destination.mkdir(parents=True, exist_ok=False)
    mixed_path = destination / "mix_processado.wav"
    vocal_path = destination / "vocal_isolado.wav"
    processed_vocal_path = destination / "vocal_corrigido.wav"
    instrumental_path = destination / "instrumental.wav"
    _write_float_wav(mixed_path, mixed, sample_rate)
    rendered_mix, rendered_sr = sf.read(
        mixed_path,
        dtype="float32",
        always_2d=True,
    )
    analysis["controle_qualidade_saida"] = evaluate_output_quality(
        original,
        np.asarray(rendered_mix, dtype=np.float32),
        int(rendered_sr),
    )
    _write_float_wav(vocal_path, vocals, sample_rate)
    _write_float_wav(processed_vocal_path, processed_vocals, sample_rate)
    _write_float_wav(instrumental_path, instrumental, sample_rate)
    report(1.0, "Processamento concluído.")
    return ProcessingResult(
        mixed_path=mixed_path,
        vocal_path=vocal_path,
        processed_vocal_path=processed_vocal_path,
        instrumental_path=instrumental_path,
        analysis=analysis,
    )
