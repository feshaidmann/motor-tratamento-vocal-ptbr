"""Separação de fontes e adaptação de áudio para o pipeline vocal."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


class DemucsExecutionError(RuntimeError):
    """Representa uma falha retornada pelo processo de separação Demucs."""


def _log(message: str) -> None:
    """Emite logs imediatamente para o processo que executa o worker."""
    print(f"[Motor Vocal PT-BR] {message}", flush=True)


def select_available_device(
    *,
    cuda_available: bool,
    mps_built: bool,
    mps_available: bool,
) -> str:
    """Seleciona o acelerador disponível, priorizando CUDA para a nuvem."""
    if cuda_available:
        return "cuda"
    if mps_built and mps_available:
        return "mps"
    return "cpu"


def preferred_demucs_device() -> str:
    """Escolhe CUDA, MPS ou CPU conforme o ambiente atual."""
    try:
        import torch

        mps = getattr(torch.backends, "mps", None)
        return select_available_device(
            cuda_available=bool(torch.cuda.is_available()),
            mps_built=bool(mps and mps.is_built()),
            mps_available=bool(mps and mps.is_available()),
        )
    except (ImportError, AttributeError):
        return "cpu"


def run_demucs(audio_path: Path, output_dir: Path, device: str) -> None:
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


def load_demucs_stem(stem_path: Path) -> tuple[np.ndarray, int]:
    """Carrega um stem WAV como float32 no formato (amostras, canais)."""
    audio, sample_rate = sf.read(stem_path, dtype="float32", always_2d=True)
    return np.asarray(audio, dtype=np.float32), int(sample_rate)


def source_audio_metadata(source_path: Path) -> tuple[int, int, int]:
    """Retorna sample rate, número de amostras e canais do arquivo original."""
    try:
        info = sf.info(source_path)
    except RuntimeError as exc:
        raise ValueError(
            f"Não foi possível ler os metadados de {source_path.name}."
        ) from exc

    if info.samplerate <= 0 or info.frames <= 0 or info.channels <= 0:
        raise ValueError("O arquivo de entrada não contém áudio válido.")
    return int(info.samplerate), int(info.frames), int(info.channels)


def adapt_channel_count(audio: np.ndarray, target_channels: int) -> np.ndarray:
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


def align_stem_to_source(
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

    audio = adapt_channel_count(np.asarray(audio, dtype=np.float32), source_channels)
    if audio.shape[0] > source_samples:
        audio = audio[:source_samples]
    elif audio.shape[0] < source_samples:
        audio = np.pad(audio, ((0, source_samples - audio.shape[0]), (0, 0)))

    return np.ascontiguousarray(audio, dtype=np.float32)


def load_source_audio(source_path: str, target_sr: int) -> np.ndarray:
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


def separate_stems(audio_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Separa ``vocals`` e ``no_vocals`` com o modelo htdemucs.

    A função usa CUDA, MPS ou CPU, nessa ordem de preferência. Se a execução
    acelerada falhar, repete a separação em CPU. Os WAVs intermediários são
    removidos automaticamente depois de carregados em memória.
    """
    _log("Etapa 1/4 — Separando voz e instrumental com Demucs/htdemucs...")
    source_path = Path(audio_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Arquivo de entrada não encontrado: {source_path}")
    if source_path.suffix.lower() not in {".wav", ".mp3"}:
        raise ValueError("Formato não suportado. Envie um arquivo WAV ou MP3.")

    source_sr, source_samples, source_channels = source_audio_metadata(source_path)
    preferred_device = preferred_demucs_device()

    with tempfile.TemporaryDirectory(prefix="motor_vocal_demucs_") as temp_dir:
        demucs_output = Path(temp_dir)
        demucs_input = source_path
        if source_path.suffix.lower() == ".mp3":
            # Use the same decoder as the reference signal. Demucs can decode
            # MP3 with a different encoder-delay offset, shifting both stems.
            demucs_input = demucs_output / "input_decoded.wav"
            with sf.SoundFile(
                demucs_input,
                mode="w",
                samplerate=source_sr,
                channels=source_channels,
                format="WAV",
                subtype="FLOAT",
            ) as decoded:
                for block in sf.blocks(
                    source_path,
                    blocksize=262_144,
                    dtype="float32",
                    always_2d=True,
                ):
                    decoded.write(block)
        try:
            run_demucs(demucs_input, demucs_output, preferred_device)
        except DemucsExecutionError:
            if preferred_device == "cpu":
                raise
            _log(
                f"A execução {preferred_device.upper()} falhou; "
                "repetindo a separação em CPU..."
            )
            run_demucs(demucs_input, demucs_output, "cpu")

        vocal_candidates = list(demucs_output.rglob("vocals.wav"))
        instrumental_candidates = list(demucs_output.rglob("no_vocals.wav"))
        if len(vocal_candidates) != 1 or len(instrumental_candidates) != 1:
            raise FileNotFoundError(
                "O Demucs não produziu exatamente um par vocals/no_vocals."
            )

        vocals, vocal_sr = load_demucs_stem(vocal_candidates[0])
        no_vocals, instrumental_sr = load_demucs_stem(instrumental_candidates[0])

    if vocal_sr != instrumental_sr:
        raise ValueError(
            "Os stems do Demucs possuem taxas de amostragem incompatíveis: "
            f"{vocal_sr} Hz e {instrumental_sr} Hz."
        )
    vocals = align_stem_to_source(
        vocals,
        vocal_sr,
        source_sr,
        source_samples,
        source_channels,
    )
    no_vocals = align_stem_to_source(
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
