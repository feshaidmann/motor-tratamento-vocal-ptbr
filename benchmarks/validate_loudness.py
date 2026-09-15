#!/usr/bin/env python3
"""Valida pares nivelados com pyloudnorm e o medidor independente do FFmpeg."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyloudnorm as pyln
import soundfile as sf


DEFAULT_MAX_RESIDUAL_LU = 0.2
DEFAULT_MAX_METER_DELTA_LU = 0.2
DEFAULT_TRUE_PEAK_CEILING_DBTP = -0.9


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_ebur128_summary(output: str) -> dict[str, float]:
    """Extrai loudness integrado e true peak do último resumo ebur128."""
    summaries = output.rsplit("Summary:", maxsplit=1)
    if len(summaries) != 2:
        raise ValueError("O FFmpeg não produziu um resumo ebur128.")
    summary = summaries[1]
    integrated = re.search(r"^\s*I:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+LUFS", summary, re.MULTILINE)
    peak = re.search(r"^\s*Peak:\s*(-?(?:\d+(?:\.\d+)?|inf))\s+dBFS", summary, re.MULTILINE)
    if integrated is None or peak is None:
        raise ValueError("Não foi possível interpretar loudness/true peak do FFmpeg.")
    return {
        "loudness_integrado_lufs": float(integrated.group(1)),
        "true_peak_dbtp": float(peak.group(1)),
    }


def measure_with_ffmpeg(path: Path, ffmpeg_binary: str = "ffmpeg") -> dict[str, float]:
    command = [
        ffmpeg_binary,
        "-hide_banner",
        "-nostats",
        "-i",
        str(path),
        "-filter_complex",
        "ebur128=peak=true",
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip().splitlines()[-1:] or ["erro sem detalhes"]
        raise RuntimeError(f"FFmpeg falhou ao medir {path.name}: {detail[0]}")
    return parse_ebur128_summary(result.stderr)


def measure_with_pyloudnorm(path: Path) -> float:
    audio, sample_rate = sf.read(path, dtype="float64", always_2d=True)
    if audio.shape[1] > 5:
        audio = audio[:, :5]
    minimum_frames = max(1, int(math.ceil(sample_rate * 0.4)))
    if len(audio) < minimum_frames:
        padding = minimum_frames - len(audio)
        audio = np.pad(audio, ((0, padding), (0, 0)))
    loudness = float(pyln.Meter(sample_rate).integrated_loudness(audio))
    if not math.isfinite(loudness):
        raise ValueError(f"{path.name} não possui loudness integrado mensurável.")
    return loudness


def evaluate_pair(
    original: dict[str, Any],
    processed: dict[str, Any],
    *,
    max_residual_lu: float = DEFAULT_MAX_RESIDUAL_LU,
    max_meter_delta_lu: float = DEFAULT_MAX_METER_DELTA_LU,
    true_peak_ceiling_dbtp: float = DEFAULT_TRUE_PEAK_CEILING_DBTP,
) -> dict[str, Any]:
    residual = abs(
        float(processed["ffmpeg"]["loudness_integrado_lufs"])
        - float(original["ffmpeg"]["loudness_integrado_lufs"])
    )
    meter_deltas = {
        condition: abs(
            float(measurement["ffmpeg"]["loudness_integrado_lufs"])
            - float(measurement["pyloudnorm_lufs"])
        )
        for condition, measurement in (("original", original), ("processado", processed))
    }
    checks = {
        "sample_rate_identico": original["sample_rate_hz"] == processed["sample_rate_hz"],
        "canais_identicos": original["canais"] == processed["canais"],
        "frames_identicos": original["frames"] == processed["frames"],
        "residual_loudness_dentro_do_limite": residual <= max_residual_lu,
        "concordancia_medidores_original": meter_deltas["original"] <= max_meter_delta_lu,
        "concordancia_medidores_processado": meter_deltas["processado"] <= max_meter_delta_lu,
        "true_peak_original_dentro_do_limite": (
            float(original["ffmpeg"]["true_peak_dbtp"]) <= true_peak_ceiling_dbtp
        ),
        "true_peak_processado_dentro_do_limite": (
            float(processed["ffmpeg"]["true_peak_dbtp"]) <= true_peak_ceiling_dbtp
        ),
    }
    return {
        "aprovado": all(checks.values()),
        "verificacoes": checks,
        "delta_loudness_ffmpeg_lu": round(residual, 3),
        "delta_medidores_lu": {key: round(value, 3) for key, value in meter_deltas.items()},
    }


def measure_file(path: Path, ffmpeg_binary: str) -> dict[str, Any]:
    info = sf.info(path)
    return {
        "arquivo": str(path),
        "sha256": _sha256(path),
        "sample_rate_hz": int(info.samplerate),
        "canais": int(info.channels),
        "frames": int(info.frames),
        "duracao_segundos": round(float(info.duration), 6),
        "ffmpeg": measure_with_ffmpeg(path, ffmpeg_binary),
        "pyloudnorm_lufs": round(measure_with_pyloudnorm(path), 3),
    }


def validate_pair(
    original_path: Path,
    processed_path: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    max_residual_lu: float = DEFAULT_MAX_RESIDUAL_LU,
    max_meter_delta_lu: float = DEFAULT_MAX_METER_DELTA_LU,
    true_peak_ceiling_dbtp: float = DEFAULT_TRUE_PEAK_CEILING_DBTP,
) -> dict[str, Any]:
    original = measure_file(original_path, ffmpeg_binary)
    processed = measure_file(processed_path, ffmpeg_binary)
    evaluation = evaluate_pair(
        original,
        processed,
        max_residual_lu=max_residual_lu,
        max_meter_delta_lu=max_meter_delta_lu,
        true_peak_ceiling_dbtp=true_peak_ceiling_dbtp,
    )
    return {"original": original, "processado": processed, **evaluation}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida pares de áudio nivelados com FFmpeg e pyloudnorm.",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        action="append",
        metavar=("ORIGINAL", "PROCESSADO"),
        required=True,
        help="Par nivelado; repita a opção para validar um corpus.",
    )
    parser.add_argument("--output", type=Path, help="Caminho do relatório JSON.")
    parser.add_argument("--max-residual-lu", type=float, default=DEFAULT_MAX_RESIDUAL_LU)
    parser.add_argument("--max-meter-delta-lu", type=float, default=DEFAULT_MAX_METER_DELTA_LU)
    parser.add_argument(
        "--true-peak-ceiling-dbtp",
        type=float,
        default=DEFAULT_TRUE_PEAK_CEILING_DBTP,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ffmpeg_binary = shutil.which("ffmpeg")
    if ffmpeg_binary is None:
        raise SystemExit("FFmpeg não encontrado no PATH.")
    if args.max_residual_lu < 0 or args.max_meter_delta_lu < 0:
        raise SystemExit("Os limites de diferença devem ser positivos ou zero.")

    pairs: list[tuple[Path, Path]] = []
    for raw_original, raw_processed in args.pair:
        original = Path(raw_original).expanduser().resolve()
        processed = Path(raw_processed).expanduser().resolve()
        for path in (original, processed):
            if not path.is_file():
                raise SystemExit(f"Arquivo não encontrado: {path}")
        pairs.append((original, processed))

    results = [
        validate_pair(
            original,
            processed,
            ffmpeg_binary=ffmpeg_binary,
            max_residual_lu=args.max_residual_lu,
            max_meter_delta_lu=args.max_meter_delta_lu,
            true_peak_ceiling_dbtp=args.true_peak_ceiling_dbtp,
        )
        for original, processed in pairs
    ]
    report = {
        "schema": "motor-clareza-loudness-validation/v1",
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "criterios": {
            "max_residual_lu": args.max_residual_lu,
            "max_meter_delta_lu": args.max_meter_delta_lu,
            "true_peak_ceiling_dbtp": args.true_peak_ceiling_dbtp,
        },
        "aprovado": all(result["aprovado"] for result in results),
        "total_pares": len(results),
        "pares_aprovados": sum(bool(result["aprovado"]) for result in results),
        "resultados": results,
    }
    default_name = f"loudness_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = (args.output or Path("benchmarks/results") / default_name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    status = "APROVADO" if report["aprovado"] else "REPROVADO"
    print(f"{status}: {report['pares_aprovados']}/{report['total_pares']} par(es).")
    print(f"Relatório salvo em: {output_path}")
    return 0 if report["aprovado"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
