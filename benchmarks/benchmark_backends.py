#!/usr/bin/env python3
"""Benchmark reproduzível do Demucs em CPU e MPS no macOS Apple Silicon."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import soundfile as sf


MODEL_NAME = "htdemucs"
POLL_INTERVAL_SECONDS = 0.20


def build_demucs_command(
    audio_path: Path,
    output_dir: Path,
    device: str,
) -> list[str]:
    """Monta a mesma chamada de separação usada pela aplicação."""
    return [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems=vocals",
        "--name",
        MODEL_NAME,
        "--device",
        device,
        "--float32",
        "--clip-mode",
        "none",
        "--out",
        str(output_dir),
        str(audio_path),
    ]


def parse_process_snapshot(snapshot: str) -> dict[int, tuple[int, int]]:
    """Converte a saída de ``ps`` em PID -> (PPID, RSS em KiB)."""
    processes: dict[int, tuple[int, int]] = {}
    for line in snapshot.splitlines():
        fields = line.split()
        if len(fields) != 3:
            continue
        try:
            pid, parent_pid, rss_kib = (int(value) for value in fields)
        except ValueError:
            continue
        processes[pid] = (parent_pid, rss_kib)
    return processes


def process_tree_rss_kib(root_pid: int) -> int:
    """Soma o RSS do processo Demucs e de todos os seus descendentes."""
    result = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss="],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return 0

    processes = parse_process_snapshot(result.stdout)
    selected = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _) in processes.items():
            if pid not in selected and parent_pid in selected:
                selected.add(pid)
                changed = True
    return sum(processes.get(pid, (0, 0))[1] for pid in selected)


def _sample_peak_memory(
    pid: int,
    stop_event: threading.Event,
    peak_holder: list[int],
) -> None:
    while not stop_event.is_set():
        peak_holder[0] = max(peak_holder[0], process_tree_rss_kib(pid))
        stop_event.wait(POLL_INTERVAL_SECONDS)
    peak_holder[0] = max(peak_holder[0], process_tree_rss_kib(pid))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision(project_dir: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def mps_status() -> dict[str, bool]:
    try:
        import torch

        return {
            "compilado": bool(torch.backends.mps.is_built()),
            "disponivel": bool(torch.backends.mps.is_available()),
        }
    except (ImportError, AttributeError):
        return {"compilado": False, "disponivel": False}


def run_once(
    audio_path: Path,
    audio_duration_seconds: float,
    device: str,
    run_number: int,
    show_demucs_output: bool,
) -> dict[str, Any]:
    """Executa uma separação isolada e mede tempo e pico de RSS do processo."""
    with tempfile.TemporaryDirectory(prefix=f"benchmark_demucs_{device}_") as temp:
        output_dir = Path(temp)
        command = build_demucs_command(audio_path, output_dir, device)
        started_at = time.perf_counter()
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=os.environ.copy(),
        )
        stop_event = threading.Event()
        peak_rss_kib = [0]
        sampler = threading.Thread(
            target=_sample_peak_memory,
            args=(process.pid, stop_event, peak_rss_kib),
            daemon=True,
        )
        sampler.start()

        output_tail: list[str] = []
        if process.stdout is not None:
            for line in process.stdout:
                clean = line.rstrip()
                if not clean:
                    continue
                output_tail.append(clean)
                output_tail = output_tail[-20:]
                if show_demucs_output:
                    print(f"[{device}] {clean}", flush=True)

        return_code = process.wait()
        elapsed_seconds = time.perf_counter() - started_at
        stop_event.set()
        sampler.join(timeout=2.0)

        stem_count = len(list(output_dir.rglob("vocals.wav"))) + len(
            list(output_dir.rglob("no_vocals.wav"))
        )
        status = "ok" if return_code == 0 and stem_count == 2 else "falha"
        return {
            "dispositivo": device,
            "execucao": run_number,
            "status": status,
            "codigo_retorno": return_code,
            "tempo_segundos": round(elapsed_seconds, 3),
            "rtf": round(elapsed_seconds / audio_duration_seconds, 4),
            "pico_rss_mb": round(peak_rss_kib[0] / 1024.0, 1),
            "stems_encontrados": stem_count,
            "log_final": output_tail if status != "ok" else [],
        }


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Agrega somente execuções concluídas, mantendo falhas no relatório bruto."""
    successful = [run for run in runs if run.get("status") == "ok"]
    if not successful:
        return {"execucoes_validas": 0, "status": "sem_resultado"}

    times = [float(run["tempo_segundos"]) for run in successful]
    rtfs = [float(run["rtf"]) for run in successful]
    memory = [float(run["pico_rss_mb"]) for run in successful]
    return {
        "execucoes_validas": len(successful),
        "status": "ok",
        "tempo_medio_segundos": round(statistics.fmean(times), 3),
        "tempo_mediano_segundos": round(statistics.median(times), 3),
        "rtf_medio": round(statistics.fmean(rtfs), 4),
        "pico_rss_maximo_mb": round(max(memory), 1),
    }


def build_report(
    audio_path: Path,
    devices: list[str],
    runs_per_device: int,
    warmup_runs: int,
    show_demucs_output: bool,
) -> dict[str, Any]:
    info = sf.info(audio_path)
    duration = float(info.duration)
    if duration <= 0:
        raise ValueError("O arquivo de benchmark não contém áudio válido.")

    project_dir = Path(__file__).resolve().parents[1]
    mps = mps_status()
    report: dict[str, Any] = {
        "schema": "motor-clareza-benchmark/v1",
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_revision(project_dir),
        "modelo": MODEL_NAME,
        "entrada": {
            "arquivo": audio_path.name,
            "sha256": _sha256(audio_path),
            "duracao_segundos": round(duration, 4),
            "sample_rate_hz": int(info.samplerate),
            "canais": int(info.channels),
            "formato": info.format,
        },
        "ambiente": {
            "sistema": platform.platform(),
            "arquitetura": platform.machine(),
            "processador": platform.processor(),
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "demucs": _package_version("demucs"),
            "mps": mps,
            "memoria_medida": "RSS do processo Demucs e descendentes; não inclui telemetria dedicada da memória unificada da GPU",
        },
        "configuracao": {
            "dispositivos_solicitados": devices,
            "execucoes_por_dispositivo": runs_per_device,
            "aquecimentos_por_dispositivo": warmup_runs,
        },
        "resultados": {},
    }

    for device in devices:
        if device == "mps" and not mps["disponivel"]:
            report["resultados"][device] = {
                "status": "indisponivel",
                "motivo": "backend MPS não disponível neste ambiente",
                "execucoes": [],
            }
            continue

        print(f"\nBenchmark {device.upper()} — aquecimento", flush=True)
        warmups = [
            run_once(audio_path, duration, device, index + 1, show_demucs_output)
            for index in range(warmup_runs)
        ]
        if any(run["status"] != "ok" for run in warmups):
            report["resultados"][device] = {
                "status": "falha_no_aquecimento",
                "aquecimentos": warmups,
                "execucoes": [],
            }
            continue

        print(f"Benchmark {device.upper()} — {runs_per_device} execução(ões)", flush=True)
        runs = [
            run_once(audio_path, duration, device, index + 1, show_demucs_output)
            for index in range(runs_per_device)
        ]
        report["resultados"][device] = {
            "status": "ok" if all(run["status"] == "ok" for run in runs) else "parcial",
            "aquecimentos": warmups,
            "execucoes": runs,
            "resumo": summarize_runs(runs),
        }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mede tempo, RTF e pico de RSS do Demucs em CPU e MPS.",
    )
    parser.add_argument("audio", type=Path, help="Arquivo WAV ou MP3 de referência.")
    parser.add_argument(
        "--devices",
        nargs="+",
        choices=("cpu", "mps"),
        default=["cpu", "mps"],
        help="Backends medidos (padrão: cpu mps).",
    )
    parser.add_argument("--runs", type=int, default=3, help="Execuções medidas por backend.")
    parser.add_argument("--warmup", type=int, default=1, help="Aquecimentos descartados por backend.")
    parser.add_argument("--output", type=Path, help="Caminho do relatório JSON.")
    parser.add_argument(
        "--show-demucs-output",
        action="store_true",
        help="Exibe toda a saída do Demucs durante as medições.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audio_path = args.audio.expanduser().resolve()
    if not audio_path.is_file():
        raise SystemExit(f"Arquivo não encontrado: {audio_path}")
    if audio_path.suffix.lower() not in {".wav", ".mp3"}:
        raise SystemExit("Use um arquivo WAV ou MP3.")
    if args.runs < 1 or args.warmup < 0:
        raise SystemExit("--runs deve ser >= 1 e --warmup deve ser >= 0.")

    report = build_report(
        audio_path,
        list(dict.fromkeys(args.devices)),
        args.runs,
        args.warmup,
        args.show_demucs_output,
    )
    default_name = f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_path = (args.output or Path("benchmarks/results") / default_name).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nRelatório salvo em: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
