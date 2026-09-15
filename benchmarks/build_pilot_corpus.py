#!/usr/bin/env python3
"""Monta um corpus-semente local a partir das sessões processadas da PoC."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import app
from benchmarks.validate_loudness import validate_pair


REQUIRED_FILES = ("vocal_antes.wav", "instrumental.wav", "mix_processado.wav", "relatorio.json")


def select_window_from_energies(energies: list[float], window_blocks: int) -> int:
    """Retorna o primeiro bloco da janela contígua com maior energia acumulada."""
    if not energies:
        raise ValueError("Não há energia vocal para selecionar um trecho.")
    width = max(1, min(window_blocks, len(energies)))
    values = np.asarray(energies, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    totals = cumulative[width:] - cumulative[:-width]
    return int(np.argmax(totals))


def select_active_window(vocal_path: Path, segment_seconds: float) -> tuple[int, int, int]:
    """Seleciona uma janela por energia vocal média em blocos de um segundo."""
    info = sf.info(vocal_path)
    sample_rate = int(info.samplerate)
    if sample_rate <= 0 or info.frames <= 0:
        raise ValueError(f"Áudio vocal inválido: {vocal_path}")
    energies: list[float] = []
    with sf.SoundFile(vocal_path) as stream:
        for block in stream.blocks(blocksize=sample_rate, dtype="float32", always_2d=True):
            energies.append(float(np.mean(np.square(block), dtype=np.float64)))
    requested_frames = max(1, int(round(segment_seconds * sample_rate)))
    segment_frames = min(requested_frames, int(info.frames))
    window_blocks = max(1, int(math.ceil(segment_frames / sample_rate)))
    start_block = select_window_from_energies(energies, window_blocks)
    start_frame = min(start_block * sample_rate, int(info.frames) - segment_frames)
    return start_frame, segment_frames, sample_rate


def read_slice(path: Path, start_frame: int, frames: int) -> tuple[np.ndarray, int]:
    with sf.SoundFile(path) as stream:
        stream.seek(start_frame)
        audio = stream.read(frames, dtype="float32", always_2d=True)
        sample_rate = int(stream.samplerate)
    if len(audio) != frames:
        raise ValueError(f"{path.name} não contém o intervalo selecionado completo.")
    return np.asarray(audio, dtype=np.float32), sample_rate


def discover_sessions(source_root: Path) -> list[Path]:
    sessions = []
    for directory in sorted(path for path in source_root.iterdir() if path.is_dir()):
        if all((directory / name).is_file() for name in REQUIRED_FILES):
            sessions.append(directory)
    return sessions


def analysis_summary(report: dict[str, Any]) -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    for key in ("nasalidade_ao_o", "estridencia_2_4khz", "sibilancia_s_x"):
        value = report.get(key, {})
        artifacts[key] = {
            "frequencia_alvo_hz": value.get("frequencia_alvo_hz"),
            "confianca": value.get("confianca"),
            "intervencao_autorizada": value.get("intervencao_autorizada"),
        }
    return {
        "preset_dsp": report.get("preset_dsp"),
        "modulos_ativos": report.get("modulos_ativos"),
        "artefatos": artifacts,
        "qualidade_separacao": report.get("qualidade_separacao"),
        "controle_qualidade_saida": report.get("controle_qualidade_saida"),
    }


def write_annotation_sheet(output_root: Path, items: list[dict[str, Any]]) -> Path:
    """Cria uma tabela simples para completar metadados que exigem revisão humana."""
    path = output_root / "anotacoes.csv"
    fields = (
        "item_id",
        "origem_local",
        "variante_ptbr",
        "regiao",
        "genero_musical",
        "perfil_vocal",
        "direitos_confirmados",
        "observacoes",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "item_id": item["id"],
                    "origem_local": item["origem_local"],
                    "variante_ptbr": "",
                    "regiao": "",
                    "genero_musical": "",
                    "perfil_vocal": "",
                    "direitos_confirmados": "não",
                    "observacoes": "",
                }
            )
    return path


def build_item(
    session_dir: Path,
    item_dir: Path,
    item_id: str,
    segment_seconds: float,
) -> dict[str, Any]:
    vocal_path = session_dir / "vocal_antes.wav"
    instrumental_path = session_dir / "instrumental.wav"
    processed_path = session_dir / "mix_processado.wav"
    start_frame, segment_frames, sample_rate = select_active_window(
        vocal_path,
        segment_seconds,
    )
    vocals, vocal_sr = read_slice(vocal_path, start_frame, segment_frames)
    instrumental, instrumental_sr = read_slice(instrumental_path, start_frame, segment_frames)
    processed, processed_sr = read_slice(processed_path, start_frame, segment_frames)
    if {vocal_sr, instrumental_sr, processed_sr} != {sample_rate}:
        raise ValueError(f"Sample rates incompatíveis em {session_dir.name}.")
    if vocals.shape != instrumental.shape or vocals.shape != processed.shape:
        raise ValueError(f"Stems ou mix desalinhados em {session_dir.name}.")

    reconstructed_original = vocals + instrumental
    matched_original, matched_processed, matching = app._level_match_abx_pair(
        reconstructed_original,
        processed,
        sample_rate,
    )
    item_dir.mkdir(parents=True, exist_ok=False)
    original_output = item_dir / "estimulo_1.wav"
    processed_output = item_dir / "estimulo_2.wav"
    sf.write(original_output, matched_original, sample_rate, subtype="FLOAT")
    sf.write(processed_output, matched_processed, sample_rate, subtype="FLOAT")
    validation = validate_pair(original_output, processed_output)

    report = json.loads((session_dir / "relatorio.json").read_text(encoding="utf-8"))
    return {
        "id": item_id,
        "origem_local": session_dir.name,
        "uso": "interno; confirmar direitos antes de compartilhar",
        "metadados_humanos": {
            "variante_ptbr": "não informado",
            "regiao": "não informado",
            "genero_musical": "não informado",
            "perfil_vocal": "não informado",
        },
        "selecao": {
            "metodo": "janela contígua de maior energia no stem vocal",
            "inicio_s": round(start_frame / sample_rate, 3),
            "duracao_s": round(segment_frames / sample_rate, 3),
        },
        "arquivos": {
            "estimulo_1": str(original_output.relative_to(item_dir.parent.parent)),
            "estimulo_2": str(processed_output.relative_to(item_dir.parent.parent)),
        },
        "equalizacao_loudness": matching,
        "validacao_independente": validation,
        "analise_motor": analysis_summary(report),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monta o corpus piloto local da PoC.")
    parser.add_argument("--source", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path, default=Path("outputs/corpus_ptbr_seed_v1"))
    parser.add_argument("--segment-seconds", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    if args.segment_seconds <= 0:
        raise SystemExit("--segment-seconds deve ser maior que zero.")
    if not source_root.is_dir():
        raise SystemExit(f"Diretório de origem não encontrado: {source_root}")
    if output_root.exists():
        raise SystemExit(f"O destino já existe; escolha outro caminho: {output_root}")

    sessions = discover_sessions(source_root)
    if not sessions:
        raise SystemExit("Nenhuma sessão completa foi encontrada.")
    items_root = output_root / "items"
    items_root.mkdir(parents=True)
    items = [
        build_item(
            session,
            items_root / f"item_{index:03d}",
            f"item_{index:03d}",
            args.segment_seconds,
        )
        for index, session in enumerate(sessions, start=1)
    ]
    approved = sum(bool(item["validacao_independente"]["aprovado"]) for item in items)
    manifest = {
        "schema": "motor-clareza-corpus/v1",
        "gerado_em_utc": datetime.now(timezone.utc).isoformat(),
        "finalidade": "corpus-semente para calibração técnica e piloto ABX",
        "politica_direitos": (
            "Corpus local. Confirmar autorização de cada gravação antes de compartilhar "
            "ou usar em publicação."
        ),
        "total_itens": len(items),
        "itens_aprovados": approved,
        "todos_aprovados": approved == len(items),
        "itens": items,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    annotation_path = write_annotation_sheet(output_root, items)
    print(f"Corpus criado: {output_root}")
    print(f"Validação: {approved}/{len(items)} item(ns) aprovado(s).")
    print(f"Manifesto: {manifest_path}")
    print(f"Anotações humanas: {annotation_path}")
    return 0 if manifest["todos_aprovados"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
