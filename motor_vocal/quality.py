"""Métricas de qualidade independentes da interface e da infraestrutura."""

from __future__ import annotations

from typing import Any

import numpy as np


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
    similarity_denominator = float(np.sqrt(reference_energy * reconstruction_energy))
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
