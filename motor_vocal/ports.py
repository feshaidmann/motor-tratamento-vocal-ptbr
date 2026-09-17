"""Portas estáveis entre o domínio de áudio e adaptadores locais/remotos.

Estas interfaces são deliberadamente pequenas. As implementações atuais continuam
em ``jobs.py`` e no filesystem local; o módulo apenas fixa as dependências que o
worker precisa, sem escolher banco, fila ou provedor.

Cada porta tem, além do adaptador real, uma implementação falsa exercitada em
``tests/test_ports.py``. Uma porta sem segundo adaptador e sem teste de contrato
parece código morto; ver ``docs/ARCHITECTURE_DECOUPLING.md``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from motor_vocal.pipeline import ProcessingResult

if TYPE_CHECKING:
    from motor_vocal.jobs import JobRecord, JobRequest


class JobStatePort(Protocol):
    """Persistência e ciclo de vida de um trabalho.

    Um adaptador remoto deve preservar a mesma semântica de tentativa: somente a
    tentativa válida conclui ou falha, e a recuperação de um trabalho abandonado
    não apaga o histórico da tentativa anterior.
    """

    def enqueue(
        self,
        request: JobRequest,
        idempotency_key: str,
        *,
        max_jobs_per_day: int | None = None,
    ) -> JobRecord: ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def claim_next(self) -> JobRecord | None: ...

    def heartbeat(self, job_id: str, attempt: int) -> bool: ...

    def complete(self, job_id: str, attempt: int, result: ProcessingResult) -> None: ...

    def fail(self, job_id: str, attempt: int, error: str) -> None: ...


class AudioExecutorPort(Protocol):
    """Executor de áudio independente da interface e da fila.

    Corresponde à assinatura de ``motor_vocal.pipeline.process_audio_file``. O
    worker depende desta porta para que o ciclo de job possa ser testado sem
    executar Demucs.
    """

    def __call__(
        self,
        audio_path: str | Path,
        output_dir: str | Path,
        *,
        preset_dsp: str = "Balanceado",
        correct_nasality: bool = True,
        correct_stridency: bool = True,
        correct_sibilance: bool = True,
    ) -> ProcessingResult: ...
