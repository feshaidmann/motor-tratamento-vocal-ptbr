#!/usr/bin/env bash

# Interrompe a execução em caso de erro, variável não definida ou falha em pipeline.
set -Eeuo pipefail

readonly ENV_DIR="env"
readonly PYTORCH_NIGHTLY_INDEX="https://download.pytorch.org/whl/nightly/cpu"
readonly PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Garante caminhos previsíveis mesmo quando o script é chamado de outra pasta.
cd "$PROJECT_DIR"

log() {
  printf '\n\033[1;34m[setup]\033[0m %s\n' "$1"
}

fail() {
  printf '\n\033[1;31m[erro]\033[0m %s\n' "$1" >&2
  exit 1
}

# Este setup foi criado exclusivamente para macOS.
[[ "$(uname -s)" == "Darwin" ]] || fail "Este script deve ser executado no macOS."

if [[ "$(uname -m)" != "arm64" ]]; then
  printf '\n\033[1;33m[aviso]\033[0m Arquitetura detectada: %s. O setup foi otimizado para Apple Silicon (arm64).\n' "$(uname -m)"
fi

# O Homebrew gerencia as dependências nativas usadas no processamento de áudio.
if ! command -v brew >/dev/null 2>&1; then
  printf '%s\n' \
    "Homebrew não encontrado." \
    "Instale-o pelo site oficial: https://brew.sh" \
    "Ou execute:" \
    '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"' \
    "Depois, abra um novo terminal e execute este script novamente."
  exit 1
fi

log "Homebrew encontrado: $(brew --version | sed -n '1p')"

# O ffmpeg permite que o Demucs e o torchaudio leiam e convertam vários formatos.
if brew list --formula ffmpeg >/dev/null 2>&1; then
  log "ffmpeg já está instalado."
else
  log "Instalando ffmpeg via Homebrew..."
  brew install ffmpeg
fi

# Usa o Python 3 disponível no sistema. Caso não exista, instala uma versão
# compatível e mantida pelo Homebrew antes de criar o ambiente virtual.
if ! command -v python3 >/dev/null 2>&1; then
  log "Python 3 não encontrado; instalando via Homebrew..."
  brew install python
fi

PYTHON_BIN="$(command -v python3)"

# PyTorch atual requer Python 3.10 ou mais recente. A checagem evita uma falha
# pouco clara durante a resolução dos pacotes pelo pip.
if ! "$PYTHON_BIN" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
  fail "Python 3.10+ é necessário. Atualize com 'brew install python' e tente novamente."
fi

log "Python selecionado: $($PYTHON_BIN --version) ($PYTHON_BIN)"

# Cria um ambiente isolado chamado env. Se ele já existir, será reutilizado.
if [[ ! -d "$ENV_DIR" ]]; then
  log "Criando o ambiente virtual em ./$ENV_DIR..."
  "$PYTHON_BIN" -m venv "$ENV_DIR"
else
  log "Ambiente virtual ./$ENV_DIR já existe; reutilizando-o."
fi

# Ativa o ambiente virtual para que todas as instalações fiquem isoladas.
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

log "Atualizando pip, setuptools e wheel..."
python -m pip install --upgrade pip setuptools wheel

# No macOS, os wheels oficiais do PyTorch incluem o backend MPS/Metal. O índice
# nightly/cpu entrega as builds de pré-lançamento sem dependências de CUDA.
log "Instalando PyTorch e torchaudio nightly com suporte a MPS/Metal..."
python -m pip install --pre --upgrade \
  torch torchaudio \
  --index-url "$PYTORCH_NIGHTLY_INDEX"

# Instala versões validadas das bibliotecas de separação de fontes, análise,
# tratamento DSP, interface web e leitura/escrita de arquivos de áudio.
log "Instalando as dependências Python do projeto..."
python -m pip install --upgrade --requirement requirements.txt

# Confirma que o PyTorch foi importado e informa se o backend MPS está disponível.
log "Validando a instalação do PyTorch e o backend MPS..."
python - <<'PY'
import platform

import torch

print(f"Python: {platform.python_version()}")
print(f"PyTorch: {torch.__version__}")
print(f"MPS compilado: {torch.backends.mps.is_built()}")
print(f"MPS disponível: {torch.backends.mps.is_available()}")
PY

printf '\n\033[1;32mSetup concluído com sucesso.\033[0m\n'
printf 'Para ativar o ambiente em uma nova sessão, execute: source %s/bin/activate\n' "$ENV_DIR"
