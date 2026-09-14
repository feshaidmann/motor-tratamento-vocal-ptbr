# Motor de Tratamento Vocal PT-BR com IA

Prova de Conceito para isolamento de voz e correção DSP de nasalidade e sibilância regionalizada em português brasileiro, combinando separação de fontes por inteligência artificial com processamento digital de áudio.

## Objetivo

Construir uma base local para experimentar um fluxo de tratamento vocal capaz de:

- isolar a voz de músicas ou gravações com o Demucs;
- analisar características espectrais com o Librosa;
- aplicar correções de nasalidade e sibilância com efeitos DSP;
- preservar particularidades fonéticas e regionais do português brasileiro;
- disponibilizar os experimentos por meio de uma interface Gradio.

## Tecnologias

- **Demucs** — separação de voz e acompanhamento por IA;
- **PyTorch Nightly** — execução dos modelos e aceleração via MPS/Metal;
- **Librosa** — análise de áudio e extração de características;
- **Pedalboard** — cadeia de efeitos e processamento DSP;
- **Gradio** — interface local para a PoC;
- **NumPy** — processamento numérico;
- **SoundFile** — leitura e gravação de áudio;
- **FFmpeg** — suporte a formatos e conversões de mídia.

## Ambiente-alvo

- macOS em Apple Silicon (`arm64`), incluindo Mac mini com chip M4;
- Python 3.10 ou superior;
- Homebrew;
- backend MPS/Metal disponível no macOS.

> No macOS, a distribuição oficial do PyTorch inclui o backend MPS. O índice `nightly/cpu` indica uma build sem CUDA; ele não desativa a aceleração MPS em equipamentos Apple Silicon compatíveis.

## Instalação

Clone o repositório, entre na pasta do projeto e execute:

```bash
chmod +x setup_mac.sh
./setup_mac.sh
```

O script verifica o Homebrew, instala o FFmpeg quando necessário, cria o ambiente virtual `env` e instala todas as dependências Python.

Se o Homebrew ainda não estiver instalado, o script exibirá as instruções oficiais e encerrará sem alterar o ambiente Python.

## Ativação do ambiente

Após a instalação, ative o ambiente virtual em cada nova sessão do terminal:

```bash
source env/bin/activate
```

Para sair do ambiente:

```bash
deactivate
```

## Verificação do MPS

O próprio setup informa ao final se o backend MPS está compilado e disponível. A verificação também pode ser repetida manualmente:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

O resultado esperado em um Mac Apple Silicon compatível é `True`.

## Estrutura inicial

```text
.
├── README.md
├── env/              # Ambiente virtual local (não versionar)
└── setup_mac.sh      # Preparação automatizada do macOS
```

## Status

Projeto em fase de Prova de Conceito. A arquitetura do pipeline, os modelos de correção regionalizada e a interface da aplicação serão evoluídos incrementalmente.

