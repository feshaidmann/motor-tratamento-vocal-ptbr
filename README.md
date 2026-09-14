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

## Executando a aplicação

Com o ambiente virtual ativo, execute:

```bash
python app.py
```

Abra [http://127.0.0.1:7860](http://127.0.0.1:7860) no navegador. Na primeira
separação, o Demucs fará o download do modelo `htdemucs`; as execuções seguintes
usarão o modelo armazenado no cache local.

O motor tenta executar a separação no backend MPS/Metal. Caso essa etapa falhe,
ela é repetida automaticamente em CPU.

## Processamento DSP

A análise MIR procura eventos sustentados nas regiões de `600–1.200 Hz`
(nasalidade) e `4–9 kHz` (sibilância). As frequências dominantes e os intervalos
detectados controlam uma cadeia regionalizada do Pedalboard:

- filtro paramétrico para atenuação de nasalidade;
- filtro paramétrico e compressor para De-Esser;
- envelopes de ataque e release para evitar clicks nas transições;
- proteção de pico no mixdown final.

A interface oferece os presets **Suave**, **Balanceado** e **Intenso**, além de
players separados para o vocal antes/depois do DSP e para o instrumental.

> A detecção é heurística e ainda não representa um classificador fonético.
> Faça sempre a comparação A/B, especialmente com material já masterizado.

## Testes

Execute a suíte de regressão com o ambiente virtual ativo:

```bash
python -m unittest discover -s tests -v
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
├── .gitignore
├── README.md
├── app.py             # Pipeline de áudio e interface Gradio
├── env/              # Ambiente virtual local (não versionar)
├── setup_mac.sh      # Preparação automatizada do macOS
└── tests/             # Testes automatizados do DSP e do mixdown
```

## Status

Projeto em fase de Prova de Conceito:

- separação real de `vocals` e `no_vocals` com `htdemucs`;
- aceleração MPS com fallback automático para CPU;
- análise MIR heurística de nasalidade e sibilância;
- EQ dinâmica e De-Esser regionalizados com três níveis de intensidade;
- interface Gradio funcional com progresso e diagnóstico;
- auditoria dos stems e do vocal antes/depois do DSP;
- mixdown e exportação WAV em 32-bit float;
- proteção automática contra pico acima de `-0,09 dBFS` no mix final.
