# Motor de Clareza Vocal PT-BR

Prova de Conceito local para isolamento de voz e correção DSP seletiva de
nasalidade, estridência e sibilância em voz cantada em português brasileiro.
O projeto combina separação de fontes por IA, análise acústica auditável e um
motor de decisão que pode se abster quando a evidência é insuficiente.

> A versão atual demonstra o pipeline de engenharia. A especialização fonética
> e o benefício perceptual ainda precisam ser validados com corpus anotado e
> testes cegos; as bandas usadas são janelas acústicas, não classificadores de
> fonemas.

## Whitepaper

O documento técnico descreve a hipótese, o estado real da implementação, os
critérios de segurança e o plano de validação para a beta:

- [Whitepaper — Motor de Clareza Vocal PT-BR](output/pdf/whitepaper_motor_clareza_vocal_ptbr.pdf)

## Pipeline

1. O Demucs `htdemucs` separa `vocals` e `no_vocals`.
2. A reconstrução dos stems é comparada à entrada; uma separação não confiável
   bloqueia o DSP.
3. O Librosa mede STFT, centroide, energia relativa, contraste e cobertura nas
   bandas candidatas.
4. Cada módulo usa um score de confiança e pode intervir ou se abster.
5. Pedalboard e envelopes NumPy aplicam DSP apenas nos intervalos autorizados.
6. Os stems são recombinados e exportados em WAV float32 com relatório de QC.

As janelas iniciais são:

- `600 Hz–1,2 kHz`: ressonâncias candidatas à nasalidade;
- `2–4 kHz`: estridência em vogais abertas e belt;
- `4–9 kHz`: sibilância e fricativas.

## Tecnologias

- **Demucs + PyTorch** — separação de fontes, com MPS/Metal e fallback para CPU;
- **Librosa + SciPy** — MIR, análise espectral, reamostragem e true peak estimado;
- **Pedalboard** — filtros paramétricos e compressor;
- **NumPy + SoundFile** — arrays, alinhamento, recombinação e WAV float32;
- **Gradio** — interface local para upload, A/B, stems e diagnóstico JSON;
- **FFmpeg** — leitura de MP3 e suporte de mídia.

## Ambiente-alvo

- macOS em Apple Silicon (`arm64`), incluindo Mac mini M4;
- Python 3.10 ou superior;
- Homebrew e FFmpeg;
- backend MPS disponível no macOS.

No macOS, a distribuição oficial do PyTorch inclui o backend MPS. O índice
`nightly/cpu` indica uma build sem CUDA; ele não desativa o MPS em equipamentos
Apple Silicon compatíveis.

## Instalação

Os comandos abaixo devem ser executados no Terminal, dentro da pasta do projeto:

```bash
cd "/caminho/para/motor-tratamento-vocal-ptbr"
chmod +x setup_mac.sh
./setup_mac.sh
```

O script verifica o Homebrew, instala o FFmpeg quando necessário, cria o
ambiente virtual `env` e instala o PyTorch nightly e as dependências registradas
em `requirements.txt`. Se o Homebrew não estiver instalado, ele mostra a
instrução oficial e encerra antes de alterar o ambiente Python.

## Execução

Em cada nova sessão do Terminal:

```bash
cd "/caminho/para/motor-tratamento-vocal-ptbr"
source env/bin/activate
python app.py
```

Abra [http://127.0.0.1:7860](http://127.0.0.1:7860). No primeiro uso, o Demucs
baixará o modelo `htdemucs`; as execuções seguintes reutilizam o cache local.

A interface oferece presets **Suave**, **Balanceado** e **Intenso**, controles
independentes para os três módulos, players da mix e dos stems, além do relatório
JSON com confiança, motivos de abstenção e métricas de qualidade.

## Segurança e controle de qualidade

- abstenção local por baixa confiança ou stem vocal abaixo de `-55 dBFS RMS`;
- abstenção global se a reconstrução tiver erro RMS relativo pior que `-15 dB`
  ou similaridade normalizada inferior a `0,95`;
- preservação de sample rate, duração e canais da entrada;
- proteção de true peak estimado por oversampling 4x, com alvo de `-1 dBTP`;
- relatório de sample peak, RMS, correlação estéreo e variação espectral.

O true peak atual é uma estimativa técnica e não substitui um medidor certificado
conforme ITU-R BS.1770. Faça sempre a comparação A/B, especialmente com material
já masterizado.

## Testes

Com o ambiente ativo:

```bash
python -m unittest discover -s tests -v
```

Para verificar o MPS:

```bash
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Benchmark CPU × MPS

O harness executa o mesmo modelo e arquivo em processos isolados, registra
tempo, RTF (tempo de processamento dividido pela duração do áudio), pico de RSS,
versões do ambiente e hash da entrada:

```bash
python benchmarks/benchmark_backends.py "/caminho/para/referencia.wav"
```

Por padrão, cada backend recebe um aquecimento descartado e três execuções
medidas. Para um teste rápido:

```bash
python benchmarks/benchmark_backends.py referencia.wav --runs 1 --warmup 0
```

Os relatórios ficam em `benchmarks/results/` e não são versionados, pois podem
conter metadados do ambiente local. O pico de memória representa o RSS do
processo Demucs e seus descendentes; ele não é uma medição dedicada de toda a
memória unificada consumida pela GPU.

## Estrutura

```text
.
├── app.py                  # Pipeline de áudio, decisão, QC e interface
├── benchmarks/
│   └── benchmark_backends.py # Harness reproduzível CPU × MPS
├── docs/
│   └── build_whitepaper.py # Fonte reproduzível do whitepaper
├── output/pdf/             # Whitepaper publicado
├── requirements.txt        # Dependências da aplicação
├── requirements-docs.txt   # Dependências da documentação
├── setup_mac.sh            # Preparação automatizada do macOS
└── tests/                  # Regressão de DSP, alinhamento, reconstrução e QC
```

Para regenerar o whitepaper:

```bash
python -m pip install -r requirements-docs.txt
python docs/build_whitepaper.py
```

## Estado da PoC

Implementado: separação real, MPS com fallback, três módulos DSP regionalizados,
confiança e abstenção, presets, auditoria dos stems, exportação float32, proteção
de pico e QC básico.

Próximos marcos para beta: executar e publicar a matriz CPU × MPS com áudio de
referência, formar um corpus piloto de canto PT-BR, calibrar os limiares,
adicionar loudness BS.1770, chave A/B sincronizada e avaliação perceptual AB/ABX.
