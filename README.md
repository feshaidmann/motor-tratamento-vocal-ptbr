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
- **pyloudnorm** — loudness integrado conforme ITU-R BS.1770-4;
- **Pedalboard** — filtros paramétricos e compressor;
- **NumPy + SoundFile** — arrays, alinhamento, recombinação e WAV float32;
- **Gradio + FastAPI** — painel técnico e superfície pública de audição;
- **Web Audio API** — reprodução A/B/X com relógio e buffers compartilhados;
- **SQLite** — sessões e votos transacionais, com exportação JSONL;
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

Abra o painel técnico em
[http://127.0.0.1:7860/admin/](http://127.0.0.1:7860/admin/). No primeiro uso,
o Demucs baixará o modelo `htdemucs`; as execuções seguintes reutilizam o cache
local. O terminal mostra a credencial administrativa gerada para aquela execução.
Para fixá-la, defina `MOTOR_VOCAL_ADMIN_USER` e `MOTOR_VOCAL_ADMIN_PASSWORD`
antes de iniciar a aplicação.

A interface oferece presets **Suave**, **Balanceado** e **Intenso**, controles
independentes para os três módulos, players da mix e dos stems, relatório JSON
com confiança, motivos de abstenção e métricas de qualidade, além do controle de
publicação das audições cegas.

O painel **Comparação técnica nivelada** gera players dedicados do original e do
processado com loudness integrado igualado. O arquivo processado bruto permanece
separado para download e auditoria, evitando confundi-lo com o estímulo de teste.

## Audição cega A/B/ABX

Depois de processar uma faixa, informe o ID do experimento e clique em **Criar
link de audição**:

- A e B recebem aleatoriamente as condições original e processada;
- X é uma cópia exata de A ou B;
- as quatro combinações A/B × X são balanceadas em blocos por experimento;
- os estímulos são alinhados em amostras e têm duração e canais idênticos;
- o sinal perceptualmente mais alto é atenuado para igualar o loudness integrado
  BS.1770, sem vantagem de volume;
- o participante recebe uma rota isolada `/audicao/<token>`, sem players
  identificados, diagnóstico ou revelação da resposta;
- o painel `/admin` exige autenticação; abra o link participante em uma janela
  privada ou em outro perfil para não reutilizar a sessão do operador;
- o Web Audio API inicia A, B e X no mesmo relógio e alterna ganhos com crossfade
  de 5 ms;
- a resposta ABX, preferência, confiança e observações são gravadas de forma
  transacional em `outputs/auditions/auditions.sqlite3`;
- **Exportar votos JSONL** gera `outputs/auditions/abx_votes.jsonl` a partir do
  banco, sem usar o arquivo como armazenamento primário.

Cada registro inclui semente, mapa cego, hashes dos estímulos, configuração e
diagnóstico do motor, commit e estado Git, hashes de código/dependências e versões
do runtime. O banco impõe um único voto por sessão, inclusive sob repetição de
requisição. Para coleta formal, ative **Modo oficial**; a publicação será recusada
se a árvore Git contiver alterações não registradas.

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

## Validação independente de loudness

O validador compara os estímulos nivelados com dois medidores: `pyloudnorm`,
usado pela aplicação, e o filtro independente `ebur128` do FFmpeg. Também exige
sample rate, canais e número de frames idênticos e verifica o teto de true peak.

Para validar um par:

```bash
python benchmarks/validate_loudness.py \
  --pair original_nivelado.wav processado_nivelado.wav
```

Para um corpus, repita `--pair` no mesmo comando. Os critérios padrão são
diferença residual máxima de `0,2 LU`, divergência máxima de `0,2 LU` entre os
medidores e true peak de até `-0,9 dBTP` (tolerância de `0,1 dB` sobre o alvo da
aplicação). O relatório JSON fica em `benchmarks/results/`; o comando retorna
código diferente de zero se algum par for reprovado.

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

### Baseline inicial (Mac mini M4)

Medição de 14/09/2026 (horário de Brasília) no commit `e45c01f`, com uma
faixa de referência em WAV estéreo, 44,1 kHz e 225 s de duração. Foram três
execuções medidas por backend, após um aquecimento descartado.

| Backend | Tempo mediano | RTF médio | Pico de RSS |
|---|---:|---:|---:|
| CPU | 55,8 s | 0,251 | 3.054 MB |
| MPS | 17,9 s | 0,081 | 2.208 MB |

No tempo mediano, o MPS foi cerca de 3,1 vezes mais rápido que a CPU.

Limites desta baseline:

- mede somente a separação com Demucs; análise, DSP, QC e exportação não entram
  no tempo;
- usa uma única faixa, sem variação de formato, taxa de amostragem ou canais;
- ambiente: macOS 26.6.2, Python 3.13.7, PyTorch 2.15.0.dev20260914 e
  Demucs 4.1.0.

## Estrutura

```text
.
├── app.py                  # Pipeline, painel técnico, API cega e persistência
├── benchmarks/
│   ├── benchmark_backends.py # Harness reproduzível CPU × MPS
│   └── validate_loudness.py  # Validação FFmpeg × pyloudnorm
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
de pico, QC básico e validação independente de loudness.

Medido: baseline inicial CPU × MPS da separação em uma faixa de referência.

Próximos marcos para beta: ampliar a matriz CPU × MPS para mais formatos e o
pipeline completo, formar um corpus piloto de canto PT-BR, calibrar os limiares,
executar a validação BS.1770 no corpus e consolidar a análise estatística das
avaliações perceptuais.
