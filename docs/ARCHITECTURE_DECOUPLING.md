# Arquitetura de desacoplamento

**Status:** proposta técnica para orientar a próxima etapa da PoC
**Atualizado em:** 17/09/2026
**Referência de código:** `main` em `83318f0`

## Objetivo

Separar as responsabilidades do motor de clareza vocal para que o processamento possa continuar rodando localmente, mas futuramente possa usar outra fila, armazenamento ou worker sem reescrever o DSP nem a interface.

Este documento define fronteiras e contratos. Não aprova AWS, SaaS, cobrança, 20 workers ou qualquer SLA.

## Princípios

1. O motor de áudio não conhece a interface web, banco de fila ou provedor de nuvem.
2. A API não executa Demucs diretamente; ela valida, registra e acompanha jobs.
3. Mensagens de fila carregam referências e metadados mínimos, nunca áudio, credenciais ou URLs privadas duradouras.
4. O estado do job é persistido antes de ser comunicado como concluído.
5. Repetição é esperada: cada execução deve ser idempotente por `job_id` e tentativa.
6. SQLite e filesystem permanecem as implementações locais de referência até que medições justifiquem substituição.
7. Cada fronteira deve ser testável com uma implementação falsa ou em memória.

## Estado de implementação

Verificado no código em 17/09/2026. Esta seção existe para que o plano não seja lido como se nada tivesse começado.

| Contrato ou porta | Situação | Onde |
| --- | --- | --- |
| `JobRequest` versionada | **Implementado** — `schema_version` com validação que rejeita versão incompatível | `motor_vocal/jobs.py` |
| `JobRecord` | **Parcial** — tem `job_id`, `idempotency_key`, request, `engine_identity`, `state`, `attempts`, timestamps, `result` e `error`; **falta progresso resumido** | `motor_vocal/jobs.py` |
| `WorkItem` | **Ausente** — não existe mensagem mínima de despacho | — |
| Porta de estado de job | **Implementado com teste de contrato** — `JobStatePort`; `LocalWorker` depende do protocolo, e `LocalJobStore` e um adaptador em memória são exercitados contra ele | `motor_vocal/ports.py`, `tests/test_ports.py` |
| Porta de executor de áudio | **Implementado com teste de contrato** — `AudioExecutorPort` injetável no `LocalWorker`, permitindo rodar o ciclo do job sem Demucs | `motor_vocal/ports.py`, `tests/test_ports.py` |
| Porta de armazenamento | **Ausente** | — |
| Porta de observabilidade | **Ausente** | — |
| Worker executável por CLI | **Ausente** — o worker ainda é iniciado pelo processo web | `motor_vocal/jobs.py` |
| Retenção | **Apenas listagem** — `preview_local_retention` não exclui nada | `motor_vocal/retention.py` |

Consequência prática: da Fase 0 só falta `WorkItem` e o progresso em `JobRecord`. As duas portas do worker existem e são carregadas por teste. Faltam as portas de armazenamento e observabilidade, e o worker executável por CLI.

Uma primeira versão das portas foi criada e removida em 17/09/2026 por não ter segundo adaptador nem teste — parecia código morto. A versão atual nasce com ambos, justamente para não se repetir.

## Componentes e responsabilidades

### 1. Admissão de entrada

Recebe upload ou referência local, valida formato, tamanho, duração, integridade e opções de processamento. Cria uma `JobRequest` versionada e uma chave de idempotência.

Não deve executar Demucs, criar resultados ou conhecer detalhes de S3/SQS.

### 2. Estado e orquestração de jobs

Mantém o ciclo de vida do trabalho, tentativas, identidade da build, erro, progresso e referências aos artefatos. Oferece operações equivalentes às atuais do `LocalJobStore`:

- `enqueue(request, idempotency_key)`;
- `get(job_id)`;
- `claim_next()`;
- atualização de progresso/heartbeat;
- conclusão ou falha condicionada à tentativa;
- recuperação de execução abandonada.

SQLite é o adaptador atual. Um banco remoto futuro deverá preservar essas semânticas, mas não será uma troca direta de driver sem testes de concorrência.

### 3. Armazenamento de objetos e arquivos

Abstrai a origem da entrada, temporários e artefatos finais. O contrato deve permitir:

- salvar ou copiar uma entrada para uma referência privada;
- abrir uma entrada para leitura;
- criar um diretório de trabalho exclusivo;
- publicar artefatos com nomes estáveis;
- listar e remover objetos de um job;
- obter metadados e tamanho;
- limpar parciais de uma tentativa.

Filesystem local é a implementação atual. Um armazenamento de objetos futuro deve manter `job_id` e tentativa como escopo de isolamento.

### 4. Worker

Consome uma unidade de trabalho, carrega a entrada, chama o executor, publica os artefatos, registra diagnóstico/QC e confirma a conclusão. Deve emitir heartbeat durante etapas longas e tratar falhas recuperáveis e permanentes separadamente.

O worker não deve importar Gradio, FastAPI ou componentes de apresentação.

### 5. Executor de áudio

O `motor_vocal.pipeline` recebe uma entrada, opções e diretório de saída exclusivo. Executa separação, alinhamento, análise, DSP, recombinação e QC, retornando um `ProcessingResult` e diagnóstico.

Ele não deve saber se foi chamado por CLI, worker local ou worker remoto. Progresso é uma função opcional; armazenamento é fornecido pelo chamador.

### 6. Audição e avaliação

Usa referências de artefatos para produzir estímulos nivelados, sessões cegas e votos. Não deve alterar o estado de processamento nem depender de caminhos internos do worker além do adaptador de armazenamento.

### 7. Observabilidade

Registra eventos estruturados de admissão, fila, execução, armazenamento, QC e audição. Logs não podem conter áudio, senha, token ou URL assinada completa.

## Fluxo lógico

```text
Entrada
  -> validação/admissão
  -> armazenamento da entrada
  -> estado: queued
  -> despacho de referência do job
  -> worker reclama uma tentativa
  -> estado: running + heartbeat
  -> executor de áudio
  -> publicação de artefatos e diagnóstico
  -> conclusão condicional da tentativa
  -> estado: succeeded ou failed
  -> audição opcional por referências privadas
```

O worker só deve confirmar a mensagem de fila depois que o estado e os artefatos necessários tiverem sido persistidos. Se houver repetição após uma falha de comunicação, a conclusão deve ser segura para repetir.

## Máquina de estados

Estados públicos atuais:

```text
queued -> running -> succeeded
                 \-> failed
running -> queued       (recuperação após stale/timeout)
```

Regras:

- `queued` pode ser reclamado uma única vez por tentativa;
- somente a tentativa atualmente válida pode atualizar progresso ou concluir;
- `succeeded` é terminal para aquela versão do job;
- `failed` conserva erro, tentativa e diagnóstico suficiente para inspeção;
- uma recuperação nunca deve apagar o histórico da tentativa anterior;
- cancelamento ainda não faz parte do contrato v1 e não deve ser simulado como falha.

## Contratos mínimos

### `JobRequest` — implementado

Deve continuar contendo, no mínimo:

- `input_ref` ou equivalente privado;
- preset DSP e módulos habilitados;
- `schema_version`;
- opcionalmente metadados não sensíveis para rastreabilidade.

O contrato não deve carregar bytes de áudio nem credenciais. A compatibilidade de versões deve ser explícita; uma mudança incompatível cria nova versão do contrato.

### `JobRecord` — parcial

Deve expor:

- `job_id` e `idempotency_key`;
- request versionada;
- identidade do engine/build;
- estado, tentativa e timestamps;
- progresso resumido — **pendente**;
- referências dos artefatos, não necessariamente caminhos locais;
- diagnóstico/QC e erro sanitizado.

### `WorkItem` — ausente

Mensagem mínima para o dispatcher:

```json
{
  "job_id": "...",
  "attempt": 1,
  "schema_version": 1
}
```

A entrada, opções e artefatos são obtidos pelo worker usando `job_id`. Isso reduz duplicação e evita colocar dados sensíveis na fila.

## Dependências permitidas

```text
interface web/API -> admission -> job state + object storage + dispatcher
worker            -> job state + object storage + audio executor
audio executor    -> áudio/DSP/QC, sem UI ou fila
audition          -> object storage + estado de audição
```

Dependências proibidas:

- `pipeline.py` importando `jobs.py` ou `app.py`;
- `processing.py` conhecendo caminhos de `outputs/`;
- UI acessando diretamente tabelas internas do SQLite;
- worker usando estado global da aplicação web;
- fila carregando o áudio ou resultado binário.

## Migração incremental proposta

### Fase 0 — documentação e contratos *(quase concluída)*

Fixar nomes, estados, semântica de tentativa e referências de artefatos. Nenhuma mudança de infraestrutura.

Resta: `WorkItem` e progresso resumido em `JobRecord`.

### Fase 1 — extrair portas locais *(em andamento)*

Introduzir protocolos pequenos para estado de job, armazenamento e despacho. Adaptar `LocalJobStore` e os diretórios atuais sem mudar o comportamento observado.

Feito: `JobStatePort` e `AudioExecutorPort`, ambos com teste de contrato. Resta: portas de armazenamento e observabilidade.

Regra aprendida: cada porta entra junto com um segundo adaptador e um teste que o exercite. Uma porta que só tem a implementação real é indistinguível de código morto e será removida por quem passar depois.

### Fase 2 — tornar o worker independente

Mover o ciclo `claim -> execute -> persist -> complete` para um módulo executável por CLI. A aplicação web apenas inicia ou supervisiona o worker local durante a PoC.

O worker deve reclamar uma tentativa, renovar heartbeat, executar `pipeline.py` sem importar UI, publicar artefatos em escopo exclusivo, registrar QC e diagnóstico, concluir somente se a tentativa ainda for válida e separar falhas recuperáveis de permanentes.

### Fase 3 — testes de contrato

Executar a mesma suíte contra implementações locais reais e falsas:

- idempotência;
- mensagem duplicada;
- tentativa antiga tentando concluir;
- falha após gerar parte dos artefatos;
- recuperação de job abandonado;
- limpeza de temporários;
- concorrência de reclamação;
- propagação de progresso e erro sanitizado.

### Fase 4 — adaptadores externos, somente após decisão

Implementar storage/fila/banco externos mantendo os contratos. A troca não deve exigir alteração em `processing.py` ou `pipeline.py`. A escolha de provedor, região e orçamento fica fora deste documento; ver [`AWS_ARCHITECTURE_PROPOSAL.md`](AWS_ARCHITECTURE_PROPOSAL.md).

## Pendências críticas

O desacoplamento é condição necessária, não suficiente. Estas pendências são independentes dele e não devem ser bloqueadas por ele — mas bloqueiam qualquer promessa de produto.

### Qualidade perceptual

- A escuta cega ainda não consolidou preferência, defeitos e abstenções em faixas completas reais.
- ABX mede distinção, mas não substitui preferência.
- O corpus disponível não deve ser tratado como representativo de estilos, timbres ou mixagens.
- Não anunciar qualidade garantida, fonética PT-BR comprovada ou ganho perceptual antes dos resultados.

**Aceite:** relatório por faixa e participante com preferência, abstenção, defeitos audíveis e condição de nivelamento.

### Dados e direitos

- Ainda não há todas as faixas autorizadas para catalogação.
- O corpus atual tem escopo interno; autorização para publicação ou compartilhamento público não está presumida.
- Antes de qualquer transferência para nuvem, registrar finalidade, autorização, hash, duração, formato e responsável.

**Aceite:** cada arquivo do piloto possui autorização verificável e finalidade de processamento registrada.

### Custo, memória e armazenamento

- O smoke test de 5 minutos usou seno sintético, não música real.
- Quatro WAVs float32 de 5 minutos podem ocupar aproximadamente 404 MiB, fora entrada e temporários.
- Faltam medições representativas de pico de RAM/VRAM, tempo por etapa, bytes e custo unitário.
- A retenção atual apenas lista candidatos; não exclui automaticamente.

**Aceite:** benchmark repetível por faixa e backend, incluindo armazenamento, falhas e custo estimado.

### Segurança e privacidade

- O modelo atual é de painel administrativo local, não identidade de clientes.
- Faltam isolamento por usuário, autorização por recurso, links expirados, armazenamento privado e auditoria.
- Segredos não podem aparecer em código, logs, mensagens ou diagnósticos.

**Aceite:** threat model mínimo, matriz de permissões, armazenamento privado e teste de expiração/exclusão.

### Escalabilidade operacional

- SQLite, caminhos locais e worker iniciado pelo processo web não formam uma fila distribuída.
- Cinco envios concorrentes não significam cinco processamentos simultâneos.
- A meta de até 20 processamentos é hipótese comercial, não capacidade entregue.

**Aceite:** carga medida com fila durável, workers idempotentes e limites definidos a partir de dados.

## Ordem recomendada de execução

1. Fechar contratos e estados do desacoplamento (`WorkItem`, progresso em `JobRecord`).
2. Instrumentar benchmark local por etapa.
3. Definir catálogo e direitos para as faixas que já estiverem disponíveis.
4. Implementar retenção e limpeza local — hoje só há listagem.
5. Tornar o worker executável sem a UI.
6. Executar um piloto em nuvem com um job não sensível ou devidamente autorizado.
7. Medir 1 job; depois 5 e 10 jobs, sem prometer capacidade.
8. Reavaliar CPU/GPU, fila, custo e armazenamento.
9. Só então discutir teto de concorrência, preços e produto comercial.

## Critérios de parada

Interromper a expansão se ocorrer qualquer um destes casos:

- artefatos audíveis relevantes ou abstenção excessiva;
- custo incompatível com a hipótese do produto;
- falha de isolamento ou exclusão de dados;
- perda de idempotência ou duplicação de cobrança/resultado;
- memória, VRAM ou armazenamento sem margem operacional;
- direitos insuficientes para o áudio usado no teste.

## Fora do escopo atual

- autenticação multiusuário e cobrança;
- upload direto para nuvem;
- autoscaling ou 20 workers;
- retenção comercial por plano;
- persistência de stems para variantes sem reprocessar;
- alteração do algoritmo DSP ou conclusão sobre qualidade perceptual;
- promessa de latência ou custo.

O código continua local, com SQLite, filesystem e um worker. Não foram aprovados nem implementados: AWS, S3, SQS, banco remoto, autenticação de clientes, cobrança, créditos, 20 workers ou SLA de processamento. As propostas de arquitetura e custo devem ser recalculadas depois de benchmarks com faixas reais autorizadas.

## Critério de aceite arquitetural

A próxima etapa estará concluída quando:

1. o pipeline puder ser executado sem importar a interface web;
2. o worker puder ser executado por uma entrada de linha de comando;
3. fila e armazenamento locais forem apenas adaptadores das portas definidas;
4. a repetição de uma tentativa não duplicar nem corromper o resultado;
5. os testes cobrirem conclusão, falha, stale, idempotência e artefatos parciais;
6. nenhuma decisão de AWS ou de escala for necessária para rodar o piloto local.

## Documentos relacionados

- [`HANDOFF.md`](HANDOFF.md) — estado do produto, limites da validação e como executar.
- [`AWS_ARCHITECTURE_PROPOSAL.md`](AWS_ARCHITECTURE_PROPOSAL.md) — fluxo, serviços candidatos, retenção e segurança do piloto em nuvem.
- [`AWS_MINIMUM_TEST_PLAN.md`](AWS_MINIMUM_TEST_PLAN.md) — estrutura mínima viável para testes.
- [`AWS_SCALING_COST_PROJECTION.md`](AWS_SCALING_COST_PROJECTION.md) — projeção de custo, a recalcular após benchmark real.
- [`FULL_SONG_SMOKE_TEST.md`](FULL_SONG_SMOKE_TEST.md) — evidência do teste local de 5 minutos.
