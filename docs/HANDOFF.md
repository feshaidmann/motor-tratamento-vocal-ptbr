# Handoff — Motor de Clareza Vocal PT-BR

**Atualizado em:** 17/09/2026

**Referência de código:** `main` em `83318f0` (`docs: record five-minute local pipeline smoke test`)

**Natureza:** passagem de contexto técnico e de produto. Propostas comerciais e de AWS abaixo não equivalem a funcionalidades entregues.

## 1. Resumo para quem assume

Esta é uma PoC local de tratamento **seletivo de voz cantada em música pronta**. O Demucs separa voz e instrumental; o motor analisa bandas acústicas candidatas a nasalidade, estridência e sibilância, decide quando agir ou se abster, aplica DSP no vocal, recombina e gera arquivos e diagnóstico. Há comparação A/B com loudness nivelado e infraestrutura de audição cega A/B/ABX para validação.

**Estado atual:** o painel admite WAV/MP3 de até **5 minutos e 150 MB**, com **50 novos jobs/dia**, fila **SQLite local com um worker** e até **cinco eventos de envio concorrentes**. Não há SaaS multiusuário, cobrança, créditos, fila AWS nem teto operacional de 20 processamentos. O teste local end-to-end de 5 minutos usou **seno sintético**, não música real: 28,1 s no Apple M4/MPS, quatro WAVs float32 somando ~404 MiB; esse número não é SLA nem benchmark CUDA/AWS. Em 17/09/2026, `env/bin/python -m unittest discover -s tests -q` passou com **50 testes**.

**Próximo gate:** processar faixas completas reais e autorizadas de 3–5 minutos, medir tempo/memória/armazenamento e realizar preferência cega com volume nivelado. Só então dimensionar e precificar uma operação AWS.

## 2. O que existe hoje

| Componente | Local | Estado |
| --- | --- | --- |
| Painel técnico, API de audição e autenticação administrativa | [`app.py`](../app.py) | Funcional para piloto local; não é autenticação de clientes. |
| Separação Demucs `htdemucs`, seleção CUDA/MPS/CPU e alinhamento | [`motor_vocal/separation.py`](../motor_vocal/separation.py) | Implementado; stems intermediários não são mantidos para reprocessar variantes. |
| Análise heurística, presets e DSP de três módulos | [`motor_vocal/processing.py`](../motor_vocal/processing.py) | Implementado com confiança e abstenção; **não** reconhece fonemas ou palavras em PT-BR. |
| Executor independente da interface | [`motor_vocal/pipeline.py`](../motor_vocal/pipeline.py) | Produz mix, vocal original, vocal corrigido e instrumental em WAV float32, com diagnóstico/QC. |
| Fila e worker locais | [`motor_vocal/jobs.py`](../motor_vocal/jobs.py) | Estados `queued/running/succeeded/failed`, idempotência e recuperação de job abandonado; um worker iniciado pelo processo web. |
| Inventário de retenção | [`motor_vocal/retention.py`](../motor_vocal/retention.py) | **Somente leitura**; não implementa exclusão automática. |
| Audição cega e campanhas | [`app.py`](../app.py), [`motor_vocal/campaign.py`](../motor_vocal/campaign.py) | Votos SQLite, comparação nivelada e exportação; evidência perceptual formal ainda não consolidada para músicas completas. |
| Corpus piloto | `outputs/corpus_ptbr_seed_v1/`, [`benchmarks/build_pilot_corpus.py`](../benchmarks/build_pilot_corpus.py) | Cinco estímulos locais de 30 s; autorização anotada para armazenamento/processamento **internos na AWS**. Não há autorização registrada para publicação/compartilhamento público. |

Fluxo: upload validado → cópia local/idempotência → fila → Demucs → checagem da reconstrução dos stems → análise/abstenção/DSP → recombinação/QC → quatro arquivos → A/B nivelado e, opcionalmente, sessão cega. O QC técnico não comprova melhora perceptual.

## 3. Como executar e verificar

Na raiz do repositório, em macOS Apple Silicon com FFmpeg e ambiente preparado por `./setup_mac.sh`:

```bash
source env/bin/activate
python app.py
```

O painel fica em `http://127.0.0.1:7860/admin/`. O usuário administrativo padrão é `admin`; uma senha temporária é gerada e mostrada no terminal ao iniciar, salvo se `MOTOR_VOCAL_ADMIN_USER` e `MOTOR_VOCAL_ADMIN_PASSWORD` estiverem definidos no ambiente. **Não registrar senhas neste documento nem expor o painel publicamente com esse modelo de acesso.** O modelo Demucs pode ser baixado no primeiro uso.

```bash
env/bin/python -m unittest discover -s tests -q
env/bin/python -m motor_vocal.retention --older-than-days 7
```

O segundo comando apenas lista candidatos antigos; `7` é exemplo de inspeção, **não uma política aprovada**. Os jobs, uploads e artefatos ficam em `outputs/jobs/`; sessões/votos e estímulos de audição em `outputs/auditions/`. `outputs/` não é versionado. Proteja esses dados e confira espaço em disco. O corpus e áudios locais não devem ser enviados ou publicados sem conferir direitos e finalidade.

Utilitários adicionais: [`benchmarks/benchmark_backends.py`](../benchmarks/benchmark_backends.py) mede separação por backend; [`benchmarks/validate_loudness.py`](../benchmarks/validate_loudness.py) confere o nivelamento contra FFmpeg; [`benchmarks/manage_abx_campaign.py`](../benchmarks/manage_abx_campaign.py) cria e resume campanhas internas. Consulte o [`README.md`](../README.md) para comandos e contrato detalhados. A árvore de estrutura/“Estado da PoC” no fim do README contém trechos históricos; prefira o código e este handoff para o estado de jobs e limites atuais.

## 4. Evidência e limites da validação

| Evidência | O que permite concluir | O que **não** permite concluir |
| --- | --- | --- |
| 50 testes automatizados passaram em 17/09/2026 | Contratos e regressões cobertos pela suíte continuam funcionando localmente. | Qualidade musical, segurança de produção ou comportamento de 20 workers. |
| Smoke test end-to-end de 5:00 em M4/MPS: 28,1 s, QC aprovado, ~404 MiB de quatro saídas | A admissão e o pipeline conseguem completar aquela entrada sintética. | Latência de música real, CUDA/AWS, SLA de 40 s ou ganho perceptual; o seno não acionou correções. |
| Baseline anterior com faixa de 225 s: Demucs 17,9 s MPS vs. 55,8 s CPU | Referência local da **separação**, na máquina e faixa medidas. | Tempo end-to-end geral ou custo unitário AWS. |
| Corpus de cinco estímulos de 30 s | Base inicial para ensaio interno autorizado. | Cobertura de estilos/timbres, material de 3–5 min ou licença pública. |

Detalhes e condições do teste longo: [`FULL_SONG_SMOKE_TEST.md`](FULL_SONG_SMOKE_TEST.md). A documentação AWS histórica que modela 30 s não deve ser extrapolada linearmente para músicas completas.

## 5. Decisões, propostas e pendências

| Tema | Estado no handoff | Decisão pendente |
| --- | --- | --- |
| Música completa | Limite local experimental de 5 min/150 MB implementado. | Confirmar limites após medir faixas reais e formatos variados. |
| Capacidade | Hoje um worker local; cinco envios concorrentes não significam cinco processamentos. | Proposta comercial de **até 20 processamentos simultâneos** com demais jobs em fila durável; dimensionar após benchmark. |
| AWS | Há desenho e plano de teste, **sem recursos criados ou áudio transferido**. Conta informada pelo usuário: `633030246669` (identificador, não prova de acesso). | Região, orçamento, permissões e implantação por IaC; primeiro teste pequeno, depois GPU e carga. |
| Preços | Hipótese documentada, sem checkout ou créditos: teste gratuito com 2 músicas **uma vez por conta**, avulso R$ 12,90/música e Básico R$ 39,90/mês até 8 músicas. | Validar demanda, conversão, custo real, impostos/taxas e regras de expiração. A proposta de 2 grátis **mensais** é cenário anterior, não decisão operacional. |
| Retenção | Só existe inventário read-only. | Política para originais, stems, resultados, estímulos e votos; implementar exclusão segura e auditável. |
| Direitos do corpus | Autorização local anotada para armazenamento/processamento AWS internos de cinco itens de Antonio Barra. | Confirmar escopo antes de qualquer uso público ou compartilhamento; obter músicas completas autorizadas para o novo teste. |

Os documentos [`PROPOSTA_PLANOS_E_PRECOS.md`](PROPOSTA_PLANOS_E_PRECOS.md) e [`ESTUDO_EVOLUCAO_PRODUTO_MOTOR_VOCAL.md`](ESTUDO_EVOLUCAO_PRODUTO_MOTOR_VOCAL.md) são estudos de produto, **não** funcionalidades lançadas. Os documentos [`AWS_MINIMUM_TEST_PLAN.md`](AWS_MINIMUM_TEST_PLAN.md), [`AWS_ARCHITECTURE_PROPOSAL.md`](AWS_ARCHITECTURE_PROPOSAL.md) e [`AWS_SCALING_COST_PROJECTION.md`](AWS_SCALING_COST_PROJECTION.md) são planejamento; as estimativas de custo dependem de premissas e precisam ser refeitas com benchmarks de faixas completas e preços vigentes.

## 6. Riscos que bloqueiam um lançamento pago

1. **Qualidade percebida não comprovada:** ABX mede distinção, não preferência. É preciso registrar preferência cega, defeitos e abstenções por música e participante, com nível igualado.
2. **Custo e armazenamento incertos:** quatro WAVs float32 de 5 min ocupam ~404 MiB em 44,1 kHz estéreo, fora upload e temporários. Cinquenta jobs desse porte gerariam ~20 GiB/dia só de resultados. Falta retenção automática e exportação enxuta.
3. **Infraestrutura local não escalável:** SQLite, caminhos locais e worker acoplado ao web não resolvem fila durável distribuída, uploads diretos, identidade por conta, prioridade justa ou 20 execuções reais.
4. **Privacidade e segurança de produção:** áudio inédito e links de audição exigem autenticação por usuário, autorização por recurso, segredos gerenciados, armazenamento privado, expiração e política de exclusão antes de exposição pública.
5. **Promessa comercial prematura:** não anunciar processamento em 40 s, “fonética PT-BR comprovada”, qualidade garantida, capacidade de 20 workers, prioridade paga ou retenção por plano como se já existissem.

## 7. Próxima execução recomendada e aceite

| Ordem | Entrega | Critério de aceite mínimo |
| --- | --- | --- |
| 1 | Matriz de faixas reais autorizadas, 3–5 min, WAV/MP3, estilos/timbres/mixagens variados. | Direitos e finalidade registrados; entrada, duração, formato e hash catalogados; sem publicar áudio por padrão. |
| 2 | Benchmark end-to-end local e em uma GPU CUDA de teste. | Por faixa e etapa: tempo, p50/p95, RAM/VRAM, falhas, QC, bytes de entrada/saída e custo estimado; repetir amostras. |
| 3 | Audição cega de preferência, A/B nivelado e ABX quando útil. | Votos por participante/faixa, relatório de preferência e taxa de artefatos; comparação cega não confundida com volume. |
| 4 | Retenção/exclusão e formato de entrega enxuto. | Política aprovada; exclusão testada para uploads, artefatos e órfãos; download do mix por padrão e stems apenas quando solicitados, se essa regra de produto for aprovada. |
| 5 | Piloto AWS pequeno e depois rajada controlada. | S3 privado, fila durável/DLQ, worker idempotente, observabilidade, orçamento e IaC; medir 1 job, depois carga até o teto decidido, incluindo espera p50/p95 e custo de ociosidade. |
| 6 | Produto comercial. | Identidade/isolamento por conta, créditos transacionais, checkout, limites e retenção por plano, suporte e preço recalculado com dados reais. |

**Critério de parada:** se as faixas reais produzirem artefatos audíveis, abstenção excessiva, custo incompatível ou falhas de privacidade, corrigir o motor/fluxo antes de aumentar capacidade ou publicar preços.
