# Plano de preparação para AWS

> O limite de 30 segundos descrito abaixo é o baseline histórico deste plano.
> A admissão local foi ampliada experimentalmente para 5 minutos/150 MB; a
> prontidão AWS para músicas completas ainda não foi demonstrada.

O desenho de serviços e as decisões abertas estão em `docs/AWS_ARCHITECTURE_PROPOSAL.md`.
Para a alternativa de menor custo e risco operacional no primeiro teste, consulte também `docs/AWS_MINIMUM_TEST_PLAN.md`.

## Parâmetros confirmados pelo usuário

- Piloto: até 50 áudios por dia, cada um com até 30 segundos.
- Até cinco usuários podem enviar áudios simultaneamente.
- Começar com um worker e uma fila; medir antes de ampliar a capacidade.
- Validar cargas de 1, 5 e 10 processamentos simultâneos; 10 é teste de pico, não capacidade reservada.
- Não há meta de tempo de resposta aprovada. A estimativa inicial de 8 segundos foi descartada.
- Os cinco itens do corpus foram informados pelo usuário como de Antonio Barra. O usuário confirmou autorização para armazenamento e processamento internos na AWS; publicação e compartilhamento públicos não foram confirmados.

## Estado verificado antes do desmembramento

- `app.py` reúne interface Gradio/FastAPI, processamento de áudio, persistência ABX e exportação.
- Demucs usa CUDA, MPS ou CPU; a seleção e o fallback estão em `motor_vocal/separation.py`.
- O pipeline e o corpus piloto passaram nas validações técnicas locais. Não há benchmark em CUDA na AWS.
- O banco ABX local é SQLite e guarda caminhos de arquivos locais. Na última inspeção havia zero sessões e zero votos; a audição satisfatória foi relatada pelo usuário, sem registro ABX no banco.
- `outputs/` é ignorado pelo Git. A anotação de autorização do corpus é local e precisa de uma fonte durável antes de qualquer migração de dados.

## Sequência de implementação

1. **Implementado localmente:** análise, DSP e controle de qualidade estão em `motor_vocal/processing.py`; a interface mantém as funções usadas pelos testes existentes.
2. **Implementado localmente:** `motor_vocal/pipeline.py` executa o áudio sem importar Gradio/FastAPI, grava quatro WAVs por trabalho e devolve diagnóstico/QC. A interface atual chama esse executor. O contrato v1 em `motor_vocal/jobs.py` registra entrada, preset, módulos, estado, artefatos, diagnóstico e impressão digital das fontes/dependências/runtime.
3. **Implementado para desenvolvimento local:** fila SQLite em `motor_vocal/jobs.py` com chave de idempotência, atribuição transacional, estados `queued`/`running`/`succeeded`/`failed` e `LocalWorker.run_once()`. O painel envia trabalhos, mostra ID/estado, permite retomar a consulta pelo ID e recebe os quatro artefatos ao concluir. O processo web inicia um worker. Os testes cobrem dez reclamações concorrentes de trabalhos; isso não mede dez processamentos de áudio simultâneos.
4. Implementar adaptadores AWS para fila, objetos e banco transacional depois de escolher serviços e região. Credenciais e segredos não devem aparecer em logs.
5. Medir o pipeline em CPU e CUDA no ambiente AWS com 1, 5 e 10 trabalhos concorrentes. Registrar espera na fila, processamento, falhas, memória e custo por áudio; escolher a capacidade a partir dessas medidas.

## Verificação após a separação

- `env/bin/python -m unittest discover -s tests -q`: 44 testes passaram após a integração da interface, a mudança de caminho dos artefatos e a prévia de retenção em 2026-09-16.
- O executor independente processou um estímulo WAV estéreo de 30,0 s do item 001 do corpus com Demucs em MPS: 6,27 s observados nessa execução, WAV float32 de 30,0 s, separação confiável e QC de saída aprovado. Esse resultado local não estabelece latência ou custo na AWS.
- Uma execução separada do mesmo estímulo pela fila local passou de `queued` a `succeeded` em uma tentativa, persistiu os quatro caminhos de artefatos e o diagnóstico com QC aprovado. Foram 5,04 s observados nessa execução local; não é benchmark de concorrência.
- O fluxo de envio local, cópia do upload e worker também processou o estímulo de 30 s após a integração: `queued` → `succeeded`, uma tentativa, quatro artefatos e QC aprovado; 6,30 s observados nessa execução MPS. O armazenamento foi posteriormente movido para `outputs/jobs/`, sem alterar o processamento.
- `env/bin/python -W error::ResourceWarning -m unittest discover -s tests -p test_jobs.py -v`: seis testes passaram, incluindo limite diário concorrente, recuperação e proteção contra conclusão de tentativa antiga.
- O aplicativo FastAPI com Gradio montado foi iniciado em processo de teste com diretórios temporários; seu worker consumiu um trabalho local e o levou a `succeeded` em uma tentativa. Nesse teste o DSP foi substituído por um resultado controlado para verificar apenas o ciclo de vida da aplicação.

## Lacunas antes do piloto em nuvem

- O painel usa a fila local. O limite de 50 novos trabalhos por dia usa o calendário de `America/Sao_Paulo`; repetição da mesma chave de idempotência não consome nova vaga. Uploads WAV/MP3 acima de 30 segundos são rejeitados antes de entrar na fila. O evento de envio admite até cinco submissões concorrentes; isso não limita nem mede as transferências de upload feitas pelo navegador.
- O banco da fila, os uploads copiados e os quatro WAVs por trabalho ficam em `outputs/jobs/`, ignorado pelo Git. Falta definir retenção e limpeza de artefatos, inclusive tentativas parciais; o volume em disco pode crescer durante o piloto.
- `motor_vocal/retention.py` oferece uma prévia somente leitura dos trabalhos antigos, artefatos e uploads não compartilhados. Exemplo: `env/bin/python -m motor_vocal.retention --older-than-days 7`; sete dias aqui são apenas um parâmetro de inspeção, não política aprovada. Na inspeção de 2026-09-16, `outputs/jobs/jobs.sqlite3` não existia e não havia candidatos na fila local.
- O worker local renova sua atividade a cada cinco segundos. Trabalhos em `running` sem renovação por mais de 15 minutos voltam à fila com nova tentativa; uma tentativa antiga não pode concluir a nova. Ainda faltam política definitiva de repetição, retenção e limpeza de arquivos parciais.
- A arquitetura atual inicia um worker por processo web. Executar vários processos web locais criaria vários workers; o piloto AWS deverá ter a contagem controlada pelo serviço de workers.
- A impressão digital identifica arquivos fonte, versões de pacotes, Python e plataforma. Ela não representa uma imagem de contêiner imutável nem substitui um identificador de release para AWS.
- Áudios, resultados, sessões ABX e votos ainda usam caminhos locais; armazenamento remoto, autenticação, retenção, observabilidade e banco gerenciado aguardam desenho e decisão.
- `app.py` ainda usa autenticação administrativa local. Se a senha vier de `MOTOR_VOCAL_ADMIN_PASSWORD`, seu valor não é registrado no terminal; uma senha gerada para uso local é exibida uma vez para permitir acesso. O tratamento de segredo e o modelo de acesso precisam ser revistos antes de qualquer exposição na nuvem.

## Limites desta fase

- Nenhuma implantação, transferência de áudio para AWS, migração de banco, commit ou publicação.
- Os metadados ainda vazios de variante PT-BR, região, gênero e perfil vocal não serão inferidos.
- Benefício perceptual formal depende de votos ABX registrados; a avaliação informal relatada não substitui essa evidência.
- Escolha de ECS, Batch, SQS, S3 ou banco gerenciado permanece recomendação técnica até uma decisão explícita.
