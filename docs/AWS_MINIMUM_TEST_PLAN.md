# Estrutura AWS mínima viável para testes

> Cenário histórico de arquivos de 30 segundos. O piloto local passou a admitir
> até 5 minutos/150 MB experimentalmente; não extrapole tempo ou custo deste
> plano sem benchmark de músicas completas.

**Status:** recomendação técnica; nenhum recurso AWS foi criado e nenhum áudio foi transferido.

**Projeção para pico de 100 usuários e 5.000 processamentos/mês:** `docs/AWS_SCALING_COST_PROJECTION.md`.

## Decisão recomendada

Para validar o contrato SaaS e medir o pipeline com o menor custo operacional, começar com **uma única instância Linux EC2**, executando API e um worker em contêineres (Docker Compose ou systemd), apoiada por serviços gerenciados mínimos:

```text
Operador --(SSM port forwarding)--> EC2: API + worker
                                  |-- S3 privado (originais/resultados)
                                  |-- SQS Standard + DLQ (somente job_id)
                                  |-- DynamoDB on-demand (estado/idempotência/ABX)
                                  |-- CloudWatch (logs/métricas)
                                  |-- SSM + IAM (acesso administrativo)
```

S3 deve permanecer privado com bloqueio de acesso público; o áudio não deve trafegar na mensagem SQS. SQS cobra por requisição e não possui tarifa mínima, e DynamoDB on-demand cobra por leitura/escrita consumida, o que evita manter capacidade reservada em um piloto pequeno ([SQS](https://aws.amazon.com/sqs/pricing/), [DynamoDB](https://aws.amazon.com/dynamodb/pricing/), [S3 Block Public Access](https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-block-public-access.html)).

## Dois perfis de teste

| Objetivo | Compute | Uso | Decisão que permite |
| --- | --- | --- | --- |
| Smoke test do contrato (1–2 trabalhos) | EC2 CPU pequena, por exemplo `c7i.large`/equivalente disponível | Validar upload, fila, idempotência, QC, entrega e retenção | Corrigir integração sem pagar GPU |
| Medição representativa do Demucs | Uma EC2 GPU por janela curta, por exemplo família G4/G6 disponível | Medir 1 job e testar 2–3 jobs por GPU apenas se RAM/VRAM permitirem; depois testar 5 e 10 jobs com a quantidade necessária de instâncias | Comparar CPU/CUDA e dimensionar o worker |

O segundo perfil deve ser ligado somente durante a medição e parado ao terminar. ECS/Fargate não é o ponto de partida para Demucs: `gpu` não é parâmetro válido em tarefas Fargate; ECS com GPU exige capacidade EC2 compatível ([diferenças de tarefas Fargate](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/fargate-tasks-services.html), [ECS GPU](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/ecs-gpu.html)). ECS pode ser a próxima etapa quando houver necessidade real de réplicas, rollout e autoscaling.

## O que fica fora do primeiro teste

- RDS/Aurora, ALB, NAT Gateway, CloudFront, EKS, ECS Service Auto Scaling e multi-AZ.
- VPN, bastion host e SSH aberto à internet. Usar Session Manager/port forwarding; ele permite administrar a instância sem portas de entrada, bastion ou chaves SSH ([Session Manager](https://docs.aws.amazon.com/systems-manager/latest/userguide/session-manager.html)).
- URLs públicas permanentes para originais ou estímulos ABX. Quando necessário, emitir URLs assinadas de curta duração.

Essa redução não é uma arquitetura de produção: é deliberadamente um ambiente descartável, de uma região e um host, suficiente para obter números de desempenho e validar as interfaces.

## Segurança e controle de custo obrigatórios

1. Uma conta/role de execução com permissões mínimas separadas para API, worker e audição; sem credenciais no código ou nos logs.
2. Bucket S3 com Block Public Access, criptografia padrão e prefixos por `job_id`; lifecycle somente depois de aprovar retenção para originais, resultados, estímulos e votos.
3. SQS Standard com DLQ, `visibility timeout` maior que o processamento máximo e consumidor idempotente.
4. DynamoDB com chaves para estado do job e unicidade de voto; habilitar TTL apenas após definir a política de retenção.
5. CloudWatch com logs de erro, idade da mensagem na fila, duração do processamento e uso de memória/GPU. Não registrar áudio, senha, token ou URL assinada.
6. AWS Budgets com alerta de custo previsto/realizado e alarme operacional para desligar o host GPU. O monitoramento de budgets é gratuito; as primeiras duas ações habilitadas também são gratuitas ([AWS Budgets](https://aws.amazon.com/aws-cost-management/aws-budgets/pricing/)).
7. Tags obrigatórias (`Project`, `Environment=test`, `Owner`, `AutoStop`) e rotina explícita de parada/terminação ao fim da janela.

## Sequência de execução

1. Decidir região (manter `sa-east-1` se residência no Brasil for requisito; comparar disponibilidade e preço antes de lançar) e teto de gasto do experimento.
2. Criar S3, SQS/DLQ, tabela DynamoDB, IAM, Budgets e EC2 via IaC; ainda sem copiar o corpus.
3. Subir a mesma imagem do worker usada localmente e executar um único WAV de teste não sensível.
4. Repetir com o corpus autorizado somente após validar acesso interno, retenção e logs.
5. Rodar cargas de 1, 5 e 10 jobs; registrar espera na fila, tempo de execução, falhas/retries, memória/GPU, bytes e custo por áudio.
6. Escolher CPU ou GPU e só então decidir se vale migrar a execução para ECS/EC2, adicionar ALB ou adotar banco PostgreSQL gerenciado.

O custo mensal será dominado pelo tempo ligado da EC2 (especialmente GPU); S3, SQS e DynamoDB acompanham o uso e não exigem capacidade fixa. Consultar o preço da família e da região escolhidas no [EC2 On-Demand](https://aws.amazon.com/ec2/pricing/on-demand/) e no [AWS Pricing Calculator](https://calculator.aws/), sem transformar preço de outra região em compromisso.
