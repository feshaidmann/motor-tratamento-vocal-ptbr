# Projeção AWS: 100 usuários no pico e 5.000 processamentos por mês

> Esta projeção usa arquivos de 30 segundos e não representa o produto de
> músicas completas. O piloto local agora admite até 5 minutos/150 MB, ainda
> sem benchmark CUDA dessa duração.

**Data da consulta:** 16/09/2026. **Região usada no exercício:** `sa-east-1` (São Paulo). **Valores:** USD, antes de impostos. Nenhum benchmark CUDA foi feito na AWS.

## Interpretação da demanda

- **100 usuários simultâneos** significa 100 pessoas conectadas ao serviço no pico. Muitas podem estar enviando, aguardando, consultando status ou ouvindo resultados. Isso não exige 100 GPUs.
- **5.000 processamentos/mês** é o volume total, equivalente a cerca de 167 por dia em um mês de 30 dias. É uma hipótese de expansão; o código do piloto atualmente limita a entrada a 50 novos áudios por dia e cinco eventos de envio concorrentes.
- Para dimensionar uma rajada forte, o exercício supõe que os 100 usuários enviem **um áudio cada no mesmo instante**. Assim, 5.000 processamentos correspondem a 50 rajadas de 100 ao longo do mês. Outra distribuição de tráfego muda o número de partidas e as horas ociosas cobradas.
- **60 segundos por áudio em uma GPU** é uma hipótese de cálculo, não uma medição. Cada `g4dn.xlarge` executa um job por vez. O teste CUDA deverá aferir duração, VRAM, RAM e se mais de um job por GPU é seguro.

## Preços oficiais adotados

Na [lista regional EC2](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/sa-east-1/index.json), Linux On-Demand em instância compartilhada custa **US$ 0,894/h** para `g4dn.xlarge` (SKU `94E26KW3K89G64UR`) e **US$ 0,16065/h** para `m7i.large` (SKU `G27QTBRXDXSMKCUJ`). A [lista ELB](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSELB/current/sa-east-1/index.json) indica **US$ 0,034/h** para Application Load Balancer e **US$ 0,011/LCU-h**. A [lista S3](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/sa-east-1/index.json) indica **US$ 0,0405/GB-mês** para S3 Standard na primeira faixa. A [lista AWS Data Transfer](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSDataTransfer/current/sa-east-1/index.json) indica **US$ 0,15/GB** na primeira faixa regional de saída para a internet, depois da franquia global de 100 GB. Conferir novamente os preços antes de contratar.

## Efeito do tamanho da frota no pico

Supondo que a frota GPU parta de zero, leve **5 minutos hipotéticos para iniciar e aquecer** em cada uma das 50 rajadas, e seja desligada logo após o último job:

`último resultado da rajada = 5 min + teto(100 / número de GPUs) × 1 min`

`GPU-horas no mês = 50 rajadas × número de GPUs × tempo cobrado da rajada / 60`

| GPUs no pico | Último resultado dos 100 envios | Custo GPU/mês | Subtotal mensal com web, S3 e saída* |
| ---: | ---: | ---: | ---: |
| 1 | ~105 min | US$ 78 | ~US$ 253 |
| 5 | ~25 min | US$ 93 | ~US$ 268 |
| 10 | ~15 min | US$ 112 | ~US$ 286 |
| 20 | ~10 min | US$ 149 | ~US$ 324 |
| 50 | ~7 min | US$ 261 | ~US$ 435 |
| 100 | ~6 min | US$ 447 | ~US$ 622 |

\*O subtotal soma o custo GPU à base ilustrativa de **~US$ 175/mês** detalhada abaixo. Não inclui EBS, NAT/endpoints, logs, requisições, backup, autenticação e impostos. Um ALB com mais de 1 LCU média e uma API maior elevam o valor. A tabela **não promete** que 100 instâncias serão obtidas e iniciadas em 5 minutos. A AWS pode levar mais tempo para subir capacidade EC2 e contêineres ([ECS cluster auto scaling](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/capacity-cluster-speed-up-ec2.html)).

Se somente 10 dos 100 usuários enviarem áudio ao mesmo tempo, 10 GPUs podem iniciar todos esses jobs após a subida da frota. A escolha da capacidade depende do **tempo máximo aceitável para o usuário receber o resultado**, não apenas da quantidade de usuários conectados.

## Composição da base mensal ilustrativa

Uma API `m7i.large` ligada 730 h/mês é um ponto de partida para servir sessões/status e emitir uploads diretos ao S3; sua capacidade para 100 usuários ainda precisa de teste de carga. Os arquivos de áudio não passam pelo ALB nesse desenho. Consideram-se cinco WAVs estéreo float32 de 44,1 kHz e 30 segundos por job: a entrada e quatro resultados. A retenção de 30 dias é **apenas hipótese desta conta**, não uma política aprovada.

| Parcela | Hipótese | Valor aproximado/mês |
| --- | --- | ---: |
| API `m7i.large` | 730 h × US$ 0,16065 | US$ 117 |
| Application Load Balancer | 730 h × US$ 0,034 + média de 1 LCU × US$ 0,011/h | US$ 33 |
| S3 Standard | ~246 GiB guardados por 30 dias em regime estável | US$ 10 |
| Saída à internet | ~197 GiB dos quatro WAVs, baixados uma vez; primeiros 100 GB disponíveis | US$ 15 |
| **Base arredondada** |  | **~US$ 175** |

Arquivos MP3, entrega apenas do mix, retenção menor, consumo da franquia de saída por outros serviços e tamanho real dos logs alteram essas parcelas. S3, SQS e DynamoDB também cobram requisições. A arquitetura precisa incluir autenticação, idempotência e controle de acesso antes de expor o serviço a usuários.

## O custo de manter GPUs prontas

Cinco mil jobs de 60 s representam **83,3 horas úteis de GPU**: o limite matemático de computação é ~**US$ 74,50/mês**, antes de ociosidade e inicialização. A capacidade pronta custa mesmo sem jobs: **uma `g4dn.xlarge` ligada 730 h custa US$ 653/mês; dez custam US$ 6.526/mês**. Uma GPU permanentemente pronta reduziria o tempo até o primeiro resultado, mas acrescentaria esse custo fixo ao desenho. [Cobrança EC2 On-Demand](https://aws.amazon.com/ec2/pricing/on-demand/).

Se o tempo real por áudio for **30 s**, os 10 workers do cenário de rajadas levariam cerca de **10 min** até o último resultado; se for **120 s**, cerca de **25 min**. A estimativa de custo GPU nesse mesmo desenho iria de **US$ 75 a US$ 186/mês**, respectivamente. A inicialização foi mantida em 5 min em ambos os casos.

## Arquitetura coerente com esse exercício

1. API pequena sempre ligada, com upload direto para S3 e consultas de status. Cem usuários conectados devem ser testados nessa camada, separadamente do processamento.
2. SQS para amortecer rajadas; estado durável e worker idempotente. Um ou mais workers GPU em EC2, com capacidade ajustada à idade da fila e ao tempo máximo de espera.
3. Um escalador converte atraso/tamanho da fila em quantidade desejada de tarefas. ECS com Auto Scaling group/capacity provider pode começar com **zero instâncias GPU** e subir quando houver tarefas; a subida da frota deve ser medida, pois acrescenta espera ([capacity providers](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/asg-capacity-providers.html), [escala com SQS](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/service-autoscaling-queue.html)).
4. Verificar quota e disponibilidade de GPU antes do teste. A quota padrão da família G/VT On-Demand é 0 vCPU por região, mas a quota efetiva da conta pode ser diferente. Dez `g4dn.xlarge` exigem pelo menos 40 vCPU de quota G/VT; 100 exigiriam 400 ([quotas EC2](https://docs.aws.amazon.com/ec2/latest/instancetypes/ec2-instance-quotas.html)).

**Próxima medição útil:** em `g4dn.xlarge`, executar o pipeline completo de 30 s, registrar duração p50/p95, VRAM, RAM, tempo de subida/descida e tamanho real dos artefatos; depois testar 100 sessões web e uma rajada de 100 jobs. Esses números substituem as hipóteses de 60 s/job e 5 min de inicialização.
