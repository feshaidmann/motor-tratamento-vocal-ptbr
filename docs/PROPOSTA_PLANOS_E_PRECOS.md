# Proposta de planos e preços — músicas completas

**Status:** proposta de produto revisada, não publicada nem implementada.
**Referência de apresentação:** [página de preços da MuvFlow](https://muvflow.com/v3/precos), consultada em 16/09/2026. A referência inspira a comparação objetiva de benefícios e a explicação de créditos; **não obriga este produto a ter três assinaturas**. O produto aqui é diferente: trata a clareza vocal de áudio enviado pelo usuário; não gera músicas, não distribui para plataformas e não concede licenças ou direitos comerciais sobre obras de terceiros.

## Texto-base para a página de preços

**Título:** Sua música inteira, com a voz mais clara.
**Subtítulo:** Envie uma faixa, compare o original com o resultado em volume nivelado e baixe a versão processada. Teste gratuitamente, pague por música quando precisar ou assine se trabalhar com várias faixas.

| | Teste grátis | Música avulsa | Assinatura Básico |
| --- | --- | --- | --- |
| Preço sugerido | **R$ 0** | **R$ 12,90 por música** | **R$ 39,90/mês** |
| Músicas completas | **2 no total para testar**, sem renovação mensal | 1 por compra | **Até 8 por mês** |
| Duração e arquivo por música | Até 5 min e 150 MB | Até 5 min e 150 MB | Até 5 min e 150 MB |
| Tratamento e comparação A/B nivelada | Incluídos | Incluídos | Incluídos |
| Download do mix processado | Incluído | Incluído | Incluído |
| Downloads técnicos de vocal e instrumental | Não incluídos | Incluídos | Incluídos |
| Fila proposta | Padrão | Prioridade paga, sem prazo garantido | Prioridade paga, sem prazo garantido |
| Histórico de resultados proposto | 48 horas | 7 dias | 7 dias |
| Envios em andamento por conta | 1 | 1 | 1 |

**Destaque do cartão Avulso:** para quem precisa de uma música agora, sem assinatura.
**Destaque do cartão Básico:** para quem processa 4 ou mais músicas no mesmo mês.
**CTA sugeridos:** `Testar 2 músicas grátis`, `Processar uma música`, `Assinar Básico`.

O algoritmo, a qualidade de áudio e a comparação A/B são os mesmos em todas as opções. A diferenciação é o volume, a fila, a retenção e os arquivos disponibilizados. Prioridade não significa início imediato nem prazo garantido. Os limites de armazenamento e de envios por conta são propostas a validar antes da publicação.

### Por que a franquia foi revista

A hipótese comercial anterior era **2 músicas grátis todo mês / 8 no Básico / 24 no Premium**. Ela conflita com a outra hipótese do projeto: **o pagante médio usa 2 músicas por mês**. Para quem usa até duas, a assinatura não traria ganho de volume; dependeria somente de fila, arquivos técnicos e retenção, cuja capacidade de conversão ainda não foi medida. O teste gratuito de **2 músicas completas por conta, uma única vez** preserva uma prova real do produto sem entregar gratuitamente o uso recorrente típico. Esta mudança é uma recomendação, não uma alteração já aprovada na plataforma. Deve ser testada contra a alternativa de 1 ou 2 músicas gratuitas renovadas mensalmente, observando ativação, conversão e retenção.

O Básico mantém **8/mês** como teto para quem trabalha com várias faixas, não como previsão de consumo. O avulso atende o usuário médio de 1–2 músicas/mês: duas compras custariam **R$ 25,80**, menos que a assinatura. A partir de quatro músicas no mesmo mês, o Básico custa menos que quatro compras avulsas (**R$ 39,90 vs. R$ 51,60**). Um Premium fica fora do lançamento até haver evidência de demanda profissional específica. Não oferecer “ilimitado” antes de medir o custo e a carga reais.

## Como funcionam os créditos

- **1 crédito = 1 processamento concluído de uma música inteira**, de até 5 minutos e 150 MB. Prévia, reprodução e novo download do mesmo resultado não gastam outro crédito.
- Os **2 créditos gratuitos são liberados uma vez por conta e não se renovam**. Créditos de assinatura são renovados a cada ciclo mensal; os não utilizados expiram ao fim do ciclo e não se acumulam. Essas regras devem aparecer antes do checkout.
- Na opção avulsa, o pagamento compra um processamento, sem mensalidade. A política de prazo para usar uma compra ainda precisa ser definida e informada antes do pagamento; não presumir que o crédito expire no fim de um mês.
- Um arquivo inválido, acima do limite ou um processamento que falha definitivamente não consome crédito. Uma nova tentativa da mesma música consome crédito somente se gerar um novo resultado a pedido do usuário.
- Pacotes de múltiplos créditos e recargas da assinatura podem ser adicionados depois da validação de custo e pagamento. **Não estão incluídos nesta projeção** nem devem ser exibidos como disponíveis antes de implementados.
- O limite de duração controla a maior parte do trabalho computacional; o limite de tamanho evita uploads excessivos. Ambos devem ser verificados no servidor, não apenas na interface.

## O que fica para uma etapa posterior

A MuvFlow apresenta opções mensais e anuais, mas este produto deve começar sem ciclo anual, Premium ou processamento em lote. Isso reduz promessas antes de medir o custo real de músicas completas, a demanda profissional, cancelamento e renovação. Essas ofertas só devem ganhar preço após validação.

## Fila e capacidade

- **Teto global inicial proposto:** 20 processamentos efetivamente em execução. Os demais aguardam em fila durável. Esse teto não é de usuários conectados e não exige manter 20 GPUs ligadas continuamente.
- Exibir no produto os estados `aguardando`, `processando`, `concluído` e `falhou`; indicar posição ou faixa estimada de espera quando houver dados confiáveis.
- A prioridade de pagantes não interrompe jobs iniciados e deve preservar uma parcela da capacidade para a fila gratuita, evitando espera indefinida.
- Não anunciar resultado em 40 segundos. Se cada música ocupar uma GPU por 4 minutos, 100 envios simultâneos com 20 workers quentes terminam em cinco grupos: cerca de 4 minutos para o primeiro e 20 minutos para o último, antes de inicialização e transferência de arquivos.

## Economia do cenário-base

As cifras abaixo são um exercício, **não uma cotação de produção**. Premissas: 40 mil contas; 15% pagando no mês (6 mil); metade comprando duas músicas avulsas e metade assinando o Básico, ambos com uso médio de 2 músicas/mês; 5 mil novas contas gratuitas por mês, das quais 50% usam os 2 testes — **a divisão entre avulso/assinatura, a aquisição e a ativação são hipóteses novas, não dados observados**; faixa média de 4 minutos; 4 minutos hipotéticos de GPU por faixa; 4 WAVs estéreo float32 de 44,1 kHz gerados e baixados uma vez por job; 7 dias de retenção para todos; dólar de R$ 5,1490. O download de quatro arquivos também no Gratuito e a retenção uniforme de 7 dias são **simplificações conservadoras**, acima do que a tabela oferece aos gratuitos. Tarifas AWS `sa-east-1` de referência: `g4dn.xlarge` US$ 0,894/h, S3 Standard US$ 0,0405/GB-mês e primeira faixa de saída US$ 0,15/GB, conforme [projeção AWS do projeto](AWS_SCALING_COST_PROJECTION.md).

| Indicador | Cenário-base |
| --- | ---: |
| Músicas dos pagantes | 12.000/mês |
| Músicas gratuitas de novos cadastros | 5.000/mês |
| **Total** | **17.000/mês** |
| Receita bruta: 3.000 × (2 × R$ 12,90) + 3.000 × R$ 39,90 | **R$ 197.100/mês** |
| Piso AWS simplificado: GPU, saída, S3 e base web | **~R$ 11.000/mês** |
| Reserva preliminar para AWS, antes de benchmark | **R$ 18.000–30.000/mês** |

Se os 6 mil pagantes comprarem apenas duas músicas avulsas cada, a receita seria **R$ 154.800/mês**; se todos assinarem o Básico, seria **R$ 239.400/mês**. A projeção central de R$ 197.100 é só um meio-termo ilustrativo. O piso AWS desconsidera partidas e ociosidade da frota, EBS, banco de dados, observabilidade, backups, autenticação, requisições e impostos. A reserva é uma faixa de planejamento, não uma garantia. Taxas de pagamento, tributos, suporte, aquisição de clientes e demais despesas da empresa também estão fora da conta. **Receita menos AWS não é lucro.** Os 15% pagantes foram preservados apenas para comparar cenários; mudar a franquia gratuita ou acrescentar avulso pode alterar essa conversão. O custo e a receita precisam ser recalculados com dados reais de ativação, divisão entre avulso/assinatura, uso da franquia e benchmark CUDA de faixas completas.

Se a decisão de produto for **manter 2 músicas grátis renovadas mensalmente**, o cenário anterior de 25% dos 34 mil gratuitos usando ambas volta a ser **29 mil músicas/mês e ~R$ 18 mil/mês de piso AWS**. Mais importante que a diferença de infraestrutura é a provável redução do incentivo para assinar entre usuários que precisam de apenas 1–2 músicas/mês; essa conversão exige teste de mercado, não pode ser inferida do custo da GPU.

Se as 20 GPUs ficarem ligadas 24 horas por dia, seu custo isolado seria cerca de **R$ 67 mil/mês**; a proposta assume escala dinâmica com fila. O preço sugerido não sustenta prometer processamento imediato em todo pico. A entrega padrão deve ser assíncrona e a expectativa de prazo deve vir de medições reais.

## O que falta antes de publicar estes planos

1. Processar músicas reais de 3 a 5 minutos na GPU AWS e medir p50/p95 de duração, memória e falhas.
2. Rever os quatro WAVs atuais: baixar apenas o mix por padrão pode reduzir a saída de dados; arquivos técnicos podem ser solicitados quando necessários.
3. Validar o limite local de 5 minutos/150 MB com músicas completas e implementar contagem transacional de créditos, fila durável, teto global de 20, prioridade justa, estados e estimativa de espera.
4. Definir política de retenção e exclusão, suporte, termos de uso de áudio enviado, prazo de uso da compra avulsa e regras de cobrança/cancelamento.
5. Recalcular preço e margem com custos reais, impostos, taxas de pagamento e uso observado do plano gratuito.

**Estado atual do repositório:** o piloto local agora admite arquivos de até 5 minutos e 150 MB, mas ainda não há benchmark GPU de músicas completas. A fila continua local, com um worker, e não existem planos comerciais, checkout ou controle de créditos. O limite de 20 workers e todas as ofertas desta proposta continuam não implementados.
