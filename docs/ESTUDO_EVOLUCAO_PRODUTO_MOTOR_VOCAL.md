# Estudo de evolução do produto a partir do motor de clareza vocal

**Data:** 16/09/2026. **Natureza:** análise de produto e viabilidade técnica, não promessa comercial nem especificação aprovada.

## Conclusão executiva

O posicionamento mais defensável é **revisão e tratamento seletivo da voz cantada dentro de uma música pronta**, com controle humano e comparação auditiva justa. Separação de stems isoladamente já é uma função difundida. A vantagem potencial do projeto está na combinação de detecção localizada, decisão de se abster, preservação da base instrumental, versões comparáveis e fluxo especializado em voz cantada em português brasileiro. Essa vantagem **ainda precisa ser demonstrada perceptualmente**; as bandas acústicas atuais não identificam fonemas nem garantem melhora.

Prioridade recomendada:

1. Transformar o pipeline de 30 segundos em um serviço confiável para músicas completas, com fila e custo medido.
2. Transformar a saída técnica atual em uma experiência de **revisão por eventos**: ouvir onde o motor agiu, ajustar intensidade e comparar versões sem reenviar a faixa.
3. Só depois ampliar DSP, fluxo profissional e integrações. Não disputar, de início, geração musical, masterização completa, clonagem de voz ou DAW generalista.

## 1. Base real do produto

| Capacidade | Estado no repositório | Consequência para novas funções |
| --- | --- | --- |
| Separação de `vocals` e `no_vocals` com Demucs `htdemucs` | Implementada em `motor_vocal/separation.py` | Permite tratamento vocal e recombinação; não equivale a separação pronta de bateria, baixo, piano etc. |
| Análise heurística de energia por bandas e eventos temporais | Implementada em `motor_vocal/processing.py` | Permite exibir trechos candidatos e intensidade; não é classificador fonético de português. |
| DSP de nasalidade, estridência e sibilância; três presets; módulo liga/desliga | Implementado | Base para controles mais simples e versões alternativas, mas a eficácia precisa de audição cega. |
| Abstenção por confiança e qualidade de reconstrução dos stems | Implementada | Pode sustentar explicação transparente; não deve virar selo automático de “resultado aprovado”. |
| Mix processado, vocal original isolado, vocal tratado e instrumental em WAV float32 | Implementados em `motor_vocal/pipeline.py` | Base para exportação profissional; quatro arquivos por música elevam armazenamento e tráfego. |
| Comparação original/processado com loudness nivelado e testes cegos A/B/X | Implementada no painel técnico | Base valiosa para revisão e avaliação; a infraestrutura de audição interna não é, ainda, um produto multiusuário. |
| Músicas completas, pagamento, créditos, 20 jobs simultâneos e fila em nuvem | **Não implementados**; piloto limita a 30 segundos e um worker local | São pré-requisitos de oferta comercial, não funcionalidades já vendáveis. |

O comando `--two-stems=vocals` do Demucs **não torna a inferência mais rápida nem economiza memória**, segundo a [documentação do próprio Demucs](https://github.com/facebookresearch/demucs). O pipeline atual também apaga os WAVs intermediários da separação; para gerar novas versões sem repetir o modelo, será preciso persistir temporariamente os stems e a análise, com controle de acesso, retenção e expiração.

## 2. O que o mercado já oferece — e a oportunidade

| Referência | Capacidades publicadas | Leitura para este produto |
| --- | --- | --- |
| [Moises](https://moises.ai/pt/features/) | Separação de stems, mudança de tom/tempo, detecção musical, masterização e estúdio criativo. | Não competir como conjunto genérico de ferramentas musicais; focar no resultado específico de voz cantada e no fluxo de revisão. |
| [LALAL.AI](https://www.lalal.ai/pricing/) | Separação e limpeza, minutos de fila rápida, download pago e recargas. | Fila/volume são mecanismos de preço, não diferenciação técnica durável. Tempo de processamento e transparência de consumo importam. |
| [iZotope RX](https://www.izotope.com/pages/rx-compare) | De-ess, de-reverb, de-plosive, reparo espectral e processamento em lote. | Reparo profundo já tem referência profissional; nosso possível diferencial é automação acessível e direcionada à música cantada, não substituir um editor especializado. |
| [Adobe Podcast](https://helpx.adobe.com/podcast/adobe-podcast-faq.html) | Limpeza e aprimoramento de **fala**, com transcrição e edição web. | Podcast/locução é um mercado adjacente, mas não deve ser extrapolado da validação de canto. |

**Inferência estratégica:** o usuário provavelmente compra “minha voz ficou mais clara sem estragar a música e consigo conferir isso”, não “acesso ao Demucs”. A maior oportunidade inicial é tornar esse julgamento rápido e confiável. Não foi feita pesquisa direta de disposição a pagar; as referências mostram capacidades oferecidas, não provam demanda pelo nosso produto.

## 3. Oportunidades diretamente ligadas ao motor

Escala de esforço relativo: **P** pequeno, **M** médio, **G** grande. Não são estimativas de prazo. “Mesmo motor” significa reaproveitar separação, análise e DSP atuais; pode ainda exigir interface, persistência e infraestrutura novas.

| Prioridade | Funcionalidade | Valor para o usuário | Reuso e trabalho novo | Risco principal / critério de liberação |
| --- | --- | --- | --- | --- |
| **0** | Música inteira com limite por duração/tamanho | Trata a obra real, sem recorte | Motor atual; **G** em memória, upload, fila, timeout e custo | Benchmark de 3–5 min em GPU, inclusive pico, sem artefatos de fronteira. |
| **0** | Linha do tempo dos eventos detectados | Mostra onde o motor interveio ou se absteve; usuário revisa apenas pontos relevantes | Usa `regioes_sustentadas` e decisões existentes; **M** de UI e mapeamento | Não chamar eventos de “fonemas errados” nem sugerir diagnóstico clínico. |
| **0** | A/B nivelado por evento, com bypass instantâneo | Facilita julgar melhora real sem viés de volume | Reaproveita comparação atual; **M** de player e sincronismo | Alinhamento exato e mesmo loudness na comparação, preservando o arquivo bruto. |
| **0** | Gerar até algumas variantes da **mesma** faixa | Usuário compara Suave/Balanceado/Intenso sem novo upload | Reusa stems e análise; **M/G** para cache, re-render e versionamento | Não refazer Demucs em cada variante; controlar retenção e custo. |
| **0** | Exportação enxuta: mix como padrão, stems sob demanda | Download mais simples e menor tráfego | Saídas já existem; **M** para formatos/entrega | Preservar WAV sem perdas para uso profissional; verificar conversões e metadados. |
| **0** | Explicação simples do processamento | Dá confiança quando o motor age pouco ou nada | Usa status de abstenção e QC; **P/M** de linguagem e UI | Evitar “nota de qualidade” não validada. |
| **1** | Fader de presença vocal e nova mixagem | Permite trazer a voz para frente ou atrás sem mexer no instrumental | Reusa dois stems; **M** de ganho/automação/re-mix | Pode revelar vazamento dos stems ou alterar a intenção artística; comparar loudness. |
| **1** | Ajustes por região, com desfazer e versão | Corrige apenas um refrão ou frase problemática | Reusa eventos e envelopes; **M/G** de editor e render | Transições audíveis, manipulação excessiva e complexidade de UX. |
| **1** | Nivelamento suave da voz entre seções | Reduz versos baixos/refrões agressivos | Reusa stem vocal; **M/G** de detector de loudness e DSP | Não achatar dinâmica musical nem confundir volume com inteligibilidade. |
| **1** | Relatório para produtor | Mostra versão, módulos, eventos, limites de pico e arquivos usados | Dados de análise/QC já existem; **M** de apresentação/exportação | Métricas objetivas não são prova de melhora perceptual. |
| **1** | Projetos, versões e compartilhamento privado | Facilita feedback entre artista e produtor | Reusa jobs/resultados; **G** em identidade, permissão e retenção | Segurança de áudio inédito, links expiráveis, remoção e auditoria. |
| **1** | Lote/álbum para estúdios | Economia de trabalho operacional | Mesmo pipeline por item; **M/G** em fila, progresso e cobrança | Pode concentrar picos e reduzir qualidade de revisão; liberar só após demanda medida. |

**Cobrança recomendada para variantes:** se uma música já foi separada e o usuário apenas muda preset ou ajusta uma região, isso não deveria consumir automaticamente outro crédito de “música completa”. Definir um número justo de versões incluídas por processamento e custo adicional apenas se o uso real justificar. A implementação deve evitar repetir a etapa GPU mais cara; o benefício técnico de cache versus custo de armazenamento precisa ser medido.

## 4. Extensões de DSP: possíveis, mas não presentes hoje

| Ideia | Por que interessa | Nova capacidade necessária | Condição antes de vender |
| --- | --- | --- | --- |
| Controle de plosivas, respirações e ruído de boca na voz cantada | Resolve defeitos pontuais de gravação | Detectores e módulos DSP específicos, com opção de abstenção | Corpus rotulado e avaliação de perda de expressividade. |
| Redução de reverberação ou vazamento de outros instrumentos no vocal | Pode melhorar material gravado em ambiente difícil | Provável modelo adicional e avaliação de artefatos; não é simples EQ | Teste separado por tipo de gravação, custo GPU e comparação perceptual. |
| Preservação/realce de consoantes do português cantado | Alinha melhor a promessa de clareza de letra | Alinhamento fonético/lyrics e/ou modelo próprio; corpus PT-BR licenciado | Demonstrar que detecta eventos linguísticos reais, não só energia em bandas. |
| Qualidade adaptativa por gênero/timbre | Evita preset único para sertanejo, gospel, pop, voz grave/aguda etc. | Dados segmentados, calibração de parâmetros e controle de viés | Melhorar preferência sem piorar subgrupos; reportar falhas. |
| Reparos locais sugeridos automaticamente | Poupa tempo de escuta | Modelo de ranking de eventos + feedback do usuário | Baixa taxa de falso positivo e correções reversíveis. |

Essas extensões **não são apenas “ligar uma opção” no motor atual**. Algumas compartilham o stem vocal e a interface de revisão, mas exigem algoritmos, dados de treino e validação adicionais.

## 5. Ideias que não priorizaria agora

- **Masterização completa:** demanda equilíbrio tonal, dinâmica e entrega de toda a música, outro escopo de produto; Moises e iZotope já oferecem soluções amplas.
- **Afinação automática, harmonias, clonagem ou conversão de voz:** outra proposta artística, novos modelos/dados e riscos de identidade/consentimento. Pode desviar a marca de “preservar a interpretação”.
- **Karaokê ou separador de stems como produto principal:** o motor consegue fornecer vocal/instrumental, mas a separação genérica já é amplamente ofertada; usá-la como função auxiliar.
- **Transcrição de letras e diagnóstico fonético prometido:** as bandas MIR atuais não reconhecem palavras nem fonemas. Exige uma linha de pesquisa própria, especialmente para canto.
- **Tratamento em tempo real ou plugin VST:** a arquitetura atual é processamento assíncrono de arquivo inteiro; baixa latência exigiria redesenho e benchmark específicos.
- **Podcasts/locução como mesma oferta:** o domínio acústico e os critérios de sucesso diferem do canto. Pode ser um produto adjacente somente após validação própria.

## 6. Proposta de sequência e decisões de investimento

| Etapa | Entrega testável | Gate para avançar |
| --- | --- | --- |
| **A. Produto básico confiável** | Faixas de até 5 min/150 MB, fila durável com teto, download do mix, A/B nivelado e política de retenção | Medir p50/p95 de tempo/custo, falhas, memória, tamanho de arquivos e qualidade em músicas inteiras. |
| **B. Revisão inteligente** | Linha do tempo, explicações de abstenção, variantes da mesma música e escuta por evento | Usuários conseguem identificar se/onde houve melhora; variantes não multiplicam custo GPU desnecessariamente. |
| **C. Fluxo profissional** | Stems sob demanda, relatório, projetos privados e lote | Evidência de uso recorrente por produtores/estúdios e suporte operacional sustentável. |
| **D. Pesquisa de novos módulos** | Plosivas, controle de nível ou fonética PT-BR, um problema por vez | Corpus adequado, avaliação cega e ganho perceptual significativo sem taxa excessiva de dano. |

Uma integração com DAW ou API pode vir após C, se o fluxo web demonstrar valor. Fila, pagamentos, controles de acesso e exclusão de áudios são **infraestrutura de produto**, não argumentos de venda isolados.

### Como isso poderia aparecer nos planos

| Oferta | Benefício principal | Funções futuras que realmente justificariam cobrança |
| --- | --- | --- |
| Teste gratuito | Ouvir o tratamento em uma música própria e baixar o mix | Revisão simples de eventos e A/B justo; não reduzir a qualidade do algoritmo para criar contraste artificial. |
| Música avulsa | Resolver uma faixa sem compromisso mensal | Arquivos técnicos, poucas variantes do mesmo upload e histórico curto. |
| Assinatura para criador recorrente | Fluxo de trabalho repetido e previsível | Franquia mensal, projetos/versões, revisão detalhada e prioridade de fila **se implementada**. |
| Oferta para estúdio — somente depois | Economia operacional em várias faixas e colaboradores | Lote, relatórios, compartilhamento privado, gestão de projetos e eventual API/integração com DAW. |

O preço não deve depender de prometer “som profissional garantido”. A qualidade central deve ser a mesma para todos; as diferenças pagas devem estar em volume, controle e fluxo. O desenho atual de planos está em [Proposta de planos e preços](PROPOSTA_PLANOS_E_PRECOS.md) e também é apenas hipótese.

## 7. Como decidir se cada função merece entrar no preço

**Validação perceptual:** usar o corpus e a infraestrutura A/B/X atuais com músicas completas, volume nivelado e diversidade de estilos/timbres. ABX mede se as versões são distinguíveis; **preferência cega** mede qual é escolhida; uma não substitui a outra. Registrar também casos em que o motor deveria abster-se e os defeitos percebidos (voz metálica, perda de consoantes, alteração de timbre, vazamento do instrumental). Repetir por participante, música e condição para não inflar evidência com votos correlacionados.

**Validação de uso:** entrevistar cantores e produtores, observar onde voltam a ouvir/ajustar, testar duas músicas gratuitas e medir conversão para avulso, proporção de resultados baixados, repetição de uso e pedidos de versão. Uma função só vira benefício de plano se alterar comportamento ou disposição a pagar, não apenas por ser tecnicamente possível.

**Validação econômica:** acompanhar GPU-segundos por faixa, horas ociosas, taxa de separação reutilizada, bytes armazenados/baixados por artefato e tempo de fila p95. Hoje o pipeline grava quatro WAVs float32; em uma faixa de 4 minutos, eles somam aproximadamente **339 MB decimais** a 44,1 kHz estéreo. Dar prioridade a download do mix e gerar/entregar arquivos técnicos sob demanda pode reduzir tráfego. Novos modelos e múltiplas separações exigem orçamento próprio.

## Recomendação final

O **primeiro pacote de funcionalidades futuras** deveria ser: música inteira, revisão temporal dos eventos, A/B sincronizado, variantes sobre stems reutilizados e exportação enxuta. É a combinação que mais transforma o motor atual em benefício perceptível sem tentar competir com plataformas musicais generalistas. Depois, decidir entre fluxo profissional e novo DSP usando evidência de usuários e testes cegos. **Não publicar alegações de clareza “comprovada” ou de tratamento fonético enquanto essa validação não existir.**
