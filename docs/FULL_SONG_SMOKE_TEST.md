# Smoke test local — faixa de 5 minutos

**Data:** 16/09/2026. **Revisão testada:** `bf25aaa`. **Máquina:** Apple M4, 16 GiB de memória; backend MPS. Este é um teste de execução, **não de qualidade musical nem de capacidade AWS**.

## Ensaio

- Entrada sintética: seno de 440 Hz, 5:00 exatos, estéreo 44,1 kHz, WAV PCM16 (~50 MiB). Nenhuma música real ou dado de usuário foi usado.
- Chamada do pipeline completo `process_audio_file`: Demucs `htdemucs`, análise, DSP, recombinação, quatro WAVs float32 e controle de qualidade.
- Resultado: **28,1 s** de tempo de parede, sem falha; separação considerada confiável e QC de saída aprovado. Nenhuma das três correções foi aplicada ao sinal sintético, como esperado para este material.
- Saídas: quatro WAVs de cerca de 101 MiB cada, **~404 MiB** no total (~423 MB decimais). Isso não inclui o upload e os arquivos temporários usados durante a separação.
- A suíte `unittest discover -s tests` passou com **50 testes**, incluindo entrada de exatamente 5 minutos, rejeição de 5:01 e limite de tamanho de arquivo.

## O que este resultado não demonstra

O seno não representa canto, instrumentos, sibilância ou vazamento entre stems. O tempo de 28,1 s no MPS não prevê tempo em CUDA, custo AWS, tempo de inicialização, memória de pico, qualidade perceptual ou resultado com 20 jobs concorrentes. A admissão local de 5 minutos/150 MB é experimental e não implica SLA de 40 segundos.

## Próximo gate de validação

1. Selecionar músicas completas de 3–5 minutos com permissão para teste, cobrindo estilos, timbres, mixagens e WAV/MP3.
2. Medir no pipeline completo tempos por etapa, pico de RAM/VRAM, falhas, tamanho de upload/saídas e QC em MPS e em GPU CUDA da AWS.
3. Fazer testes cegos de preferência com loudness nivelado; ABX isolado só mostra distinguibilidade, não melhora.
4. Ensaiar uma rajada até o teto proposto de 20 workers e fila de 100 envios, registrando espera p50/p95 e custo total da capacidade ociosa.
5. Definir retenção automática antes de operação continuada: 50 faixas de 5 minutos poderiam gerar da ordem de **20 GiB/dia apenas nos quatro resultados**, se todas ocuparem o limite e usarem 44,1 kHz estéreo float32.
