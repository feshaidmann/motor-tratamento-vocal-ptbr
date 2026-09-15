"""Gera o whitepaper técnico do Motor de Clareza Vocal PT-BR."""

from __future__ import annotations

from pathlib import Path

from reportlab.graphics.shapes import Drawing, Line, Polygon, Rect, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "output" / "pdf" / "whitepaper_motor_clareza_vocal_ptbr.pdf"

NAVY = colors.HexColor("#102A43")
BLUE = colors.HexColor("#1769AA")
CYAN = colors.HexColor("#2CB7C9")
INK = colors.HexColor("#243B53")
MUTED = colors.HexColor("#627D98")
PALE = colors.HexColor("#EAF4FB")
PALE_GRAY = colors.HexColor("#F4F7FA")
LINE_COLOR = colors.HexColor("#C7D5E2")
WHITE = colors.white


class CoverCard(Flowable):
    """Cartão vetorial da capa."""

    def __init__(self, width: float, height: float) -> None:
        super().__init__()
        self.width = width
        self.height = height

    def draw(self) -> None:
        canvas = self.canv
        canvas.setFillColor(NAVY)
        canvas.roundRect(0, 0, self.width, self.height, 4 * mm, fill=1, stroke=0)
        canvas.setFillColor(CYAN)
        canvas.roundRect(0, 0, 4 * mm, self.height, 2 * mm, fill=1, stroke=0)

        canvas.setFont("Helvetica-Bold", 10)
        canvas.drawString(30 * mm, self.height - 28 * mm, "WHITEPAPER TÉCNICO · PROVA DE CONCEITO")
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 28)
        canvas.drawString(30 * mm, self.height - 65 * mm, "Motor de Clareza")
        canvas.drawString(30 * mm, self.height - 80 * mm, "Vocal PT-BR")

        canvas.setFont("Helvetica", 12)
        canvas.setFillColor(colors.HexColor("#D9E8F2"))
        canvas.drawString(30 * mm, self.height - 99 * mm, "IA, análise acústica e DSP seletivo")
        canvas.drawString(30 * mm, self.height - 107 * mm, "para voz cantada em português brasileiro")

        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(CYAN)
        canvas.drawString(30 * mm, 10 * mm, "VERSÃO 1.0 · SETEMBRO 2026")


def paragraph_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=28,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=23,
            textColor=NAVY,
            spaceBefore=5,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12.5,
            leading=15,
            textColor=BLUE,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=13.5,
            textColor=INK,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.7,
            leading=10.2,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.8,
            leading=10,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.5,
            leading=9,
            textColor=WHITE,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.5,
            textColor=INK,
        ),
        "center": ParagraphStyle(
            "Center",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
        ),
    }


STYLES = paragraph_styles()


def P(text: str, style: str = "body") -> Paragraph:
    return Paragraph(text, STYLES[style])


def callout(text: str) -> Table:
    table = Table([[P(text, "callout")]], colWidths=[171 * mm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9CC7E6")),
                ("LINEBEFORE", (0, 0), (0, -1), 3, CYAN),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table


def data_table(
    rows: list[list[str]],
    widths: list[float],
    repeat_rows: int = 1,
) -> Table:
    formatted: list[list[Paragraph]] = []
    for row_index, row in enumerate(rows):
        style = "table_header" if row_index == 0 else "table"
        formatted.append([P(cell, style) for cell in row])
    table = Table(formatted, colWidths=widths, repeatRows=repeat_rows, hAlign="LEFT")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("GRID", (0, 0), (-1, -1), 0.4, LINE_COLOR),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    for row_index in range(1, len(rows)):
        if row_index % 2 == 0:
            commands.append(("BACKGROUND", (0, row_index), (-1, row_index), PALE_GRAY))
    table.setStyle(TableStyle(commands))
    return table


def bullet(text: str) -> Paragraph:
    return P(f"- {text}", "body")


def pipeline_drawing() -> Drawing:
    drawing = Drawing(470, 430)

    def box(x: float, y: float, w: float, h: float, title: str, subtitle: str, dark: bool = False) -> None:
        fill = NAVY if dark else PALE
        stroke = NAVY if dark else BLUE
        text_color = WHITE if dark else INK
        drawing.add(Rect(x, y, w, h, 9, 9, fillColor=fill, strokeColor=stroke, strokeWidth=1.4))
        drawing.add(String(x + w / 2, y + h - 20, title, textAnchor="middle", fontName="Helvetica-Bold", fontSize=9, fillColor=text_color))
        drawing.add(String(x + w / 2, y + 14, subtitle, textAnchor="middle", fontName="Helvetica", fontSize=7.5, fillColor=text_color))

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        drawing.add(Line(x1, y1, x2, y2, strokeColor=BLUE, strokeWidth=1.3))
        drawing.add(Polygon([x2, y2, x2 - 4, y2 + 8, x2 + 4, y2 + 8], fillColor=BLUE, strokeColor=BLUE))

    box(155, 360, 160, 55, "1. ENTRADA", "WAV/MP3 finalizado", dark=True)
    box(155, 270, 160, 60, "2. SEPARAÇÃO", "htdemucs: vocals + no_vocals")
    box(15, 160, 175, 65, "ACOMPANHAMENTO", "Preservado para recombinação")
    box(280, 160, 175, 65, "3. DIAGNÓSTICO", "STFT + confiança + abstenção")
    box(280, 65, 175, 65, "4. DSP SELETIVO", "EQ, compressor e envelopes")
    box(155, 0, 160, 55, "5. SAÍDA", "Recombinação + QC + WAV float", dark=True)

    arrow(235, 360, 235, 330)
    arrow(205, 270, 105, 225)
    arrow(265, 270, 367, 225)
    arrow(367, 160, 367, 130)
    arrow(105, 160, 205, 55)
    arrow(367, 65, 265, 55)
    return drawing


def page_decor(canvas, document) -> None:
    if document.page == 1:
        return
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(LINE_COLOR)
    canvas.setLineWidth(0.5)
    canvas.line(20 * mm, height - 16 * mm, width - 20 * mm, height - 16 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, height - 12 * mm, "WHITEPAPER | MOTOR DE CLAREZA VOCAL PT-BR")
    canvas.drawRightString(width - 20 * mm, 11 * mm, str(document.page))
    canvas.restoreState()


def build_story() -> list:
    story: list = []

    story.extend(
        [
            Spacer(1, 35 * mm),
            CoverCard(171 * mm, 127 * mm),
            Spacer(1, 20 * mm),
            P("DOCUMENTO DE VISÃO TECNOLÓGICA E ESTADO DA IMPLEMENTAÇÃO", "center"),
            PageBreak(),
        ]
    )

    story.extend(
        [
            P("1. Resumo executivo", "h1"),
            P(
                "O <b>Motor de Clareza Vocal PT-BR</b> é uma prova de conceito local em Python para investigar tratamento vocal contextual em português brasileiro cantado. O sistema combina separação de fontes por aprendizado profundo, análise no domínio tempo-frequência, decisão baseada em confiança e processamento digital seletivo.",
            ),
            P(
                "A proposta não é reconhecer automaticamente fonemas nesta fase. Ela detecta <b>eventos acústicos candidatos</b> em regiões associadas a nasalidade, estridência e sibilância, decide quando há evidência suficiente para intervir e preserva o sinal quando a confiança é baixa.",
            ),
            callout(
                "<b>Escopo correto:</b> a implementação atual prova o pipeline e fornece métricas auditáveis. Superioridade comercial, especialização fonética e benefício perceptual dependem de corpus anotado, benchmarks e testes cegos posteriores."
            ),
            Spacer(1, 5 * mm),
            P("2. Estado atual da PoC", "h1"),
            data_table(
                [
                    ["CAPACIDADE", "ESTADO", "EVIDÊNCIA NA IMPLEMENTAÇÃO"],
                    ["Separação", "Entregue", "htdemucs real; MPS com fallback automático para CPU"],
                    ["Diagnóstico", "Entregue / heurístico", "STFT, centroide, energia por banda, contraste, cobertura e confiança"],
                    ["Motor de decisão", "Entregue / inicial", "Abstenção por silêncio, baixa confiança ou reconstrução não confiável"],
                    ["DSP", "Entregue / experimental", "EQ regionalizada, compressor, envelopes e presets por intensidade"],
                    ["QC", "Entregue / básico", "Reconstrução, RMS, sample peak, true peak estimado 4x, correlação e delta espectral"],
                    ["Validação PT-BR", "Pendente", "Corpus anotado, precisão/recall/F1 e avaliação AB/ABX"],
                ],
                [35 * mm, 31 * mm, 105 * mm],
            ),
            Spacer(1, 4 * mm),
            P(
                "<b>Terminologia:</b> a saída é uma <i>mix processada</i> ou <i>versão corrigida</i>. O termo master exige uma etapa integral de masterização e escuta crítica.",
                "small",
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            P("3. Problema de engenharia", "h1"),
            P(
                "Processadores vocais genéricos raramente declaram validação específica para canto em PT-BR. Isso motiva uma hipótese de pesquisa: descritores acústicos e dados representativos podem orientar intervenções mais contextuais do que presets fixos. As frequências variam com intérprete, registro, microfone, arranjo, dinâmica e técnica; portanto, as faixas abaixo são <b>janelas iniciais de busca</b>, não equivalências fonéticas.",
            ),
            data_table(
                [
                    ["FENÔMENO", "JANELA INICIAL", "RISCO NA MIX", "ESTADO"],
                    ["Ressonâncias candidatas à nasalidade", "600 Hz-1,2 kHz", "Coloração fechada e perda de definição", "DSP ativo"],
                    ["Estridência em vogais abertas e belt", "2-4 kHz", "Fadiga e competição com caixa/guitarras", "DSP ativo"],
                    ["Sibilância e fricativas", "4-9 kHz", "Aspereza e excesso de alta frequência", "DSP ativo"],
                ],
                [47 * mm, 29 * mm, 65 * mm, 30 * mm],
            ),
            P("3.1 Limite fonético", "h2"),
            P(
                "Energia em uma banda não identifica por si só /ã/, /õ/, /s/ ou /ʃ/. Estudos acústicos de vogais nasais do português brasileiro descrevem estrutura temporal, formantes, antiformantes e variação entre falantes. A heurística atual deve ser entendida como detector de candidatos; um diagnóstico fonema-específico exigirá alinhamento ou modelo acústico treinado para canto.",
            ),
            P("3.2 Regra de segurança", "h2"),
            P(
                "Cada banda recebe um score de confiança derivado de energia relativa, contraste e cobertura temporal. O motor se abstém quando o score fica abaixo do limiar, quando o stem vocal está abaixo de -55 dBFS RMS ou quando a soma dos stems falha no teste global de reconstrução.",
            ),
            callout(
                "<b>Princípio de projeto:</b> uma intervenção evitada é preferível a uma correção agressiva sem evidência. Controles independentes permitem desligar nasalidade, estridência ou sibilância."
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            P("4. Arquitetura híbrida implementada", "h1"),
            P("1. Isolamento por IA", "h2"),
            P(
                "O Demucs gera <i>vocals</i> e <i>no_vocals</i> com o modelo htdemucs. O sistema tenta MPS, repete em CPU quando necessário e restaura sample rate, duração e canais da entrada.",
            ),
            P("2. Análise MIR e diagnóstico", "h2"),
            P(
                "O Librosa calcula STFT, centroide espectral, energia por banda, picos dominantes e regiões sustentadas. Janelas temporais distintas refletem a diferença entre vogais prolongadas e fricativas curtas.",
            ),
            P("3. Motor de decisão", "h2"),
            P(
                "Regras parametrizadas transformam descritores em regiões candidatas, score de confiança e autorização de intervenção. A reconstrução pré-DSP deve apresentar erro RMS relativo de até -15 dB e similaridade normalizada mínima de 0,95.",
            ),
            P("4. Tratamento DSP seletivo", "h2"),
            P(
                "O Pedalboard cria variantes processadas com filtros paramétricos e compressor. Envelopes de ataque/release e crossfades NumPy aplicam cada variante apenas durante eventos autorizados. Os presets Suave, Balanceado e Intenso alteram ganho, Q, threshold, ratio e wet mix.",
            ),
            P("5. Recombinação e controle de qualidade", "h2"),
            P(
                "Vocal processado e acompanhamento são somados em float32. A saída recebe proteção de true peak estimado em 4x com alvo de -1 dBTP e relatório de sample peak, RMS, correlação estéreo, delta RMS e variação do centroide. A estimativa não substitui medidor certificado BS.1770.",
            ),
            callout(
                "<b>Auditoria:</b> a interface disponibiliza mix processada, vocal antes/depois do DSP, instrumental, módulos ativos e diagnóstico JSON."
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            P("5. Pipeline de processamento", "h1"),
            Spacer(1, 2 * mm),
            pipeline_drawing(),
            P(
                "Figura 1. Fluxo lógico atual. O acompanhamento permanece fora do DSP vocal; o motor de decisão pode se abster antes da intervenção.",
                "center",
            ),
            Spacer(1, 5 * mm),
            callout(
                "A separação reduz interferência instrumental na análise, mas pode introduzir artefatos. Por isso, reconstrução e escuta dos stems fazem parte do contrato do pipeline."
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            P("6. Viabilidade em Apple Silicon", "h1"),
            P(
                "A execução local em Apple Silicon está demonstrada na PoC. O backend MPS direciona operações compatíveis à GPU integrada, enquanto Librosa, SciPy e Pedalboard permanecem majoritariamente na CPU. Disponibilidade de MPS não implica aceleração integral: cobertura de operadores, memória unificada, modelo e duração do áudio afetam o resultado.",
            ),
            P("6.1 Benchmark necessário para a beta", "h2"),
            data_table(
                [
                    ["MÉTRICA", "COMO MEDIR", "ESTADO"],
                    ["Latência / RTF", "Mesmo arquivo e parâmetros em CPU e MPS", "Pendente como harness reproduzível"],
                    ["Memória", "Pico de RAM e memória unificada", "Pendente"],
                    ["Reconstrução", "Erro RMS relativo e similaridade", "Implementado"],
                    ["Separação", "SDR/SI-SDR mais escuta cega", "Pendente por falta de referência"],
                    ["Robustez", "Gênero, sotaque, registro, microfone e compressão", "Pendente por corpus"],
                ],
                [38 * mm, 91 * mm, 42 * mm],
            ),
            P("6.2 Controle de qualidade atual", "h2"),
            bullet("Abstenção global se a reconstrução dos stems não for confiável."),
            bullet("Abstenção local por score de confiança e nível mínimo do vocal."),
            bullet("Proteção de true peak estimado por oversampling 4x, alvo -1 dBTP."),
            bullet("Preservação de sample rate, duração, canais e exportação WAV float32."),
            bullet("Oito testes automatizados de DSP, alinhamento, reconstrução e QC."),
            callout(
                "<b>Limite:</b> ainda faltam loudness integrado conforme ITU-R BS.1770, true peak certificado, detecção automática de artefatos e métricas com stems de referência."
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            P("7. Plano de validação e evolução", "h1"),
            data_table(
                [
                    ["ETAPA", "PERGUNTA", "EVIDÊNCIA DE SAÍDA"],
                    ["Dataset", "Há diversidade representativa?", "Trechos licenciados e anotados por fenômeno, região vocal, sotaque e cadeia de gravação"],
                    ["Detecção", "O motor localiza eventos relevantes?", "Precisão, recall, F1, calibração da confiança e análise de erros"],
                    ["Processamento", "A correção ajuda sem degradar?", "AB/ABX, artefatos, naturalidade e preferência de especialistas"],
                    ["Sistema", "É rápido e estável no M4?", "RTF, memória, falhas, fallback e reprodutibilidade"],
                ],
                [28 * mm, 55 * mm, 88 * mm],
            ),
            P("7.1 Critérios de saída para beta", "h2"),
            bullet("Processar WAV/MP3 mono ou estéreo em 44,1 e 48 kHz sem perder sincronismo."),
            bullet("Registrar confiança e motivo de abstenção para cada módulo."),
            bullet("Não produzir NaN, clipping acima do alvo ou mudança estrutural de duração."),
            bullet("Fornecer comparação A/B auditável, stems e relatório de QC."),
            bullet("Executar um corpus mínimo de 30 gravações PT-BR sem falha bloqueante."),
            bullet("Concluir avaliação perceptual inicial com protocolo e amostra documentados."),
            P("7.2 Próximos incrementos", "h2"),
            data_table(
                [
                    ["PRIORIDADE", "INCREMENTO", "RESULTADO"],
                    ["P0", "Harness CPU x MPS e telemetria local", "RTF e memória reproduzíveis"],
                    ["P0", "Corpus piloto e ferramenta de anotação", "Calibração de confiança e falsos positivos"],
                    ["P1", "Loudness BS.1770 e QC espectral por bandas", "A/B mais rigoroso"],
                    ["P1", "Chave A/B sincronizada e teste cego", "Avaliação perceptual consistente"],
                    ["P2", "Detector acústico treinado para canto PT-BR", "Evolução além das heurísticas"],
                ],
                [23 * mm, 73 * mm, 75 * mm],
            ),
            PageBreak(),
        ]
    )

    story.extend(
        [
            P("8. Vantagem competitiva e limites", "h1"),
            P(
                "A diferenciação não está apenas em Demucs, Librosa ou Pedalboard, que são acessíveis a terceiros. O ativo potencial está na combinação de dados licenciados de canto em PT-BR, critérios acústicos calibrados, motor de decisão, experiência de uso e histórico de avaliações perceptuais. Um moat só existe quando esses elementos demonstram desempenho difícil de reproduzir.",
            ),
            P("Principais limites", "h2"),
            bullet("Separação de fontes pode criar artefatos e alterar transientes."),
            bullet("Bandas de frequência não equivalem automaticamente a fonemas."),
            bullet("Sotaque, técnica vocal e intenção estética impedem regras universais."),
            bullet("True peak atual é estimado; loudness integrado e ABX ainda são roadmap."),
            bullet("Uma correção mensurável pode não ser artisticamente desejável."),
            P("9. Conclusão", "h1"),
            P(
                "O Motor de Clareza Vocal PT-BR já demonstra um pipeline modular e auditável: isolamento, diagnóstico, confiança, DSP seletivo, recombinação e QC básico. A tese de especialização linguística continua sendo uma hipótese de produto. O próximo salto de valor depende de dados representativos e testes cegos que comprovem benefício sem perda de naturalidade.",
            ),
            callout(
                "<b>Síntese:</b> a PoC prova o pipeline; os testes medem engenharia; dados e avaliação perceptual deverão provar qualidade e diferenciação."
            ),
            Spacer(1, 5 * mm),
            P("Referências técnicas essenciais", "h2"),
            P(
                "1. Défossez, A. <i>Hybrid Spectrogram and Waveform Source Separation</i>. arXiv:2111.03600, 2021. <link href='https://arxiv.org/abs/2111.03600' color='#1769AA'>arxiv.org/abs/2111.03600</link>",
                "small",
            ),
            P(
                "2. McFee, B. et al. <i>librosa: Audio and Music Signal Analysis in Python</i>. Proceedings of SciPy, 2015. DOI 10.25080/Majora-7b98e3ed-003.",
                "small",
            ),
            P(
                "3. Spotify. <i>Pedalboard: Python Audio Effects Library</i>. <link href='https://spotify.github.io/pedalboard/' color='#1769AA'>spotify.github.io/pedalboard</link>",
                "small",
            ),
            P(
                "4. PyTorch. <i>MPS backend documentation</i>. <link href='https://docs.pytorch.org/docs/stable/notes/mps.html' color='#1769AA'>docs.pytorch.org/docs/stable/notes/mps.html</link>",
                "small",
            ),
            P(
                "5. Correa, B. T.; Gonçalves, G. F.; Seara, I. C. <i>Brazilian Portuguese nasal vowels and their acoustic moments</i>. Estudios de Fonética Experimental, XXXI, 2022.",
                "small",
            ),
            P(
                "6. ITU-R BS.1770. <i>Algorithms to measure audio programme loudness and true-peak audio level</i>.",
                "small",
            ),
        ]
    )
    return story


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=23 * mm,
        bottomMargin=18 * mm,
        title="Motor de Clareza Vocal PT-BR",
        author="Projeto Motor de Clareza Vocal PT-BR",
        subject="Whitepaper técnico e estado da implementação",
    )
    document.build(build_story(), onFirstPage=page_decor, onLaterPages=page_decor)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
