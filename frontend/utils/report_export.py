from typing import List, Dict, Any
from datetime import datetime
from io import BytesIO

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle


def build_drug_advisor_pdf(patient: Dict[str, Any],
                           extracted: List[Dict[str, Any]],
                           interactions: List[Dict[str, Any]],
                           dosage_notes: List[str],
                           alternatives: List[Dict[str, Any]]) -> bytes:
    """Generate a PDF report summarizing Drug Advisor results.

    patient: {name, age}
    extracted: list of drug dicts from backend (with optional normalization fields)
    interactions: list of interaction dicts
    dosage_notes: list of textual dosage guidance/alerts
    alternatives: list of {drug, category, alternatives}
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    story = []
    title = f"Drug Advisor Report"
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 8))

    # Header: patient info
    pname = patient.get('name', 'Unknown')
    page = [
        ["Patient", pname],
        ["Age", str(patient.get('age', '-'))],
        ["Date", datetime.now().strftime('%Y-%m-%d %H:%M')],
    ]
    t = Table(page, colWidths=[120, 360])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke),
        ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
        ('INNERGRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
    ]))
    story.append(t)
    story.append(Spacer(1, 12))

    # Extracted drugs table
    story.append(Paragraph("Drugs Extracted", styles['Heading2']))
    if extracted:
        rows = [["Drug", "Strength", "Frequency", "Duration", "Confidence", "Source"]]
        for d in extracted:
            rows.append([
                str(d.get('name', '')),
                str(d.get('strength', '')),
                str(d.get('frequency', '')),
                str(d.get('duration', '')),
                f"{round(float(d.get('score', 0.0))*100):d}%" if isinstance(d.get('score', 0.0), (int, float)) else "",
                d.get('source', 'HF/regex')
            ])
        tbl = Table(rows, colWidths=[120, 90, 90, 70, 70, 80])
        tbl.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8f0fe')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("No drugs detected.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Interactions
    story.append(Paragraph("Interactions", styles['Heading2']))
    if interactions:
        rows = [["Pair", "Severity", "Source", "Note"]]
        for it in interactions:
            pair = " / ".join(it.get('pair', []))
            rows.append([pair, it.get('severity', ''), it.get('source', ''), it.get('note', '')[:120]])
        tbl = Table(rows, colWidths=[160, 80, 80, 250])
        tbl.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#fef3c7')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("No interactions flagged.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Dosage guidance
    story.append(Paragraph("Dosage Guidance", styles['Heading2']))
    if dosage_notes:
        for note in dosage_notes:
            story.append(Paragraph(f"• {note}", styles['Normal']))
    else:
        story.append(Paragraph("No specific dosage warnings.", styles['Normal']))
    story.append(Spacer(1, 12))

    # Alternatives
    story.append(Paragraph("Safer Alternatives", styles['Heading2']))
    if alternatives:
        rows = [["Drug", "Category", "Alternatives"]]
        for a in alternatives:
            rows.append([a.get('drug',''), a.get('category',''), ", ".join(a.get('alternatives', []))])
        tbl = Table(rows, colWidths=[120, 120, 330])
        tbl.setStyle(TableStyle([
            ('BOX', (0,0), (-1,-1), 0.5, colors.grey),
            ('INNERGRID', (0,0), (-1,-1), 0.25, colors.lightgrey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#dcfce7')),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("No alternatives suggested.", styles['Normal']))
    story.append(Spacer(1, 16))

    # Disclaimer
    disclaimer = "This report is for informational purposes only and does not replace professional medical advice."
    story.append(Paragraph(disclaimer, styles['Italic']))

    doc.build(story)
    return buf.getvalue()
