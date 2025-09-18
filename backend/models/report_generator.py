"""
Medical Report Generator Module
Generates comprehensive PDF reports from medical analysis results
"""

import os
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Report generation libraries
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.platypus import Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie

# Data processing
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import tempfile

from .medical_report_analyzer import MedicalReport, MedicalEntity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MedicalReportGenerator:
    """Generates comprehensive PDF reports from medical analysis"""
    
    def __init__(self):
        self.setup_styles()
        
    def setup_styles(self):
        """Initialize report styles"""
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.darkblue
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        ))
        
        self.styles.add(ParagraphStyle(
            name='HighlightBox',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            borderWidth=1,
            borderColor=colors.grey,
            borderPadding=10,
            backColor=colors.lightgrey
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskWarning',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            borderWidth=2,
            borderColor=colors.red,
            borderPadding=10,
            backColor=colors.mistyrose,
            textColor=colors.darkred
        ))
        
        self.styles.add(ParagraphStyle(
            name='RecommendationBox',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            rightIndent=20,
            spaceAfter=10,
            borderWidth=1,
            borderColor=colors.green,
            borderPadding=10,
            backColor=colors.lightgreen
        ))
    
    def create_header_section(self, story: List, patient_name: str = "Patient"):
        """Create report header section"""
        # Title
        title = Paragraph("Medical Report Analysis", self.styles['CustomTitle'])
        story.append(title)
        story.append(Spacer(1, 20))
        
        # Patient info table
        patient_data = [
            ['Patient Name:', patient_name],
            ['Report Date:', datetime.now().strftime('%B %d, %Y')],
            ['Analysis Time:', datetime.now().strftime('%I:%M %p')],
            ['Report Type:', 'Comprehensive Medical Analysis']
        ]
        
        patient_table = Table(patient_data, colWidths=[2*inch, 4*inch])
        patient_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.darkblue),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(patient_table)
        story.append(Spacer(1, 30))
    
    def create_summary_section(self, story: List, report: MedicalReport):
        """Create executive summary section"""
        story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        # Analysis confidence
        confidence_text = f"Analysis Confidence: {report.confidence_score:.1%}"
        if report.confidence_score >= 0.8:
            confidence_color = colors.green
            confidence_desc = "High confidence - Analysis based on clear medical text"
        elif report.confidence_score >= 0.6:
            confidence_color = colors.orange
            confidence_desc = "Moderate confidence - Some ambiguity in source text"
        else:
            confidence_color = colors.red
            confidence_desc = "Low confidence - Limited or unclear medical information"
        
        confidence_para = Paragraph(
            f"<font color='{confidence_color}'><b>{confidence_text}</b></font><br/>{confidence_desc}",
            self.styles['HighlightBox']
        )
        story.append(confidence_para)
        
        # Key findings summary
        summary_data = [
            ['Medical Conditions Found:', str(len(report.conditions))],
            ['Medications Identified:', str(len(report.medications))],
            ['Symptoms Noted:', str(len(report.symptoms))],
            ['Lab Values Extracted:', str(len(report.lab_values))],
            ['Future Risk Factors:', str(len(report.future_risks))],
            ['Recommendations Generated:', str(len(report.recommendations))]
        ]
        
        summary_table = Table(summary_data, colWidths=[3*inch, 2*inch])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(summary_table)
        story.append(Spacer(1, 20))
    
    def create_conditions_section(self, story: List, conditions: List[MedicalEntity]):
        """Create medical conditions section"""
        story.append(Paragraph("Medical Conditions Identified", self.styles['SectionHeader']))
        
        if not conditions:
            story.append(Paragraph("No specific medical conditions were clearly identified in the report.", 
                                 self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Group conditions by type
        condition_groups = {}
        for condition in conditions:
            label = condition.label.replace('CONDITION_', '').replace('_', ' ').title()
            if label not in condition_groups:
                condition_groups[label] = []
            condition_groups[label].append(condition)
        
        for group_name, group_conditions in condition_groups.items():
            story.append(Paragraph(f"<b>{group_name}:</b>", self.styles['Normal']))
            
            condition_data = [['Condition', 'Confidence', 'Context']]
            for condition in group_conditions:
                context_preview = condition.context[:100] + "..." if len(condition.context) > 100 else condition.context
                condition_data.append([
                    condition.text,
                    f"{condition.confidence:.1%}",
                    context_preview
                ])
            
            condition_table = Table(condition_data, colWidths=[2*inch, 1*inch, 3*inch])
            condition_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
            ]))
            
            story.append(condition_table)
            story.append(Spacer(1, 15))
    
    def create_medications_section(self, story: List, medications: List[MedicalEntity]):
        """Create medications section"""
        story.append(Paragraph("Medications and Treatments", self.styles['SectionHeader']))
        
        if not medications:
            story.append(Paragraph("No specific medications were clearly identified in the report.", 
                                 self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        med_data = [['Medication/Treatment', 'Confidence', 'Context']]
        for med in medications:
            context_preview = med.context[:100] + "..." if len(med.context) > 100 else med.context
            med_data.append([
                med.text,
                f"{med.confidence:.1%}",
                context_preview
            ])
        
        med_table = Table(med_data, colWidths=[2*inch, 1*inch, 3*inch])
        med_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkgreen),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(med_table)
        story.append(Spacer(1, 20))
    
    def create_lab_values_section(self, story: List, lab_values: List[MedicalEntity]):
        """Create lab values section"""
        story.append(Paragraph("Laboratory Values and Test Results", self.styles['SectionHeader']))
        
        if not lab_values:
            story.append(Paragraph("No specific laboratory values were identified in the report.", 
                                 self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        lab_data = [['Test/Parameter', 'Value', 'Context']]
        for lab in lab_values:
            context_preview = lab.context[:80] + "..." if len(lab.context) > 80 else lab.context
            lab_data.append([
                lab.label.replace('LAB_', '').replace('_', ' ').title(),
                lab.text,
                context_preview
            ])
        
        lab_table = Table(lab_data, colWidths=[2*inch, 1.5*inch, 2.5*inch])
        lab_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.purple),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
        ]))
        
        story.append(lab_table)
        story.append(Spacer(1, 20))
    
    def create_risk_assessment_section(self, story: List, future_risks: List[str]):
        """Create risk assessment section"""
        story.append(Paragraph("Future Health Risk Assessment", self.styles['SectionHeader']))
        
        if not future_risks:
            story.append(Paragraph("No specific future health risks were identified based on current conditions.", 
                                 self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Risk warning
        warning_text = """
        <b>Important:</b> The following risk assessments are based on statistical correlations and medical literature. 
        They are not definitive predictions and should be discussed with your healthcare provider for personalized evaluation.
        """
        story.append(Paragraph(warning_text, self.styles['RiskWarning']))
        
        # Risk list
        for i, risk in enumerate(future_risks, 1):
            risk_para = Paragraph(f"<b>{i}.</b> {risk}", self.styles['Normal'])
            story.append(risk_para)
        
        story.append(Spacer(1, 20))
    
    def create_recommendations_section(self, story: List, recommendations: List[str]):
        """Create recommendations section"""
        story.append(Paragraph("Health Recommendations", self.styles['SectionHeader']))
        
        if not recommendations:
            story.append(Paragraph("No specific recommendations were generated.", self.styles['Normal']))
            story.append(Spacer(1, 20))
            return
        
        # Recommendation intro
        intro_text = """
        The following recommendations are based on the identified conditions and established medical guidelines. 
        Please consult with your healthcare provider before making any changes to your treatment plan.
        """
        story.append(Paragraph(intro_text, self.styles['RecommendationBox']))
        
        # Recommendations list
        for i, recommendation in enumerate(recommendations, 1):
            rec_para = Paragraph(f"<b>{i}.</b> {recommendation}", self.styles['Normal'])
            story.append(rec_para)
            story.append(Spacer(1, 5))
        
        story.append(Spacer(1, 20))
    
    def create_disclaimer_section(self, story: List):
        """Create medical disclaimer section"""
        story.append(Paragraph("Medical Disclaimer", self.styles['SectionHeader']))
        
        disclaimer_text = """
        <b>IMPORTANT MEDICAL DISCLAIMER:</b><br/><br/>
        
        This report is generated by an AI-powered medical text analysis system and is intended for informational purposes only. 
        It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment.<br/><br/>
        
        <b>Key Points:</b><br/>
        • This analysis is based on automated text processing and may contain errors or omissions<br/>
        • Medical decisions should always be made in consultation with qualified healthcare professionals<br/>
        • The accuracy of this analysis depends on the quality and completeness of the source document<br/>
        • Future risk predictions are statistical estimates and may not apply to individual cases<br/>
        • Always seek immediate medical attention for urgent health concerns<br/><br/>
        
        <b>For Medical Emergencies:</b> Contact your local emergency services immediately.<br/>
        <b>For Medical Questions:</b> Consult with your healthcare provider or primary care physician.
        """
        
        story.append(Paragraph(disclaimer_text, self.styles['RiskWarning']))
        story.append(Spacer(1, 20))
    
    def create_technical_details_section(self, story: List, report: MedicalReport):
        """Create technical analysis details section"""
        story.append(Paragraph("Technical Analysis Details", self.styles['SectionHeader']))
        
        # Analysis metadata
        tech_data = [
            ['Analysis Timestamp:', report.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')],
            ['Text Length (characters):', str(len(report.original_text))],
            ['Entities Extracted:', str(len(report.conditions) + len(report.medications) + 
                                      len(report.symptoms) + len(report.lab_values))],
            ['Analysis Confidence:', f"{report.confidence_score:.1%}"],
            ['Processing Method:', 'AI-powered NLP with medical domain models']
        ]
        
        tech_table = Table(tech_data, colWidths=[2.5*inch, 3.5*inch])
        tech_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        story.append(tech_table)
        story.append(Spacer(1, 20))
    
    def generate_report(self, report: MedicalReport, patient_name: str = "Patient") -> bytes:
        """Generate complete PDF report"""
        try:
            # Create PDF buffer
            buffer = io.BytesIO()
            
            # Create document
            doc = SimpleDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=18
            )
            
            # Build story (content)
            story = []
            
            # Add sections
            self.create_header_section(story, patient_name)
            self.create_summary_section(story, report)
            self.create_conditions_section(story, report.conditions)
            self.create_medications_section(story, report.medications)
            self.create_lab_values_section(story, report.lab_values)
            self.create_risk_assessment_section(story, report.future_risks)
            self.create_recommendations_section(story, report.recommendations)
            
            # Add page break before technical details
            story.append(PageBreak())
            self.create_technical_details_section(story, report)
            self.create_disclaimer_section(story)
            
            # Build PDF
            doc.build(story)
            
            # Get PDF bytes
            pdf_bytes = buffer.getvalue()
            buffer.close()
            
            logger.info("✓ PDF report generated successfully")
            return pdf_bytes
            
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            raise ValueError(f"Failed to generate PDF report: {e}")
    
    def generate_summary_report(self, report: MedicalReport, patient_name: str = "Patient") -> Dict[str, Any]:
        """Generate a summary report in dictionary format for API responses"""
        return {
            'patient_name': patient_name,
            'analysis_date': report.analysis_timestamp.isoformat(),
            'confidence_score': report.confidence_score,
            'summary': {
                'conditions_found': len(report.conditions),
                'medications_identified': len(report.medications),
                'symptoms_noted': len(report.symptoms),
                'lab_values_extracted': len(report.lab_values),
                'future_risks': len(report.future_risks),
                'recommendations': len(report.recommendations)
            },
            'conditions': [
                {
                    'text': condition.text,
                    'type': condition.label,
                    'confidence': condition.confidence,
                    'context': condition.context[:100]
                } for condition in report.conditions
            ],
            'medications': [
                {
                    'text': med.text,
                    'confidence': med.confidence,
                    'context': med.context[:100]
                } for med in report.medications
            ],
            'lab_values': [
                {
                    'test': lab.label.replace('LAB_', '').replace('_', ' ').title(),
                    'value': lab.text,
                    'context': lab.context[:100]
                } for lab in report.lab_values
            ],
            'future_risks': report.future_risks,
            'recommendations': report.recommendations,
            'text_preview': report.original_text[:500] + "..." if len(report.original_text) > 500 else report.original_text
        }
