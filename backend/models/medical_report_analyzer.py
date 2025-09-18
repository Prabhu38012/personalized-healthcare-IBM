"""
Medical Report Analysis Module
Handles text extraction, medical NLP, and clinical analysis of medical reports
"""

import os
import io
import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import tempfile

# Text extraction libraries
import pdfplumber
import pytesseract
from PIL import Image
import fitz  # PyMuPDF
from pdf2image import convert_from_bytes

# NLP and Medical Analysis (deferred imports for optional deps)
import nltk
from transformers import AutoTokenizer, AutoModel, pipeline
import torch
import numpy as np
# Do not import spacy or sentence_transformers at module level; they may be absent
spacy = None  # type: ignore
SentenceTransformer = None  # type: ignore

# Data processing
import pandas as pd
import json
from fuzzywuzzy import fuzz, process

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from text"""
    text: str
    label: str
    start: int
    end: int
    confidence: float
    context: str = ""

@dataclass
class MedicalReport:
    """Represents a complete medical report analysis"""
    original_text: str
    conditions: List[MedicalEntity]
    medications: List[MedicalEntity]
    symptoms: List[MedicalEntity]
    lab_values: List[MedicalEntity]
    recommendations: List[str]
    risk_factors: List[str]
    future_risks: List[str]
    confidence_score: float
    analysis_timestamp: datetime

class MedicalReportAnalyzer:
    """Main class for analyzing medical reports"""
    
    def __init__(self):
        self.setup_models()
        self.setup_medical_knowledge()
        
    def setup_models(self):
        """Initialize NLP models and pipelines with graceful fallbacks"""
        try:
            # Load spaCy models with multiple fallbacks
            self.nlp_sci = None
            models_to_try = ["en_core_sci_md", "en_core_sci_sm", "en_core_web_sm", "en_core_web_md"]
            
            for model_name in models_to_try:
                try:
                    global spacy
                    if spacy is None:
                        import spacy as _spacy  # local import
                        spacy = _spacy
                    self.nlp_sci = spacy.load(model_name)
                    logger.info(f"✓ Loaded {model_name} model")
                    break
                except (OSError, ImportError) as e:
                    logger.warning(f"⚠️ Could not load {model_name}: {e}")
                    continue
            
            if self.nlp_sci is None:
                # Create a basic English model as last resort
                try:
                    if spacy is None:
                        import spacy as _spacy
                        spacy = _spacy
                    self.nlp_sci = spacy.blank("en")
                    logger.warning("⚠️ Using basic English model - medical analysis will be limited")
                except ImportError:
                    logger.error("❌ spaCy not available - text analysis will be basic")
                    self.nlp_sci = None
            
            # Initialize Clinical BERT for medical text understanding (optional)
            self.clinical_bert = None
            try:
                from transformers import pipeline
                self.clinical_bert = pipeline(
                    "text-classification",
                    model="emilyalsentzer/Bio_ClinicalBERT",
                    tokenizer="emilyalsentzer/Bio_ClinicalBERT"
                )
                logger.info("✓ Loaded Clinical BERT model")
            except Exception as e:
                logger.warning(f"⚠️ Could not load Clinical BERT: {e}")
                self.clinical_bert = None
            
            # Initialize sentence transformer for semantic similarity (optional)
            self.sentence_model = None
            try:
                global SentenceTransformer
                if SentenceTransformer is None:
                    from sentence_transformers import SentenceTransformer as _ST
                    SentenceTransformer = _ST
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✓ Loaded sentence transformer model")
            except Exception as e:
                logger.warning(f"⚠️ Could not load sentence transformer: {e}")
                self.sentence_model = None
            
            # Download required NLTK data (optional)
            try:
                import nltk
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('averaged_perceptron_tagger', quiet=True)
                logger.info("✓ NLTK data downloaded")
            except Exception as e:
                logger.warning(f"⚠️ NLTK download issues: {e}")
                
        except Exception as e:
            logger.warning(f"Model setup completed with limitations: {e}")
            # Don't raise - allow system to work with basic functionality
    
    def setup_medical_knowledge(self):
        """Initialize medical knowledge bases and dictionaries"""
        # Common medical conditions
        self.medical_conditions = {
            'diabetes': ['diabetes', 'diabetic', 'hyperglycemia', 'high blood sugar', 'dm', 'type 1 diabetes', 'type 2 diabetes'],
            'hypertension': ['hypertension', 'high blood pressure', 'htn', 'elevated bp', 'systolic pressure', 'diastolic pressure'],
            'heart_disease': ['heart disease', 'cardiac', 'coronary', 'myocardial', 'angina', 'heart attack', 'mi', 'chd'],
            'kidney_disease': ['kidney disease', 'renal', 'nephropathy', 'ckd', 'chronic kidney disease', 'kidney failure'],
            'liver_disease': ['liver disease', 'hepatic', 'hepatitis', 'cirrhosis', 'liver failure', 'jaundice'],
            'respiratory': ['asthma', 'copd', 'pneumonia', 'bronchitis', 'respiratory', 'lung disease', 'dyspnea'],
            'cancer': ['cancer', 'tumor', 'malignancy', 'carcinoma', 'oncology', 'metastasis', 'neoplasm'],
            'mental_health': ['depression', 'anxiety', 'bipolar', 'schizophrenia', 'ptsd', 'mental health', 'psychiatric']
        }
        
        # Common medications
        self.medication_patterns = [
            r'\b\w+cillin\b',  # antibiotics
            r'\b\w+statin\b',  # cholesterol medications
            r'\b\w+pril\b',    # ACE inhibitors
            r'\b\w+sartan\b',  # ARBs
            r'\b\w+olol\b',    # beta blockers
            r'\b\w+pine\b',    # calcium channel blockers
            r'\bmg\b|\bml\b|\btablet\b|\bcapsule\b'  # dosage indicators
        ]
        
        # Lab value patterns
        self.lab_patterns = {
            'glucose': r'glucose\s*:?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mmol/l)?',
            'cholesterol': r'cholesterol\s*:?\s*(\d+(?:\.\d+)?)\s*(mg/dl|mmol/l)?',
            'blood_pressure': r'(?:bp|blood pressure)\s*:?\s*(\d+)/(\d+)',
            'hemoglobin': r'(?:hb|hemoglobin)\s*:?\s*(\d+(?:\.\d+)?)\s*(g/dl|g/l)?',
            'creatinine': r'creatinine\s*:?\s*(\d+(?:\.\d+)?)\s*(mg/dl|umol/l)?'
        }
        
        # Risk factors and complications
        self.risk_mappings = {
            'diabetes': {
                'risk_factors': ['obesity', 'family history', 'sedentary lifestyle', 'poor diet'],
                'complications': ['diabetic retinopathy', 'diabetic nephropathy', 'diabetic neuropathy', 'cardiovascular disease'],
                'recommendations': ['monitor blood sugar', 'healthy diet', 'regular exercise', 'medication compliance']
            },
            'hypertension': {
                'risk_factors': ['smoking', 'obesity', 'high sodium intake', 'stress', 'family history'],
                'complications': ['stroke', 'heart attack', 'kidney disease', 'heart failure'],
                'recommendations': ['reduce sodium intake', 'regular exercise', 'stress management', 'medication compliance']
            }
        }
    
    def extract_text_from_pdf(self, file_content: bytes) -> str:
        """Extract text from PDF using multiple methods"""
        extracted_text = ""
        
        try:
            # Method 1: pdfplumber (best for structured text)
            with io.BytesIO(file_content) as pdf_file:
                with pdfplumber.open(pdf_file) as pdf:
                    for page in pdf.pages:
                        page_text = page.extract_text()
                        if page_text:
                            extracted_text += page_text + "\n"
            
            if extracted_text.strip():
                logger.info("✓ Text extracted using pdfplumber")
                return extracted_text
                
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")
        
        try:
            # Method 2: PyMuPDF fallback
            pdf_document = fitz.open(stream=file_content, filetype="pdf")
            for page_num in range(pdf_document.page_count):
                page = pdf_document[page_num]
                extracted_text += page.get_text() + "\n"
            pdf_document.close()
            
            if extracted_text.strip():
                logger.info("✓ Text extracted using PyMuPDF")
                return extracted_text
                
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")
        
        try:
            # Method 3: OCR fallback for scanned documents
            images = convert_from_bytes(file_content)
            for image in images:
                ocr_text = pytesseract.image_to_string(image, config='--psm 6')
                extracted_text += ocr_text + "\n"
            
            if extracted_text.strip():
                logger.info("✓ Text extracted using OCR")
                return extracted_text
                
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
        
        if not extracted_text.strip():
            raise ValueError("Could not extract text from PDF using any method")
        
        return extracted_text
    
    def extract_text_from_image(self, file_content: bytes) -> str:
        """Extract text from image using OCR"""
        try:
            image = Image.open(io.BytesIO(file_content))
            # Enhance image for better OCR
            image = image.convert('RGB')
            text = pytesseract.image_to_string(image, config='--psm 6')
            logger.info("✓ Text extracted from image using OCR")
            return text
        except Exception as e:
            logger.error(f"Image text extraction failed: {e}")
            raise ValueError(f"Could not extract text from image: {e}")
    
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep medical notation
        text = re.sub(r'[^\w\s\-\./:%()]', ' ', text)
        
        # Normalize medical abbreviations
        text = re.sub(r'\bBP\b', 'blood pressure', text, flags=re.IGNORECASE)
        text = re.sub(r'\bHR\b', 'heart rate', text, flags=re.IGNORECASE)
        text = re.sub(r'\bDM\b', 'diabetes mellitus', text, flags=re.IGNORECASE)
        text = re.sub(r'\bHTN\b', 'hypertension', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[MedicalEntity]]:
        """Extract medical entities using NLP with fallbacks"""
        entities = {
            'conditions': [],
            'medications': [],
            'symptoms': [],
            'lab_values': []
        }
        
        try:
            # Process with spaCy if available
            if self.nlp_sci is not None:
                doc = self.nlp_sci(text)
                
                # Extract named entities
                for ent in doc.ents:
                    entity = MedicalEntity(
                        text=ent.text,
                        label=ent.label_,
                        start=ent.start_char,
                        end=ent.end_char,
                        confidence=0.8,  # Default confidence for spaCy entities
                        context=text[max(0, ent.start_char-50):ent.end_char+50]
                    )
                    
                    # Categorize entities
                    if ent.label_ in ['DISEASE', 'DISORDER', 'CONDITION']:
                        entities['conditions'].append(entity)
                    elif ent.label_ in ['DRUG', 'MEDICATION', 'CHEMICAL']:
                        entities['medications'].append(entity)
                    elif ent.label_ in ['SYMPTOM', 'SIGN']:
                        entities['symptoms'].append(entity)
            
            # Extract lab values using regex patterns
            for lab_type, pattern in self.lab_patterns.items():
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = MedicalEntity(
                        text=match.group(0),
                        label=f'LAB_{lab_type.upper()}',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.9,
                        context=text[max(0, match.start()-30):match.end()+30]
                    )
                    entities['lab_values'].append(entity)
            
            # Extract medications using pattern matching
            for pattern in self.medication_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = MedicalEntity(
                        text=match.group(0),
                        label='MEDICATION',
                        start=match.start(),
                        end=match.end(),
                        confidence=0.7,
                        context=text[max(0, match.start()-30):match.end()+30]
                    )
                    entities['medications'].append(entity)
            
            # Extract conditions using fuzzy matching
            for condition_type, keywords in self.medical_conditions.items():
                for keyword in keywords:
                    matches = re.finditer(re.escape(keyword), text, re.IGNORECASE)
                    for match in matches:
                        entity = MedicalEntity(
                            text=match.group(0),
                            label=f'CONDITION_{condition_type.upper()}',
                            start=match.start(),
                            end=match.end(),
                            confidence=0.8,
                            context=text[max(0, match.start()-40):match.end()+40]
                        )
                        entities['conditions'].append(entity)
            
        except Exception as e:
            logger.error(f"Entity extraction error: {e}")
        
        return entities
    
    def predict_future_risks(self, entities: Dict[str, List[MedicalEntity]]) -> List[str]:
        """Predict future health risks based on current conditions"""
        risks = []
        
        # Extract condition types
        conditions = [entity.text.lower() for entity in entities['conditions']]
        
        # Risk prediction logic
        if any('diabetes' in condition for condition in conditions):
            risks.extend([
                'Diabetic retinopathy (eye complications)',
                'Diabetic nephropathy (kidney damage)',
                'Cardiovascular disease risk',
                'Diabetic neuropathy (nerve damage)',
                'Increased infection risk'
            ])
        
        if any('hypertension' in condition or 'blood pressure' in condition for condition in conditions):
            risks.extend([
                'Stroke risk',
                'Heart attack risk',
                'Kidney disease progression',
                'Heart failure risk'
            ])
        
        if any('heart' in condition for condition in conditions):
            risks.extend([
                'Arrhythmia risk',
                'Heart failure progression',
                'Sudden cardiac death risk'
            ])
        
        # Check lab values for additional risks
        lab_values = [entity.text.lower() for entity in entities['lab_values']]
        if any('cholesterol' in lab for lab in lab_values):
            risks.append('Atherosclerosis progression')
        
        return list(set(risks))  # Remove duplicates
    
    def generate_recommendations(self, entities: Dict[str, List[MedicalEntity]]) -> List[str]:
        """Generate personalized health recommendations using comprehensive lifestyle engine"""
        try:
            # Import lifestyle engine
            from ..utils.lifestyle_recommender import lifestyle_engine
            
            # Extract conditions and lab values
            conditions = [entity.text for entity in entities['conditions']]
            lab_values = [entity.text for entity in entities['lab_values']]
            
            # Generate comprehensive lifestyle plan
            lifestyle_plan = lifestyle_engine.generate_comprehensive_plan(
                conditions=conditions,
                lab_values=lab_values
            )
            
            # Compile all recommendations
            recommendations = []
            
            # Add priority actions first
            recommendations.extend(lifestyle_plan.priority_actions)
            
            # Add diet recommendations
            for diet_rec in lifestyle_plan.diet_recommendations:
                recommendations.append(f"Diet: Follow {diet_rec.category}")
                recommendations.extend(diet_rec.nutritional_goals[:2])  # Top 2 goals
            
            # Add exercise recommendations
            for exercise_rec in lifestyle_plan.exercise_recommendations:
                recommendations.append(f"Exercise: {exercise_rec.recommended_activities[0]} - {exercise_rec.frequency}")
                recommendations.extend(exercise_rec.precautions[:2])  # Top 2 precautions
            
            # Add lifestyle recommendations
            for lifestyle_rec in lifestyle_plan.lifestyle_recommendations:
                recommendations.extend(lifestyle_rec.recommendations[:2])  # Top 2 from each category
            
            # Add monitoring suggestions
            recommendations.extend(lifestyle_plan.monitoring_suggestions[:3])  # Top 3 monitoring items
            
            # Add professional consultations
            recommendations.extend([f"Consider consultation with: {consult}" for consult in lifestyle_plan.professional_consultations[:2]])
            
            return recommendations[:15]  # Limit to 15 recommendations to avoid overwhelming
            
        except Exception as e:
            logger.warning(f"Error generating lifestyle recommendations: {e}")
            # Fallback to basic recommendations
            return self._generate_basic_recommendations(entities)
    
    def _generate_basic_recommendations(self, entities: Dict[str, List[MedicalEntity]]) -> List[str]:
        """Fallback method for basic recommendations if lifestyle engine fails"""
        recommendations = []
        
        # Extract conditions for targeted recommendations
        conditions = [entity.text.lower() for entity in entities['conditions']]
        
        # General recommendations
        recommendations.extend([
            'Regular follow-up with healthcare provider',
            'Maintain a healthy, balanced diet',
            'Engage in regular physical activity as approved by physician',
            'Take medications as prescribed',
            'Monitor vital signs regularly'
        ])
        
        # Condition-specific recommendations
        if any('diabetes' in condition for condition in conditions):
            recommendations.extend([
                'Monitor blood glucose levels daily',
                'Follow diabetic diet plan',
                'Regular eye and foot examinations',
                'Maintain HbA1c levels below 7%'
            ])
        
        if any('hypertension' in condition for condition in conditions):
            recommendations.extend([
                'Limit sodium intake to less than 2300mg daily',
                'Monitor blood pressure regularly',
                'Manage stress through relaxation techniques',
                'Limit alcohol consumption'
            ])
        
        if any('heart' in condition for condition in conditions):
            recommendations.extend([
                'Follow heart-healthy diet (low saturated fat)',
                'Avoid smoking and secondhand smoke',
                'Take prescribed cardiac medications consistently',
                'Monitor for chest pain or shortness of breath'
            ])
        
        return recommendations
    
    def calculate_confidence_score(self, entities: Dict[str, List[MedicalEntity]], text: str) -> float:
        """Calculate overall confidence score for the analysis"""
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        
        if total_entities == 0:
            return 0.3  # Low confidence if no entities found
        
        # Calculate average entity confidence
        all_confidences = []
        for entity_list in entities.values():
            all_confidences.extend([entity.confidence for entity in entity_list])
        
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.5
        
        # Adjust based on text quality indicators
        text_quality_score = 1.0
        if len(text) < 100:
            text_quality_score *= 0.7  # Short text reduces confidence
        
        # Check for medical terminology density
        medical_terms = sum(1 for word in text.lower().split() 
                          if any(term in word for condition_list in self.medical_conditions.values() 
                                for term in condition_list))
        
        if medical_terms / len(text.split()) > 0.1:
            text_quality_score *= 1.2  # High medical term density increases confidence
        
        final_confidence = min(avg_confidence * text_quality_score, 1.0)
        return round(final_confidence, 2)
    
    def analyze_report(self, file_content: bytes, filename: str) -> MedicalReport:
        """Main method to analyze a medical report"""
        try:
            # Extract text based on file type
            if filename.lower().endswith('.pdf'):
                text = self.extract_text_from_pdf(file_content)
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
                text = self.extract_text_from_image(file_content)
            else:
                raise ValueError(f"Unsupported file type: {filename}")
            
            # Preprocess text
            processed_text = self.preprocess_text(text)
            
            # Extract medical entities
            entities = self.extract_medical_entities(processed_text)
            
            # Generate predictions and recommendations
            future_risks = self.predict_future_risks(entities)
            recommendations = self.generate_recommendations(entities)
            
            # Calculate confidence score
            confidence = self.calculate_confidence_score(entities, processed_text)
            
            # Create medical report object
            report = MedicalReport(
                original_text=processed_text,
                conditions=entities['conditions'],
                medications=entities['medications'],
                symptoms=entities['symptoms'],
                lab_values=entities['lab_values'],
                recommendations=recommendations,
                risk_factors=[],  # Will be populated based on conditions
                future_risks=future_risks,
                confidence_score=confidence,
                analysis_timestamp=datetime.now()
            )
            
            logger.info(f"✓ Medical report analysis completed with {confidence:.2f} confidence")
            return report
            
        except Exception as e:
            logger.error(f"Report analysis failed: {e}")
            raise ValueError(f"Failed to analyze medical report: {e}")
