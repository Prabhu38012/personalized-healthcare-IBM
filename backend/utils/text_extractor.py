"""
Advanced Text Extraction and Analysis Utility
Combines multiple libraries for comprehensive text processing
"""

import pdfplumber
import PyMuPDF  # fitz
import easyocr
import cv2
import numpy as np
from PIL import Image
import pytesseract
import re
import logging
from typing import Dict, List, Optional, Union
import io
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextExtractor:
    """Advanced text extraction and analysis utility"""
    
    def __init__(self):
        self.ocr_reader = easyocr.Reader(['en'])
        logger.info("TextExtractor initialized with OCR capabilities")
    
    def extract_from_pdf(self, pdf_path: str, method: str = 'pdfplumber') -> Dict[str, any]:
        """
        Extract text from PDF using multiple methods
        
        Args:
            pdf_path: Path to PDF file
            method: 'pdfplumber', 'pymupdf', or 'both'
        
        Returns:
            Dictionary with extracted text and metadata
        """
        results = {
            'text': '',
            'pages': [],
            'metadata': {},
            'method_used': method
        }
        
        try:
            if method in ['pdfplumber', 'both']:
                results.update(self._extract_with_pdfplumber(pdf_path))
            
            if method in ['pymupdf', 'both']:
                pymupdf_results = self._extract_with_pymupdf(pdf_path)
                if method == 'both':
                    results['pymupdf_text'] = pymupdf_results['text']
                else:
                    results.update(pymupdf_results)
                    
        except Exception as e:
            logger.error(f"Error extracting from PDF {pdf_path}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> Dict[str, any]:
        """Extract text using pdfplumber (excellent for structured PDFs)"""
        results = {'text': '', 'pages': [], 'metadata': {}}
        
        with pdfplumber.open(pdf_path) as pdf:
            results['metadata'] = {
                'total_pages': len(pdf.pages),
                'creator': pdf.metadata.get('Creator', ''),
                'producer': pdf.metadata.get('Producer', ''),
                'subject': pdf.metadata.get('Subject', '')
            }
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ''
                results['pages'].append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'tables': page.extract_tables(),
                    'words': len(page_text.split())
                })
                results['text'] += page_text + '\n'
        
        return results
    
    def _extract_with_pymupdf(self, pdf_path: str) -> Dict[str, any]:
        """Extract text using PyMuPDF (good for complex layouts)"""
        results = {'text': '', 'pages': [], 'metadata': {}}
        
        doc = fitz.open(pdf_path)
        results['metadata'] = {
            'total_pages': doc.page_count,
            'metadata': doc.metadata
        }
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            page_text = page.get_text()
            results['pages'].append({
                'page_number': page_num + 1,
                'text': page_text,
                'words': len(page_text.split())
            })
            results['text'] += page_text + '\n'
        
        doc.close()
        return results
    
    def extract_from_image(self, image_path: str, method: str = 'easyocr') -> Dict[str, any]:
        """
        Extract text from images using OCR
        
        Args:
            image_path: Path to image file
            method: 'easyocr', 'tesseract', or 'both'
        
        Returns:
            Dictionary with extracted text and confidence scores
        """
        results = {
            'text': '',
            'confidence': 0,
            'method_used': method,
            'words': []
        }
        
        try:
            if method in ['easyocr', 'both']:
                results.update(self._extract_with_easyocr(image_path))
            
            if method in ['tesseract', 'both']:
                tesseract_results = self._extract_with_tesseract(image_path)
                if method == 'both':
                    results['tesseract_text'] = tesseract_results['text']
                    results['tesseract_confidence'] = tesseract_results['confidence']
                else:
                    results.update(tesseract_results)
                    
        except Exception as e:
            logger.error(f"Error extracting from image {image_path}: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def _extract_with_easyocr(self, image_path: str) -> Dict[str, any]:
        """Extract text using EasyOCR (good for handwriting)"""
        results = self.ocr_reader.readtext(image_path)
        
        text_parts = []
        confidences = []
        words = []
        
        for (bbox, text, confidence) in results:
            text_parts.append(text)
            confidences.append(confidence)
            words.append({
                'text': text,
                'confidence': confidence,
                'bbox': bbox
            })
        
        return {
            'text': ' '.join(text_parts),
            'confidence': np.mean(confidences) if confidences else 0,
            'words': words
        }
    
    def _extract_with_tesseract(self, image_path: str) -> Dict[str, any]:
        """Extract text using Tesseract OCR"""
        image = Image.open(image_path)
        
        # Get text with confidence
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        text_parts = []
        confidences = []
        words = []
        
        for i in range(len(data['text'])):
            if int(data['conf'][i]) > 0:  # Only include confident predictions
                text = data['text'][i].strip()
                if text:
                    text_parts.append(text)
                    confidences.append(int(data['conf'][i]))
                    words.append({
                        'text': text,
                        'confidence': int(data['conf'][i]),
                        'bbox': (data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i])
                    })
        
        return {
            'text': ' '.join(text_parts),
            'confidence': np.mean(confidences) if confidences else 0,
            'words': words
        }
    
    def preprocess_image_for_ocr(self, image_path: str, output_path: str = None) -> str:
        """
        Preprocess image to improve OCR accuracy
        
        Args:
            image_path: Input image path
            output_path: Output path for processed image
        
        Returns:
            Path to processed image
        """
        # Read image
        image = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Apply threshold to get binary image
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save processed image
        if not output_path:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_processed{ext}"
        
        cv2.imwrite(output_path, binary)
        logger.info(f"Preprocessed image saved to: {output_path}")
        
        return output_path
    
    def analyze_text(self, text: str) -> Dict[str, any]:
        """
        Analyze extracted text for insights
        
        Args:
            text: Text to analyze
        
        Returns:
            Dictionary with text analysis results
        """
        if not text:
            return {'error': 'No text provided'}
        
        # Basic statistics
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        paragraphs = text.split('\n\n')
        
        # Find medical terms (basic pattern matching)
        medical_patterns = [
            r'\b\d+\s*mg\b',  # Dosages
            r'\b\d+\s*ml\b',  # Volumes
            r'\btablet[s]?\b',  # Tablets
            r'\bcapsule[s]?\b',  # Capsules
            r'\bdaily\b|\btwice daily\b|\bthrice daily\b',  # Frequency
            r'\bafter meals\b|\bbefore meals\b',  # Instructions
        ]
        
        medical_terms = []
        for pattern in medical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            medical_terms.extend(matches)
        
        # Extract numbers and dates
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        
        return {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'paragraph_count': len([p for p in paragraphs if p.strip()]),
            'character_count': len(text),
            'medical_terms': medical_terms,
            'numbers_found': numbers,
            'dates_found': dates,
            'avg_words_per_sentence': len(words) / max(len(sentences), 1),
            'readability_score': self._calculate_readability(text)
        }
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate simple readability score (Flesch Reading Ease approximation)"""
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        syllables = sum([self._count_syllables(word) for word in words])
        
        if len(sentences) == 0 or len(words) == 0:
            return 0
        
        # Simplified Flesch Reading Ease formula
        score = 206.835 - (1.015 * (len(words) / len(sentences))) - (84.6 * (syllables / len(words)))
        return max(0, min(100, score))
    
    def _count_syllables(self, word: str) -> int:
        """Estimate syllable count in a word"""
        word = word.lower()
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False
        
        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False
        
        # Handle silent 'e'
        if word.endswith('e'):
            syllable_count -= 1
        
        return max(1, syllable_count)

# Example usage and testing
if __name__ == "__main__":
    extractor = TextExtractor()
    
    # Test with a sample text
    sample_text = "Take 2 tablets of Aspirin 325mg twice daily after meals. Follow up on 12/25/2024."
    analysis = extractor.analyze_text(sample_text)
    print("Text Analysis:", analysis)
