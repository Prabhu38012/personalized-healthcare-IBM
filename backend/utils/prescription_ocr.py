"""
Prescription OCR and Analysis Module
Handles image processing, text extraction, and medicine analysis from prescription images
"""

import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import base64
import io
import fitz  # PyMuPDF for PDF processing
from fuzzywuzzy import fuzz, process
import difflib

# Configure logging
logger = logging.getLogger(__name__)

class PrescriptionOCR:
    """Advanced OCR system for prescription analysis"""
    
    def __init__(self):
        """Initialize OCR readers and medicine database"""
        try:
            # Initialize EasyOCR reader with optimized settings
            import warnings
            warnings.filterwarnings("ignore", message=".*pin_memory.*")
            
            # Suppress torch warnings
            import os
            os.environ['TORCH_WARN'] = '0'
            
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.easyocr_reader = None
        
        # Check if Tesseract is available
        self.tesseract_available = self._check_tesseract_availability()
        if not self.tesseract_available:
            logger.info("Tesseract not available - using EasyOCR only for faster processing")
        
        # Load medicine database
        self.medicine_database = self._load_medicine_database()
        
        # Common medical terms and patterns
        self.medical_patterns = {
            'dosage': r'\b\d+\s*(mg|ml|g|mcg|units?|iu|tablets?|caps?|drops?)\b',
            'frequency': r'\b(once|twice|thrice|\d+\s*times?)\s*(daily|per\s*day|a\s*day|bid|tid|qid|od|bd|td)\b',
            'duration': r'\b(for\s*)?\d+\s*(days?|weeks?|months?|years?)\b',
            'instructions': r'\b(before|after|with|without)\s*(meals?|food|breakfast|lunch|dinner)\b'
        }
    
    def _check_tesseract_availability(self) -> bool:
        """Check if Tesseract is available and working"""
        try:
            import pytesseract
            # Try a simple test
            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
    
    def _load_medicine_database(self) -> Dict[str, Any]:
        """Load comprehensive medicine database with brand names and variations"""
        return {
            'antibiotics': {
                'generic': ['amoxicillin', 'azithromycin', 'ciprofloxacin', 'doxycycline', 
                           'erythromycin', 'penicillin', 'cephalexin', 'clindamycin', 'ampicillin',
                           'clarithromycin', 'levofloxacin', 'trimethoprim', 'sulfamethoxazole'],
                'brand': ['amoxil', 'zithromax', 'cipro', 'vibramycin', 'ery-tab', 
                         'keflex', 'cleocin', 'biaxin', 'levaquin', 'bactrim']
            },
            'pain_relievers': {
                'generic': ['ibuprofen', 'acetaminophen', 'paracetamol', 'aspirin', 
                           'naproxen', 'diclofenac', 'tramadol', 'codeine', 'morphine',
                           'oxycodone', 'hydrocodone', 'ketorolac', 'celecoxib'],
                'brand': ['advil', 'motrin', 'tylenol', 'panadol', 'crocin', 'aleve',
                         'voltaren', 'ultram', 'percocet', 'vicodin', 'toradol', 'celebrex']
            },
            'cardiovascular': {
                'generic': ['lisinopril', 'metoprolol', 'amlodipine', 'atorvastatin', 
                           'simvastatin', 'warfarin', 'clopidogrel', 'losartan', 'enalapril',
                           'propranolol', 'diltiazem', 'verapamil', 'rosuvastatin'],
                'brand': ['prinivil', 'zestril', 'lopressor', 'toprol', 'norvasc',
                         'lipitor', 'zocor', 'coumadin', 'plavix', 'cozaar', 'crestor']
            },
            'diabetes': {
                'generic': ['metformin', 'insulin', 'glipizide', 'glyburide', 
                           'sitagliptin', 'pioglitazone', 'acarbose', 'glimepiride',
                           'repaglinide', 'rosiglitazone', 'empagliflozin'],
                'brand': ['glucophage', 'humulin', 'novolin', 'glucotrol', 'diabeta',
                         'januvia', 'actos', 'precose', 'amaryl', 'prandin', 'jardiance']
            },
            'respiratory': {
                'generic': ['albuterol', 'fluticasone', 'montelukast', 'budesonide', 
                           'ipratropium', 'theophylline', 'prednisone', 'salbutamol',
                           'beclomethasone', 'salmeterol', 'formoterol'],
                'brand': ['proventil', 'ventolin', 'flovent', 'singulair', 'pulmicort',
                         'atrovent', 'theo-dur', 'qvar', 'serevent', 'foradil']
            },
            'gastrointestinal': {
                'generic': ['omeprazole', 'ranitidine', 'metoclopramide', 'loperamide', 
                           'simethicone', 'lansoprazole', 'pantoprazole', 'esomeprazole',
                           'famotidine', 'cimetidine', 'sucralfate'],
                'brand': ['prilosec', 'zantac', 'reglan', 'imodium', 'gas-x',
                         'prevacid', 'protonix', 'nexium', 'pepcid', 'tagamet', 'carafate']
            },
            'vitamins_supplements': {
                'generic': ['vitamin d', 'vitamin b12', 'folic acid', 'iron', 'calcium',
                           'magnesium', 'zinc', 'omega 3', 'multivitamin', 'vitamin c'],
                'brand': ['centrum', 'one-a-day', 'nature-made', 'kirkland', 'vitafusion']
            }
        }
    
    def process_pdf(self, pdf_data: bytes) -> List[bytes]:
        """Extract images from PDF pages"""
        try:
            pdf_document = fitz.open(stream=pdf_data, filetype="pdf")
            images = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                # Convert page to image
                mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                images.append(img_data)
                
                logger.info(f"Extracted page {page_num + 1} from PDF")
            
            pdf_document.close()
            return images
            
        except Exception as e:
            logger.error(f"PDF processing failed: {e}")
            raise
    
    def preprocess_image(self, image_data: bytes) -> Tuple[np.ndarray, np.ndarray]:
        """
        Advanced image preprocessing for better OCR accuracy
        Returns original and processed images
        """
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to OpenCV format
            opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            original = opencv_image.copy()
            
            # Resize image if too large
            height, width = opencv_image.shape[:2]
            if width > 1500 or height > 1500:
                scale = min(1500/width, 1500/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                opencv_image = cv2.resize(opencv_image, (new_width, new_height))
            
            # Convert to grayscale
            gray = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Enhance contrast using CLAHE
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(blurred)
            
            # Apply adaptive thresholding
            thresh = cv2.adaptiveThreshold(
                enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, 11, 2
            )
            
            # Morphological operations to clean up the image
            kernel = np.ones((2, 2), np.uint8)
            processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            
            return original, processed
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            raise
    
    def extract_text_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using EasyOCR"""
        if not self.easyocr_reader:
            return []
        
        try:
            results = self.easyocr_reader.readtext(image)
            extracted_text = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence results
                    extracted_text.append({
                        'text': text.strip(),
                        'confidence': confidence,
                        'bbox': bbox,
                        'method': 'easyocr'
                    })
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def extract_text_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using Tesseract OCR"""
        try:
            # Convert to PIL Image for Tesseract
            pil_image = Image.fromarray(image)
            
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
            
            extracted_text = []
            n_boxes = len(data['level'])
            
            for i in range(n_boxes):
                confidence = int(data['conf'][i])
                text = data['text'][i].strip()
                
                if confidence > 30 and text:  # Filter low confidence and empty results
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    
                    extracted_text.append({
                        'text': text,
                        'confidence': confidence / 100.0,  # Normalize to 0-1
                        'bbox': bbox,
                        'method': 'tesseract'
                    })
            
            return extracted_text
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    def combine_ocr_results(self, easyocr_results: List[Dict], tesseract_results: List[Dict]) -> List[Dict[str, Any]]:
        """Combine and deduplicate OCR results from multiple engines"""
        all_results = easyocr_results + tesseract_results
        
        # Sort by confidence
        all_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates and low-quality results
        unique_results = []
        seen_texts = set()
        
        for result in all_results:
            text_lower = result['text'].lower().strip()
            if text_lower and text_lower not in seen_texts and result['confidence'] > 0.4:
                unique_results.append(result)
                seen_texts.add(text_lower)
        
        return unique_results
    
    def identify_medicines(self, extracted_text: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify medicines from extracted text using fuzzy matching"""
        identified_medicines = []
        
        # Combine all text and create individual words
        all_text = ' '.join([item['text'] for item in extracted_text]).lower()
        words = re.findall(r'\b[a-zA-Z]+\b', all_text)
        
        # Create flat list of all medicines for fuzzy matching
        all_medicines = []
        for category, medicine_types in self.medicine_database.items():
            for med_type, medicines in medicine_types.items():
                for medicine in medicines:
                    all_medicines.append({
                        'name': medicine.lower(),
                        'category': category,
                        'type': med_type  # generic or brand
                    })
        
        # Use fuzzy matching to find medicines
        for word in words:
            if len(word) < 3:  # Skip very short words
                continue
                
            # Find best matches using fuzzy string matching
            matches = process.extract(word, [med['name'] for med in all_medicines], 
                                    scorer=fuzz.ratio, limit=3)
            
            for match_name, score in matches:
                if score >= 75:  # Minimum similarity threshold
                    # Find the medicine info
                    medicine_info = next((med for med in all_medicines if med['name'] == match_name), None)
                    if medicine_info:
                        # Find the original text item that contains this word
                        for text_item in extracted_text:
                            if word in text_item['text'].lower():
                                # Check if we already found this medicine
                                already_found = any(
                                    med['medicine_name'].lower() == match_name.lower() 
                                    for med in identified_medicines
                                )
                                
                                if not already_found:
                                    identified_medicines.append({
                                        'medicine_name': match_name.title(),
                                        'category': medicine_info['category'],
                                        'type': medicine_info['type'],
                                        'confidence': min(text_item['confidence'], score/100),
                                        'original_text': text_item['text'],
                                        'bbox': text_item['bbox'],
                                        'fuzzy_score': score
                                    })
                                break
        
        # Sort by confidence and remove duplicates
        identified_medicines.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Remove duplicates based on medicine name
        unique_medicines = []
        seen_names = set()
        for med in identified_medicines:
            name_lower = med['medicine_name'].lower()
            if name_lower not in seen_names:
                unique_medicines.append(med)
                seen_names.add(name_lower)
        
        return unique_medicines
    
    def extract_dosage_info(self, extracted_text: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract dosage, frequency, and instruction information"""
        all_text = ' '.join([item['text'] for item in extracted_text]).lower()
        
        dosage_info = {
            'dosages': [],
            'frequencies': [],
            'durations': [],
            'instructions': []
        }
        
        # Extract dosage information
        for pattern_type, pattern in self.medical_patterns.items():
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            if matches:
                if pattern_type == 'dosage':
                    dosage_info['dosages'].extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
                elif pattern_type == 'frequency':
                    dosage_info['frequencies'].extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
                elif pattern_type == 'duration':
                    dosage_info['durations'].extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
                elif pattern_type == 'instructions':
                    dosage_info['instructions'].extend([' '.join(match) if isinstance(match, tuple) else match for match in matches])
        
        return dosage_info
    
    def analyze_prescription(self, file_data: bytes, file_type: str = 'image') -> Dict[str, Any]:
        """
        Complete prescription analysis pipeline
        """
        try:
            logger.info(f"Starting prescription analysis for {file_type}")
            
            # Validate file data
            if not file_data or len(file_data) == 0:
                raise ValueError("Empty file data provided")
            
            all_combined_results = []
            
            if file_type.lower() == 'pdf':
                # Process PDF with optimized batch processing
                pdf_images = self.process_pdf(file_data)
                logger.info(f"Extracted {len(pdf_images)} pages from PDF")
                
                # Process pages with timeout and optimization
                import signal
                from concurrent.futures import ThreadPoolExecutor, TimeoutError
                import time
                
                def process_page_with_timeout(page_data):
                    i, image_data = page_data
                    try:
                        logger.info(f"Processing PDF page {i+1}")
                        
                        # Preprocess image
                        original_image, processed_image = self.preprocess_image(image_data)
                        
                        # Use EasyOCR for text extraction with timeout
                        start_time = time.time()
                        easyocr_results = self.extract_text_easyocr(processed_image)
                        
                        # Skip Tesseract for faster processing
                        tesseract_results = []
                        
                        # Combine OCR results for this page
                        page_results = self.combine_ocr_results(easyocr_results, tesseract_results)
                        
                        processing_time = time.time() - start_time
                        logger.info(f"Page {i+1} processed in {processing_time:.2f}s")
                        
                        return page_results
                    except Exception as e:
                        logger.error(f"Error processing page {i+1}: {e}")
                        return []
                
                # Process pages with limited concurrency to avoid memory issues
                max_workers = min(2, len(pdf_images))  # Limit to 2 workers
                
                try:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        page_data = [(i, image_data) for i, image_data in enumerate(pdf_images)]
                        
                        # Process with timeout per page
                        futures = []
                        for data in page_data:
                            future = executor.submit(process_page_with_timeout, data)
                            futures.append(future)
                        
                        # Collect results with timeout
                        for future in futures:
                            try:
                                page_results = future.result(timeout=30)  # 30 second timeout per page
                                all_combined_results.extend(page_results)
                            except TimeoutError:
                                logger.warning("Page processing timed out, skipping")
                                continue
                            except Exception as e:
                                logger.error(f"Page processing failed: {e}")
                                continue
                                
                except Exception as e:
                    logger.error(f"Batch processing failed, falling back to sequential: {e}")
                    # Fallback to sequential processing
                    for i, image_data in enumerate(pdf_images[:5]):  # Limit to first 5 pages
                        try:
                            page_results = process_page_with_timeout((i, image_data))
                            all_combined_results.extend(page_results)
                        except Exception as e:
                            logger.error(f"Sequential processing failed for page {i+1}: {e}")
                            continue
                    
                combined_results = all_combined_results
                logger.info(f"Combined OCR results from all PDF pages: {len(combined_results)} unique text items")
            else:
                # Process single image
                original_image, processed_image = self.preprocess_image(file_data)
                logger.info(f"Image preprocessing completed - original shape: {original_image.shape}, processed shape: {processed_image.shape}")
                
                # Extract text using multiple OCR engines
                easyocr_results = self.extract_text_easyocr(processed_image)
                logger.info(f"EasyOCR extracted {len(easyocr_results)} text items")
                
                # Only use Tesseract if it's available
                tesseract_results = []
                if self.tesseract_available:
                    try:
                        tesseract_results = self.extract_text_tesseract(processed_image)
                        logger.info(f"Tesseract extracted {len(tesseract_results)} text items")
                    except Exception as e:
                        logger.warning(f"Tesseract failed: {e}")
                        tesseract_results = []
                else:
                    logger.debug("Tesseract not available, using EasyOCR only")
                
                # Combine OCR results
                combined_results = self.combine_ocr_results(easyocr_results, tesseract_results)
                logger.info(f"Combined OCR results: {len(combined_results)} unique text items")
            
            # Identify medicines
            identified_medicines = self.identify_medicines(combined_results)
            logger.info(f"Identified {len(identified_medicines)} medicines")
            
            # Extract dosage information
            dosage_info = self.extract_dosage_info(combined_results)
            logger.info(f"Extracted dosage info: {dosage_info}")
            
            # Generate analysis summary
            analysis_summary = self._generate_analysis_summary(identified_medicines, dosage_info)
            
            # Generate comprehensive medical advice
            medical_advice = self._generate_medical_advice(identified_medicines, dosage_info, combined_results)
            
            # Convert numpy types to Python native types for JSON serialization
            result = {
                'success': True,
                'extracted_text': [str(item['text']) for item in combined_results],
                'identified_medicines': self._convert_numpy_types(identified_medicines),
                'dosage_info': self._convert_numpy_types(dosage_info),
                'analysis_summary': str(analysis_summary),
                'medical_advice': str(medical_advice),
                'confidence_score': float(self._calculate_overall_confidence(combined_results)),
                'total_medicines_found': int(len(identified_medicines))
            }
            
            logger.info(f"Analysis completed successfully - found {len(identified_medicines)} medicines with confidence {result['confidence_score']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"Prescription analysis failed: {e}", exc_info=True)
            # Generate fallback medical advice
            medical_advice = 'Unable to provide detailed medical advice due to analysis failure. Please consult with a healthcare professional and try uploading a clearer image.'
            
            return {
                'success': False,
                'error': str(e),
                'extracted_text': [],
                'identified_medicines': [],
                'dosage_info': {},
                'analysis_summary': f"Analysis failed due to technical error: {str(e)}",
                'medical_advice': 'Unable to provide medical advice due to analysis failure. Please try again with a clearer image.',
                'confidence_score': 0.0,
                'total_medicines_found': 0
            }
    
    def _calculate_overall_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score"""
        if not results:
            return 0.0
        
        total_confidence = sum(item['confidence'] for item in results)
        return total_confidence / len(results)
    
    def _generate_analysis_summary(self, medicines: List[Dict], dosage_info: Dict) -> str:
        """Generate comprehensive medical analysis summary"""
        if not medicines:
            return """**Medical Analysis:** No medications were clearly identified in this prescription. This could be due to:
            
• **Image Quality Issues:** Poor lighting, blur, or low resolution
• **Handwriting Clarity:** Difficult to read handwritten prescriptions
• **Document Format:** Unusual prescription layout or format
• **OCR Limitations:** Technical limitations in text recognition

**Recommendation:** Please ensure the prescription image is clear, well-lit, and taken from directly above the document. Consider retaking the photo or providing a clearer image."""
        
        summary_parts = []
        
        # Professional medical header
        summary_parts.append("**🩺 Medical Analysis Results:**")
        summary_parts.append("")
        
        # Medicine summary with medical context
        medicine_names = [med['medicine_name'] for med in medicines]
        avg_confidence = sum(med['confidence'] for med in medicines) / len(medicines) * 100
        
        summary_parts.append(f"**✅ Successfully Identified:** {len(medicine_names)} medication(s)")
        summary_parts.append(f"**📊 Analysis Confidence:** {avg_confidence:.1f}%")
        summary_parts.append(f"**💊 Medications:** {', '.join(medicine_names)}")
        summary_parts.append("")
        
        # Category breakdown with medical significance
        categories = {}
        for med in medicines:
            category = med['category'].replace('_', ' ').title()
            categories[category] = categories.get(category, 0) + 1
        
        if categories:
            summary_parts.append("**🏥 Therapeutic Categories:**")
            for category, count in categories.items():
                summary_parts.append(f"• **{category}:** {count} medication(s)")
            summary_parts.append("")
        
        # Dosage and administration information
        if dosage_info.get('dosages') or dosage_info.get('frequencies'):
            summary_parts.append("**📋 Dosage Information Detected:**")
            
            if dosage_info.get('dosages'):
                summary_parts.append(f"• **Dosages:** {', '.join(dosage_info['dosages'][:3])}")
            
            if dosage_info.get('frequencies'):
                summary_parts.append(f"• **Frequencies:** {', '.join(dosage_info['frequencies'][:3])}")
            
            if dosage_info.get('durations'):
                summary_parts.append(f"• **Duration:** {', '.join(dosage_info['durations'][:2])}")
            
            summary_parts.append("")
        
        # Medical recommendations
        summary_parts.append("**⚕️ Next Steps:**")
        summary_parts.append("• Verify all medications with your healthcare provider")
        summary_parts.append("• Confirm dosages and administration instructions")
        summary_parts.append("• Check for potential drug interactions")
        summary_parts.append("• Follow prescribed treatment schedule exactly")
        
        return '\n'.join(summary_parts)
    
    def _generate_medical_advice(self, medicines: List[Dict], dosage_info: Dict, extracted_text: List[Dict]) -> str:
        """Generate comprehensive medical advice based on identified medicines"""
        if not medicines:
            return """**🩺 Medical Assessment:**

I was unable to clearly identify specific medications from this prescription. This analysis limitation could be due to:

**📋 Possible Causes:**
• **Image Quality:** Insufficient lighting, blur, or low resolution
• **Handwriting Legibility:** Difficult to read handwritten prescriptions
• **Document Angle:** Photo taken at an angle or with distortion
• **OCR Technical Limits:** Current technology limitations

**🔧 Recommended Actions:**
1. **Retake Photo:** Ensure good lighting and direct overhead angle
2. **Higher Resolution:** Use your device's highest camera quality
3. **Steady Hands:** Avoid camera shake and ensure sharp focus
4. **Consult Pharmacist:** For immediate prescription reading assistance
5. **Contact Prescriber:** Verify medications directly with your doctor

**⚠️ Important Safety Note:**
Never guess about medication names or dosages. Always verify with a healthcare professional before taking any medication."""
        
        advice_parts = []
        
        # Professional medical header
        advice_parts.append("**🩺 Professional Medical Assessment:**")
        advice_parts.append("")
        advice_parts.append(f"I have successfully identified **{len(medicines)} medication(s)** from your prescription with detailed analysis:")
        advice_parts.append("")
        
        # Individual medicine analysis
        for i, medicine in enumerate(medicines, 1):
            medicine_name = medicine['medicine_name']
            category = medicine['category'].replace('_', ' ').title()
            med_type = medicine.get('type', 'generic').title()
            confidence = medicine.get('confidence', 0) * 100
            
            advice_parts.append(f"**{i}. {medicine_name}** ({med_type} - {category})")
            advice_parts.append(f"   📊 **Detection Confidence:** {confidence:.1f}%")
            
            # Get specific medical advice for this medicine
            medicine_advice = self._get_medicine_specific_advice(medicine_name, category)
            advice_parts.append(f"   {medicine_advice}")
            advice_parts.append("")
        
        # Dosage and administration guidance
        if dosage_info and any(dosage_info.values()):
            advice_parts.append("**📋 Dosage & Administration Guidance:**")
            
            if dosage_info.get('dosages'):
                advice_parts.append(f"• **Identified Dosages:** {', '.join(dosage_info['dosages'][:3])}")
                advice_parts.append("  ⚠️ Always verify exact dosage with your healthcare provider")
            
            if dosage_info.get('frequencies'):
                advice_parts.append(f"• **Administration Schedule:** {', '.join(dosage_info['frequencies'][:3])}")
                advice_parts.append("  ⏰ Set medication reminders to maintain consistent timing")
            
            if dosage_info.get('instructions'):
                advice_parts.append(f"• **Special Instructions:** {', '.join(dosage_info['instructions'][:2])}")
            
            advice_parts.append("")
        
        # Category-specific medical recommendations
        categories = [med['category'] for med in medicines]
        category_advice = self._get_category_specific_advice(categories)
        if category_advice:
            advice_parts.append("**🏥 Category-Specific Medical Recommendations:**")
            advice_parts.extend(category_advice)
            advice_parts.append("")
        
        # General medical recommendations
        advice_parts.append("**⚕️ Essential Medical Recommendations:**")
        advice_parts.append("1. **Verify with Healthcare Provider:** Confirm all medications and dosages with your doctor or pharmacist")
        advice_parts.append("2. **Complete Treatment Course:** Especially important for antibiotics - finish the entire prescription")
        advice_parts.append("3. **Monitor for Side Effects:** Report any unusual symptoms immediately to your healthcare provider")
        advice_parts.append("4. **Drug Interaction Check:** Inform all healthcare providers about ALL medications you're taking")
        advice_parts.append("5. **Proper Storage:** Store medications as directed (temperature, humidity, light exposure)")
        advice_parts.append("6. **Medication Schedule:** Take medications at consistent times for optimal effectiveness")
        advice_parts.append("")
        
        # Safety warnings
        advice_parts.append("**🚨 Important Safety Reminders:**")
        advice_parts.append("• Never share prescription medications with others")
        advice_parts.append("• Don't stop medications abruptly without medical consultation")
        advice_parts.append("• Keep medications away from children and pets")
        advice_parts.append("• Check expiration dates regularly")
        advice_parts.append("• Dispose of unused medications properly")
        advice_parts.append("")
        
        # Medical disclaimer
        advice_parts.append("**📋 Medical Disclaimer:**")
        advice_parts.append("This AI analysis is for informational purposes only and does not replace professional medical advice. Always consult with your healthcare provider or pharmacist for medication guidance, dosage confirmation, and any questions about your treatment plan.")
        
        return '\n'.join(advice_parts)
    
    def _get_medicine_specific_advice(self, medicine_name: str, category: str) -> str:
        """Get specific medical advice for a medicine"""
        medicine_lower = medicine_name.lower()
        
        # Common medicine-specific advice
        specific_advice = {
            'paracetamol': '💊 Take with or without food. Maximum 4g per day. Monitor liver function with long-term use.',
            'acetaminophen': '💊 Take with or without food. Maximum 4g per day. Avoid alcohol to prevent liver damage.',
            'ibuprofen': '💊 Take with food to reduce stomach irritation. Monitor for GI bleeding and kidney function.',
            'aspirin': '💊 Take with food. Watch for bleeding tendencies. Avoid if allergic to NSAIDs.',
            'amoxicillin': '🦠 Complete full course even if feeling better. Take with food if stomach upset occurs.',
            'azithromycin': '🦠 Can be taken with or without food. Complete entire course for effectiveness.',
            'metformin': '🩸 Take with meals to reduce GI side effects. Monitor blood glucose regularly.',
            'lisinopril': '❤️ Monitor blood pressure regularly. Rise slowly from sitting to prevent dizziness.',
            'atorvastatin': '💓 Take in evening. Monitor liver enzymes and muscle pain. Avoid grapefruit juice.',
            'omeprazole': '🍽️ Take 30-60 minutes before first meal. Long-term use may affect nutrient absorption.'
        }
        
        if medicine_lower in specific_advice:
            return specific_advice[medicine_lower]
        
        # Category-based generic advice
        category_advice = {
            'antibiotics': '🦠 Complete full course. Take at evenly spaced intervals. Monitor for allergic reactions.',
            'pain_relievers': '💊 Use lowest effective dose. Monitor for side effects. Don\'t exceed recommended duration.',
            'cardiovascular': '❤️ Monitor vital signs regularly. Maintain consistent timing. Report dizziness or chest pain.',
            'diabetes': '🩸 Monitor blood glucose. Maintain consistent meal timing. Watch for hypoglycemia symptoms.',
            'respiratory': '🫁 Use proper inhaler technique. Rinse mouth after use. Monitor breathing patterns.',
            'gastrointestinal': '🍽️ Take as directed with meals. Monitor for symptom improvement. Report persistent issues.'
        }
        
        return category_advice.get(category, '💊 Follow prescribed dosage and timing. Monitor for side effects.')
    
    def _get_category_specific_advice(self, categories: List[str]) -> List[str]:
        """Get category-specific medical advice"""
        advice = []
        unique_categories = set(categories)
        
        if 'antibiotics' in unique_categories:
            advice.extend([
                "• **Antibiotic Therapy:** Complete the full course even if symptoms improve",
                "• **Probiotic Support:** Consider probiotics to maintain gut health during treatment",
                "• **Alcohol Avoidance:** Avoid alcohol during antibiotic treatment"
            ])
        
        if 'pain_relievers' in unique_categories:
            advice.extend([
                "• **Pain Management:** Use lowest effective dose for shortest duration",
                "• **GI Protection:** Take with food to prevent stomach irritation",
                "• **Alternative Therapies:** Consider non-drug pain management techniques"
            ])
        
        if 'cardiovascular' in unique_categories:
            advice.extend([
                "• **Heart Health Monitoring:** Regular blood pressure and heart rate checks",
                "• **Lifestyle Factors:** Maintain heart-healthy diet and regular exercise",
                "• **Emergency Signs:** Seek immediate care for chest pain or severe dizziness"
            ])
        
        if 'diabetes' in unique_categories:
            advice.extend([
                "• **Blood Sugar Management:** Regular glucose monitoring as prescribed",
                "• **Meal Planning:** Consistent carbohydrate intake and meal timing",
                "• **Hypoglycemia Awareness:** Keep glucose tablets or snacks available"
            ])
        
        if 'respiratory' in unique_categories:
            advice.extend([
                "• **Inhaler Technique:** Ensure proper inhaler use for maximum effectiveness",
                "• **Respiratory Monitoring:** Track symptoms and peak flow if applicable",
                "• **Environmental Control:** Avoid known triggers and allergens"
            ])
        
        return advice
    
    def _convert_numpy_types(self, obj):
        """Convert numpy types to Python native types for JSON serialization"""
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

# Medicine information database for detailed analysis
class MedicineDatabase:
    """Comprehensive medicine information database"""
    
    def __init__(self):
        self.medicine_info = self._load_detailed_medicine_info()
    
    def _load_detailed_medicine_info(self) -> Dict[str, Dict[str, Any]]:
        """Load detailed medicine information"""
        return {
            'paracetamol': {
                'generic_name': 'Acetaminophen',
                'brand_names': ['Tylenol', 'Panadol', 'Crocin'],
                'category': 'Pain Reliever/Fever Reducer',
                'uses': ['Pain relief', 'Fever reduction', 'Headache', 'Muscle aches'],
                'common_dosage': '500-1000mg every 4-6 hours',
                'max_daily_dose': '4000mg',
                'side_effects': ['Nausea', 'Liver damage (overdose)', 'Allergic reactions'],
                'contraindications': ['Severe liver disease', 'Alcohol dependency'],
                'interactions': ['Warfarin', 'Alcohol', 'Phenytoin']
            },
            'ibuprofen': {
                'generic_name': 'Ibuprofen',
                'brand_names': ['Advil', 'Motrin', 'Brufen'],
                'category': 'NSAID Pain Reliever',
                'uses': ['Pain relief', 'Inflammation', 'Fever reduction', 'Arthritis'],
                'common_dosage': '200-400mg every 4-6 hours',
                'max_daily_dose': '1200mg (OTC), 2400mg (prescription)',
                'side_effects': ['Stomach upset', 'Heartburn', 'Dizziness', 'Kidney problems'],
                'contraindications': ['Peptic ulcers', 'Severe heart failure', 'Kidney disease'],
                'interactions': ['Blood thinners', 'ACE inhibitors', 'Lithium']
            },
            'amoxicillin': {
                'generic_name': 'Amoxicillin',
                'brand_names': ['Amoxil', 'Trimox', 'Moxatag'],
                'category': 'Antibiotic (Penicillin)',
                'uses': ['Bacterial infections', 'Respiratory infections', 'Urinary tract infections'],
                'common_dosage': '250-500mg every 8 hours',
                'max_daily_dose': '3000mg',
                'side_effects': ['Nausea', 'Diarrhea', 'Allergic reactions', 'Rash'],
                'contraindications': ['Penicillin allergy', 'Mononucleosis'],
                'interactions': ['Birth control pills', 'Methotrexate', 'Probenecid']
            }
        }
    
    def get_medicine_info(self, medicine_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a medicine"""
        medicine_lower = medicine_name.lower()
        
        # Direct match
        if medicine_lower in self.medicine_info:
            return self.medicine_info[medicine_lower]
        
        # Check brand names
        for generic_name, info in self.medicine_info.items():
            brand_names = [name.lower() for name in info.get('brand_names', [])]
            if medicine_lower in brand_names:
                return info
        
        return None
    
    def generate_medicine_advice(self, medicine_name: str) -> str:
        """Generate professional medical advice for a medicine"""
        info = self.get_medicine_info(medicine_name)
        
        if not info:
            return f"Information about {medicine_name} is not available in our database. Please consult your healthcare provider for detailed information."
        
        advice = f"""
**{info['generic_name']} ({medicine_name.title()})**

**Category:** {info['category']}

**Primary Uses:** {', '.join(info['uses'])}

**Typical Dosage:** {info['common_dosage']}
**Maximum Daily Dose:** {info['max_daily_dose']}

**Important Safety Information:**
- **Common Side Effects:** {', '.join(info['side_effects'])}
- **Contraindications:** {', '.join(info['contraindications'])}
- **Drug Interactions:** {', '.join(info['interactions'])}

**Professional Recommendation:** Always follow your doctor's prescribed dosage and duration. Do not exceed the maximum daily dose. If you experience any unusual side effects, contact your healthcare provider immediately.
        """
        
        return advice.strip()
