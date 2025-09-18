"""
Fuzzy matching utilities for medical text analysis
"""

from typing import List, Dict, Any, Tuple, Optional
from rapidfuzz import fuzz, process
import json
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MatchResult:
    """Represents a fuzzy match result"""
    term: str
    matched_text: str
    score: float
    category: str
    metadata: Optional[Dict[str, Any]] = None

class MedicalFuzzyMatcher:
    """Handles fuzzy matching of medical terms with configurable thresholds"""
    
    def __init__(self, medical_terms_path: str = None):
        """
        Initialize the fuzzy matcher with medical terms
        
        Args:
            medical_terms_path: Path to JSON file containing medical terms
        """
        self.medical_terms = {}
        self.initialize_terms(medical_terms_path)
        
        # Default thresholds for different match types
        self.thresholds = {
            'exact': 100,
            'high_confidence': 90,
            'medium_confidence': 80,
            'low_confidence': 70
        }
    
    def initialize_terms(self, file_path: str = None) -> None:
        """Load medical terms from JSON file"""
        if not file_path:
            # Use default path if none provided
            file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'data',
                'medical_terms.json'
            )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.medical_terms = json.load(f)
            logger.info(f"Loaded medical terms from {file_path}")
        except Exception as e:
            logger.error(f"Failed to load medical terms: {e}")
            self.medical_terms = {}
    
    def find_best_matches(self, text: str, category: str = None, 
                         score_cutoff: float = 80.0, limit: int = 3) -> List[MatchResult]:
        """
        Find best fuzzy matches for text in medical terms
        
        Args:
            text: Text to match against medical terms
            category: Specific category to search in (conditions, lab_tests, etc.)
            score_cutoff: Minimum score to consider a match (0-100)
            limit: Maximum number of matches to return
            
        Returns:
            List of MatchResult objects
        """
        if not text or not text.strip():
            return []
            
        text = text.lower().strip()
        results = []
        
        # Determine which categories to search
        categories_to_search = [category] if category else self.medical_terms.keys()
        
        for cat in categories_to_search:
            if cat not in self.medical_terms:
                continue
                
            # Handle different term structures
            if cat == 'conditions':
                for condition, data in self.medical_terms[cat].items():
                    # Check main terms
                    for term in data.get('terms', []) + data.get('abbreviations', []) + data.get('common_misspellings', []):
                        score = fuzz.ratio(text, term.lower())
                        if score >= score_cutoff:
                            results.append(MatchResult(
                                term=term,
                                matched_text=text,
                                score=score,
                                category=f"{cat}.{condition}",
                                metadata={"type": "condition"}
                            ))
            
            elif cat in ['lab_tests', 'medications', 'symptoms']:
                if cat == 'lab_tests':
                    for test_name, test_data in self.medical_terms[cat].items():
                        # Match test name
                        score = fuzz.ratio(text, test_name.lower())
                        if score >= score_cutoff:
                            results.append(MatchResult(
                                term=test_name,
                                matched_text=text,
                                score=score,
                                category=f"{cat}.{test_name}",
                                metadata={"type": "lab_test", "full_name": test_data.get('name', '')}
                            ))
                        
                        # Match test components
                        for component in test_data.get('components', []):
                            score = fuzz.ratio(text, component.lower())
                            if score >= score_cutoff:
                                results.append(MatchResult(
                                    term=component,
                                    matched_text=text,
                                    score=score,
                                    category=f"{cat}.{test_name}",
                                    metadata={"type": "lab_component", "test_name": test_name}
                                ))
                
                elif cat == 'medications':
                    for med_class, med_list in self.medical_terms[cat].items():
                        for med in med_list:
                            score = fuzz.ratio(text, med.lower())
                            if score >= score_cutoff:
                                results.append(MatchResult(
                                    term=med,
                                    matched_text=text,
                                    score=score,
                                    category=f"{cat}.{med_class}",
                                    metadata={"type": "medication", "class": med_class}
                                ))
                
                elif cat == 'symptoms':
                    for symptom_type, symptom_list in self.medical_terms[cat].items():
                        for symptom in symptom_list:
                            score = fuzz.ratio(text, symptom.lower())
                            if score >= score_cutoff:
                                results.append(MatchResult(
                                    term=symptom,
                                    matched_text=text,
                                    score=score,
                                    category=f"{cat}.{symptom_type}",
                                    metadata={"type": "symptom", "category": symptom_type}
                                ))
        
        # Sort by score (highest first) and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:limit]
    
    def is_medical_term(self, text: str, category: str = None, 
                       min_confidence: float = 80.0) -> bool:
        """Check if text is likely a medical term"""
        if not text or not text.strip():
            return False
            
        matches = self.find_best_matches(text, category=category, score_cutoff=min_confidence, limit=1)
        return len(matches) > 0 and matches[0].score >= min_confidence

# Create a singleton instance
fuzzy_matcher = MedicalFuzzyMatcher()
