"""
Drug Advisor Service Layer
- NLP-based extraction using Hugging Face transformers (with graceful fallback)
- Drug interaction checks via openFDA/DrugBank (fallback to curated rules)
- Age-specific dosage guidance (fallback heuristics)
- Alternative medication suggestions (category-based)
- Optional IBM Watson NLU integration for medical context understanding
"""
from __future__ import annotations

import os
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache

import requests

try:
    # optional cache for API requests
    import requests_cache  # type: ignore
    requests_cache.install_cache("drug_api_cache", backend="memory", expire_after=60 * 60)
except Exception:
    pass

logger = logging.getLogger(__name__)

# -----------------------------
# Environment / Config
# -----------------------------
HF_MODEL = os.getenv("HF_DRUG_NER_MODEL", "kamalkraj/BioBERT-NER")
ENABLE_HF_NER = os.getenv("ENABLE_HF_NER", "0") in {"1", "true", "True"}
WATSON_API_KEY = os.getenv("WATSON_API_KEY")
WATSON_URL = os.getenv("WATSON_URL")  # e.g., NLU or Discovery endpoint
OPENFDA_BASE = os.getenv("OPENFDA_BASE", "https://api.fda.gov")
RXNAV_BASE = os.getenv("RXNAV_BASE", "https://rxnav.nlm.nih.gov/REST")

# Simple curated interaction rules as a safe fallback
CURATED_INTERACTIONS: Dict[Tuple[str, str], str] = {
    ("ibuprofen", "aspirin"): "Increased risk of gastrointestinal bleeding when combined.",
    ("warfarin", "aspirin"): "Both increase bleeding risk. Avoid or monitor closely.",
    ("metformin", "contrast dye"): "Contrast agents may increase risk of lactic acidosis.",
    ("lisinopril", "spironolactone"): "Combination can raise potassium levels (hyperkalemia).",
}

# Expanded categories for alternatives
DRUG_CATEGORIES: Dict[str, str] = {
    # Pain/fever
    "ibuprofen": "NSAID", "naproxen": "NSAID", "aspirin": "NSAID",
    "acetaminophen": "Analgesic", "paracetamol": "Analgesic",
    # Antibiotics
    "amoxicillin": "Antibiotic", "azithromycin": "Antibiotic", "doxycycline": "Antibiotic",
    # Lipids
    "atorvastatin": "Statin", "simvastatin": "Statin", "rosuvastatin": "Statin",
    # Diabetes
    "metformin": "Antidiabetic", "glipizide": "Antidiabetic", "empagliflozin": "Antidiabetic",
    # BP/Cardio
    "lisinopril": "ACE inhibitor", "enalapril": "ACE inhibitor",
    "losartan": "ARB", "valsartan": "ARB",
    "metoprolol": "Beta blocker", "atenolol": "Beta blocker",
    "amlodipine": "Calcium channel blocker",
    # GI
    "omeprazole": "PPI", "pantoprazole": "PPI",
    "ranitidine": "H2 blocker", "famotidine": "H2 blocker",
    # Psych
    "sertraline": "SSRI", "fluoxetine": "SSRI", "venlafaxine": "SNRI",
}

CATEGORY_ALTERNATIVES: Dict[str, List[str]] = {
    "NSAID": ["acetaminophen", "paracetamol"],
    "Analgesic": ["ibuprofen", "naproxen"],
    "Antibiotic": ["azithromycin", "amoxicillin", "doxycycline"],
    "Statin": ["atorvastatin", "rosuvastatin", "simvastatin"],
    "Antidiabetic": ["metformin", "empagliflozin", "glipizide"],
    "ACE inhibitor": ["lisinopril", "enalapril"],
    "ARB": ["losartan", "valsartan"],
    "Beta blocker": ["metoprolol", "atenolol"],
    "Calcium channel blocker": ["amlodipine"],
    "PPI": ["omeprazole", "pantoprazole"],
    "H2 blocker": ["famotidine"],
    "SSRI": ["sertraline", "fluoxetine"],
    "SNRI": ["venlafaxine"],
}

# Enhanced age-specific dosage rules with more comprehensive coverage
AGE_DOSAGE_RULES: Dict[str, Dict[str, str]] = {
    "ibuprofen": {
        "infant": "4-10 mg/kg every 6-8 hours (max 40 mg/kg/day). Not recommended under 6 months.",
        "child": "5-10 mg/kg every 6-8 hours (max 40 mg/kg/day).",
        "adult": "200-400 mg every 4-6 hours as needed (max 1200 mg OTC/day).",
        "older": "Use lowest effective dose; monitor renal function and GI risk. Consider 200mg every 8 hours.",
    },
    "acetaminophen": {
        "infant": "10-15 mg/kg every 4-6 hours (max 75 mg/kg/day).",
        "child": "10-15 mg/kg every 4-6 hours (max 75 mg/kg/day).",
        "adult": "325-1000 mg every 4-6 hours (max 3000 mg/day typical).",
        "older": "Do not exceed 3000 mg/day; consider liver status. May need dose reduction.",
    },
    "amoxicillin": {
        "infant": "20-40 mg/kg/day divided every 12 hours (under 3 months: 20-30 mg/kg/day).",
        "child": "25-45 mg/kg/day divided every 12 hours depending on indication.",
        "adult": "500 mg every 8-12 hours depending on infection.",
        "older": "Adjust based on renal function. Consider 250mg every 12 hours if CrCl <30.",
    },
    "aspirin": {
        "infant": "Not recommended under 12 years due to Reye's syndrome risk.",
        "child": "Not recommended under 12 years due to Reye's syndrome risk.",
        "adult": "325-650 mg every 4-6 hours (max 4000 mg/day).",
        "older": "Use lowest effective dose. Consider 81mg daily for cardiovascular protection.",
    },
    "metformin": {
        "infant": "Not typically used in pediatric patients under 10 years.",
        "child": "500mg twice daily, increase gradually (10-16 years).",
        "adult": "500-1000mg twice daily with meals.",
        "older": "Start with 500mg daily, adjust based on renal function. Monitor eGFR.",
    },
    "lisinopril": {
        "infant": "Not recommended under 6 years.",
        "child": "0.07 mg/kg once daily (6-16 years), max 5mg daily.",
        "adult": "10-40 mg once daily.",
        "older": "Start with 2.5-5mg daily. Monitor blood pressure and renal function.",
    },
    "atorvastatin": {
        "infant": "Not recommended under 10 years.",
        "child": "10mg daily (10-17 years) for familial hypercholesterolemia.",
        "adult": "10-80 mg once daily.",
        "older": "Start with 10-20mg daily. Monitor liver enzymes and muscle symptoms.",
    },
    "omeprazole": {
        "infant": "0.5-1 mg/kg once daily (under 1 year).",
        "child": "0.5-1 mg/kg once daily (1-16 years).",
        "adult": "20-40 mg once daily before breakfast.",
        "older": "20mg daily. Consider lower dose if renal impairment.",
    },
    "warfarin": {
        "infant": "0.1-0.2 mg/kg daily, adjust based on INR.",
        "child": "0.1-0.2 mg/kg daily, adjust based on INR.",
        "adult": "2-10 mg daily, adjust based on INR (target 2-3).",
        "older": "Start with 2-5mg daily. More sensitive to dose changes.",
    },
    "prednisone": {
        "infant": "0.5-2 mg/kg/day divided every 6-12 hours.",
        "child": "0.5-2 mg/kg/day divided every 6-12 hours.",
        "adult": "5-60 mg daily depending on condition.",
        "older": "Use lowest effective dose. Monitor for osteoporosis and diabetes.",
    }
}

# -----------------------------
# NLP Extraction (Hugging Face)
# -----------------------------

class DrugNER:
    def __init__(self) -> None:
        self.pipeline = None
        if ENABLE_HF_NER:
            try:
                from transformers import pipeline  # type: ignore
                self.pipeline = pipeline(
                    "token-classification",
                    model=HF_MODEL,
                    aggregation_strategy="simple",
                )
                logger.info("Loaded Hugging Face NER model: %s", HF_MODEL)
            except Exception as e:
                logger.warning("Falling back to regex NER due to error loading model: %s", e)
                self.pipeline = None
        else:
            logger.info("ENABLE_HF_NER is disabled; using fast regex-based extractor")

    def extract(self, text: str) -> List[Dict[str, Any]]:
        if not text:
            return []
        if self.pipeline is None:
            # Enhanced regex-based fallback with higher recall and medical context
            candidates = set()
            vocab = set(DRUG_CATEGORIES.keys())
            lower_text = text.lower()
            
            # 1) Known vocabulary hits with context awareness
            for name in vocab:
                if name in lower_text:
                    # Check if it's in a medical context
                    context_start = max(0, lower_text.find(name) - 50)
                    context_end = min(len(lower_text), lower_text.find(name) + len(name) + 50)
                    context = lower_text[context_start:context_end]
                    
                    # Medical context keywords
                    medical_keywords = [
                        'prescription', 'medication', 'drug', 'tablet', 'capsule', 'mg', 'dose', 'dosage',
                        'take', 'twice', 'daily', 'bid', 'tid', 'qid', 'prn', 'with food', 'after meals',
                        'rx', 'pharmacy', 'doctor', 'physician', 'prescribed', 'medicine'
                    ]
                    
                    if any(keyword in context for keyword in medical_keywords):
                        candidates.add(name)
            
            # 2) Enhanced suffix/pattern heuristics for common drug families
            suffix_patterns = [
                r"\b([A-Za-z][A-Za-z\-]{2,}cillin)\b",   # amoxicillin, ampicillin
                r"\b([A-Za-z][A-Za-z\-]{2,}statin)\b",   # atorvastatin, simvastatin
                r"\b([A-Za-z][A-Za-z\-]{2,}pril)\b",     # lisinopril, enalapril
                r"\b([A-Za-z][A-Za-z\-]{2,}sartan)\b",   # losartan, valsartan
                r"\b([A-Za-z][A-Za-z\-]{2,}olol)\b",     # metoprolol, atenolol
                r"\b([A-Za-z][A-Za-z\-]{2,}pine)\b",     # amlodipine, nifedipine
                r"\b([A-Za-z][A-Za-z\-]{2,}zole)\b",     # omeprazole, pantoprazole
                r"\b([A-Za-z][A-Za-z\-]{2,}mycin)\b",    # azithromycin, erythromycin
                r"\b([A-Za-z][A-Za-z\-]{2,}cycline)\b",  # doxycycline, tetracycline
                r"\b([A-Za-z][A-Za-z\-]{2,}pam)\b",      # diazepam, lorazepam
                r"\b([A-Za-z][A-Za-z\-]{2,}zepam)\b",    # diazepam, clonazepam
                r"\b([A-Za-z][A-Za-z\-]{2,}zine)\b",     # promethazine, chlorpromazine
                r"\b(paracetamol|acetaminophen)\b",
                r"\b(ibuprofen|naproxen|diclofenac)\b",
                r"\b(metformin|glipizide|sitagliptin)\b",
                r"\b(warfarin|heparin|clopidogrel)\b",
                r"\b(prednisone|hydrocortisone|dexamethasone)\b",
                r"\b(insulin|glargine|lispro|aspart)\b",
                r"\b(morphine|fentanyl|oxycodone|hydrocodone)\b",
                r"\b(loratadine|cetirizine|fexofenadine)\b",
                r"\b(omeprazole|lansoprazole|pantoprazole)\b",
                r"\b(metoprolol|propranolol|atenolol)\b"
            ]
            
            for pat in suffix_patterns:
                for m in re.finditer(pat, text, flags=re.IGNORECASE):
                    candidates.add(m.group(1).lower())
            
            # 3) Names followed by a strength (very common prescription pattern)
            strength_pattern = r"\b([A-Za-z][A-Za-z\-]{2,})\s*(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu|units?))\b"
            for m in re.finditer(strength_pattern, text, flags=re.IGNORECASE):
                candidates.add(m.group(1).lower())
            
            # 4) Brand name patterns (often capitalized or in quotes)
            brand_patterns = [
                r"\b([A-Z][a-z]+(?:in|ol|ide|ine|pam|zole))\b",  # Common brand name patterns
                r'"([A-Za-z][A-Za-z\-]{2,})"',  # Names in quotes
                r"\b([A-Z][A-Z\-]+)\b",  # All caps or hyphenated names
            ]
            
            for pat in brand_patterns:
                for m in re.finditer(pat, text):
                    candidates.add(m.group(1).lower())
            
            # 5) Generic name patterns (often lowercase, scientific naming)
            generic_patterns = [
                r"\b([a-z]+(?:mycin|cycline|cillin|statin|pril|sartan|olol|pine|zole))\b",
                r"\b([a-z]+(?:pam|zepam|zine|fen|profen|metin|formin))\b"
            ]
            
            for pat in generic_patterns:
                for m in re.finditer(pat, text):
                    candidates.add(m.group(1).lower())
            
            # Filter candidates to remove common false positives
            false_positives = {
                'daily', 'twice', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
                'morning', 'evening', 'night', 'day', 'week', 'month', 'year', 'time', 'times',
                'tablet', 'capsule', 'pill', 'dose', 'dosage', 'mg', 'ml', 'g', 'mcg', 'iu',
                'take', 'with', 'after', 'before', 'food', 'water', 'meal', 'breakfast', 'lunch', 'dinner'
            }
            
            candidates = candidates - false_positives
            
            # Score candidates based on context and patterns
            scored_candidates = []
            for candidate in candidates:
                score = 0.5  # Base score
                
                # Increase score for medical context
                context_start = max(0, lower_text.find(candidate) - 30)
                context_end = min(len(lower_text), lower_text.find(candidate) + len(candidate) + 30)
                context = lower_text[context_start:context_end]
                
                if any(keyword in context for keyword in ['prescription', 'medication', 'drug', 'rx']):
                    score += 0.2
                if any(keyword in context for keyword in ['mg', 'dose', 'tablet', 'capsule']):
                    score += 0.2
                if any(keyword in context for keyword in ['take', 'twice', 'daily', 'bid', 'tid']):
                    score += 0.1
                
                # Increase score for known drug patterns
                if candidate in vocab:
                    score += 0.3
                
                scored_candidates.append({
                    "entity_group": "DRUG",
                    "word": candidate,
                    "score": min(score, 1.0),
                    "start": text.lower().find(candidate),
                    "end": text.lower().find(candidate) + len(candidate),
                })
            
            # Sort by score and return top candidates
            scored_candidates.sort(key=lambda x: x['score'], reverse=True)
            return scored_candidates[:20]  # Limit to top 20 candidates
        try:
            outputs = self.pipeline(text)
            results: List[Dict[str, Any]] = []
            for item in outputs:
                word = item.get("word") or item.get("entity")
                if not word:
                    continue
                results.append(
                    {
                        "entity_group": item.get("entity_group", "DRUG"),
                        "word": str(word),
                        "score": float(item.get("score", 0.0)),
                        "start": int(item.get("start", 0)),
                        "end": int(item.get("end", 0)),
                    }
                )
            return results
        except Exception as e:
            logger.error("NER pipeline failed: %s", e)
            return []

# -----------------------------
# External API helpers
# -----------------------------

def query_openfda_label(drug_name: str) -> Optional[Dict[str, Any]]:
    try:
        url = f"{OPENFDA_BASE}/drug/label.json"
        params = {
            "search": f"openfda.brand_name:{drug_name}^ OR openfda.generic_name:{drug_name}^",
            "limit": 1,
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if data.get("results"):
                return data["results"][0]
    except Exception as e:
        logger.debug("openFDA query failed for %s: %s", drug_name, e)
    return None


def detect_interactions_openfda(drugs: List[str]) -> List[Dict[str, Any]]:
    interactions: List[Dict[str, Any]] = []
    # naive pairwise check using label 'drug_interactions' text
    normalized = [d.lower().strip() for d in drugs if d]
    for i in range(len(normalized)):
        for j in range(i + 1, len(normalized)):
            a, b = normalized[i], normalized[j]
            # quick curated first
            note = CURATED_INTERACTIONS.get((a, b)) or CURATED_INTERACTIONS.get((b, a))
            if note:
                interactions.append({"pair": [a, b], "severity": "moderate", "source": "curated", "note": note})
                continue
            # try openFDA labels for each drug and look for the other's name in interaction text
            for primary, other in [(a, b), (b, a)]:
                label = query_openfda_label(primary)
                if not label:
                    continue
                text_fields = [
                    *(label.get("drug_interactions", []) or []),
                    *(label.get("warnings", []) or []),
                    *(label.get("precautions", []) or []),
                ]
                hay = "\n".join(text_fields).lower()
                if other in hay:
                    interactions.append(
                        {
                            "pair": [a, b],
                            "severity": "potential",
                            "source": "openfda_label",
                            "note": f"Label of {primary} mentions {other} in interactions/warnings.",
                        }
                    )
                    break
    # de-duplicate
    uniq = {tuple(sorted(item["pair"])): item for item in interactions}
    return list(uniq.values())


def detect_interactions_rxnav(drugs: List[str]) -> List[Dict[str, Any]]:
    """Use RxNav Interaction API for higher-coverage interactions.
    Tries the names endpoint to avoid explicit RXCUI lookups.
    """
    interactions: List[Dict[str, Any]] = []
    try:
        names = ",".join([d.strip() for d in drugs if d])
        if not names:
            return []
        url = f"{RXNAV_BASE}/interaction/interaction.json"
        params = {"names": names}
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return []
        data = r.json() or {}
        # The structure contains 'interactionTypeGroup' → groups → 'interactionType' → 'interactionPair'
        groups = data.get("interactionTypeGroup", []) or []
        for g in groups:
            types = g.get("interactionType", []) or []
            for t in types:
                pairs = t.get("interactionPair", []) or []
                for p in pairs:
                    desc = p.get("description") or "Interaction noted by RxNav"
                    # pull the pair names (minConcept item list)
                    minc = p.get("interactionConcept", []) or []
                    pair_names = []
                    for mc in minc:
                        nm = mc.get("minConceptItem", {}).get("name")
                        if nm:
                            pair_names.append(nm.lower())
                    if len(pair_names) >= 2:
                        a, b = pair_names[0], pair_names[1]
                    else:
                        continue
                    severity = (p.get("severity") or "unknown").lower()
                    interactions.append({
                        "pair": sorted([a, b]),
                        "severity": severity,
                        "source": "rxnav",
                        "note": desc,
                    })
        # de-duplicate
        uniq = {tuple(sorted(item["pair"])): item for item in interactions}
        return list(uniq.values())
    except Exception:
        return []


def watson_enrich_context(text: str) -> Dict[str, Any]:
    if not WATSON_API_KEY or not WATSON_URL:
        return {"used": False, "reason": "Missing IBM Watson credentials"}
    try:
        # Example NLU sentiment/categories call placeholder
        headers = {"Content-Type": "application/json"}
        auth = ("apikey", WATSON_API_KEY)
        payload = {"text": text, "features": {"keywords": {"limit": 5}, "entities": {"limit": 5}}}
        r = requests.post(WATSON_URL, headers=headers, auth=auth, data=json.dumps(payload), timeout=15)
        if r.status_code == 200:
            return {"used": True, "data": r.json()}
        return {"used": False, "reason": f"HTTP {r.status_code}"}
    except Exception as e:
        return {"used": False, "reason": str(e)}

# -----------------------------
# Public Service Functions
# -----------------------------

ner_singleton: Optional[DrugNER] = None

def get_ner() -> DrugNER:
    global ner_singleton
    if ner_singleton is None:
        ner_singleton = DrugNER()
    return ner_singleton


@lru_cache(maxsize=128)
def _extract_drug_info_cached(text: str) -> Dict[str, Any]:
    ner = get_ner()
    ents = ner.extract(text)
    # Normalize and group by unique drug words
    drugs: List[Dict[str, Any]] = []
    seen = set()

    # Precompute simple regex captures for dosage elements
    # strength like 200 mg, 5mg, 2 g, 10ml
    strength_re = re.compile(r"(\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu))", re.IGNORECASE)
    # frequency shorthand: bid/tid/qid/qhs/qam/qd/qod/prn, or textual frequencies
    freq_re = re.compile(r"\b((?:q\d+h)|(?:q\d+hr)|(?:q\d+hours)|(?:bid)|(?:tid)|(?:qid)|(?:qd)|(?:qod)|(?:qam)|(?:qpm)|(?:qhs)|(?:hs)|(?:prn)|(?:once\s+daily)|(?:twice\s+daily)|(?:every\s+\d+\s*(?:h|hr|hour|hours|days|day)))\b", re.IGNORECASE)
    # duration like for 5 days / x days / x weeks
    duration_re = re.compile(r"(?:for\s+)?(\d+\s*(?:day|days|week|weeks))", re.IGNORECASE)
    # simple instruction keywords
    instr_re = re.compile(r"\b(with food|after meals|before meals|with water|do not drive|avoid alcohol)\b", re.IGNORECASE)

    lower_text = text.lower()
    strengths = list(strength_re.finditer(text))
    freqs = list(freq_re.finditer(text))
    durations = list(duration_re.finditer(text))
    instrs = [m.group(1) for m in instr_re.finditer(text)]

    def nearest(match_list: List[re.Match], pos: int) -> Optional[str]:
        if not match_list:
            return None
        best = None
        best_dist = 10**9
        for m in match_list:
            d = abs(m.start() - pos)
            if d < best_dist:
                best_dist = d
                best = m.group(1) if m.lastindex else m.group(0)
        return best

    for e in ents:
        name = e.get("word", "").lower()
        if not name or name in seen:
            continue
        seen.add(name)
        pos = lower_text.find(name)
        item: Dict[str, Any] = {
            "name": name,
            "score": e.get("score", 0.0),
        }
        # Attach nearby dosage/frequency/duration
        near_strength = nearest(strengths, pos)
        near_freq = nearest(freqs, pos)
        near_dur = nearest(durations, pos)
        if near_strength:
            item["strength"] = near_strength.strip()
            # Normalize strength
            val, unit = _normalize_strength(near_strength)
            if val is not None:
                item["strength_value"] = val
                item["strength_unit"] = unit
                if unit in ("mg", "mcg", "g"):
                    item["strength_mg"] = _to_mg(val, unit)
        if near_freq:
            item["frequency"] = near_freq.strip()
            per_day = _normalize_frequency_per_day(near_freq)
            if per_day is not None:
                item["frequency_per_day"] = per_day
        if near_dur:
            item["duration"] = near_dur.strip()
            days = _normalize_duration_days(near_dur)
            if days is not None:
                item["duration_days"] = days
        if instrs:
            item["instructions"] = list(sorted(set([s.lower() for s in instrs])))
        drugs.append(item)
    watson = watson_enrich_context(text)
    return {"extracted": drugs, "raw": ents, "watson": watson}


def extract_drug_info(text: str) -> Dict[str, Any]:
    """Public entry with caching for speed on repeated texts."""
    text = (text or "").strip()
    if not text:
        return {"extracted": [], "raw": [], "watson": {"used": False, "reason": "empty"}}
    return _extract_drug_info_cached(text)


def check_interactions(drugs: List[str]) -> Dict[str, Any]:
    if not drugs or len(drugs) < 2:
        return {"interactions": [], "summary": "Need at least two drugs to check interactions."}
    
    # Enhanced interaction checking with multiple sources
    normalized = [d.lower().strip() for d in drugs if d]
    
    # Get interactions from multiple sources
    rx = detect_interactions_rxnav(normalized)
    ofda = detect_interactions_openfda(normalized)
    curated = _get_curated_interactions(normalized)
    category_interactions = _get_category_interactions(normalized)
    
    # Merge all interaction sources
    all_items = rx + ofda + curated + category_interactions
    merged = {}
    
    for it in all_items:
        key = tuple(sorted(it.get("pair", [])))
        if not key:
            continue
        # Prefer RxNav > openFDA > curated > category
        priority = {"rxnav": 4, "openfda_label": 3, "curated": 2, "category": 1}
        current_priority = priority.get(merged.get(key, {}).get("source", ""), 0)
        new_priority = priority.get(it.get("source", ""), 0)
        
        if key not in merged or new_priority > current_priority:
            merged[key] = it
    
    interactions = list(merged.values())
    
    # Enhanced summary with severity analysis
    if not interactions:
        summary = "✅ No known interactions detected between these medications. However, this does not guarantee complete safety - always consult a healthcare professional."
    else:
        # Categorize by severity
        severe = [i for i in interactions if i.get("severity", "").lower() in ["severe", "major", "high"]]
        moderate = [i for i in interactions if i.get("severity", "").lower() in ["moderate", "medium"]]
        minor = [i for i in interactions if i.get("severity", "").lower() in ["minor", "low", "potential"]]
        
        summary_parts = []
        if severe:
            pairs = [" / ".join(item["pair"]) for item in severe]
            summary_parts.append(f"🚨 SEVERE interactions: {', '.join(pairs)}")
        if moderate:
            pairs = [" / ".join(item["pair"]) for item in moderate]
            summary_parts.append(f"⚠️ MODERATE interactions: {', '.join(pairs)}")
        if minor:
            pairs = [" / ".join(item["pair"]) for item in minor]
            summary_parts.append(f"ℹ️ MINOR interactions: {', '.join(pairs)}")
        
        summary = ". ".join(summary_parts) + ". Review detailed notes for each interaction."
    
    return {
        "interactions": interactions, 
        "summary": summary,
        "total_drugs": len(normalized),
        "interaction_count": len(interactions),
        "severity_breakdown": _get_severity_breakdown(interactions),
        "recommendations": _get_interaction_recommendations(interactions)
    }

def _get_curated_interactions(drugs: List[str]) -> List[Dict[str, Any]]:
    """Get curated interactions for the drug list"""
    interactions = []
    for i in range(len(drugs)):
        for j in range(i + 1, len(drugs)):
            a, b = drugs[i], drugs[j]
            note = CURATED_INTERACTIONS.get((a, b)) or CURATED_INTERACTIONS.get((b, a))
            if note:
                interactions.append({
                    "pair": sorted([a, b]), 
                    "severity": "moderate", 
                    "source": "curated", 
                    "note": note
                })
    return interactions

def _get_category_interactions(drugs: List[str]) -> List[Dict[str, Any]]:
    """Get category-based interactions (e.g., multiple NSAIDs, multiple statins)"""
    interactions = []
    
    # Group drugs by category
    categories = {}
    for drug in drugs:
        cat = DRUG_CATEGORIES.get(drug, "Unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(drug)
    
    # Check for same-category interactions
    for category, drug_list in categories.items():
        if len(drug_list) > 1:
            # Multiple drugs in same category
            if category == "NSAID":
                interactions.append({
                    "pair": sorted(drug_list),
                    "severity": "moderate",
                    "source": "category",
                    "note": f"Multiple NSAIDs detected: {', '.join(drug_list)}. Increased risk of GI bleeding and kidney damage."
                })
            elif category == "Statin":
                interactions.append({
                    "pair": sorted(drug_list),
                    "severity": "moderate",
                    "source": "category",
                    "note": f"Multiple statins detected: {', '.join(drug_list)}. No additional benefit, increased side effect risk."
                })
            elif category == "ACE inhibitor":
                interactions.append({
                    "pair": sorted(drug_list),
                    "severity": "moderate",
                    "source": "category",
                    "note": f"Multiple ACE inhibitors detected: {', '.join(drug_list)}. No additional benefit, increased side effect risk."
                })
    
    return interactions

def _get_severity_breakdown(interactions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Get breakdown of interactions by severity"""
    breakdown = {"severe": 0, "moderate": 0, "minor": 0, "unknown": 0}
    
    for interaction in interactions:
        severity = interaction.get("severity", "unknown").lower()
        if severity in ["severe", "major", "high"]:
            breakdown["severe"] += 1
        elif severity in ["moderate", "medium"]:
            breakdown["moderate"] += 1
        elif severity in ["minor", "low", "potential"]:
            breakdown["minor"] += 1
        else:
            breakdown["unknown"] += 1
    
    return breakdown

def _get_interaction_recommendations(interactions: List[Dict[str, Any]]) -> List[str]:
    """Get general recommendations based on interactions found"""
    recommendations = []
    
    if not interactions:
        recommendations.append("✅ No interactions detected - continue current medication regimen as prescribed.")
        return recommendations
    
    # Check for severe interactions
    severe_count = sum(1 for i in interactions if i.get("severity", "").lower() in ["severe", "major", "high"])
    if severe_count > 0:
        recommendations.append("🚨 IMMEDIATE ACTION REQUIRED: Severe interactions detected. Contact healthcare provider immediately.")
        recommendations.append("📞 Consider emergency consultation if experiencing adverse effects.")
    
    # Check for moderate interactions
    moderate_count = sum(1 for i in interactions if i.get("severity", "").lower() in ["moderate", "medium"])
    if moderate_count > 0:
        recommendations.append("⚠️ MODERATE INTERACTIONS: Schedule appointment with healthcare provider within 1-2 weeks.")
        recommendations.append("📋 Monitor for side effects and therapeutic response.")
    
    # General recommendations
    recommendations.extend([
        "📚 Review all medication labels and patient information sheets.",
        "🔄 Consider timing adjustments to minimize interactions.",
        "📊 Regular monitoring may be required for certain drug combinations.",
        "💊 Never stop or change medications without healthcare provider approval."
    ])
    
    return recommendations


def dosage_recommendation(drug: str, age: int) -> Dict[str, Any]:
    name = (drug or "").lower().strip()
    if not name or age is None:
        return {"drug": drug, "age": age, "recommendation": "Insufficient input."}
    
    # Enhanced age band determination with more granular categories
    if age < 1:
        band = "infant"
    elif age < 12:
        band = "child"
    elif age >= 65:
        band = "older"
    else:
        band = "adult"
    
    # Get dosage recommendation
    rule = AGE_DOSAGE_RULES.get(name)
    if rule:
        rec = rule.get(band)
        if rec:
            # Add additional context based on age
            additional_context = _get_age_specific_context(age, band)
            full_recommendation = f"{rec}\n\n{additional_context}"
            return {
                "drug": name, 
                "age": age, 
                "age_band": band, 
                "recommendation": full_recommendation, 
                "source": "heuristic",
                "weight_based": "infant" in band or "child" in band,
                "monitoring_required": _get_monitoring_requirements(name, age)
            }
    
    # Try openFDA label for dosage and administration text
    label = query_openfda_label(name)
    if label:
        dosing_text = "\n".join(label.get("dosage_and_administration", []) or [])
        if dosing_text:
            return {
                "drug": name, 
                "age": age, 
                "age_band": band, 
                "recommendation": dosing_text[:800] + ("..." if len(dosing_text) > 800 else ""), 
                "source": "openfda_label",
                "monitoring_required": _get_monitoring_requirements(name, age)
            }
    
    return {
        "drug": name, 
        "age": age, 
        "age_band": band, 
        "recommendation": "No trusted dosing guidance found. Consult official label and clinician.", 
        "source": "none",
        "monitoring_required": _get_monitoring_requirements(name, age)
    }

def _get_age_specific_context(age: int, band: str) -> str:
    """Get age-specific context and warnings"""
    context_parts = []
    
    if band == "infant":
        context_parts.append("⚠️ INFANT DOSING: Weight-based dosing required. Consult pediatrician for exact calculations.")
        context_parts.append("📋 Monitor for: Feeding tolerance, weight gain, developmental milestones.")
    elif band == "child":
        context_parts.append("👶 PEDIATRIC DOSING: Weight-based dosing may be required. Verify with pediatrician.")
        context_parts.append("📋 Monitor for: Growth parameters, school performance, behavioral changes.")
    elif band == "older":
        context_parts.append("👴 GERIATRIC CONSIDERATIONS: Start with lowest effective dose.")
        context_parts.append("📋 Monitor for: Renal function, liver function, cognitive changes, fall risk.")
        context_parts.append("💊 Consider: Drug interactions, polypharmacy, adherence challenges.")
    else:
        context_parts.append("👤 ADULT DOSING: Standard adult dosing guidelines apply.")
        context_parts.append("📋 Monitor for: Therapeutic response, side effects, adherence.")
    
    return "\n".join(context_parts)

def _get_monitoring_requirements(drug: str, age: int) -> List[str]:
    """Get specific monitoring requirements for drug and age"""
    monitoring = []
    
    # Drug-specific monitoring
    if drug in ["warfarin", "heparin"]:
        monitoring.extend(["INR levels", "Bleeding signs", "Bruising"])
    elif drug in ["digoxin", "lanoxin"]:
        monitoring.extend(["Heart rate", "Digoxin levels", "Potassium levels"])
    elif drug in ["lithium"]:
        monitoring.extend(["Lithium levels", "Thyroid function", "Renal function"])
    elif drug in ["methotrexate"]:
        monitoring.extend(["Liver function", "Blood counts", "Renal function"])
    elif drug in ["metformin"]:
        monitoring.extend(["Renal function", "B12 levels", "Lactic acid"])
    elif drug in ["statins", "atorvastatin", "simvastatin", "rosuvastatin"]:
        monitoring.extend(["Liver enzymes", "Muscle symptoms", "CK levels"])
    
    # Age-specific monitoring
    if age < 12:
        monitoring.extend(["Growth parameters", "Development milestones"])
    elif age >= 65:
        monitoring.extend(["Cognitive function", "Fall risk", "Drug interactions"])
    
    return list(set(monitoring))  # Remove duplicates


def alternative_suggestions(drug: str) -> Dict[str, Any]:
    name = (drug or "").lower().strip()
    cat = DRUG_CATEGORIES.get(name)
    if not cat:
        # try to infer from suffix/patterns
        if name.endswith("statin"):
            cat = "Statin"
        elif name in {"paracetamol", "acetaminophen"}:
            cat = "Analgesic"
    if not cat:
        return {"drug": name, "alternatives": [], "reason": "Unknown category; unable to suggest.", "source": "none"}
    return {"drug": name, "category": cat, "alternatives": CATEGORY_ALTERNATIVES.get(cat, []), "source": "curated"}


# -----------------------------
# Normalization helpers
# -----------------------------

def _normalize_strength(text: str) -> Tuple[Optional[float], Optional[str]]:
    try:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|iu)", text.strip(), re.IGNORECASE)
        if not m:
            return None, None
        val = float(m.group(1))
        unit = m.group(2).lower()
        return val, unit
    except Exception:
        return None, None


def _to_mg(val: float, unit: str) -> Optional[float]:
    unit = unit.lower()
    if unit == "mg":
        return val
    if unit == "g":
        return val * 1000.0
    if unit == "mcg":
        return val / 1000.0
    return None


def _normalize_frequency_per_day(text: str) -> Optional[float]:
    s = text.strip().lower()
    mapping = {
        "qd": 1, "qam": 1, "qpm": 1, "qhs": 1, "hs": 1, "once daily": 1,
        "bid": 2, "twice daily": 2,
        "tid": 3,
        "qid": 4,
        "prn": None,
    }
    if s in mapping:
        return mapping[s]
    # q12h, q8h, q6h, every 12 hours
    m = re.search(r"q\s*(\d+)\s*(h|hr|hour|hours)$", s)
    if not m:
        m = re.search(r"every\s+(\d+)\s*(h|hr|hour|hours)$", s)
    if m:
        hours = float(m.group(1))
        if hours > 0:
            return round(24.0 / hours, 2)
    # every N days
    m = re.search(r"every\s+(\d+)\s*(day|days)$", s)
    if m:
        days = float(m.group(1))
        if days > 0:
            return round(1.0 / days, 3)
    return None


def _normalize_duration_days(text: str) -> Optional[int]:
    s = text.strip().lower()
    m = re.search(r"(\d+)\s*(day|days)$", s)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)\s*(week|weeks)$", s)
    if m:
        return int(m.group(1)) * 7
    return None

