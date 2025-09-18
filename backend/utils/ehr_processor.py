
import pandas as pd
import json
import glob
from tqdm import tqdm
import logging
import re
import gc
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from typing import Generator, List, Dict, Any, Union
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EHRDataProcessor:
    def __init__(self, data_path=None, max_workers=None):
        if data_path:
            self.data_path = data_path
            self.files = glob.glob(f'{self.data_path}/**/*.json', recursive=True)
            logger.info(f"Found {len(self.files)} JSON files in {self.data_path}")
        else:
            self.data_path = None
            self.files = []
        
        # Set optimal number of workers for parallel processing
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        logger.info(f"Using {self.max_workers} workers for parallel processing")
    
    def estimate_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage and recommend batch size"""
        if not self.files:
            return {"total_size_gb": 0, "recommended_batch_size": 100}
        
        # Sample first few files to estimate average size
        sample_size = min(10, len(self.files))
        total_sample_size = 0
        
        for file_path in self.files[:sample_size]:
            total_sample_size += os.path.getsize(file_path)
        
        avg_file_size = total_sample_size / sample_size
        total_size_gb = (avg_file_size * len(self.files)) / (1024**3)
        
        # Recommend batch size based on available memory (assuming 16GB RAM)
        available_memory_gb = 8  # Conservative estimate
        recommended_batch_size = max(50, int((available_memory_gb * 0.5 * 1024**3) / avg_file_size))
        
        return {
            "total_size_gb": total_size_gb,
            "avg_file_size_mb": avg_file_size / (1024**2),
            "recommended_batch_size": min(recommended_batch_size, 1000),
            "total_files": len(self.files)
        }

    def process_file_parallel(self, file_path: str) -> Dict[str, Any]:
        """Process a single JSON file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return self.extract_patient_info(data)
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not process file {file_path}: {str(e)}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error processing {file_path}: {str(e)}")
            return {}
    
    def process_in_batches(self, batch_size=None, use_multiprocessing=True) -> Generator[pd.DataFrame, None, None]:
        """Process EHR files in batches with optimized memory usage"""
        if not batch_size:
            memory_info = self.estimate_memory_usage()
            batch_size = memory_info['recommended_batch_size']
            logger.info(f"Auto-selected batch size: {batch_size} based on dataset analysis")
            logger.info(f"Dataset info: {memory_info['total_files']} files, {memory_info['total_size_gb']:.2f} GB total")
        
        total_batches = (len(self.files) + batch_size - 1) // batch_size
        
        for batch_idx in tqdm(range(0, len(self.files), int(batch_size)), 
                              desc="Processing EHR batches", 
                              total=total_batches):
            
            batch_files = self.files[batch_idx:batch_idx + batch_size]
            batch_data = []
            
            if use_multiprocessing and len(batch_files) > 10:
                # Use parallel processing for larger batches
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    results = list(executor.map(self.process_file_parallel, batch_files))
                batch_data = [result for result in results if result]
            else:
                # Sequential processing for smaller batches
                for file_path in batch_files:
                    result = self.process_file_parallel(file_path)
                    if result:
                        batch_data.append(result)
            
            if batch_data:
                batch_df = self.flatten_data(batch_data)
                logger.info(f"Processed batch {batch_idx//batch_size + 1}/{total_batches}: {len(batch_df)} records")
                
                # Add target variable if missing (for training)
                if 'target' not in batch_df.columns:
                    batch_df = self.add_synthetic_target(batch_df)
                
                yield batch_df
                
                # Force garbage collection to free memory
                del batch_data, batch_df
                gc.collect()
            else:
                logger.warning(f"No valid data in batch {batch_idx//batch_size + 1}")

    def extract_patient_info(self, data) -> Dict[str, Any]:
        patient_info: Dict[str, Any] = {'observations': {}}
        for entry in data.get('entry', []):
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')

            if resource_type == 'Patient':
                patient_info['patient_id'] = resource.get('id')
                birth_date = resource.get('birthDate')
                if birth_date:
                    patient_info['birth_date'] = birth_date
                patient_info['gender'] = resource.get('gender')

            elif resource_type == 'Condition':
                if 'conditions' not in patient_info:
                    patient_info['conditions'] = []
                condition = resource.get('code', {}).get('text')
                if condition and isinstance(patient_info.get('conditions'), list):
                    patient_info['conditions'].append(condition)

            elif resource_type == 'Observation':
                code = resource.get('code', {}).get('coding', [{}])[0].get('display')
                if code:
                    if 'valueQuantity' in resource:
                        value = resource['valueQuantity'].get('value')
                        patient_info['observations'][code] = value
                    elif 'component' in resource:
                        for component in resource['component']:
                            comp_code = component.get('code', {}).get('coding', [{}])[0].get('display')
                            if comp_code:
                                value = component['valueQuantity'].get('value')
                                patient_info['observations'][comp_code] = value
        return patient_info

    def flatten_data(self, batch_data):
        flat_data = []
        for record in batch_data:
            flat_record = {}
            for key, value in record.items():
                if key == 'observations':
                    for obs_key, obs_value in value.items():
                        # Sanitize column names
                        col_name = re.sub(r'[^A-Za-z0-9_]+', '', obs_key)
                        flat_record[col_name] = obs_value
                elif key == 'conditions':
                    flat_record['conditions'] = ' '.join(value)
                else:
                    flat_record[key] = value
            flat_data.append(flat_record)
        return pd.DataFrame(flat_data)

    def add_synthetic_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add realistic target variable based on health indicators"""
        if df.empty:
            return df
        
        # Create risk score based on available health indicators
        risk_score = np.zeros(len(df))
        
        # Age factor (higher age = higher risk)
        if 'birth_date' in df.columns:
            try:
                ages = pd.to_datetime('today') - pd.to_datetime(df['birth_date'], errors='coerce')
                ages_years = ages.dt.days / 365.25
                risk_score += np.where(ages_years > 60, 0.3, np.where(ages_years > 45, 0.2, 0.1))
            except Exception:
                pass
        
        # Blood pressure indicators - simplified approach
        bp_cols = [col for col in df.columns if 'BloodPressure' in col or 'bloodpressure' in col.lower()]
        for col in bp_cols:
            try:
                # Convert to numeric and handle NaN values
                values = pd.to_numeric(df[col], errors='coerce')
                values_array = np.array(values, dtype=float)
                values_filled = np.nan_to_num(values_array, nan=0.0)
                risk_score += np.where(values_filled > 140, 0.25, np.where(values_filled > 120, 0.15, 0))
            except Exception:
                pass
        
        # Cholesterol indicators - simplified approach
        chol_cols = [col for col in df.columns if 'Cholesterol' in col or 'cholesterol' in col.lower()]
        for col in chol_cols:
            try:
                # Convert to numeric and handle NaN values
                values = pd.to_numeric(df[col], errors='coerce')
                values_array = np.array(values, dtype=float)
                values_filled = np.nan_to_num(values_array, nan=0.0)
                risk_score += np.where(values_filled > 240, 0.2, np.where(values_filled > 200, 0.1, 0))
            except Exception:
                pass
        
        # Glucose/Diabetes indicators - simplified approach
        glucose_cols = [col for col in df.columns if 'Glucose' in col or 'glucose' in col.lower() or 'HbA1c' in col]
        for col in glucose_cols:
            try:
                # Convert to numeric and handle NaN values
                values = pd.to_numeric(df[col], errors='coerce')
                values_array = np.array(values, dtype=float)
                values_filled = np.nan_to_num(values_array, nan=0.0)
                risk_score += np.where(values_filled > 126, 0.2, np.where(values_filled > 100, 0.1, 0))
            except Exception:
                pass
        
        # Add some randomness to make it more realistic
        np.random.seed(42)
        noise = np.random.normal(0, 0.1, len(df))
        risk_score += noise
        
        # Convert to binary classification (0 or 1)
        target = (risk_score > np.median(risk_score)).astype(int)
        df['target'] = target
        
        logger.info(f"Added target variable: {target.sum()}/{len(target)} positive cases ({target.mean()*100:.1f}%)")
        return df
    
    def get_feature_summary(self, sample_size=1000) -> Dict[str, Any]:
        """Analyze dataset features for model training insights"""
        if not self.files:
            return {"error": "No files found"}
        
        logger.info(f"Analyzing dataset features from {min(sample_size, len(self.files))} files...")
        
        all_features = set()
        feature_types = {}
        sample_files = self.files[:min(sample_size, len(self.files))]
        
        for file_path in tqdm(sample_files, desc="Analyzing features"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    patient_data = self.extract_patient_info(data)
                    if patient_data:
                        flat_data = self.flatten_data([patient_data])
                        if not flat_data.empty:
                            for col in flat_data.columns:
                                all_features.add(col)
                                if col not in feature_types:
                                    feature_types[col] = flat_data[col].dtype
            except Exception as e:
                continue
        
        return {
            "total_features": len(all_features),
            "feature_names": sorted(list(all_features)),
            "feature_types": {k: str(v) for k, v in feature_types.items()},
            "analyzed_files": len(sample_files)
        }
    
    def process_single_record(self, record_json):
        """Process a single FHIR record for prediction"""
        patient_data = self.extract_patient_info(record_json)
        if patient_data:
            return self.flatten_data([patient_data])
        return pd.DataFrame()


if __name__ == '__main__':
    processor = EHRDataProcessor('d:/personalized-healthcare/data/ehr')
    for batch_df in processor.process_in_batches(batch_size=100):
        print(batch_df.head())
        print(f"Processed batch of size {len(batch_df)}")
        # Add a mock target column for demonstration
        if not batch_df.empty:
            batch_df['target'] = [1, 0] * (len(batch_df) // 2) if len(batch_df) > 1 else [1]
            if len(batch_df) % 2 != 0 and len(batch_df) > 1:
                batch_df.loc[batch_df.index[-1], 'target'] = 0


        print(batch_df.head())
        break # Process only one batch for demonstration
