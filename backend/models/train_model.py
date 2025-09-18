import os
import sys
import time
import gc
import logging
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

from backend.utils.preprocess import HealthDataPreprocessor
from backend.utils.ehr_processor import EHRDataProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthRiskModelTrainer:
    def __init__(self):
        self.preprocessor = HealthDataPreprocessor()
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42, n_jobs=-1),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'logistic_regression': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
            # Removed SVM for large datasets due to computational complexity
        }
        self.best_model = None
        self.best_score = 0
        self.training_history = []
        
    def analyze_dataset(self, data_path):
        """Analyze the EHR dataset before training"""
        logger.info("=== DATASET ANALYSIS ===")
        processor = EHRDataProcessor(data_path)
        
        # Get memory and size estimates
        memory_info = processor.estimate_memory_usage()
        logger.info(f"Dataset Size: {memory_info['total_size_gb']:.2f} GB")
        logger.info(f"Total Files: {memory_info['total_files']:,}")
        logger.info(f"Recommended Batch Size: {memory_info['recommended_batch_size']}")
        
        # Analyze features
        feature_info = processor.get_feature_summary(sample_size=500)
        logger.info(f"Total Features Found: {feature_info['total_features']}")
        logger.info(f"Sample Features: {feature_info['feature_names'][:10]}...")
        
        return memory_info, feature_info
        
    def load_data(self, data_path=None, max_samples=None):
        """Load EHR data efficiently with streaming approach"""
        if data_path and os.path.exists(data_path):
            logger.info(f"Loading EHR data from {data_path}")
            
            # Analyze dataset first
            memory_info, feature_info = self.analyze_dataset(data_path)
            
            processor = EHRDataProcessor(data_path)
            all_data = []
            total_samples = 0
            
            start_time = time.time()
            
            # Process in optimized batches
            for batch_idx, batch_df in enumerate(processor.process_in_batches(use_multiprocessing=True)):
                if not batch_df.empty:
                    all_data.append(batch_df)
                    total_samples += len(batch_df)
                    
                    # Log progress
                    elapsed = time.time() - start_time
                    logger.info(f"Loaded {total_samples:,} samples in {elapsed:.1f}s")
                    
                    # Optional: Limit samples for testing
                    if max_samples and total_samples >= max_samples:
                        logger.info(f"Reached sample limit: {max_samples:,}")
                        break
                    
                    # Memory management for very large datasets
                    if len(all_data) > 10:  # Combine every 10 batches
                        combined_df = pd.concat(all_data, ignore_index=True)
                        all_data = [combined_df]
                        gc.collect()
            
            if all_data:
                data = pd.concat(all_data, ignore_index=True)
                logger.info(f"Final dataset: {len(data):,} samples with {len(data.columns)} features")
                
                # Log feature distribution
                if 'target' in data.columns:
                    target_dist = data['target'].value_counts()
                    logger.info(f"Target distribution: {dict(target_dist)}")
                
                return data
            else:
                logger.warning("No valid data found in EHR files")
                return self.create_fallback_data()
        else:
            logger.info("EHR data path not found, creating synthetic data")
            return self.create_fallback_data()
    
    def create_fallback_data(self):
        """Create synthetic data as fallback"""
        logger.info("Creating synthetic patient data as fallback")
        from backend.utils.preprocess import create_synthetic_patient_data
        return create_synthetic_patient_data(5000)  # Larger synthetic dataset
    
    def train_models(self, data, target_column='target'):
        """Train multiple models and select the best one"""
        logger.info("Starting model training process")
        
        # Prepare features
        X, y = self.preprocessor.prepare_features(data, target_column, fit=True)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else 0
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'predictions': y_pred,
                'probabilities': y_pred_proba
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")
            
            # Update best model
            if f1 > self.best_score:
                self.best_score = f1
                self.best_model = model
                
        return results, X_test, y_test
    
    def hyperparameter_tuning(self, X_train, y_train):
        """Perform hyperparameter tuning for the best model"""
        logger.info("Performing hyperparameter tuning")
        
        # Define parameter grids
        param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }
        
        best_models = {}
        
        for name, model in [('random_forest', RandomForestClassifier(random_state=42)),
                           ('gradient_boosting', GradientBoostingClassifier(random_state=42))]:
            if name in param_grids:
                logger.info(f"Tuning {name}")
                
                grid_search = GridSearchCV(
                    model, param_grids[name], 
                    cv=5, scoring='f1', n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                best_models[name] = {
                    'model': grid_search.best_estimator_,
                    'params': grid_search.best_params_,
                    'score': grid_search.best_score_
                }
                
                logger.info(f"{name} best score: {grid_search.best_score_:.4f}")
                
        return best_models
    
    def save_model(self, model_path='models/risk_model.pkl'):
        """Save the trained model and preprocessor"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'preprocessor': self.preprocessor,
            'feature_columns': self.preprocessor.feature_columns
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
        
    def generate_model_report(self, results, X_test, y_test):
        """Generate comprehensive model evaluation report"""
        report = "# Healthcare Risk Model Training Report\n\n"
        
        for name, result in results.items():
            report += f"## {name.replace('_', ' ').title()}\n"
            report += f"- Accuracy: {result['accuracy']:.4f}\n"
            report += f"- Precision: {result['precision']:.4f}\n"
            report += f"- Recall: {result['recall']:.4f}\n"
            report += f"- F1 Score: {result['f1']:.4f}\n"
            report += f"- AUC Score: {result['auc']:.4f}\n\n"
            
        # Feature importance for tree-based models
        if self.best_model is not None and hasattr(self.best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.preprocessor.feature_columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            report += "## Feature Importance (Top 10)\n"
            for _, row in feature_importance.head(10).iterrows():
                report += f"- {row['feature']}: {row['importance']:.4f}\n"
        else:
            report += "## Feature Importance\n"
            report += "Feature importance not available for this model type.\n"
                
        return report

def main():
    """Main training pipeline"""
    trainer = HealthRiskModelTrainer()
    
    # Load data
    data = trainer.load_data(data_path='d:/personalized-healthcare/data/ehr')
    
    # Train models
    results, X_test, y_test = trainer.train_models(data)
    
    # Save best model
    trainer.save_model()
    
    # Generate report
    report = trainer.generate_model_report(results, X_test, y_test)
    
    with open('models/training_report.md', 'w') as f:
        f.write(report)
        
    logger.info("Training completed successfully!")
    
    return trainer.best_model, trainer.preprocessor

if __name__ == "__main__":
    main()
