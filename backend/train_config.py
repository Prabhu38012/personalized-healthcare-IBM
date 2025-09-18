"""
Training Configuration for Large EHR Dataset
============================================

This script helps you configure and start training on your 10GB EHR dataset.
"""

import os
import sys
import logging
from pathlib import Path

# Add backend to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.models.train_model import HealthRiskModelTrainer
from backend.utils.ehr_processor import EHRDataProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration for training on large EHR dataset"""
    
    def __init__(self):
        # Dataset Configuration
        self.ehr_data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "ehr")
        self.max_samples = None  # Set to limit samples for testing (e.g., 50000)
        
        # Training Configuration
        self.test_size = 0.2
        self.validation_size = 0.1
        self.random_state = 42
        
        # Performance Configuration
        self.use_parallel_processing = True
        self.optimize_hyperparameters = True
        self.save_intermediate_models = True
        
        # Output Configuration
        self.model_save_path = "models/ehr_risk_model.pkl"
        self.report_save_path = "models/ehr_training_report.md"
    
    def validate_setup(self):
        """Validate the training setup"""
        issues = []
        
        # Check if EHR data exists
        if not os.path.exists(self.ehr_data_path):
            issues.append(f"EHR data path does not exist: {self.ehr_data_path}")
        else:
            # Check if there are JSON files
            json_files = list(Path(self.ehr_data_path).rglob("*.json"))
            if not json_files:
                issues.append(f"No JSON files found in {self.ehr_data_path}")
            else:
                logger.info(f"Found {len(json_files)} JSON files in dataset")
        
        # Check available memory
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            if available_memory_gb < 4:
                issues.append(f"Low memory warning: {available_memory_gb:.1f}GB available. Consider reducing batch size.")
            else:
                logger.info(f"Available memory: {available_memory_gb:.1f}GB")
        except ImportError:
            logger.warning("psutil not installed. Cannot check memory usage.")
        
        # Create output directories
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.report_save_path), exist_ok=True)
        
        return issues
    
    def run_training(self):
        """Run the complete training pipeline"""
        logger.info("=" * 60)
        logger.info("STARTING EHR MODEL TRAINING")
        logger.info("=" * 60)
        
        # Validate setup
        issues = self.validate_setup()
        if issues:
            logger.error("Setup validation failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        try:
            # Initialize trainer
            trainer = HealthRiskModelTrainer()
            
            # Load data
            logger.info("Loading EHR dataset...")
            data = trainer.load_data(
                data_path=self.ehr_data_path,
                max_samples=self.max_samples
            )
            
            if data is None or data.empty:
                logger.error("Failed to load data")
                return False
            
            # Train models
            logger.info("Starting model training...")
            results, X_test, y_test = trainer.train_models(data)
            
            # Save best model
            logger.info("Saving trained model...")
            trainer.save_model(self.model_save_path)
            
            # Generate report
            logger.info("Generating training report...")
            report = trainer.generate_model_report(results, X_test, y_test)
            
            with open(self.report_save_path, 'w') as f:
                f.write(report)
            
            logger.info("=" * 60)
            logger.info("TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"Model saved to: {self.model_save_path}")
            logger.info(f"Report saved to: {self.report_save_path}")
            logger.info(f"Best model: {type(trainer.best_model).__name__}")
            logger.info(f"Best F1 score: {trainer.best_score:.4f}")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed with error: {str(e)}", exc_info=True)
            return False

def main():
    """Main training function"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                EHR Healthcare Risk Model Training                ║
    ║                                                                  ║
    ║  This will train a machine learning model on your 10GB EHR      ║
    ║  dataset to predict healthcare risks.                           ║
    ║                                                                  ║
    ║  Prerequisites:                                                  ║
    ║  1. EHR JSON files should be placed in data/ehr/               ║
    ║  2. Ensure you have sufficient RAM (recommended: 8GB+)          ║
    ║  3. Training may take 30 minutes to several hours               ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    config = TrainingConfig()
    
    # Interactive configuration
    print("\n🔧 TRAINING CONFIGURATION:")
    print(f"   EHR Data Path: {config.ehr_data_path}")
    print(f"   Sample Limit: {'No limit' if config.max_samples is None else f'{config.max_samples:,}'}")
    print(f"   Parallel Processing: {'Enabled' if config.use_parallel_processing else 'Disabled'}")
    
    response = input("\nStart training? (y/n): ").lower().strip()
    
    if response == 'y':
        success = config.run_training()
        if success:
            print("\n✅ Training completed successfully!")
            print("\n📊 Next steps:")
            print("   1. Check the training report for model performance")
            print("   2. Test the model using the API endpoints")
            print("   3. Deploy the model for production use")
        else:
            print("\n❌ Training failed. Check the logs for details.")
    else:
        print("Training cancelled.")

if __name__ == "__main__":
    main()