import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthDataPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.feature_columns = None
        self.categorical_columns = []
        self.numerical_columns = []
        
    def clean_data(self, df):
        """Clean and prepare healthcare data"""
        logger.info("Starting data cleaning process")
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Handle missing values
        self.numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        # Fill numeric missing values with median
        for col in self.numerical_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].median())
                
        # Fill categorical missing values with mode
        for col in self.categorical_columns:
            if col in df.columns and col not in ['target', 'diagnosis']:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna('Unknown')
                    
        logger.info(f"Data cleaned. Shape: {df.shape}")
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical variables"""
        for col in self.categorical_columns:
            if col in df.columns and col not in ['target', 'diagnosis']:
                if fit:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    # Handle unseen categories during fit
                    df[col] = df[col].astype(str)
                    self.label_encoders[col].fit(df[col])
                    df[col] = self.label_encoders[col].transform(df[col])
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories during transform
                        df[col] = df[col].astype(str)
                        # Map unknown categories to a default value
                        known_categories = set(self.label_encoders[col].classes_)
                        current_categories = set(df[col].unique())
                        unknown_categories = current_categories - known_categories
                        
                        if unknown_categories:
                            logger.warning(f"Unknown categories found in {col}: {unknown_categories}")
                            # Replace unknown categories with the most frequent one
                            df[col] = df[col].replace(list(unknown_categories), self.label_encoders[col].classes_[0])
                        
                        df[col] = self.label_encoders[col].transform(df[col])
                        
        return df
    
    def scale_features(self, df, fit=True):
        """Scale numerical features"""
        if fit:
            df[self.numerical_columns] = self.scaler.fit_transform(df[self.numerical_columns])
        else:
            df[self.numerical_columns] = self.scaler.transform(df[self.numerical_columns])
            
        return df
    
    def prepare_features(self, df, target_column=None, fit=True):
        """Complete preprocessing pipeline"""
        logger.info("Starting feature preparation")
        
        # Clean data
        df = self.clean_data(df.copy())
        
        # Separate features and target
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
            if target_column in self.numerical_columns:
                self.numerical_columns.remove(target_column)
        else:
            X = df.copy()
            y = None
            
        # Encode categorical features
        X = self.encode_categorical_features(X, fit=fit)
        
        # Scale features
        X = self.scale_features(X, fit=fit)
        
        if fit:
            self.feature_columns = X.columns.tolist()
            
        logger.info(f"Feature preparation complete. Features: {len(X.columns)}")
        
        return X, y
    
    def transform_single_record(self, record_dict):
        """Transform a single patient record for prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([record_dict])
            
            # Apply the same preprocessing steps
            df = self.clean_data(df)
            df = self.encode_categorical_features(df, fit=False)
            df = self.scale_features(df, fit=False)
            
            # Ensure all expected features are present
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in df.columns:
                        df[col] = 0  # Default value for missing features
                df = df[self.feature_columns]  # Reorder columns
                
            return df.values.reshape(1, -1)
            
        except Exception as e:
            logger.error(f"Error transforming single record: {e}")
            raise ValueError(f"Data transformation error: {e}")

def create_synthetic_patient_data(n_samples=1000):
    """Create synthetic patient data for demonstration"""
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 90, n_samples),
        'sex': np.random.choice(['M', 'F'], n_samples),
        'chest_pain_type': np.random.choice(['typical', 'atypical', 'non_anginal', 'asymptomatic'], n_samples),
        'resting_bp': np.random.randint(90, 200, n_samples),
        'cholesterol': np.random.randint(120, 400, n_samples),
        'fasting_blood_sugar': np.random.choice([0, 1], n_samples),
        'resting_ecg': np.random.choice(['normal', 'abnormal', 'hypertrophy'], n_samples),
        'max_heart_rate': np.random.randint(60, 220, n_samples),
        'exercise_angina': np.random.choice([0, 1], n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.choice(['upsloping', 'flat', 'downsloping'], n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.choice(['normal', 'fixed', 'reversible'], n_samples),
        'bmi': np.random.uniform(18.5, 40, n_samples),
        'smoking': np.random.choice([0, 1], n_samples),
        'diabetes': np.random.choice([0, 1], n_samples),
        'family_history': np.random.choice([0, 1], n_samples)
    }
    
    # Create target variable (heart disease risk)
    risk_factors = (
        (np.array(data['age']) > 50).astype(int) +
        (np.array(data['resting_bp']) > 140).astype(int) +
        (np.array(data['cholesterol']) > 240).astype(int) +
        np.array(data['smoking']) +
        np.array(data['diabetes']) +
        np.array(data['family_history']) +
        (np.array(data['bmi']) > 30).astype(int)
    )
    
    # Add some randomness
    target = (risk_factors >= 3).astype(int)
    noise = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    data['target'] = np.logical_xor(target, noise).astype(int)
    
    return pd.DataFrame(data)