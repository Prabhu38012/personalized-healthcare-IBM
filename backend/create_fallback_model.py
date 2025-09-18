"""
Create a fallback model for the healthcare prediction system.
This script generates a basic machine learning model for cardiovascular risk prediction.
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

def create_synthetic_data(n_samples=1000):
    """Create synthetic healthcare data for training"""
    np.random.seed(42)
    
    # Generate synthetic patient data
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'sex': np.random.choice([0, 1], n_samples),  # 0=Female, 1=Male
        'systolic_bp': np.random.normal(130, 20, n_samples),
        'diastolic_bp': np.random.normal(80, 10, n_samples),
        'cholesterol': np.random.normal(200, 40, n_samples),
        'bmi': np.random.normal(25, 5, n_samples),
        'smoking': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        'diabetes': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'family_history': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'exercise_hours': np.random.exponential(2, n_samples),
        'stress_level': np.random.randint(1, 11, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic target variable based on risk factors
    risk_score = (
        (df['age'] - 20) / 60 * 0.3 +  # Age factor
        df['sex'] * 0.1 +  # Male higher risk
        np.maximum(0, (df['systolic_bp'] - 120) / 60) * 0.2 +  # High BP
        np.maximum(0, (df['cholesterol'] - 200) / 100) * 0.15 +  # High cholesterol
        np.maximum(0, (df['bmi'] - 25) / 10) * 0.1 +  # Obesity
        df['smoking'] * 0.15 +  # Smoking
        df['diabetes'] * 0.2 +  # Diabetes
        df['family_history'] * 0.1 +  # Family history
        np.maximum(0, (10 - df['exercise_hours']) / 10) * 0.1 +  # Low exercise
        (df['stress_level'] - 1) / 9 * 0.05  # Stress
    )
    
    # Add some noise and convert to binary classification
    risk_score += np.random.normal(0, 0.1, n_samples)
    df['target'] = (risk_score > 0.5).astype(int)
    
    return df

def train_fallback_model():
    """Train and save a fallback model"""
    print("Creating synthetic training data...")
    df = create_synthetic_data(2000)
    
    # Prepare features and target
    feature_columns = [
        'age', 'sex', 'systolic_bp', 'diastolic_bp', 'cholesterol', 
        'bmi', 'smoking', 'diabetes', 'family_history', 
        'exercise_hours', 'stress_level'
    ]
    
    X = df[feature_columns]
    y = df['target']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train the model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model accuracy: {accuracy:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Create model data structure
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'model_type': 'RandomForestClassifier',
        'accuracy': accuracy,
        'version': '1.0.0',
        'created_date': pd.Timestamp.now().isoformat()
    }
    
    return model_data

def save_model(model_data, model_path):
    """Save the model to disk"""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_data, model_path)
    print(f"Model saved to: {model_path}")

if __name__ == "__main__":
    # Define model path
    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "ehr_risk_model.pkl")
    
    print("Creating fallback healthcare risk prediction model...")
    print("=" * 50)
    
    # Train the model
    model_data = train_fallback_model()
    
    # Save the model
    save_model(model_data, model_path)
    
    print("=" * 50)
    print("Fallback model created successfully!")
    print(f"Model path: {os.path.abspath(model_path)}")
    print("\nYou can now start the backend server and test the risk analysis functionality.")
