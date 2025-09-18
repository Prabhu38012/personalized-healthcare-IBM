# 🏥 Personalized Healthcare Prediction System

A comprehensive AI-powered healthcare recommendation system that predicts cardiovascular disease risk using machine learning models and Electronic Health Records (EHR) data.

## 🚀 Quick Start for Friends/Contributors

### Prerequisites

1. **Python 3.8+** (recommended: Python 3.9-3.11)
2. **Git** installed on your system
3. **10GB+ free disk space** (for datasets and models)
4. **8GB+ RAM** (16GB recommended for large datasets)

### 🔧 Setup Instructions

#### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd personalized-healthcare
```

#### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 4. Verify Installation
```bash
python -c "import streamlit, fastapi, sklearn, pandas; print('✅ All dependencies installed successfully!')"
```

## 🏃‍♂️ Running the Application

### Option 1: Quick Demo (Recommended)
Run both frontend and backend with sample data:

```bash
# Terminal 1: Start Backend API
cd backend
python app.py

# Terminal 2: Start Frontend (new terminal)
cd frontend
streamlit run app.py
```

**Access the app at:**
- 🌐 **Frontend**: http://localhost:8501
- 🔧 **Backend API**: http://localhost:8000
- 📖 **API Docs**: http://localhost:8000/docs

### Option 2: Docker (One-Command Setup)
```bash
# Run everything with Docker
docker-compose up

# Or run in background
docker-compose up -d
```

### Option 3: Production Mode
```bash
# Single container with both services
docker-compose --profile production up
```

## 📊 Training Your Own Model

### 1. Prepare Dataset
- Download EHR dataset from Kaggle (JSON format)
- Place files in: `data/ehr/`
- **Currently includes sample data** for immediate testing

### 2. Train Model
```bash
cd backend
python train_config.py
```

### 3. Monitor Training
- Watch console output for progress
- Check `training.log` for detailed logs
- Training time: 30 minutes to 4+ hours

📖 **Detailed training guide**: See [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)

## 🌟 Features

### Frontend (Streamlit)
- 📋 **Patient Data Input Form** - Easy-to-use interface
- 📊 **Real-time Risk Assessment** - Visual risk indicators
- 📈 **Interactive Dashboards** - Health metrics visualization
- 💊 **Personalized Recommendations** - AI-generated health advice

### Backend (FastAPI)
- 🤖 **ML Prediction API** - REST endpoints for predictions
- 📁 **EHR Data Processing** - Handle large healthcare datasets
- 🏥 **Risk Assessment Engine** - Cardiovascular disease prediction
- 🔗 **Model Management** - Load/save trained models

### Machine Learning
- 🌲 **Random Forest** - Primary prediction model
- 📈 **Gradient Boosting** - Enhanced accuracy
- 📉 **Logistic Regression** - Interpretable baseline
- 🔍 **Feature Engineering** - Automated EHR processing

## 📁 Project Structure

```
personalized-healthcare/
├── backend/                 # FastAPI backend
│   ├── routes/             # API endpoints
│   ├── models/             # ML models and training
│   ├── utils/              # Data processing utilities
│   └── app.py              # Main backend application
├── frontend/               # Streamlit frontend
│   ├── components/         # UI components
│   ├── pages/              # Application pages
│   ├── utils/              # Frontend utilities
│   └── app.py              # Main frontend application
├── data/                   # Sample datasets
├── deployment/             # Docker and deployment configs
├── models/                 # Trained model files and reports
├── requirements.txt        # Python dependencies
└── TRAINING_GUIDE.md       # Detailed training instructions
```

## 🧪 Testing the System

### 1. Test Backend API
```bash
# Check if backend is running
curl http://localhost:8000/api/health

# Test prediction endpoint
curl -X POST "http://localhost:8000/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "sex": 1, "chest_pain_type": 2, "resting_bp": 130}'
```

### 2. Test Frontend
1. Open http://localhost:8501
2. Fill in the patient form
3. Click "Predict Risk"
4. View results and recommendations

### 3. Run Automated Tests
```bash
cd backend
python -m pytest tests/ -v
```

## 🛠️ Development

### Adding New Features
1. **Backend**: Add endpoints in `backend/routes/`
2. **Frontend**: Add components in `frontend/components/`
3. **ML Models**: Modify `backend/models/train_model.py`

### Environment Variables
Create `.env` file for configuration:
```env
BACKEND_URL=http://localhost:8000
MODEL_PATH=models/
DEBUG=True
```

## 🐛 Troubleshooting

### Common Issues

#### "ModuleNotFoundError"
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt
```

#### "Port already in use"
```bash
# Change ports in the commands
streamlit run app.py --server.port 8502
uvicorn app:app --port 8001
```

#### "Model not found"
```bash
# Train a model first or use sample data
cd backend
python train_config.py
```

#### "Memory errors during training"
```bash
# Reduce dataset size in train_config.py
# Set max_samples = 10000 for testing
```

### Getting Help
1. Check the [TRAINING_GUIDE.md](./TRAINING_GUIDE.md) for training issues
2. Review API documentation at http://localhost:8000/docs
3. Look at sample data in `data/` directory

## 📋 System Requirements

### Minimum
- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 10GB free space
- **OS**: Windows, macOS, Linux

### Recommended
- **Python**: 3.9-3.11
- **RAM**: 16GB+
- **Storage**: 20GB+ free space
- **CPU**: Multi-core processor

## 🔒 Data Privacy

- All data processing happens locally
- No patient data is transmitted externally
- Models are trained and stored locally
- Compliant with healthcare data standards

## 📄 License

This project is for educational and research purposes. Please ensure compliance with healthcare regulations when using with real patient data.

---

## 🆘 Quick Help

**Can't get it running?** Try this minimal setup:

1. `git clone <repo-url> && cd personalized-healthcare`
2. `python -m venv venv && venv\Scripts\activate` (Windows) or `source venv/bin/activate` (Mac/Linux)
3. `pip install -r requirements.txt`
4. `cd backend && python app.py` (in one terminal)
5. `cd frontend && streamlit run app.py` (in another terminal)
6. Open http://localhost:8501

**Still stuck?** The system includes sample data, so you can test immediately without downloading large datasets!