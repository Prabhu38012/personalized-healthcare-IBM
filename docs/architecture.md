# System Architecture

## Overview
The Personalized Healthcare Recommendation System follows a microservices architecture with clear separation between frontend, backend, and ML components.

## Architecture Diagram
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │    Backend      │    │   ML Models     │
│   (Streamlit)   │◄──►│   (FastAPI)     │◄──►│  (Scikit-learn) │
│                 │    │                 │    │                 │
│ • Patient Forms │    │ • API Routes    │    │ • Risk Model    │
│ • Dashboard     │    │ • Data Proc.    │    │ • Preprocessor  │
│ • Visualizations│    │ • Predictions   │    │ • Feature Eng.  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Layer    │    │  Business Logic │    │   Data Layer    │
│                 │    │                 │    │                 │
│ • Healthcare    │    │ • Risk Analysis │    │ • Heart Dataset │
│   Providers     │    │ • Recommendations│    │ • Synthetic Data│
│ • Patients      │    │ • Validation    │    │ • Model Storage │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Component Details

### Frontend Layer (Streamlit)
- **Purpose**: User interface for healthcare providers and patients
- **Technology**: Streamlit with Plotly visualizations
- **Features**:
  - Patient data input forms
  - Risk assessment dashboard
  - Interactive analytics
  - Real-time predictions

### Backend Layer (FastAPI)
- **Purpose**: API services and business logic
- **Technology**: FastAPI with Pydantic models
- **Features**:
  - RESTful API endpoints
  - Data validation and preprocessing
  - ML model integration
  - Error handling and logging

### ML Layer (Scikit-learn)
- **Purpose**: Machine learning models and predictions
- **Technology**: Scikit-learn, Pandas, NumPy
- **Features**:
  - Risk prediction models
  - Feature engineering
  - Model training and evaluation
  - Batch and real-time inference

## Data Flow

1. **Input**: Patient data entered through Streamlit interface
2. **Validation**: FastAPI validates and preprocesses data
3. **Processing**: ML models analyze patient data
4. **Prediction**: Risk scores and recommendations generated
5. **Output**: Results displayed in interactive dashboard

## Security Architecture

- **Authentication**: JWT-based authentication (ready for implementation)
- **Authorization**: Role-based access control
- **Data Protection**: Input validation and sanitization
- **Network Security**: CORS configuration and HTTPS support

## Scalability Considerations

- **Horizontal Scaling**: Containerized services with Docker
- **Load Balancing**: Multiple backend instances
- **Caching**: Model predictions and static data
- **Database**: SQLite for development, PostgreSQL for production
