# 🏥 Training Guide for 10GB EHR Dataset

## 📋 Prerequisites

### 1. **Dataset Preparation**
- Download your 10GB EHR dataset from Kaggle
- Extract all JSON files to: `d:\personalized-healthcare\data\ehr\`
- Ensure the files are in FHIR format or similar EHR JSON structure

### 2. **System Requirements**
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Storage**: 15GB+ free space (for dataset + models)
- **CPU**: Multi-core processor recommended
- **Python**: 3.8+ (you have 3.13.5 ✅)

### 3. **Install Additional Dependencies**
```bash
pip install psutil tqdm
```

## 🚀 Training Steps

### Step 1: Place Your Dataset
1. Download your Kaggle EHR dataset
2. Extract all JSON files to: `data\ehr\`
3. Verify files are present:
```bash
ls data\ehr\*.json | wc -l
```

### Step 2: Run Dataset Analysis (Optional)
```bash
cd backend
python -c "
from utils.ehr_processor import EHRDataProcessor
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ehr_path = os.path.join(project_root, 'data', 'ehr')
processor = EHRDataProcessor(ehr_path)
memory_info = processor.estimate_memory_usage()
feature_info = processor.get_feature_summary(sample_size=100)
print('Dataset Analysis:')
print(f'Total Size: {memory_info[\"total_size_gb\"]:.2f} GB')
print(f'Total Files: {memory_info[\"total_files\"]:,}')
print(f'Recommended Batch Size: {memory_info[\"recommended_batch_size\"]}')
print(f'Total Features: {feature_info[\"total_features\"]}')
"
```

### Step 3: Start Training
```bash
cd backend
python train_config.py
```

### Step 4: Monitor Training Progress
- Watch the console output for progress
- Check `training.log` for detailed logs
- Training time: 30 minutes to 4+ hours depending on dataset size

## ⚙️ Configuration Options

### Memory Optimization
If you encounter memory issues, edit `train_config.py`:
```python
# Limit samples for testing
self.max_samples = 50000  # Start with 50K samples

# Reduce batch size
processor = EHRDataProcessor(data_path, max_workers=2)
```

### Model Selection
For large datasets, the script uses:
- ✅ **Random Forest** (Best for large datasets)
- ✅ **Gradient Boosting** (Good accuracy)
- ✅ **Logistic Regression** (Fast, interpretable)
- ❌ **SVM** (Removed - too slow for large datasets)

## 📊 Expected Output

### 1. **Console Output**
```
Found 50,000 JSON files in data/ehr
Dataset Size: 10.2 GB
Recommended Batch Size: 500
Processing EHR batches: 100%|████████| 100/100
Loaded 1,250,000 samples in 45.2s
Training random_forest: Accuracy: 0.8245, F1: 0.8156
Best model: RandomForestClassifier
Training completed successfully!
```

### 2. **Generated Files**
- `models/ehr_risk_model.pkl` - Trained model
- `models/ehr_training_report.md` - Performance report
- `training.log` - Detailed training logs

### 3. **Performance Expectations**
With 10GB of quality EHR data, expect:
- **Accuracy**: 75-90% (depending on data quality)
- **F1 Score**: 70-85%
- **Training Time**: 1-4 hours
- **Model Size**: 50-200MB

## 🔧 Troubleshooting

### Common Issues:

#### 1. **Out of Memory Error**
```python
# In train_config.py, reduce batch size:
self.max_samples = 10000  # Start smaller
```

#### 2. **JSON Parsing Errors**
- Check if your JSON files are valid FHIR format
- The processor will skip invalid files automatically

#### 3. **Low Accuracy**
- Increase `max_samples` to use more data
- Check if target variable generation is appropriate for your use case
- Consider feature engineering

#### 4. **Slow Training**
```python
# Reduce parallel workers:
processor = EHRDataProcessor(data_path, max_workers=2)
```

## 📈 Model Evaluation

After training, check the generated report:
```bash
cat models/ehr_training_report.md
```

Look for:
- **F1 Score > 0.7**: Good performance
- **Feature Importance**: Top predictive features
- **Cross-validation results**: Model stability

## 🎯 Next Steps

1. **Test the Model**: Use the API endpoints to test predictions
2. **Fine-tune**: Adjust hyperparameters based on results  
3. **Deploy**: Use the trained model in your healthcare application
4. **Monitor**: Track model performance with new data

## 💡 Pro Tips

1. **Start Small**: Begin with `max_samples=10000` to test the pipeline
2. **Monitor Resources**: Watch RAM usage during training
3. **Backup Models**: Save intermediate models during long training
4. **Validate Data**: Ensure your EHR data has relevant health indicators
5. **Feature Engineering**: Consider domain-specific feature extraction

---

**Ready to train?** Run `python backend/train_config.py` and follow the prompts!