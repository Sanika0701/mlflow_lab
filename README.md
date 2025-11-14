# mlflow_lab

# Wine Quality Prediction - MLflow Project

A comprehensive machine learning project comparing multiple models for wine quality classification with full MLflow tracking and monitoring.

## 🎯 Project Overview

This project extends traditional wine quality prediction by:
- Comparing **4 models**: Random Forest, XGBoost, LightGBM, and CatBoost
- Implementing **complete MLflow tracking** for all experiments
- Creating **custom visualizations** (ROC curves, confusion matrices, feature importance)
- Building a **monitoring system** with prediction analysis and calibration curves
- Organizing code into **reusable modules**

## 📊 Results

| Model | AUC | Status |
|-------|-----|--------|
| **Random Forest** | **0.8993** | ✅ Best |
| [Other models] | [TBD] | After upload |

## 🗂️ Project Structure
```
mlflow_lab/
├── data/                          # Wine datasets
├── notebooks/                     # Jupyter notebooks
│   ├── 01_data_preparation.ipynb
│   ├── 02_model_training.ipynb
│   ├── 03_model_comparison.ipynb
│   └── 04_model_monitoring.ipynb
├── src/                           # Source code modules
│   ├── data_loader.py
│   ├── model_trainer.py
│   └── visualization.py
└── requirements.txt
```

## 🚀 Quick Start

1. **Setup Environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

2. **Start MLflow UI**
```bash
mlflow ui --port=5001
```
Access at: http://localhost:5001

3. **Run Notebooks** (in order)
- Data Preparation → Model Training → Comparison → Monitoring

## 🎓 Key Features

✅ **4 Model Comparison** - RF, XGBoost, LightGBM, CatBoost  
✅ **Complete MLflow Integration** - All metrics, parameters, and artifacts tracked  
✅ **Custom Visualizations** - ROC curves, confusion matrices, feature importance  
✅ **Monitoring Dashboard** - Prediction analysis and calibration  
✅ **Modular Code** - Organized, reusable Python modules  

## 📈 Best Model Performance

- **Model**: Random Forest
- **Test AUC**: 0.8993
- **Accuracy**: 86.1%
- **High Confidence Predictions**: 77.9%

## 🛠️ Technologies

- MLflow for experiment tracking
- Scikit-learn, XGBoost, LightGBM, CatBoost
- Matplotlib & Seaborn for visualizations
