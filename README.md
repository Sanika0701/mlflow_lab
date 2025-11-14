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

## Mlflow Output Screenshots: Metrics Tab
- F1 Score
<img width="1200" height="600" alt="f1_score" src="https://github.com/user-attachments/assets/688bf18c-51ea-4eac-bcd4-43cdf4e766af" />

- Precision
<img width="1200" height="600" alt="precision" src="https://github.com/user-attachments/assets/d5b4212c-0237-4bcd-9d11-7755201947a5" />

- Recall
<img width="1200" height="600" alt="recall" src="https://github.com/user-attachments/assets/343a25e5-9db3-40bf-8ea6-5ef6b7bdbbc6" />

- Accuracy
<img width="1200" height="600" alt="accuracy" src="https://github.com/user-attachments/assets/8f1af200-042a-47cb-944a-f7455f7e7574" />

- AUC
<img width="1200" height="600" alt="auc" src="https://github.com/user-attachments/assets/77ebd8cd-f741-4ce7-a790-411a205bb9e1" />
