"""
Model training utilities
"""
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from mlflow.models.signature import infer_signature
from src.visualization import log_visualizations_to_mlflow

def log_metrics(y_true, y_pred, y_pred_proba):
    """
    Calculate and log multiple evaluation metrics
    """
    metrics = {
        'auc': roc_auc_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    for metric_name, metric_value in metrics.items():
        mlflow.log_metric(metric_name, metric_value)
    
    return metrics

def train_random_forest(X_train, y_train, X_test, y_test, run_name='random_forest'):
    """
    Train Random Forest model with MLflow tracking
    """
    with mlflow.start_run(run_name=run_name):
        # Parameters
        n_estimators = 100
        max_depth = 10
        
        # Train model
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Log parameters
        mlflow.log_param('n_estimators', n_estimators)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('model_type', 'RandomForest')
        
        # Log metrics
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        # Log visualizations
        log_visualizations_to_mlflow(y_test, y_pred_proba, model, 
                                     X_train.columns, 'RandomForest')
        
        # Log model
        signature = infer_signature(X_train, y_pred_proba)
        mlflow.sklearn.log_model(model, "model", signature=signature)
        
        print(f"✅ Random Forest - AUC: {metrics['auc']:.4f}")
        return model, metrics

def train_xgboost(X_train, y_train, X_test, y_test, run_name='xgboost'):
    """
    Train XGBoost model with MLflow tracking
    """
    with mlflow.start_run(run_name=run_name):
        # Parameters
        params = {
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 100,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'random_state': 42
        }
        
        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param('model_type', 'XGBoost')
        
        # Log metrics
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        # Log visualizations
        log_visualizations_to_mlflow(y_test, y_pred_proba, model, 
                                     X_train.columns, 'XGBoost')
        
        # Log model
        signature = infer_signature(X_train, y_pred_proba)
        mlflow.xgboost.log_model(model, "model", signature=signature)
        
        print(f"✅ XGBoost - AUC: {metrics['auc']:.4f}")
        return model, metrics

def train_lightgbm(X_train, y_train, X_test, y_test, run_name='lightgbm'):
    """
    Train LightGBM model with MLflow tracking
    """
    with mlflow.start_run(run_name=run_name):
        # Parameters
        params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 6,
            'num_leaves': 31,
            'random_state': 42
        }
        
        # Train model
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param('model_type', 'LightGBM')
        
        # Log metrics
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        # Log visualizations
        log_visualizations_to_mlflow(y_test, y_pred_proba, model, 
                                     X_train.columns, 'LightGBM')
        
        # Log model
        signature = infer_signature(X_train, y_pred_proba)
        mlflow.lightgbm.log_model(model, "model", signature=signature)
        
        print(f"✅ LightGBM - AUC: {metrics['auc']:.4f}")
        return model, metrics

def train_catboost(X_train, y_train, X_test, y_test, run_name='catboost'):
    """
    Train CatBoost model with MLflow tracking
    """
    with mlflow.start_run(run_name=run_name):
        # Parameters
        params = {
            'iterations': 100,
            'learning_rate': 0.1,
            'depth': 6,
            'random_state': 42,
            'verbose': False
        }
        
        # Train model
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Log parameters
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)
        mlflow.log_param('model_type', 'CatBoost')
        
        # Log metrics
        metrics = log_metrics(y_test, y_pred, y_pred_proba)
        
        # Log visualizations
        log_visualizations_to_mlflow(y_test, y_pred_proba, model, 
                                     X_train.columns, 'CatBoost')
        
        # Log model
        signature = infer_signature(X_train, y_pred_proba)
        mlflow.catboost.log_model(model, "model", signature=signature)
        
        print(f"✅ CatBoost - AUC: {metrics['auc']:.4f}")
        return model, metrics