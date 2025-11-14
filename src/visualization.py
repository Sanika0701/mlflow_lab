"""
Visualization utilities for model evaluation
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import mlflow

def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """
    Create and save confusion matrix plot
    """
    cm = confusion_matrix(y_true, (y_pred > 0.5).astype(int))
    disp = ConfusionMatrixDisplay(cm, display_labels=['Low Quality', 'High Quality'])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def plot_roc_curve(y_true, y_pred_proba, model_name='Model', save_path='roc_curve.png'):
    """
    Create and save ROC curve plot
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def plot_feature_importance(model, feature_names, top_n=10, save_path='feature_importance.png'):
    """
    Create and save feature importance plot
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(importances['feature'], importances['importance'])
    plt.xlabel('Importance', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Feature Importances', fontsize=14)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def plot_prediction_distribution(y_true, y_pred_proba, save_path='prediction_distribution.png'):
    """
    Create and save prediction distribution plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(y_pred_proba[y_true == 0], bins=30, alpha=0.6, label='Low Quality', color='red')
    axes[0].hist(y_pred_proba[y_true == 1], bins=30, alpha=0.6, label='High Quality', color='green')
    axes[0].set_xlabel('Predicted Probability', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Prediction Distribution by True Class', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Box plot
    df_plot = pd.DataFrame({
        'Probability': y_pred_proba,
        'True Class': ['High Quality' if x == 1 else 'Low Quality' for x in y_true]
    })
    sns.boxplot(data=df_plot, x='True Class', y='Probability', ax=axes[1])
    axes[1].set_title('Prediction Probability by True Class', fontsize=14)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def log_visualizations_to_mlflow(y_true, y_pred_proba, model, feature_names, model_name='Model'):
    """
    Create all visualizations and log them to MLflow
    """
    # Confusion Matrix
    cm_path = plot_confusion_matrix(y_true, y_pred_proba, f'{model_name}_confusion_matrix.png')
    mlflow.log_artifact(cm_path)
    
    # ROC Curve
    roc_path = plot_roc_curve(y_true, y_pred_proba, model_name, f'{model_name}_roc_curve.png')
    mlflow.log_artifact(roc_path)
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        fi_path = plot_feature_importance(model, feature_names, top_n=10, 
                                         save_path=f'{model_name}_feature_importance.png')
        mlflow.log_artifact(fi_path)
    
    # Prediction Distribution
    pred_dist_path = plot_prediction_distribution(y_true, y_pred_proba, 
                                                  f'{model_name}_prediction_distribution.png')
    mlflow.log_artifact(pred_dist_path)