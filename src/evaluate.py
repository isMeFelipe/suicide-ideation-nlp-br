import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
try:
    from .config import RESULTS_DIR
except ImportError:
    from config import RESULTS_DIR

def calculate_metrics(y_true, y_pred, y_proba=None):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
        'precision_class_0': precision_score(y_true, y_pred, pos_label=0),
        'recall_class_0': recall_score(y_true, y_pred, pos_label=0),
        'f1_class_0': f1_score(y_true, y_pred, pos_label=0),
        'precision_class_1': precision_score(y_true, y_pred, pos_label=1),
        'recall_class_1': recall_score(y_true, y_pred, pos_label=1),
        'f1_class_1': f1_score(y_true, y_pred, pos_label=1),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sem Risco', 'Risco'],
                yticklabels=['Sem Risco', 'Risco'])
    plt.title(f'Matriz de Confusão - {model_name}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curve(y_true, y_proba, model_name="Model", save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    roc_auc = roc_auc_score(y_true, y_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curve(y_true, y_proba, model_name="Model", save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
    pr_auc = average_precision_score(y_true, y_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR (AUC = {pr_auc:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curva Precision-Recall - {model_name}')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def evaluate_model(y_true, y_pred, y_proba=None, model_name="Model", save_results=True):
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    print(f"\n{'='*60}")
    print(f"AVALIAÇÃO DO MODELO: {model_name}")
    print(f"{'='*60}")
    print(f"\nMétricas Gerais:")
    print(f"  Acurácia: {metrics['accuracy']:.4f}")
    print(f"  Precision (weighted): {metrics['precision']:.4f}")
    print(f"  Recall (weighted): {metrics['recall']:.4f}")
    print(f"  F1-Score (weighted): {metrics['f1_score']:.4f}")
    
    if y_proba is not None:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    
    print(f"\nMétricas por Classe:")
    print(f"  Classe 0 (Sem Risco):")
    print(f"    Precision: {metrics['precision_class_0']:.4f}")
    print(f"    Recall: {metrics['recall_class_0']:.4f}")
    print(f"    F1-Score: {metrics['f1_class_0']:.4f}")
    print(f"  Classe 1 (Risco):")
    print(f"    Precision: {metrics['precision_class_1']:.4f}")
    print(f"    Recall: {metrics['recall_class_1']:.4f}")
    print(f"    F1-Score: {metrics['f1_class_1']:.4f}")
    
    print(f"\n{classification_report(y_true, y_pred)}")
    
    if save_results:
        results_dir = RESULTS_DIR / model_name.lower().replace(' ', '_')
        results_dir.mkdir(exist_ok=True)
        
        metrics_path = results_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
        
        plot_confusion_matrix(y_true, y_pred, model_name, 
                            results_dir / "confusion_matrix.png")
        
        if y_proba is not None:
            plot_roc_curve(y_true, y_proba, model_name,
                          results_dir / "roc_curve.png")
            plot_precision_recall_curve(y_true, y_proba, model_name,
                                      results_dir / "pr_curve.png")
    
    return metrics

