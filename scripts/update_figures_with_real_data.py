#!/usr/bin/env python3
"""
Script para atualizar todas as figuras usando dados reais dos modelos treinados
Recalcula matrizes de confusão e gera todas as métricas a partir dos modelos
"""

import sys
import os
from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / 'src'))

try:
    from src.config import MODELS_DIR, RESULTS_DIR, DATA_DIR, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
    from src.train_all_models import load_reddit_en, load_twitter_en, load_merged_en
    from src.train_all_models import load_reddit_pt, load_twitter_pt, load_merged_pt
    from src.train_all_models import preprocess_df
except ImportError:
    from config import MODELS_DIR, RESULTS_DIR, DATA_DIR, RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE
    from train_all_models import load_reddit_en, load_twitter_en, load_merged_en
    from train_all_models import load_reddit_pt, load_twitter_pt, load_merged_pt
    from train_all_models import preprocess_df

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

OUTPUT_DIR = base_dir / 'tcc_figures'
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_model_and_vectorizer(dataset_name, language):
    model_dir = MODELS_DIR / f"{dataset_name}_{language}"
    
    if not model_dir.exists():
        return None, None, None
    
    try:
        with open(model_dir / "vectorizer.pkl", "rb") as f:
            vectorizer = pickle.load(f)
        
        models = {}
        for model_type in ['logistic_regression', 'svm', 'random_forest']:
            model_path = model_dir / f"{model_type}.pkl"
            if model_path.exists():
                with open(model_path, "rb") as f:
                    models[model_type] = pickle.load(f)
        
        return vectorizer, models, model_dir
    except Exception as e:
        print(f"Erro ao carregar modelos de {dataset_name}_{language}: {e}")
        return None, None, None

def calculate_real_confusion_matrix(dataset_name, language, model_type='logistic_regression'):
    loaders = {
        ('reddit', 'en'): load_reddit_en,
        ('twitter', 'en'): load_twitter_en,
        ('merged', 'en'): load_merged_en,
        ('reddit', 'pt'): load_reddit_pt,
        ('twitter', 'pt'): load_twitter_pt,
        ('merged', 'pt'): load_merged_pt,
    }
    
    loader = loaders.get((dataset_name, language))
    if not loader:
        return None
    
    try:
        df = loader()
        if df is None or len(df) == 0:
            return None
        
        lang = 'portuguese' if language == 'pt' else 'english'
        df_processed = preprocess_df(df, lang)
        X = df_processed['clean_text']
        y = df_processed['label']
        
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=TEST_SIZE + VALIDATION_SIZE, 
            stratify=y, random_state=RANDOM_STATE
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=TEST_SIZE/(TEST_SIZE + VALIDATION_SIZE),
            stratify=y_temp, random_state=RANDOM_STATE
        )
        
        vectorizer, models, _ = load_model_and_vectorizer(dataset_name, language)
        if vectorizer is None or model_type not in models:
            return None
        
        model = models[model_type]
        X_test_tfidf = vectorizer.transform(X_test)
        y_pred = model.predict(X_test_tfidf)
        
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        
        if cm.shape != (2, 2):
            print(f"     ⚠️ Matriz tem shape {cm.shape}, expandindo para 2x2...")
            new_cm = np.zeros((2, 2), dtype=int)
            if cm.shape == (1, 1):
                if 0 in np.unique(y_test) and 0 in np.unique(y_pred):
                    new_cm[0, 0] = cm[0, 0]
                elif 1 in np.unique(y_test) and 1 in np.unique(y_pred):
                    new_cm[1, 1] = cm[0, 0]
            elif cm.shape == (2, 1):
                new_cm[:, 0] = cm[:, 0]
            elif cm.shape == (1, 2):
                new_cm[0, :] = cm[0, :]
            else:
                new_cm[:cm.shape[0], :cm.shape[1]] = cm
            cm = new_cm
        
        return cm, y_test, y_pred, model.predict_proba(X_test_tfidf)
    except Exception as e:
        print(f"Erro ao calcular matriz de confusão para {dataset_name}_{language}: {e}")
        return None

def load_all_results():
    all_results = []
    
    for result_file in RESULTS_DIR.glob("*_results.json"):
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result = json.load(f)
                all_results.append(result)
        except Exception as e:
            print(f"Erro ao carregar {result_file}: {e}")
    
    comparison_file = RESULTS_DIR / 'model_comparison_all.json'
    if comparison_file.exists():
        try:
            with open(comparison_file, 'r', encoding='utf-8') as f:
                comparison_results = json.load(f)
                all_results.extend(comparison_results)
        except Exception as e:
            print(f"Erro ao carregar {comparison_file}: {e}")
    
    return all_results

def save_figure(filename, dpi=300):
    path = OUTPUT_DIR / filename
    plt.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ {filename}")
    return path

def plot_confusion_matrix(cm, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if cm.shape == (1, 1):
        print(f"   ⚠️ Matriz de confusão {filename} tem apenas 1 classe. Expandindo para 2x2...")
        new_cm = np.zeros((2, 2), dtype=int)
        if len(np.unique(cm)) == 1:
            unique_val = cm[0, 0]
            if unique_val == 0:
                new_cm[0, 0] = cm[0, 0]
            else:
                new_cm[1, 1] = cm[0, 0]
        cm = new_cm
    elif cm.shape == (2, 1) or cm.shape == (1, 2):
        print(f"   ⚠️ Matriz de confusão {filename} tem formato irregular {cm.shape}. Corrigindo...")
        new_cm = np.zeros((2, 2), dtype=int)
        if cm.shape == (2, 1):
            new_cm[:, 0] = cm[:, 0]
        else:
            new_cm[0, :] = cm[0, :]
        cm = new_cm
    
    if cm.shape != (2, 2):
        print(f"   ⚠️ AVISO: Matriz não é 2x2! Shape={cm.shape}, expandindo...")
        new_cm = np.zeros((2, 2), dtype=int)
        if cm.size > 0:
            new_cm[:min(2, cm.shape[0]), :min(2, cm.shape[1])] = cm[:min(2, cm.shape[0]), :min(2, cm.shape[1])]
        cm = new_cm
    
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues', aspect='auto')
    ax.figure.colorbar(im, ax=ax, shrink=0.8)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, format(cm[i, j], 'd'),
                   horizontalalignment='center',
                   verticalalignment='center',
                   color=text_color, fontsize=16, fontweight='bold')
    
    ax.set_xticks(np.arange(cm.shape[1]))
    ax.set_yticks(np.arange(cm.shape[0]))
    ax.set_xticklabels(['Não Suicida', 'Suicida'])
    ax.set_yticklabels(['Não Suicida', 'Suicida'])
    
    ax.set_xlabel('Predição', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    save_figure(filename)

def plot_cv_scores(scores, title, filename):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(1, len(scores) + 1)
    mean = np.mean(scores)
    std = np.std(scores)
    
    ax.plot(x, scores, 'o-', linewidth=2, markersize=10, label='Score por Fold', color='#2E86AB')
    ax.axhline(mean, color='#A23B72', linestyle='--', linewidth=2, label=f'Média: {mean:.4f}')
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color='#A23B72', 
                    label=f'±1 std: {std:.4f}')
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Acurácia', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([min(scores) - 0.02, max(scores) + 0.02])
    
    save_figure(filename)

def main():
    print("="*80)
    print("ATUALIZANDO FIGURAS COM DADOS REAIS DOS MODELOS")
    print("="*80)
    
    all_results = load_all_results()
    print(f"\nCarregados {len(all_results)} resultados de modelos")
    
    datasets_config = [
        ('reddit', 'en'), ('reddit', 'pt'),
        ('twitter', 'en'), ('twitter', 'pt'),
        ('merged', 'en'), ('merged', 'pt')
    ]
    
    print("\n1. Gerando matrizes de confusão reais...")
    confusion_matrices = {}
    
    for dataset_name, language in datasets_config:
        key = f"{dataset_name}_{language}"
        result = calculate_real_confusion_matrix(dataset_name, language, 'logistic_regression')
        
        if result:
            cm, _, _, _ = result
            confusion_matrices[key] = cm
            dataset_display = key.replace('_', ' ').title()
            plot_confusion_matrix(cm, f'Matriz de Confusão - {dataset_display}', 
                                 f'{key}_confusion_matrix.png')
        else:
            print(f"  ⚠️ Não foi possível gerar matriz para {key}")
    
    print("\n2. Gerando gráficos de validação cruzada...")
    cv_scores_data = {}
    
    for result in all_results:
        dataset = result.get('dataset')
        language = result.get('language')
        key = f"{dataset}_{language}"
        
        cv_data = result.get('cross_validation', {})
        if 'logistic_regression' in cv_data:
            scores = cv_data['logistic_regression'].get('scores', [])
            if scores:
                cv_scores_data[key] = scores
                dataset_display = key.replace('_', ' ').title()
                plot_cv_scores(scores, f'Validação Cruzada - {dataset_display}', 
                              f'{key}_cv_scores.png')
    
    print("\n3. Gerando comparação de modelos com dados reais...")
    models_comparison = {}
    
    for result in all_results:
        dataset = result.get('dataset')
        language = result.get('language')
        key = f"{dataset} ({language.upper()})"
        
        if key not in models_comparison:
            models_comparison[key] = {}
        
        test_metrics = result.get('test_metrics', {})
        for model_type, metrics in test_metrics.items():
            model_short = {'logistic_regression': 'LR', 'svm': 'SVM', 'random_forest': 'RF'}.get(model_type)
            if model_short:
                accuracy = metrics.get('accuracy', 0) * 100
                models_comparison[key][model_short] = accuracy
    
    if models_comparison:
        fig, ax = plt.subplots(figsize=(14, 7))
        
        datasets_list = list(models_comparison.keys())
        x = np.arange(len(datasets_list))
        width = 0.25
        colors = ['#3498db', '#e74c3c', '#2ecc71']
        
        for i, model in enumerate(['LR', 'SVM', 'RF']):
            values = [models_comparison[ds].get(model, 0) for ds in datasets_list]
            bars = ax.bar(x + i * width, values, width, label=model, color=colors[i], alpha=0.8)
            
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
        ax.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Acurácia entre Modelos e Datasets', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(datasets_list, rotation=15, ha='right')
        ax.legend(fontsize=11, title='Modelo', title_fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([75, 95])
        
        save_figure('model_comparison_bar_chart.png')
    
    print("\n4. Gerando curvas ROC e PR com dados reais...")
    roc_data = {}
    pr_data = {}
    
    for dataset_name, language in datasets_config:
        key = f"{dataset_name}_{language}"
        result_data = calculate_real_confusion_matrix(dataset_name, language, 'logistic_regression')
        
        if result_data:
            _, y_test, _, y_proba = result_data
            
            fpr, tpr, _ = roc_curve(y_test, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)
            roc_data[key] = (fpr, tpr, roc_auc)
            
            precision, recall, _ = precision_recall_curve(y_test, y_proba[:, 1])
            pr_auc = auc(recall, precision)
            pr_data[key] = (recall, precision, pr_auc)
    
    if roc_data:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors_map = {
            'reddit_en': '#3498db', 'reddit_pt': '#2980b9',
            'twitter_en': '#e74c3c', 'twitter_pt': '#c0392b',
            'merged_en': '#2ecc71', 'merged_pt': '#27ae60'
        }
        
        for key, (fpr, tpr, roc_auc) in roc_data.items():
            label = key.replace('_', ' ').title()
            color = colors_map.get(key, '#95a5a6')
            ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.3f})', 
                   linewidth=2.5, color=color)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12, fontweight='bold')
        ax.set_title('Curvas ROC - Todos os Datasets', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower right')
        ax.grid(True, alpha=0.3)
        
        save_figure('roc_curves_all_models.png')
    
    if pr_data:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for key, (recall, precision, pr_auc) in pr_data.items():
            label = key.replace('_', ' ').title()
            color = colors_map.get(key, '#95a5a6')
            ax.plot(recall, precision, label=f'{label} (AUC = {pr_auc:.3f})', 
                   linewidth=2.5, color=color)
        
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Curvas Precision-Recall - Todos os Datasets', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='lower left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0.6, 1.0])
        
        save_figure('pr_curves_all_models.png')
    
    print("\n5. Gerando impacto da tradução com dados reais...")
    translation_comparison = {}
    
    for result in all_results:
        dataset = result.get('dataset')
        language = result.get('language')
        
        if dataset not in translation_comparison:
            translation_comparison[dataset] = {}
        
        test_metrics = result.get('test_metrics', {})
        best_model = result.get('best_model', 'logistic_regression')
        if best_model in test_metrics:
            accuracy = test_metrics[best_model].get('accuracy', 0) * 100
            translation_comparison[dataset][language] = accuracy
    
    if translation_comparison:
        datasets = []
        en_scores = []
        pt_scores = []
        
        for dataset, scores in translation_comparison.items():
            if 'en' in scores and 'pt' in scores:
                datasets.append(dataset.title())
                en_scores.append(scores['en'])
                pt_scores.append(scores['pt'])
        
        if datasets:
            fig, ax = plt.subplots(figsize=(10, 7))
            
            x = np.arange(len(datasets))
            width = 0.35
            
            bars1 = ax.bar(x - width/2, en_scores, width, label='Inglês (Original)', 
                          color='#3498db', alpha=0.8)
            bars2 = ax.bar(x + width/2, pt_scores, width, label='Português (Traduzido)', 
                          color='#e74c3c', alpha=0.8)
            
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            for i, (en, pt) in enumerate(zip(en_scores, pt_scores)):
                diff = ((pt - en) / en) * 100
                ax.text(i, max(en, pt) + 1, f'{diff:+.1f}%', ha='center', 
                       fontsize=9, color='green' if diff >= 0 else 'red', fontweight='bold')
            
            ax.set_xlabel('Dataset', fontsize=12, fontweight='bold')
            ax.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
            ax.set_title('Impacto da Tradução Automática no Desempenho', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(datasets)
            ax.legend(fontsize=11)
            ax.grid(axis='y', alpha=0.3)
            ax.set_ylim([75, 95])
            
            save_figure('translation_impact_comparison.png')
    
    print("\n6. Gerando sumário final de métricas...")
    metrics_summary = {}
    
    for result in all_results:
        dataset = result.get('dataset')
        language = result.get('language')
        key = f"{dataset} ({language.upper()})"
        
        test_metrics = result.get('test_metrics', {})
        best_model = result.get('best_model', 'logistic_regression')
        
        if best_model in test_metrics:
            metrics = test_metrics[best_model]
            metrics_summary[key] = {
                'Acurácia': metrics.get('accuracy', 0) * 100,
                'F1-Score': metrics.get('f1_score', 0) * 100,
                'ROC-AUC': metrics.get('roc_auc', 0) * 100,
                'PR-AUC': metrics.get('pr_auc', 0) * 100
            }
    
    if metrics_summary:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sumário Final de Métricas de Desempenho', fontsize=16, fontweight='bold', y=0.995)
        
        datasets_list = list(metrics_summary.keys())
        colors_map = plt.cm.Set3(np.linspace(0, 1, len(datasets_list)))
        
        for idx, metric in enumerate(['Acurácia', 'F1-Score', 'ROC-AUC', 'PR-AUC']):
            ax = axes[idx // 2, idx % 2]
            values = [metrics_summary[ds][metric] for ds in datasets_list]
            
            bars = ax.barh(datasets_list, values, color=colors_map, alpha=0.8, edgecolor='black')
            
            for bar, val in zip(bars, values):
                width = bar.get_width()
                ax.text(width, bar.get_y() + bar.get_height()/2, 
                       f' {val:.2f}%', va='center', fontsize=10, fontweight='bold')
            
            ax.set_xlabel('Valor (%)', fontsize=11, fontweight='bold')
            ax.set_title(metric, fontsize=13, fontweight='bold')
            ax.set_xlim([75, 100])
            ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_figure('final_metrics_summary.png')
    
    print("\n" + "="*80)
    print("FIGURAS ATUALIZADAS COM SUCESSO!")
    print("="*80)
    print(f"Total de figuras geradas: {len(list(OUTPUT_DIR.glob('*.png')))}")
    print(f"Localização: {OUTPUT_DIR.absolute()}/")

if __name__ == "__main__":
    main()

