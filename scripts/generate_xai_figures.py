#!/usr/bin/env python3
"""
Script para gerar todas as figuras XAI (SHAP e LIME) para o TCC
"""

import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

base_dir = Path(__file__).parent.parent
sys.path.insert(0, str(base_dir))
sys.path.insert(0, str(base_dir / 'src'))

try:
    from src.xai import ModelExplainer
    from src.xai_multi_model import MultiModelExplainer
    from src.train_all_models import load_reddit_en, preprocess_df
    from src.config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE, MODELS_DIR
except ImportError:
    from xai import ModelExplainer
    from xai_multi_model import MultiModelExplainer
    from train_all_models import load_reddit_en, preprocess_df
    from config import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE, MODELS_DIR

OUTPUT_DIR = base_dir / 'tcc_figures'
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("GERANDO FIGURAS XAI PARA O TCC")
print("="*80)

high_risk_text = "I feel hopeless and don't want to live anymore. Life has no meaning."
low_risk_text = "Had an amazing day with family. Life is beautiful and I'm grateful!"

dataset_to_use = 'reddit_en'
model_dir = MODELS_DIR / dataset_to_use

if not model_dir.exists():
    print(f"⚠️ Diretório do modelo não encontrado: {model_dir}")
    print("   Pulando geração de figuras XAI...")
    sys.exit(1)

print(f"\nUsando modelos de: {dataset_to_use}")
print(f"Diretório: {model_dir}")

try:
    print("\n1. Gerando global_importance_plot.png...")
    try:
        explainer = ModelExplainer(
            model_path=str(model_dir / 'logistic_regression.pkl'),
            vectorizer_path=str(model_dir / 'vectorizer.pkl')
        )
        
        df = load_reddit_en()
        df_processed = preprocess_df(df, 'english')
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
        
        explainer.plot_global_importance(X_test, y_test, 
                                        save_path=str(OUTPUT_DIR / 'global_importance_plot.png'), 
                                        top_n=30)
        print("   ✓ global_importance_plot.png gerado")
    except Exception as e:
        print(f"   ⚠️ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Gerando shap_high_risk_example.png...")
    try:
        explainer.plot_shap_explanation(high_risk_text, 
                                       save_path=str(OUTPUT_DIR / 'shap_high_risk_example.png'), 
                                       max_features=15)
        print("   ✓ shap_high_risk_example.png gerado")
    except Exception as e:
        print(f"   ⚠️ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Gerando shap_low_risk_example.png...")
    try:
        explainer.plot_shap_explanation(low_risk_text, 
                                       save_path=str(OUTPUT_DIR / 'shap_low_risk_example.png'), 
                                       max_features=15)
        print("   ✓ shap_low_risk_example.png gerado")
    except Exception as e:
        print(f"   ⚠️ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Gerando comparison_shap_lime.png...")
    try:
        explainer.plot_comparison(high_risk_text, 
                                 save_path=str(OUTPUT_DIR / 'comparison_shap_lime.png'), 
                                 max_features=10)
        print("   ✓ comparison_shap_lime.png gerado")
    except Exception as e:
        print(f"   ⚠️ Erro: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n5. Gerando multi_model_shap_comparison.png...")
    try:
        multi = MultiModelExplainer(dataset_to_use)
        multi.plot_model_comparison(high_risk_text, 
                                   save_path=str(OUTPUT_DIR / 'multi_model_shap_comparison.png'), 
                                   top_n=10)
        print("   ✓ multi_model_shap_comparison.png gerado")
    except Exception as e:
        print(f"   ⚠️ Erro: {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"\n⚠️ Erro geral ao gerar figuras XAI: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("GERAÇÃO DE FIGURAS XAI CONCLUÍDA!")
print("="*80)

