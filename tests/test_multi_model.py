import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.xai_multi_model import MultiModelExplainer
import pandas as pd

print("="*80)
print("TESTE DE COMPARAÇÃO MULTI-MODELO")
print("="*80)

datasets = ['reddit_en', 'twitter_en']

test_text = "I feel hopeless and don't want to live anymore. Life has no meaning."

for dataset in datasets:
    print(f"\n{'='*80}")
    print(f"DATASET: {dataset.upper()}")
    print(f"{'='*80}\n")
    
    try:
        multi_explainer = MultiModelExplainer(dataset=dataset)
        
        print(f"Modelos carregados: {', '.join(multi_explainer.get_model_names())}\n")
        
        print("-"*80)
        print("TEXTO DE TESTE:")
        print("-"*80)
        print(f"{test_text}\n")
        
        print("-"*80)
        print("COMPARAÇÃO DE PREDIÇÕES:")
        print("-"*80)
        df_pred = multi_explainer.compare_predictions(test_text)
        print(df_pred.to_string(index=False))
        
        print("\n" + "-"*80)
        print("TOP 5 FEATURES SHAP POR MODELO:")
        print("-"*80)
        
        comparison = multi_explainer.compare_shap_features(test_text, top_n=5)
        
        for model_name, features in comparison.items():
            if isinstance(features, dict) and 'error' in features:
                print(f"\n{model_name.upper()}: Erro - {features['error']}")
                continue
            
            print(f"\n{model_name.upper()}:")
            for i, feat in enumerate(features[:5], 1):
                icon = "🔴" if feat['shap_value'] > 0 else "🟢"
                print(f"  {i}. {icon} {feat['feature']:15s} | {feat['shap_value']:+7.4f}")
        
        print("\n" + "="*80)
        print("GERANDO VISUALIZAÇÕES...")
        print("="*80)
        
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            fig = multi_explainer.plot_predictions_comparison(test_text)
            filename = f'test_multi_predictions_{dataset}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Gráfico de predições salvo: {filename}")
            
            fig = multi_explainer.plot_model_comparison(test_text, top_n=10)
            filename = f'test_multi_shap_{dataset}.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"✓ Comparação SHAP salva: {filename}")
            
        except Exception as e:
            print(f"✗ Erro ao gerar visualizações: {e}")
        
    except Exception as e:
        print(f"✗ Erro ao processar {dataset}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TESTE CONCLUÍDO!")
print("="*80)
print("\nPara usar a interface web interativa:")
print("  streamlit run app_multi_model.py")
print("\nPara gerar análises completas:")
print("  python3 src/xai_multi_model.py")

