import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.xai import ModelExplainer
from src.preprocess import load_datasets, preprocess
from src.config import RESULTS_DIR, RANDOM_STATE, TEST_SIZE
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def generate_global_analysis():
    print("="*80)
    print("GERANDO ANÁLISE GLOBAL DE EXPLICABILIDADE (XAI)")
    print("="*80)
    
    xai_dir = RESULTS_DIR / 'xai_analysis'
    xai_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n1. Carregando modelo e dados...")
    explainer = ModelExplainer()
    
    df = preprocess(load_datasets())
    X = df['clean_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    
    print(f"   ✓ Dados de teste: {len(X_test)} amostras")
    
    print("\n2. Calculando feature importance global...")
    importance_df = explainer.get_global_feature_importance(X_test, y_test, top_n=50)
    
    importance_path = xai_dir / 'global_feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"   ✓ Tabela salva: {importance_path}")
    
    print("\n   Top 20 Features Mais Importantes:")
    for i, row in importance_df.head(20).iterrows():
        print(f"   {i+1:2d}. {row['feature']:25s} | Importância: {row['importance']:.6f}")
    
    print("\n3. Gerando gráfico de importância global...")
    fig = explainer.plot_global_importance(X_test, y_test, top_n=30)
    plot_path = xai_dir / 'global_importance_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Gráfico salvo: {plot_path}")
    
    print("\n4. Gerando explicações para exemplos representativos...")
    
    suicidal_samples = X_test[y_test == 1].head(3)
    non_suicidal_samples = X_test[y_test == 0].head(3)
    
    examples_dir = xai_dir / 'example_explanations'
    examples_dir.mkdir(exist_ok=True)
    
    print("\n   Exemplos de textos SUICIDAS:")
    for i, text in enumerate(suicidal_samples, 1):
        print(f"   {i}. {text[:80]}...")
        try:
            report = explainer.generate_report(text, save_dir=examples_dir)
            print(f"      ✓ Relatório gerado")
        except Exception as e:
            print(f"      ✗ Erro: {e}")
    
    print("\n   Exemplos de textos NÃO SUICIDAS:")
    for i, text in enumerate(non_suicidal_samples, 1):
        print(f"   {i}. {text[:80]}...")
        try:
            report = explainer.generate_report(text, save_dir=examples_dir)
            print(f"      ✓ Relatório gerado")
        except Exception as e:
            print(f"      ✗ Erro: {e}")
    
    print("\n5. Gerando análise comparativa SHAP vs LIME...")
    
    comparison_samples = [
        ("Suicidal", X_test[y_test == 1].iloc[0]),
        ("Non-Suicidal", X_test[y_test == 0].iloc[0])
    ]
    
    for label, text in comparison_samples:
        print(f"\n   Gerando comparação para: {label}")
        print(f"   Texto: {text[:80]}...")
        
        try:
            comparison_path = xai_dir / f'comparison_{label.lower()}.png'
            fig = explainer.plot_comparison(text, save_path=str(comparison_path), max_features=15)
            plt.close()
            print(f"   ✓ Comparação salva: {comparison_path}")
        except Exception as e:
            print(f"   ✗ Erro: {e}")
    
    print("\n6. Gerando resumo estatístico...")
    
    summary = {
        'total_samples_test': len(X_test),
        'suicidal_samples': sum(y_test == 1),
        'non_suicidal_samples': sum(y_test == 0),
        'total_features': len(explainer.feature_names),
        'top_50_features': importance_df.head(50)['feature'].tolist(),
        'model_type': type(explainer.model).__name__
    }
    
    summary_path = xai_dir / 'analysis_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("RESUMO DA ANÁLISE GLOBAL DE XAI\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Modelo: {summary['model_type']}\n")
        f.write(f"Total de Features: {summary['total_features']}\n")
        f.write(f"Amostras de Teste: {summary['total_samples_test']}\n")
        f.write(f"  - Suicida: {summary['suicidal_samples']}\n")
        f.write(f"  - Não Suicida: {summary['non_suicidal_samples']}\n\n")
        
        f.write("-"*80 + "\n")
        f.write("TOP 50 FEATURES MAIS IMPORTANTES\n")
        f.write("-"*80 + "\n")
        for i, feat in enumerate(summary['top_50_features'], 1):
            f.write(f"{i:2d}. {feat}\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"   ✓ Resumo salvo: {summary_path}")
    
    print("\n" + "="*80)
    print("ANÁLISE XAI CONCLUÍDA!")
    print("="*80)
    print(f"\nTodos os resultados foram salvos em: {xai_dir}")
    print("\nArquivos gerados:")
    print(f"  - global_feature_importance.csv")
    print(f"  - global_importance_plot.png")
    print(f"  - analysis_summary.txt")
    print(f"  - example_explanations/ (múltiplos relatórios)")
    print(f"  - comparison_*.png")
    
    return xai_dir


def demonstrate_xai_methods():
    print("\n" + "="*80)
    print("DEMONSTRAÇÃO DOS MÉTODOS DE XAI")
    print("="*80)
    
    explainer = ModelExplainer()
    
    test_cases = [
        {
            'label': 'Alto Risco',
            'text': "I can't take it anymore. Life is meaningless and I want to end it all. Nobody cares."
        },
        {
            'label': 'Sem Risco',
            'text': "Had an amazing day at the beach with family. Life is beautiful and I'm so grateful!"
        },
        {
            'label': 'Risco Moderado',
            'text': "Feeling really down and hopeless lately. Everything seems pointless but I'm trying to cope."
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"CASO {i}: {case['label']}")
        print(f"{'='*80}")
        print(f"Texto: {case['text']}")
        print("-"*80)
        
        result = explainer.explain_text(case['text'], methods=['shap', 'lime'])
        
        print(f"\n🎯 PREDIÇÃO:")
        print(f"   Classe: {result['prediction_label']}")
        print(f"   Confiança: {result['probability']:.2%}")
        print(f"   Prob. Não Suicida: {result['probabilities']['Não Suicida']:.2%}")
        print(f"   Prob. Suicida: {result['probabilities']['Suicida']:.2%}")
        
        if 'shap' in result['explanations'] and 'error' not in result['explanations']['shap']:
            print(f"\n📊 TOP 10 FEATURES (SHAP):")
            for j, feat in enumerate(result['explanations']['shap']['top_features'][:10], 1):
                icon = "🔴" if feat['shap_value'] > 0 else "🟢"
                print(f"   {j:2d}. {icon} {feat['feature']:20s} | {feat['shap_value']:+8.4f} | {feat['contribution']}")
        
        if 'lime' in result['explanations'] and 'error' not in result['explanations']['lime']:
            print(f"\n📊 TOP 10 FEATURES (LIME):")
            for j, feat in enumerate(result['explanations']['lime']['top_features'][:10], 1):
                icon = "🔴" if feat['weight'] > 0 else "🟢"
                print(f"   {j:2d}. {icon} {feat['feature']:20s} | {feat['weight']:+8.4f} | {feat['contribution']}")
    
    print("\n" + "="*80)
    print("DEMONSTRAÇÃO CONCLUÍDA!")
    print("="*80)


def main():
    print("\n🧠 ANÁLISE DE EXPLICABILIDADE (XAI) - TCC Ideação Suicida\n")
    
    print("Escolha uma opção:")
    print("1. Gerar análise global completa (recomendado)")
    print("2. Demonstração de métodos XAI")
    print("3. Ambos")
    
    choice = input("\nOpção (1/2/3): ").strip()
    
    if choice == '1':
        generate_global_analysis()
    elif choice == '2':
        demonstrate_xai_methods()
    elif choice == '3':
        demonstrate_xai_methods()
        print("\n" + "="*80 + "\n")
        generate_global_analysis()
    else:
        print("Opção inválida. Executando análise global...")
        generate_global_analysis()


if __name__ == "__main__":
    main()


