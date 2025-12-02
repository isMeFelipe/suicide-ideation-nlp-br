import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

print("="*80)
print("TESTE RÁPIDO DE XAI")
print("="*80)

print("\n1. Verificando dependências...")
try:
    import shap
    print("   ✓ SHAP instalado:", shap.__version__)
except ImportError:
    print("   ✗ SHAP NÃO instalado! Execute: pip install shap==0.42.1")
    sys.exit(1)

try:
    import lime
    print("   ✓ LIME instalado")
except ImportError:
    print("   ✗ LIME NÃO instalado! Execute: pip install lime==0.2.0.1")
    sys.exit(1)

print("\n2. Carregando ModelExplainer...")
try:
    from src.xai import ModelExplainer
    explainer = ModelExplainer()
    print("   ✓ ModelExplainer carregado com sucesso")
    print(f"   ✓ Modelo: {type(explainer.model).__name__}")
    print(f"   ✓ Features: {len(explainer.feature_names)}")
except Exception as e:
    print(f"   ✗ Erro ao carregar: {e}")
    sys.exit(1)

print("\n3. Testando predição simples...")
test_text = "I feel hopeless and don't want to live anymore"
try:
    result = explainer.predict_with_explanation(test_text)
    print(f"   ✓ Texto: {test_text}")
    print(f"   ✓ Predição: {result['prediction_label']}")
    print(f"   ✓ Confiança: {result['probability']:.2%}")
except Exception as e:
    print(f"   ✗ Erro na predição: {e}")
    sys.exit(1)

print("\n4. Testando explicação SHAP...")
try:
    shap_result = explainer.explain_with_shap(test_text, max_features=5)
    print(f"   ✓ SHAP funcionando!")
    print(f"   ✓ Expected value: {shap_result['expected_value']:.4f}")
    print("   ✓ Top 3 features:")
    for i, feat in enumerate(shap_result['top_features'][:3], 1):
        print(f"      {i}. {feat['feature']:15s} | {feat['shap_value']:+7.4f}")
except Exception as e:
    print(f"   ✗ Erro no SHAP: {e}")
    import traceback
    traceback.print_exc()

print("\n5. Testando explicação LIME...")
try:
    lime_result = explainer.explain_with_lime(test_text, num_features=5)
    print(f"   ✓ LIME funcionando!")
    print("   ✓ Top 3 features:")
    for i, feat in enumerate(lime_result['top_features'][:3], 1):
        print(f"      {i}. {feat['feature']:15s} | {feat['weight']:+7.4f}")
except Exception as e:
    print(f"   ✗ Erro no LIME: {e}")
    import traceback
    traceback.print_exc()

print("\n6. Testando visualização SHAP...")
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig = explainer.plot_shap_explanation(test_text, max_features=10)
    plt.savefig('test_shap_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Gráfico SHAP gerado: test_shap_plot.png")
except Exception as e:
    print(f"   ✗ Erro ao gerar gráfico SHAP: {e}")

print("\n7. Testando visualização LIME...")
try:
    fig = explainer.plot_lime_explanation(test_text, num_features=10)
    plt.savefig('test_lime_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Gráfico LIME gerado: test_lime_plot.png")
except Exception as e:
    print(f"   ✗ Erro ao gerar gráfico LIME: {e}")

print("\n8. Testando comparação SHAP vs LIME...")
try:
    fig = explainer.plot_comparison(test_text, max_features=8)
    plt.savefig('test_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Gráfico de comparação gerado: test_comparison.png")
except Exception as e:
    print(f"   ✗ Erro ao gerar comparação: {e}")

print("\n" + "="*80)
print("TESTE CONCLUÍDO COM SUCESSO! ✓")
print("="*80)
print("\nArquivos gerados:")
print("  - test_shap_plot.png")
print("  - test_lime_plot.png")
print("  - test_comparison.png")
print("\nPróximos passos:")
print("  1. Testar interface web: streamlit run app.py")
print("  2. Gerar análise completa: python src/generate_xai_analysis.py")
print("="*80)


