# Explicabilidade (XAI)

## Visão Geral

O projeto implementa duas técnicas de Explainable AI para interpretar predições dos modelos de Machine Learning.

## Métodos

### SHAP (SHapley Additive exPlanations)

Baseado em teoria dos jogos, atribui contribuição justa para cada feature.

**Características:**
- Valores positivos indicam contribuição para classe "Suicida"
- Valores negativos indicam contribuição para classe "Não Suicida"
- Funciona melhor com modelos lineares

**Uso:**
```python
from src.xai import ModelExplainer

explainer = ModelExplainer()
result = explainer.explain_with_shap(text, max_features=15)
```

### LIME (Local Interpretable Model-agnostic Explanations)

Cria explicações locais treinando modelo simples ao redor da predição.

**Características:**
- Agnóstico ao modelo
- Identifica frases além de palavras isoladas
- Pesos indicam importância

**Uso:**
```python
result = explainer.explain_with_lime(text, num_features=15)
```

## Features Mais Importantes

Top 10 identificadas via SHAP no dataset completo:

1. suicide
2. die
3. kill
4. hopeless
5. life
6. anymore
7. want
8. feel
9. depressed
10. worthless

## Visualizações

### Explicação SHAP

```python
explainer.plot_shap_explanation(text, max_features=15)
```

Gera gráfico de barras horizontais com contribuições de cada feature.

### Comparação SHAP vs LIME

```python
explainer.plot_comparison(text, max_features=10)
```

Mostra explicações lado a lado para validação cruzada.

### Importância Global

```python
from src.preprocess import load_datasets, preprocess
from sklearn.model_selection import train_test_split

df = preprocess(load_datasets())
X_train, X_test, y_train, y_test = train_test_split(...)

explainer.plot_global_importance(X_test, y_test, top_n=30)
```

Mostra features mais importantes em todo o dataset.

## Análise Multi-Modelo

```python
from src.xai_multi_model import MultiModelExplainer

multi = MultiModelExplainer(dataset='reddit_en')
multi.plot_model_comparison(text, top_n=10)
```

Compara explicações SHAP entre Logistic Regression, SVM e Random Forest.

## Análise Completa

```bash
python3 src/generate_xai_analysis.py
```

Gera:
- Feature importance global (CSV e gráfico)
- Exemplos representativos com explicações
- Comparações SHAP vs LIME
- Relatórios detalhados

Saída em `results/xai_analysis/`.

## Interface Web

Ambas as interfaces (`app.py` e `app_multi_model.py`) incluem XAI integrado:

- Explicações SHAP e LIME em tempo real
- Visualizações interativas
- Configuração de número de features
- Comparação entre métodos

## Referências

- Lundberg & Lee (2017). A unified approach to interpreting model predictions. NeurIPS.
- Ribeiro et al. (2016). "Why should I trust you?" Explaining the predictions of any classifier. KDD.

