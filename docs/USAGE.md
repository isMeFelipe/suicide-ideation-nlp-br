# Uso

## Interface Web

### Modelo Único

```bash
streamlit run app.py
```

Interface para análise de textos com Regressão Logística e explicações XAI.

### Múltiplos Modelos

```bash
streamlit run app_multi_model.py
```

Interface para comparação entre Logistic Regression, SVM e Random Forest.

## Scripts

### Treinar Modelo

```bash
python3 src/train_model.py
```

Treina Regressão Logística no dataset mesclado.

### Treinar Todos os Modelos

```bash
python3 src/train_all_models.py
```

Treina LR, SVM e RF em todos os datasets (Reddit, Twitter, Merged) em inglês e português.

### Comparar Modelos

```bash
python3 src/compare_models.py
```

Gera comparação completa entre modelos salvando em `results/model_comparison_all.json`.

### Gerar Figuras para TCC

```bash
python3 scripts/generate_tcc_figures.py
```

Gera 20 figuras de alta qualidade (300 DPI) em `tcc_figures/`.

### Análise XAI

```bash
python3 src/generate_xai_analysis.py
```

Gera análise completa de explicabilidade com SHAP e LIME.

## Testes

```bash
python3 tests/test_xai.py
python3 tests/test_multi_model.py
```

Testa funcionalidades de XAI e comparação multi-modelo.

## Estrutura de Saída

- `models/` - Modelos treinados (.pkl)
- `results/` - Métricas e análises (.json, .png)
- `tcc_figures/` - Figuras para TCC (300 DPI)

