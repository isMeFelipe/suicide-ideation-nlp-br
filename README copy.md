# Detecção de Ideação Suicida em Redes Sociais

Sistema de classificação automática de textos para identificação de ideação suicida utilizando Machine Learning e técnicas de Explainable AI (XAI).

## Resumo

Este projeto implementa e compara três algoritmos clássicos de ML (Logistic Regression, SVM, Random Forest) para detecção de ideação suicida em textos de Reddit e Twitter. Inclui:

- Dataset bilíngue (inglês e português) com 28.760 amostras
- Tradução automática via Ollama 3.1
- Explicabilidade via SHAP e LIME
- Interface web interativa

## Resultados

| Dataset | Melhor Modelo | Acurácia | ROC-AUC |
|---------|---------------|----------|---------|
| Reddit | Logistic Regression | 87.62% | 94.50% |
| Twitter | SVM | 90.73% | 96.01% |
| Merged | Logistic Regression | 88.18% | 94.45% |

## Início Rápido

### 1. Instalação

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Treinar Modelo

```bash
python3 src/train_model.py
```

### 3. Interface Web

```bash
streamlit run app.py
```

Acesse http://localhost:8501

## Estrutura

```
.
├── src/                Código fonte
├── scripts/            Scripts utilitários
├── tests/              Scripts de teste
├── data/               Datasets
├── models/             Modelos treinados
├── results/            Resultados e métricas
├── tcc_figures/        Figuras para TCC (300 DPI)
├── docs/               Documentação
├── app.py              Interface web modelo único
└── app_multi_model.py  Interface web multi-modelo
```

## Uso

### Interface Web

**Modelo Único:**
```bash
streamlit run app.py
```

**Comparação de Modelos:**
```bash
streamlit run app_multi_model.py
```

### Treinar Modelos

**Modelo único (LR no merged):**
```bash
python3 src/train_model.py
```

**Todos os modelos:**
```bash
python3 src/train_all_models.py
```

### Gerar Figuras

```bash
python3 scripts/generate_tcc_figures.py
```

Gera 20 figuras em `tcc_figures/` prontas para uso no TCC.

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

## Datasets

### Tamanhos

- Reddit: 12.601 amostras (6.560 positivas, 6.041 negativas)
- Twitter: 1.779 amostras (658 positivas, 1.121 negativas)
- Merged: 14.380 amostras (7.218 positivas, 7.162 negativas)

### Idiomas

- Inglês (original)
- Português (tradução automática via Ollama 3.1)

## Modelos

### Algoritmos

1. **Logistic Regression**
   - Melhor no Reddit e Merged
   - Alta interpretabilidade
   - Funciona perfeitamente com SHAP

2. **SVM (kernel RBF)**
   - Melhor no Twitter
   - Excelente com alta dimensionalidade
   - ROC-AUC de 96.01%

3. **Random Forest**
   - Baseline robusto
   - Ensemble de árvores
   - Menor overfitting

### Hiperparâmetros

- TF-IDF: max_features=5000, ngram_range=(1,2)
- LR: C=1.0, max_iter=1000
- SVM: kernel='rbf', C=1.0
- RF: n_estimators=100

## Explicabilidade (XAI)

### Métodos

**SHAP:** Baseado em teoria dos jogos, atribui contribuição justa para cada feature.

**LIME:** Cria explicações locais treinando modelo simples ao redor da predição.

### Features Mais Importantes

Top 10 identificadas via SHAP:
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

## Métricas

- Acurácia
- Precisão, Recall, F1-Score (geral e por classe)
- ROC-AUC
- PR-AUC
- Matriz de Confusão
- Validação Cruzada (5-fold)

## Contribuições

1. Primeiro dataset público em português para ideação suicida
2. Validação de tradução automática para criação de recursos linguísticos
3. Comparação sistemática de algoritmos clássicos de ML
4. Implementação completa de XAI (SHAP e LIME)
5. Sistema demonstrativo com interface web

## Documentação

- [Instalação](docs/INSTALLATION.md)
- [Uso](docs/USAGE.md)
- [Arquitetura](docs/ARCHITECTURE.md)

## Requisitos do Sistema

- Python 3.8+
- 4GB RAM mínimo
- ~500MB espaço em disco

## Licença

[Especificar licença]

## Citação

```bibtex
@mastersthesis{seu_tcc,
  title={Detecção Automática de Ideação Suicida em Redes Sociais},
  author={Seu Nome},
  year={2024},
  school={Sua Universidade}
}
```

## Contato

[Seus contatos]

## Aviso Importante

Este sistema é uma ferramenta de triagem, não diagnóstico. Não substitui avaliação profissional qualificada em saúde mental.

