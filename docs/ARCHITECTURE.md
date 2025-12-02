# Arquitetura do Projeto

## Estrutura de Diretórios

```
tcc_suicial_ideation/
├── src/                    Código fonte principal
│   ├── config.py          Configurações e constantes
│   ├── preprocess.py      Pré-processamento de texto
│   ├── train_model.py     Treinamento modelo único
│   ├── train_all_models.py Treinamento todos os modelos
│   ├── compare_models.py  Comparação de modelos
│   ├── evaluate.py        Funções de avaliação
│   ├── xai.py            XAI para modelo único
│   ├── xai_multi_model.py XAI para múltiplos modelos
│   └── generate_xai_analysis.py Análise XAI completa
├── scripts/               Scripts utilitários
│   ├── generate_tcc_figures.py Gerar figuras TCC
│   ├── run_analysis.py    Análise exploratória
│   ├── install_dependencies.sh
│   └── install_xai.sh
├── tests/                 Scripts de teste
│   ├── test_xai.py
│   └── test_multi_model.py
├── data/                  Datasets
├── models/                Modelos treinados
├── results/               Resultados e métricas
├── tcc_figures/           Figuras para TCC (300 DPI)
├── docs/                  Documentação
├── app.py                 Interface web modelo único
└── app_multi_model.py     Interface web multi-modelo
```

## Componentes Principais

### Pré-processamento

**Arquivo:** `src/preprocess.py`

- Carregamento de datasets
- Limpeza de texto (URLs, menções, caracteres especiais)
- Normalização e tokenização
- Vetorização TF-IDF

### Treinamento

**Arquivos:** 
- `src/train_model.py` - Modelo único (LR no merged)
- `src/train_all_models.py` - Todos os modelos em todos os datasets

**Modelos:**
- Logistic Regression
- SVM (kernel RBF)
- Random Forest

**Validação:**
- Split 70/15/15 (train/val/test)
- 5-fold cross-validation
- Estratificação por classe

### Avaliação

**Arquivo:** `src/evaluate.py`

**Métricas:**
- Acurácia, Precisão, Recall, F1-Score
- ROC-AUC, PR-AUC
- Métricas por classe
- Matriz de confusão

### Explicabilidade (XAI)

**Arquivos:**
- `src/xai.py` - Classe ModelExplainer
- `src/xai_multi_model.py` - Classe MultiModelExplainer

**Métodos:**
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)

**Funcionalidades:**
- Explicações locais (por texto)
- Importância global de features
- Visualizações comparativas
- Geração de relatórios

### Interfaces

**Arquivos:**
- `app.py` - Interface modelo único
- `app_multi_model.py` - Interface comparativa

**Tecnologia:** Streamlit

## Fluxo de Dados

```
Dataset Bruto (CSV)
    ↓
Pré-processamento (preprocess.py)
    ↓
Vetorização TF-IDF
    ↓
Treinamento (train_*.py)
    ↓
Modelos Salvos (.pkl)
    ↓
Avaliação (evaluate.py)
    ↓
Resultados (JSON, PNG)
    ↓
XAI (xai.py)
    ↓
Explicações e Visualizações
```

## Configurações

**Arquivo:** `src/config.py`

- Parâmetros de treino (RANDOM_STATE, TEST_SIZE, etc)
- Configurações TF-IDF (MAX_FEATURES, NGRAM_RANGE, etc)
- Hiperparâmetros dos modelos
- Caminhos de arquivos

## Formato de Dados

### Input (CSV)

```
text,label
"I feel hopeless...",1
"Great day!",0
```

### Output (JSON)

```json
{
  "dataset": "reddit",
  "language": "en",
  "test_metrics": {
    "logistic_regression": {
      "accuracy": 0.8762,
      "f1_score": 0.8762,
      "roc_auc": 0.9450
    }
  }
}
```

## Performance

### Datasets

- Reddit: 12.601 amostras
- Twitter: 1.779 amostras
- Merged: 14.380 amostras

### Melhores Resultados

- Reddit: Logistic Regression (87.62% acurácia)
- Twitter: SVM (90.73% acurácia)
- Merged: Logistic Regression (88.18% acurácia)

