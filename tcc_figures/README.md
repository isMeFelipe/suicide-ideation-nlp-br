# 📊 Figuras do TCC - Guia de Referência

## ✅ Todas as Figuras Geradas (20 total)

Este diretório contém todas as figuras necessárias para o TCC, prontas para uso no LaTeX.

---

## 📋 Lista Completa de Figuras

### **Capítulo: Metodologia**

| # | Arquivo | Citação LaTeX | Descrição |
|---|---------|---------------|-----------|
| 1 | `class_distribution.png` | `\ref{fig:class_dist}` | Distribuição de classes nos 3 datasets |
| 2 | `preprocessing_pipeline.png` | `\ref{fig:preprocessing}` | Pipeline de pré-processamento (diagrama) |

---

### **Capítulo: Resultados**

#### **Seção: Reddit**

| # | Arquivo | Citação LaTeX | Descrição |
|---|---------|---------------|-----------|
| 3 | `reddit_en_confusion_matrix.png` | `\ref{fig:reddit_en_cm}` | Matriz de confusão - Reddit EN |
| 4 | `reddit_en_cv_scores.png` | `\ref{fig:reddit_en_cv}` | Validação cruzada - Reddit EN |
| 5 | `reddit_pt_confusion_matrix.png` | `\ref{fig:reddit_pt_cm}` | Matriz de confusão - Reddit PT |

#### **Seção: Twitter**

| # | Arquivo | Citação LaTeX | Descrição |
|---|---------|---------------|-----------|
| 6 | `twitter_en_confusion_matrix.png` | `\ref{fig:twitter_en_cm}` | Matriz de confusão - Twitter EN |
| 7 | `twitter_en_cv_scores.png` | `\ref{fig:twitter_en_cv}` | Validação cruzada - Twitter EN |
| 8 | `twitter_pt_confusion_matrix.png` | `\ref{fig:twitter_pt_cm}` | Matriz de confusão - Twitter PT |

#### **Seção: Dataset Mesclado**

| # | Arquivo | Citação LaTeX | Descrição |
|---|---------|---------------|-----------|
| 9 | `merged_en_confusion_matrix.png` | `\ref{fig:merged_en_cm}` | Matriz de confusão - Merged EN |

#### **Seção: Análise Comparativa**

| # | Arquivo | Citação LaTeX | Descrição |
|---|---------|---------------|-----------|
| 10 | `model_comparison_bar_chart.png` | `\ref{fig:model_comparison}` | Comparação visual de acurácia |
| 11 | `roc_curves_all_models.png` | `\ref{fig:roc_curves}` | Curvas ROC de todos os modelos |
| 12 | `pr_curves_all_models.png` | `\ref{fig:pr_curves}` | Curvas Precision-Recall |
| 13 | `translation_impact_comparison.png` | `\ref{fig:translation_impact}` | Impacto da tradução automática |

#### **Seção: Análise de Explicabilidade (XAI)**

| # | Arquivo | Citação LaTeX | Descrição |
|---|---------|---------------|-----------|
| 14 | `global_importance_plot.png` | `\ref{fig:global_importance}` | Top 30 features globalmente (SHAP) |
| 15 | `shap_high_risk_example.png` | `\ref{fig:shap_high_risk}` | Explicação SHAP - texto alto risco |
| 16 | `comparison_shap_lime.png` | `\ref{fig:comparison_methods}` | Comparação SHAP vs LIME |
| 17 | `shap_low_risk_example.png` | `\ref{fig:shap_low_risk}` | Explicação SHAP - texto sem risco |
| 18 | `multi_model_shap_comparison.png` | `\ref{fig:model_comparison_shap}` | Comparação SHAP entre modelos |

#### **Seção: Interface**

| # | Arquivo | Citação LaTeX | Descrição |
|---|---------|---------------|-----------|
| 19 | `interface_screenshot.png` | `\ref{fig:interface}` | Interface web do sistema |

---

### **Capítulo: Conclusão**

| # | Arquivo | Citação LaTeX | Descrição |
|---|---------|---------------|-----------|
| 20 | `final_metrics_summary.png` | `\ref{fig:final_summary}` | Sumário final de métricas |

---

## 🔧 Como Usar no LaTeX

### Passo 1: Copiar Figuras

```bash
# Opção 1: Copiar todas
cp tcc_figures/*.png /caminho/do/latex/figuras/

# Opção 2: Criar link simbólico
ln -s $(pwd)/tcc_figures /caminho/do/latex/figuras
```

### Passo 2: Usar no LaTeX

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.7\textwidth]{figuras/class_distribution.png}
\caption{Distribuição de classes nos datasets utilizados}
\label{fig:class_dist}
\end{figure}
```

### Passo 3: Referenciar no Texto

```latex
Como mostra a Figura~\ref{fig:class_dist}, a distribuição...
```

---

## 📊 Estatísticas das Figuras

- **Total de figuras:** 20
- **Formato:** PNG (alta resolução, 300 DPI)
- **Tamanho total:** ~3.5 MB
- **Tamanho médio:** ~175 KB por figura
- **Maior figura:** `final_metrics_summary.png` (298 KB)
- **Menor figura:** `twitter_en_confusion_matrix.png` (84 KB)

---

## 🎨 Características das Figuras

### Qualidade
- ✅ **Resolução:** 300 DPI (qualidade de impressão)
- ✅ **Formato:** PNG com fundo branco
- ✅ **Cores:** Paleta profissional e consistente
- ✅ **Fontes:** Legíveis e padronizadas

### Conteúdo
- ✅ **Títulos:** Todos descritivos e informativos
- ✅ **Legendas:** Eixos claramente rotulados
- ✅ **Valores:** Números exibidos quando relevante
- ✅ **Grid:** Auxilia leitura de valores

---

## 🔄 Regenerar Figuras

Para regenerar todas as figuras (útil se dados mudarem):

```bash
python3 generate_tcc_figures.py
```

### Regenerar Figuras Específicas

Para regenerar apenas figuras XAI (se treinar novo modelo):

```bash
python3 -c "
from src.xai import ModelExplainer
from src.preprocess import load_datasets, preprocess
from sklearn.model_selection import train_test_split
from src.config import RANDOM_STATE, TEST_SIZE

explainer = ModelExplainer()
df = preprocess(load_datasets())
X_train, X_test, y_train, y_test = train_test_split(
    df['clean_text'], df['label'], 
    test_size=TEST_SIZE, stratify=df['label'], random_state=RANDOM_STATE
)

explainer.plot_global_importance(X_test, y_test, 
    save_path='tcc_figures/global_importance_plot.png', top_n=30)

high_risk = 'I feel hopeless and dont want to live anymore'
explainer.plot_shap_explanation(high_risk, 
    save_path='tcc_figures/shap_high_risk_example.png', max_features=15)

print('✓ Figuras XAI regeneradas!')
"
```

---

## 📝 Checklist para o TCC

Antes de entregar/apresentar, verifique:

- [ ] Todas as 20 figuras estão no diretório `figuras/` do LaTeX
- [ ] Todas as figuras compilam sem erros no LaTeX
- [ ] Todos os `\ref{}` apontam para labels corretos
- [ ] Legendas (captions) são descritivas
- [ ] Figuras estão mencionadas/discutidas no texto
- [ ] Qualidade visual adequada para impressão
- [ ] Figuras estão na ordem correta do documento

---

## 🎓 Dicas para Apresentação

### Figuras Essenciais para Slides

Se tiver que escolher 5-7 figuras para apresentação:

1. ✅ `class_distribution.png` - Contextualiza dados
2. ✅ `model_comparison_bar_chart.png` - Mostra resultados principais
3. ✅ `roc_curves_all_models.png` - Performance visual
4. ✅ `global_importance_plot.png` - XAI global
5. ✅ `shap_high_risk_example.png` - XAI local (exemplo)
6. ✅ `translation_impact_comparison.png` - Contribuição única (dataset PT)
7. ✅ `interface_screenshot.png` - Aplicação prática

### Ordem Sugerida de Apresentação

1. Introdução → `class_distribution.png`
2. Metodologia → `preprocessing_pipeline.png`
3. Resultados → `model_comparison_bar_chart.png` + `roc_curves_all_models.png`
4. Diferencial (PT) → `translation_impact_comparison.png`
5. XAI → `global_importance_plot.png` + `shap_high_risk_example.png`
6. Demo → `interface_screenshot.png`
7. Conclusão → `final_metrics_summary.png`

---

## 🔍 Troubleshooting

### Figura não aparece no LaTeX

```latex
% Verifique o caminho
\includegraphics[width=0.7\textwidth]{figuras/class_distribution.png}

% Ou tente caminho absoluto temporariamente
\includegraphics[width=0.7\textwidth]{/caminho/completo/tcc_figures/class_distribution.png}
```

### Figura muito grande/pequena

```latex
% Ajuste o width
\includegraphics[width=0.5\textwidth]{figuras/...}   % 50%
\includegraphics[width=0.9\textwidth]{figuras/...}   % 90%
\includegraphics[width=\textwidth]{figuras/...}      % 100%
```

### Qualidade ruim ao compilar

```latex
% Use pdflatex (não latex)
pdflatex seu_documento.tex

% Ou especifique DPI no graphicx
\usepackage[pdftex]{graphicx}
```

---

## 📚 Referências

- **Script gerador:** `generate_tcc_figures.py`
- **Documentação XAI:** `XAI_README.md`
- **Instruções:** `INSTRUCOES_XAI.md`

---

**Gerado automaticamente em:** 02/12/2024  
**Última atualização:** 02/12/2024  
**Status:** ✅ Pronto para uso no TCC

