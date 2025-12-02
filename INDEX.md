# Índice do Projeto

## Documentação Principal

1. **README.md** - Visão geral do projeto, resultados e início rápido
2. **QUICKSTART.md** - Do zero em 3 passos (8 minutos)
3. **HOW_TO_RUN.md** - Guia detalhado passo a passo

## Documentação Técnica

- **docs/INSTALLATION.md** - Requisitos e instalação
- **docs/USAGE.md** - Comandos e uso dos scripts
- **docs/ARCHITECTURE.md** - Estrutura e componentes do sistema
- **docs/XAI.md** - Explicabilidade (SHAP e LIME)

## Código Fonte

### Principal (src/)
- `config.py` - Configurações e constantes
- `preprocess.py` - Pré-processamento de texto
- `train_model.py` - Treina modelo único (LR no merged)
- `train_all_models.py` - Treina todos os modelos
- `compare_models.py` - Compara modelos
- `evaluate.py` - Funções de avaliação
- `xai.py` - XAI para modelo único
- `xai_multi_model.py` - XAI para múltiplos modelos
- `generate_xai_analysis.py` - Análise XAI completa

### Scripts Utilitários (scripts/)
- `generate_tcc_figures.py` - Gera 20 figuras para TCC
- `run_analysis.py` - Análise exploratória
- `install_dependencies.sh` - Instala dependências base
- `install_xai.sh` - Instala SHAP e LIME

### Testes (tests/)
- `test_xai.py` - Testa funcionalidades XAI
- `test_multi_model.py` - Testa comparação multi-modelo

## Interfaces Web

- `app.py` - Interface modelo único (Logistic Regression + XAI)
- `app_multi_model.py` - Interface comparativa (LR vs SVM vs RF + XAI)

## Dados

### Datasets (data/)
- Reddit: 12.601 amostras
- Twitter: 1.779 amostras
- Merged: 14.380 amostras
- Versões em inglês e português

### Modelos (models/)
- Modelos treinados por dataset e idioma
- Formato: .pkl (pickle)

### Resultados (results/)
- `model_comparison_all.json` - Comparação completa
- Métricas por modelo e dataset
- Curvas ROC e PR

### Figuras (tcc_figures/)
- 20 figuras de alta qualidade (300 DPI)
- Nomes exatos das citações LaTeX
- Prontas para uso no TCC

## Fluxo de Trabalho

### Para Desenvolvimento

1. Ler `docs/ARCHITECTURE.md` para entender estrutura
2. Modificar código em `src/`
3. Testar com `tests/test_*.py`
4. Treinar com `src/train_model.py`

### Para Uso

1. Ler `QUICKSTART.md`
2. Instalar dependências
3. Treinar modelo
4. Executar `streamlit run app.py`

### Para TCC

1. Ler `README.md` para contexto
2. Gerar figuras: `python3 scripts/generate_tcc_figures.py`
3. Usar figuras de `tcc_figures/`
4. Consultar métricas em `results/model_comparison_all.json`

## Comandos Rápidos

```bash
# Instalar
pip install -r requirements.txt

# Treinar
python3 src/train_model.py

# Interface web
streamlit run app.py

# Gerar figuras TCC
python3 scripts/generate_tcc_figures.py

# Testar
python3 tests/test_xai.py

# Análise XAI
python3 src/generate_xai_analysis.py
```

## Requisitos

- Python 3.8+
- 4GB RAM
- 500MB espaço
- Dependências em `requirements.txt`

## Navegação Rápida

**Quero entender o projeto:**
→ Leia `README.md`

**Quero rodar do zero:**
→ Leia `HOW_TO_RUN.md`

**Quero entender a estrutura:**
→ Leia `docs/ARCHITECTURE.md`

**Quero entender XAI:**
→ Leia `docs/XAI.md`

**Quero usar os scripts:**
→ Leia `docs/USAGE.md`

**Quero as figuras do TCC:**
→ Execute `python3 scripts/generate_tcc_figures.py`
→ Veja `tcc_figures/README.md`

