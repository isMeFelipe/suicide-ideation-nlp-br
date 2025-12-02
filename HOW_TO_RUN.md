# Como Rodar o Projeto do Zero

## Passo 1: Clone e Setup

```bash
cd ~/projects
git clone <url> tcc_suicial_ideation
cd tcc_suicial_ideation
```

## Passo 2: Ambiente Virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

## Passo 3: Instalar Dependências

```bash
pip install -r requirements.txt
```

Tempo estimado: 2-5 minutos

## Passo 4: Verificar Instalação

```bash
python3 -c "import pandas, sklearn, shap, lime, streamlit; print('OK')"
```

Deve exibir: `OK`

## Passo 5: Treinar Modelo

```bash
python3 src/train_model.py
```

Tempo estimado: 1-3 minutos

Saída:
- `models/logistic_model.pkl`
- `models/tfidf_vectorizer.pkl`

## Passo 6: Executar Interface

```bash
streamlit run app.py
```

Acesse: http://localhost:8501

## Alternativas

### Interface Multi-Modelo

```bash
streamlit run app_multi_model.py
```

Requer modelos treinados:
```bash
python3 src/train_all_models.py
```

### Gerar Figuras para TCC

```bash
python3 scripts/generate_tcc_figures.py
```

Saída: `tcc_figures/*.png` (20 figuras, 300 DPI)

### Análise XAI

```bash
python3 src/generate_xai_analysis.py
```

Saída: `results/xai_analysis/`

## Testes

```bash
python3 tests/test_xai.py
```

Verifica que XAI está funcionando.

## Solução de Problemas

### jinja2 version error

```bash
pip install --upgrade jinja2
```

### Modelo não encontrado

```bash
python3 src/train_model.py
```

### LIME não instalado

```bash
pip install lime==0.2.0.1
```

## Resumo dos Comandos

```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Treinar
python3 src/train_model.py

# 3. Usar
streamlit run app.py

# 4. Testar
python3 tests/test_xai.py

# 5. Figuras TCC
python3 scripts/generate_tcc_figures.py
```

## Tempo Total

- Instalação: ~5 minutos
- Treinamento: ~3 minutos
- **Total: ~8 minutos do zero ao funcionando**

## Requisitos

- Python 3.8+
- 4GB RAM
- 500MB espaço em disco
- Conexão de internet (instalação)

