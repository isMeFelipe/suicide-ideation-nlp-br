# Instalação

## Requisitos

- Python 3.8+
- pip

## Passos

### 1. Clonar repositório

```bash
git clone <url>
cd tcc_suicial_ideation
```

### 2. Criar ambiente virtual

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

## Dependências Principais

- pandas 2.1.1
- scikit-learn 1.3.2
- shap 0.42.1
- lime 0.2.0.1
- streamlit 1.26.0

## Verificação

```bash
python3 -c "import pandas, sklearn, shap, lime, streamlit; print('OK')"
```

