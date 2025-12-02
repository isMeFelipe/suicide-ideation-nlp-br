#!/bin/bash

echo "Instalando dependências do projeto..."

echo "1. Dependências base..."
pip install -r requirements_base.txt

echo "2. APIs (OpenAI, Anthropic)..."
pip install -r requirements_apis.txt

echo "3. PyTorch (versão CPU)..."
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

echo "4. ML (SHAP, Transformers)..."
pip install -r requirements_ml.txt

echo "5. Interface (Streamlit)..."
pip install -r requirements_ui.txt

echo "✅ Todas as dependências instaladas!"

