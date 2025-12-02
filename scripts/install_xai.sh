#!/bin/bash

echo "=================================="
echo "Instalando dependências de XAI"
echo "=================================="

if [ ! -d "venv" ]; then
    echo "⚠️  Ambiente virtual não encontrado!"
    echo "Criando ambiente virtual..."
    python3 -m venv venv
fi

echo "Ativando ambiente virtual..."
source venv/bin/activate

echo "Instalando SHAP e LIME..."
pip install shap==0.42.1
pip install lime==0.2.0.1

echo ""
echo "✓ Instalação concluída!"
echo ""
echo "Para usar o XAI:"
echo "  1. Interface web: streamlit run app.py"
echo "  2. Análise global: python src/generate_xai_analysis.py"
echo "  3. Demo programático: python src/xai.py"


