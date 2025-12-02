import streamlit as st
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.preprocess import clean_text
from src.config import MODEL_FILES
from src.xai import ModelExplainer

st.set_page_config(
    page_title="Detecção de Ideação Suicida - TCC",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_explainer():
    try:
        return ModelExplainer()
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        st.stop()

explainer = load_explainer()

st.title("Detecção de Ideação Suicida em Textos Curtos")
st.markdown("### Sistema de Classificação com Explicabilidade (XAI)")

with st.sidebar:
    st.header("Configurações")
    
    show_shap = st.checkbox("Mostrar Explicação SHAP", value=True)
    show_lime = st.checkbox("Mostrar Explicação LIME", value=True)
    num_features = st.slider("Número de features a exibir", 5, 30, 10)
    
    st.markdown("---")
    st.markdown("### Sobre o Modelo")
    st.info(f"**Modelo:** {explainer.model_name}")
    st.info(f"**Features:** {len(explainer.feature_names)}")
    
    st.markdown("---")
    st.markdown("### Legenda")
    st.markdown("**Vermelho**: Contribui para classificação 'Suicida'")
    st.markdown("**Verde**: Contribui para classificação 'Não Suicida'")
    
    st.markdown("---")
    st.warning("Este sistema é apenas uma ferramenta de triagem. Não substitui avaliação profissional.")

st.markdown("### Digite o texto para análise:")
user_input = st.text_area(
    "Texto:",
    height=150,
    placeholder="Digite aqui o texto que deseja analisar...",
    help="O texto será pré-processado e analisado pelo modelo"
)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    analyze_button = st.button("Analisar Texto", use_container_width=True, type="primary")

with col2:
    clear_button = st.button("Limpar", use_container_width=True)

if clear_button:
    st.experimental_rerun()

if analyze_button:
    if user_input.strip() == "":
        st.warning("Por favor, digite algum texto para análise!")
    else:
        with st.spinner("Analisando texto..."):
            
            result = explainer.predict_with_explanation(user_input)
            
            st.markdown("---")
            st.markdown("## Resultados da Análise")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                st.markdown("### Classificação")
                if result['prediction'] == 1:
                    st.error(f"**{result['prediction_label']}**")
            else:
                    st.success(f"**{result['prediction_label']}**")
                
                st.metric("Confiança", f"{result['probability']:.1%}")
            
            with col_result2:
                st.markdown("### Probabilidades")
                prob_df = pd.DataFrame({
                    'Classe': list(result['probabilities'].keys()),
                    'Probabilidade': [f"{v:.1%}" for v in result['probabilities'].values()],
                    'Valor': list(result['probabilities'].values())
                })
                st.dataframe(prob_df[['Classe', 'Probabilidade']], hide_index=True, use_container_width=True)
            
            st.markdown("### Texto Pré-processado")
            with st.expander("Ver texto limpo"):
                st.code(result['cleaned_text'])
            
            st.markdown("---")
            st.markdown("## Explicabilidade (XAI)")
            
            methods = []
            if show_shap:
                methods.append('shap')
            if show_lime:
                methods.append('lime')
            
            if not methods:
                st.info("Selecione pelo menos um método de explicabilidade na barra lateral.")
            else:
                explanations = {}
                
                if 'shap' in methods:
                    with st.spinner("Gerando explicação SHAP..."):
                        try:
                            explanations['shap'] = explainer.explain_with_shap(user_input, max_features=num_features)
                        except Exception as e:
                            st.error(f"Erro ao gerar explicação SHAP: {e}")
                
                if 'lime' in methods:
                    with st.spinner("Gerando explicação LIME..."):
                        try:
                            explanations['lime'] = explainer.explain_with_lime(user_input, num_features=num_features)
                        except Exception as e:
                            st.error(f"Erro ao gerar explicação LIME: {e}")
                
                if len(methods) == 2 and all(k in explanations for k in ['shap', 'lime']):
                    tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "Comparação"])
                elif 'shap' in explanations:
                    tab1 = st.container()
                    explanations_tabs = {'SHAP': tab1}
                elif 'lime' in explanations:
                    tab2 = st.container()
                    explanations_tabs = {'LIME': tab2}
                else:
                    st.error("Nenhuma explicação disponível.")
                    explanations_tabs = {}
                
                if 'shap' in explanations and len(methods) == 2:
                    with tab1:
                        st.markdown("### Explicação SHAP (SHapley Additive exPlanations)")
                        st.markdown("""
                        SHAP é baseado em teoria dos jogos e atribui a cada feature uma contribuição justa para a predição.
                        Valores positivos aumentam a probabilidade de 'Suicida', valores negativos de 'Não Suicida'.
                        """)
                        
                        shap_data = explanations['shap']
                        
                        st.markdown(f"**Valor Base (Expected Value):** {shap_data['expected_value']:.4f}")
                        
                        features_df = pd.DataFrame(shap_data['top_features'])
                        features_df['shap_abs'] = features_df['shap_value'].abs()
                        features_df = features_df.sort_values('shap_abs', ascending=False)
                        
                        st.dataframe(
                            features_df[['feature', 'shap_value', 'contribution']].rename(columns={
                                'feature': 'Palavra',
                                'shap_value': 'Valor SHAP',
                                'contribution': 'Contribuição'
                            }),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        try:
                            fig = explainer.plot_shap_explanation(user_input, max_features=num_features)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.warning(f"Não foi possível gerar gráfico SHAP: {e}")
                
                if 'lime' in explanations and len(methods) == 2:
                    with tab2:
                        st.markdown("### Explicação LIME (Local Interpretable Model-agnostic Explanations)")
                        st.markdown("""
                        LIME cria explicações locais ao treinar um modelo simples ao redor da predição específica.
                        Pesos positivos indicam contribuição para 'Suicida', negativos para 'Não Suicida'.
                        """)
                        
                        lime_data = explanations['lime']
                        
                        features_df = pd.DataFrame(lime_data['top_features'])
                        features_df['weight_abs'] = features_df['weight'].abs()
                        features_df = features_df.sort_values('weight_abs', ascending=False)
                        
                        st.dataframe(
                            features_df[['feature', 'weight', 'contribution']].rename(columns={
                                'feature': 'Palavra/Frase',
                                'weight': 'Peso',
                                'contribution': 'Contribuição'
                            }),
                            hide_index=True,
                            use_container_width=True
                        )
                        
                        try:
                            fig = explainer.plot_lime_explanation(user_input, num_features=num_features)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.warning(f"Não foi possível gerar gráfico LIME: {e}")
                
                if len(methods) == 2 and all(k in explanations for k in ['shap', 'lime']):
                    with tab3:
                        st.markdown("### Comparação SHAP vs LIME")
                        st.markdown("""
                        Comparação lado a lado das duas metodologias de explicabilidade.
                        Ambas podem destacar features diferentes devido às suas abordagens distintas.
                        """)
                        
                        try:
                            fig = explainer.plot_comparison(user_input, max_features=min(num_features, 10))
                            st.pyplot(fig)
                            plt.close()
        except Exception as e:
                            st.warning(f"Não foi possível gerar gráfico de comparação: {e}")
                        
                        col_comp1, col_comp2 = st.columns(2)
                        
                        with col_comp1:
                            st.markdown("#### Top 5 SHAP")
                            shap_top5 = explanations['shap']['top_features'][:5]
                            for i, feat in enumerate(shap_top5, 1):
                                direction = "+" if feat['shap_value'] > 0 else "-"
                                st.write(f"{i}. [{direction}] **{feat['feature']}**: {feat['shap_value']:+.4f}")
                        
                        with col_comp2:
                            st.markdown("#### Top 5 LIME")
                            lime_top5 = explanations['lime']['top_features'][:5]
                            for i, feat in enumerate(lime_top5, 1):
                                direction = "+" if feat['weight'] > 0 else "-"
                                st.write(f"{i}. [{direction}] **{feat['feature']}**: {feat['weight']:+.4f}")

st.markdown("---")
with st.expander("Exemplos de Textos para Teste"):
    st.markdown("""
    **Exemplo 1 (Risco Alto):**
    > I feel hopeless and don't want to live anymore. Life has no meaning.
    
    **Exemplo 2 (Sem Risco):**
    > Just had a great day with friends! Feeling blessed and happy.
    
    **Exemplo 3 (Risco Moderado):**
    > I'm feeling really depressed lately but trying to get help.
    
    **Exemplo 4 (Risco Alto):**
    > Can't take it anymore. Nobody would miss me if I was gone.
    
    **Exemplo 5 (Sem Risco):**
    > Looking forward to my vacation next week. So excited!
    """)
