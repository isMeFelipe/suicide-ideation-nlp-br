import streamlit as st
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(__file__))

from src.preprocess import clean_text
from src.config import MODELS_DIR
from src.xai_multi_model import MultiModelExplainer

st.set_page_config(
    page_title="Detecção de Ideação Suicida - Comparação de Modelos",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_multi_explainer(dataset):
    try:
        return MultiModelExplainer(dataset=dataset)
    except Exception as e:
        st.error(f"Erro ao carregar modelos: {e}")
        return None

st.title("Detecção de Ideação Suicida - Comparação de Modelos")
st.markdown("### Sistema Multi-Modelo com Explicabilidade (XAI)")

with st.sidebar:
    st.header("Configurações")
    
    dataset = st.selectbox(
        "Dataset",
        options=['reddit_en', 'twitter_en', 'merged_en'],
        index=0,
        help="Selecione o dataset usado para treinar os modelos"
    )
    
    st.markdown("---")
    
    multi_explainer = load_multi_explainer(dataset)
    
    if multi_explainer:
        available_models = multi_explainer.get_model_names()
        
        st.markdown(f"### Modelos Disponíveis ({len(available_models)})")
        for model in available_models:
            st.success(f"{model.replace('_', ' ').title()}")
        
        st.markdown("---")
        
        selected_model = st.selectbox(
            "Modelo para Explicação Detalhada",
            options=available_models,
            index=0 if 'logistic_regression' in available_models else 0,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        show_shap = st.checkbox("Mostrar Explicação SHAP", value=True)
        show_lime = st.checkbox("Mostrar Explicação LIME", value=False)
        num_features = st.slider("Número de features", 5, 30, 10)
        
        st.markdown("---")
        st.markdown("### Legenda")
        st.markdown("**Vermelho**: Contribui para 'Suicida'")
        st.markdown("**Verde**: Contribui para 'Não Suicida'")
    else:
        st.error("Nenhum modelo disponível")
        st.stop()
    
    st.markdown("---")
    st.warning("Sistema de triagem. Não substitui avaliação profissional.")

st.markdown("### Digite o texto para análise:")
user_input = st.text_area(
    "Texto:",
    height=150,
    placeholder="Digite aqui o texto que deseja analisar...",
    help="O texto será analisado por múltiplos modelos"
)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    analyze_button = st.button("Analisar com Todos os Modelos", use_container_width=True, type="primary")

with col2:
    clear_button = st.button("Limpar", use_container_width=True)

if clear_button:
    st.experimental_rerun()

if analyze_button:
    if user_input.strip() == "":
        st.warning("Por favor, digite algum texto para análise!")
    else:
        with st.spinner("Analisando texto com múltiplos modelos..."):
            
            st.markdown("---")
            st.markdown("## Comparação de Predições entre Modelos")
            
            df_pred = multi_explainer.compare_predictions(user_input)
            
            st.dataframe(
                df_pred,
                use_container_width=True,
                hide_index=True
            )
            
            for idx, row in df_pred.iterrows():
                if 'Suicida' in row['Predição']:
                    st.markdown(f"**{row['Modelo']}**: {row['Predição']} ({row['Confiança']})")
                else:
                    st.markdown(f"**{row['Modelo']}**: {row['Predição']} ({row['Confiança']})")
            
            col_viz1, col_viz2 = st.columns(2)
            
            with col_viz1:
                st.markdown("### Probabilidades por Modelo")
                try:
                    fig = multi_explainer.plot_predictions_comparison(user_input)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Erro ao gerar gráfico: {e}")
            
            with col_viz2:
                st.markdown("### Comparação SHAP entre Modelos")
                try:
                    fig = multi_explainer.plot_model_comparison(user_input, top_n=10)
                    st.pyplot(fig)
                    plt.close()
                except Exception as e:
                    st.error(f"Erro ao gerar comparação SHAP: {e}")
            
            st.markdown("---")
            st.markdown(f"## Análise Detalhada: {selected_model.replace('_', ' ').title()}")
            
            explainer = multi_explainer.explainers[selected_model]
            result = explainer.predict_with_explanation(user_input)
            
            col_detail1, col_detail2 = st.columns(2)
            
            with col_detail1:
                st.markdown("### Predição")
                if result['prediction'] == 1:
                    st.error(f"**{result['prediction_label']}**")
                else:
                    st.success(f"**{result['prediction_label']}**")
                
                st.metric("Confiança", f"{result['probability']:.1%}")
            
            with col_detail2:
                st.markdown("### Probabilidades")
                prob_df = pd.DataFrame({
                    'Classe': list(result['probabilities'].keys()),
                    'Probabilidade': [f"{v:.1%}" for v in result['probabilities'].values()],
                })
                st.dataframe(prob_df, hide_index=True, use_container_width=True)
            
            st.markdown("### Texto Pré-processado")
            with st.expander("Ver texto limpo"):
                st.code(result['cleaned_text'])
            
            if show_shap or show_lime:
                st.markdown("---")
                st.markdown(f"## Explicabilidade (XAI) - {selected_model.replace('_', ' ').title()}")
                
                if show_shap and show_lime:
                    tab1, tab2, tab3 = st.tabs(["SHAP", "LIME", "Comparação"])
                elif show_shap:
                    tab1 = st.container()
                    tabs_dict = {'SHAP': tab1}
                elif show_lime:
                    tab2 = st.container()
                    tabs_dict = {'LIME': tab2}
                
                if show_shap:
                    with (tab1 if show_shap and show_lime else tabs_dict['SHAP']):
                        st.markdown("### Explicação SHAP")
                        st.markdown("""
                        SHAP atribui contribuições justas para cada feature baseado em teoria dos jogos.
                        """)
                        
                        try:
                            shap_data = explainer.explain_with_shap(user_input, max_features=num_features)
                            
                            features_df = pd.DataFrame(shap_data['top_features'][:num_features])
                            st.dataframe(
                                features_df[['feature', 'shap_value', 'contribution']].rename(columns={
                                    'feature': 'Palavra',
                                    'shap_value': 'Valor SHAP',
                                    'contribution': 'Contribuição'
                                }),
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            fig = explainer.plot_shap_explanation(user_input, max_features=num_features)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.error(f"Erro SHAP: {e}")
                
                if show_lime:
                    with (tab2 if show_shap and show_lime else tabs_dict['LIME']):
                        st.markdown("### Explicação LIME")
                        st.markdown("""
                        LIME cria explicações locais treinando modelo simples ao redor da predição.
                        """)
                        
                        try:
                            lime_data = explainer.explain_with_lime(user_input, num_features=num_features)
                            
                            features_df = pd.DataFrame(lime_data['top_features'][:num_features])
                            st.dataframe(
                                features_df[['feature', 'weight', 'contribution']].rename(columns={
                                    'feature': 'Palavra/Frase',
                                    'weight': 'Peso',
                                    'contribution': 'Contribuição'
                                }),
                                hide_index=True,
                                use_container_width=True
                            )
                            
                            fig = explainer.plot_lime_explanation(user_input, num_features=num_features)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.error(f"Erro LIME: {e}")
                
                if show_shap and show_lime:
                    with tab3:
                        st.markdown("### Comparação SHAP vs LIME")
                        
                        try:
                            fig = explainer.plot_comparison(user_input, max_features=min(num_features, 10))
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.error(f"Erro na comparação: {e}")
            
st.markdown("---")
with st.expander("Exemplos de Textos para Teste"):
    col_ex1, col_ex2 = st.columns(2)
    
    with col_ex1:
        st.markdown("**Alto Risco:**")
        st.code("I feel hopeless and don't want to live anymore. Life has no meaning.")
        st.code("i want to die so hard, i can't take it anymore.")
        
    with col_ex2:
        st.markdown("**Sem Risco:**")
        st.code("Just had a great day with friends! Feeling blessed and happy.")
        st.code("Looking forward to my vacation next week. So excited!")

with st.expander("Sobre os Modelos"):
    st.markdown("""
    ### Modelos Disponíveis:
    
    **Logistic Regression:**
    - Melhor desempenho no Reddit (87.62% acurácia)
    - Interpretável e eficiente
    - ROC-AUC: 94.50%
    
    **SVM (Support Vector Machine):**
    - Melhor desempenho no Twitter (90.73% acurácia)
    - Excelente com espaços de alta dimensionalidade
    - ROC-AUC: 96.01%
    
    **Random Forest:**
    - Ensemble de árvores de decisão
    - Robustez contra overfitting
    - Bom baseline geral
    
    ### Métodos de Explicabilidade:
    
    **SHAP:**
    - Baseado em teoria dos jogos (valores de Shapley)
    - Atribuições justas e consistentes
    - Melhor para modelos lineares
    
    **LIME:**
    - Explicações locais agnósticas
    - Identifica frases contextuais
    - Flexível para qualquer modelo
    """)

