
"""
App Unificado - Análise ITBI Recife
Integra análise exploratória, clustering, modelo de classificação e explicabilidade.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from PIL import Image
import streamlit.components.v1 as components

# Importações dos scripts locais
from data_processing import load_and_preprocess_data
from clustering_analysis import get_clustering_data_optimized, create_cluster_visualizations
from data.geo_clustering import build_regions_for_recife

st.set_page_config(
    page_title="ITBI Recife - Análise Completa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CACHE E DADOS ====================

@st.cache_data(show_spinner=False)
def get_data():
    """Carrega dados gerais do ITBI."""
    # A função já resolve o diretório internamente
    return load_and_preprocess_data()

@st.cache_data(show_spinner=False)
def get_clustering_data():
    """Carrega dados de clustering de perfis (K-means)."""
    return get_clustering_data_optimized()

@st.cache_resource(show_spinner="Carregando modelo de classificação...")
def load_model():
    """Carrega o modelo de classificação treinado."""
    try:
        # O arquivo está na raiz do projeto
        model = joblib.load('property_classifier_model_optimized.joblib')
        return model
    except FileNotFoundError:
        st.error("Arquivo do modelo 'property_classifier_model_optimized.joblib' não encontrado. Execute o script de treinamento do modelo primeiro.")
        return None

# ==================== NAVEGAÇÃO PRINCIPAL ====================

st.title("📊 Análise Completa do Mercado Imobiliário de Recife")
st.caption("ITBI 2015-2023 • Dados Residenciais (Apartamentos e Casas)")

# Adicionamos a nova tab de ML
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 EDA Exploratória",
    "🎯 Clustering de Perfis",
    "🗺️ Dashboard Regional",
    "🔥 Análise Integrada",
    "🤖 Predição & Explicabilidade"
])

# ==================== TAB 1: EDA EXPLORATÓRIA ====================
with tab1:
    st.header("Análise Exploratória de Dados")
    df_eda = get_data()
    # ... (O restante do código da tab1 permanece o mesmo)
    st.plotly_chart(px.bar(df_eda.head(10), x='bairro', y='valor_m2'), use_container_width=True)


# ==================== TAB 2: CLUSTERING DE PERFIS ====================
with tab2:
    st.header("🎯 Clustering de Perfis de Mercado")
    df_clustered, silhouette_score, features = get_clustering_data()
    # ... (O restante do código da tab2 permanece o mesmo)
    st.metric("Silhouette Score", f"{silhouette_score:.3f}")


# ==================== TAB 3: DASHBOARD REGIONAL ====================
with tab3:
    st.header("🗺️ Dashboard Regional (IBGE)")
    # ... (O restante do código da tab3 permanece o mesmo)
    st.info("Análise por regiões geográficas com agrupamento de subdistritos.")


# ==================== TAB 4: ANÁLISE INTEGRADA ====================
with tab4:
    st.header("🔥 Análise Integrada: Perfis × Regiões")
    # ... (O restante do código da tab4 permanece o mesmo)
    st.info("Cruzamento dos clusters de mercado com regiões geográficas.")


# ==================== TAB 5: PREDIÇÃO & EXPLICABILIDADE ====================
with tab5:
    st.header("🤖 Predição de Categoria de Valor & Explicabilidade (XAI)")
    st.markdown("Entendendo e utilizando o modelo de Machine Learning para prever a categoria de valor de um imóvel.")

    model = load_model()

    if model:
        st.subheader("🧠 Explicando as Decisões do Modelo com SHAP")
        st.markdown("""
        Os gráficos a seguir foram gerados com a biblioteca SHAP para nos ajudar a entender o comportamento do modelo de classificação. Eles mostram quais características (features) são mais importantes para as decisões do modelo.
        """)

        # Exibir gráficos SHAP
        try:
            st.image(Image.open('shap_summary_bar.png'), caption='Importância Global das Features (SHAP)', use_column_width=True)

            with st.expander("Ver análise detalhada por classe (Beeswarm plots)"):
                st.image(Image.open('shap_summary_beeswarm_Alto Valor.png'), caption='Impacto das Features na Classe: Alto Valor', use_column_width=True)
                st.image(Image.open('shap_summary_beeswarm_Médio.png'), caption='Impacto das Features na Classe: Médio', use_column_width=True)
                st.image(Image.open('shap_summary_beeswarm_Econômico.png'), caption='Impacto das Features na Classe: Econômico', use_column_width=True)

            st.subheader("🔬 Análise de uma Predição Individual (Force Plot)")
            st.markdown("O gráfico abaixo é interativo e mostra como cada feature contribuiu para uma predição específica.")
            
            with open('shap_force_plot_local.html', 'r', encoding='utf-8') as f:
                html_string = f.read()
            components.html(html_string, height=200, scrolling=True)

        except FileNotFoundError:
            st.warning("Gráficos de SHAP não encontrados. Execute o script `shap_explainer.py` para gerá-los.")

        st.divider()

        # Simulador de Previsão
        st.subheader("🔮 Simulador de Categoria de Valor")
        st.markdown("Insira os dados de um imóvel para obter uma previsão da sua categoria de valor.")
        
        # Obter opções para os seletores a partir dos dados de treino
        df_base, _, _ = get_clustering_data()
        bairros_options = sorted(df_base['bairro'].unique())
        tipo_imovel_options = sorted(df_base['tipo_imovel'].unique())
        padrao_acabamento_options = sorted(df_base['padrao_acabamento'].unique())
        cluster_options = sorted(df_base['cluster'].unique())

        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                area_construida = st.number_input("Área Construída (m²)", min_value=10, max_value=1000, value=100, step=10)
                area_terreno = st.number_input("Área do Terreno (m²)", min_value=10, max_value=5000, value=200, step=10)
                ano_construcao = st.number_input("Ano de Construção", min_value=1950, max_value=2024, value=2010, step=1)
            with col2:
                bairro = st.selectbox("Bairro", options=bairros_options, index=bairros_options.index("BOA VIAGEM") if "BOA VIAGEM" in bairros_options else 0)
                tipo_imovel = st.selectbox("Tipo de Imóvel", options=tipo_imovel_options)
                padrao_acabamento = st.selectbox("Padrão de Acabamento", options=padrao_acabamento_options)
            with col3:
                cluster = st.selectbox("Cluster (Perfil de Mercado)", options=cluster_options, help="Selecione o perfil de imóvel mais próximo. Veja a aba 'Clustering' para detalhes.")

            submit_button = st.form_submit_button(label='🚀 Prever Categoria')

        if submit_button:
            # Criar DataFrame com os dados do formulário
            input_data = pd.DataFrame({
                'area_construida': [area_construida],
                'area_terreno': [area_terreno],
                'ano_construcao': [ano_construcao],
                'padrao_acabamento': [padrao_acabamento],
                'cluster': [cluster],
                'bairro': [bairro],
                'tipo_imovel': [tipo_imovel]
            })
            
            # Fazer a predição
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0]
            classes = model.classes_

            # Exibir o resultado
            st.success(f"**Categoria Prevista: {prediction}**")
            
            # Exibir probabilidades em um formato visual
            prob_df = pd.DataFrame({'Classe': classes, 'Probabilidade': probabilities})
            prob_df = prob_df.sort_values('Probabilidade', ascending=False)
            
            fig_prob = px.bar(prob_df, x='Probabilidade', y='Classe', orientation='h', 
                              title='Probabilidades da Predição', text=prob_df['Probabilidade'].apply(lambda x: f'{x:.1%}'))
            fig_prob.update_layout(xaxis_title="Probabilidade", yaxis_title="Categoria", uniformtext_minsize=8, uniformtext_mode='hide')
            st.plotly_chart(fig_prob, use_container_width=True)


# Restaurar o código original das outras abas para garantir que continuem funcionando
with tab1:
    st.header("Análise Exploratória de Dados")
    st.markdown("Visão geral do mercado imobiliário de Recife (todos os tipos de imóveis)")
    
    df = get_data()
    
    with st.sidebar:
        st.subheader("🔍 Filtros - EDA")
        bairros_disponiveis = sorted(df["bairro"].unique().tolist())
        selected_bairro = st.selectbox(
            "Bairro (para referência)",
            bairros_disponiveis,
            index=bairros_disponiveis.index("BOA VIAGEM") if "BOA VIAGEM" in bairros_disponiveis else 0,
            key="eda_bairro"
        )
        
        if st.checkbox("Mostrar Dados Brutos"):
            st.dataframe(df.head(100), use_container_width=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Transações", f"{len(df):,}".replace(",", "."))
    col2.metric("Valor Médio", f"R$ {df['valor_avaliacao'].mean():,.2f}")
    col3.metric("Valor m² Mediano", f"R$ {df['valor_m2'].median():,.2f}")
    col4.metric("Período", "2015-2023")
    
    # ... Adicionar aqui o restante dos gráficos da tab1 se necessário

with tab2:
    st.header("🎯 Clustering de Perfis de Mercado")
    st.markdown("Segmentação inteligente em 5 perfis usando K-means (dados residenciais)")
    
    with st.spinner("Carregando clustering de perfis..."):
        df_clustered, silhouette_score, features = get_clustering_data()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Imóveis Analisados", f"{len(df_clustered):,}".replace(",", "."))
    col2.metric("Silhouette Score", f"{silhouette_score:.3f}")
    col3.metric("Clusters", "5 perfis")
    col4.metric("Features", len(features))

    figs = create_cluster_visualizations(df_clustered)
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(figs[0], use_container_width=True)
    with col_right:
        st.plotly_chart(figs[1], use_container_width=True)
    st.plotly_chart(figs[2], use_container_width=True)
