import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from clustering_analysis import (
    filter_residential_data,
    prepare_clustering_features,
    perform_clustering,
    analyze_clusters,
    create_cluster_visualizations
)
from data_processing import load_and_preprocess_data

st.set_page_config(
    page_title="Clusterização - ITBI Recife", 
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_clustered_data():
    """Carrega e processa dados com clusterização."""
    # Carregar dados originais
    df = load_and_preprocess_data()
    
    # Filtrar apenas residenciais
    df_residential = filter_residential_data(df)
    
    # Preparar features
    df_features, features = prepare_clustering_features(df_residential)
    
    # Executar clusterização
    df_clustered, kmeans, scaler, silhouette = perform_clustering(df_features, features)
    
    return df_clustered, silhouette, features

# Carregar dados
df_clustered, silhouette_score, features = load_clustered_data()

# Título principal
st.title("🏠 Análise de Clusterização - Mercado Residencial de Recife")
st.markdown("*Análise focada apenas em dados residenciais (Apartamentos e Casas)*")

# Sidebar com informações
st.sidebar.header("📊 Informações da Análise")
st.sidebar.metric("Total de Imóveis Analisados", f"{len(df_clustered):,}")
st.sidebar.metric("Silhouette Score", f"{silhouette_score:.3f}")
st.sidebar.metric("Número de Clusters", "5")

# Justificativa da filtragem
with st.expander("❓ Por que filtramos apenas dados residenciais?"):
    st.markdown("""
    **Motivos para excluir imóveis não-residenciais:**
    
    🏥 **Hospitais e Instituições**: Valores extremamente altos que distorcem a análise
    
    🏢 **Edifícios Comerciais**: Padrões de preço completamente diferentes do mercado residencial
    
    🏪 **Lojas e Salas**: Valores por m² não comparáveis com residências
    
    📈 **Resultado**: Dataset mais homogêneo e análise mais precisa do mercado residencial
    
    **Dados filtrados:**
    - ✅ Apartamentos (91.4% dos dados residenciais)
    - ✅ Casas (8.6% dos dados residenciais)
    - ❌ Salas comerciais, lojas, hospitais, etc. (13.6% removidos)
    """)

# Análise dos clusters
st.header("📋 Características dos Clusters Identificados")

# Criar métricas para cada cluster
clusters_info = df_clustered.groupby('cluster').agg({
    'valor_m2': 'median',
    'area_construida': 'median', 
    'ano_construcao': 'median',
    'tipo_imovel': lambda x: x.value_counts().index[0]
}).round(0)

clusters_info['count'] = df_clustered.groupby('cluster').size()
clusters_info['bairro_principal'] = df_clustered.groupby('cluster')['bairro'].apply(
    lambda x: x.value_counts().index[0]
)

# Criar colunas para mostrar clusters
col1, col2, col3, col4, col5 = st.columns(5)
columns = [col1, col2, col3, col4, col5]

for i, (cluster_id, col) in enumerate(zip(range(5), columns)):
    with col:
        cluster_data = clusters_info.loc[cluster_id]
        
        st.metric(
            label=f"🏘️ Cluster {cluster_id}",
            value=f"R$ {cluster_data['valor_m2']:,.0f}/m²"
        )
        
        st.write(f"**{cluster_data['count']:,} imóveis**")
        st.write(f"Área: {cluster_data['area_construida']:.0f} m²")
        st.write(f"Ano: {cluster_data['ano_construcao']:.0f}")
        st.write(f"Tipo: {cluster_data['tipo_imovel']}")
        st.write(f"Bairro: {cluster_data['bairro_principal']}")

# Visualizações
st.header("📈 Visualizações dos Clusters")

# Criar as visualizações
fig1, fig2, fig3 = create_cluster_visualizations(df_clustered)

# Tabs para organizar gráficos
tab1, tab2, tab3 = st.tabs(["🔍 Scatter Plot", "📊 Distribuição Valores", "🏠 Tipos por Cluster"])

with tab1:
    st.subheader("Valor m² vs Área Construída por Cluster")
    st.plotly_chart(fig1, use_container_width=True)
    st.info("💡 Cada ponto representa um imóvel. Cores diferentes = clusters diferentes. Passe o mouse para ver detalhes!")

with tab2:
    st.subheader("Distribuição do Valor m² por Cluster")
    st.plotly_chart(fig2, use_container_width=True)
    st.info("📋 Box plots mostram a mediana, quartis e outliers para cada cluster.")

with tab3:
    st.subheader("Distribuição de Tipos de Imóveis por Cluster")
    st.plotly_chart(fig3, use_container_width=True)
    st.info("🏘️ Veja como apartamentos e casas se distribuem entre os clusters.")

# Análise detalhada
st.header("🔍 Interpretação dos Clusters")

interpretacoes = {
    0: {
        "nome": "🌟 Apartamentos Premium Novos",
        "descricao": "Apartamentos com alto valor/m², área média, construção recente",
        "perfil": "Imóveis de padrão médio-alto em bairros valorizados"
    },
    1: {
        "nome": "🏠 Apartamentos Econômicos Novos", 
        "descricao": "Apartamentos com valor/m² moderado, área menor, construção recente",
        "perfil": "Imóveis mais acessíveis em diferentes bairros"
    },
    2: {
        "nome": "🕰️ Imóveis Antigos Diversos",
        "descricao": "Mix de apartamentos e casas, valor médio, construção mais antiga",
        "perfil": "Imóveis estabelecidos com boa localização"
    },
    3: {
        "nome": "🏛️ Apartamentos Grandes Premium",
        "descricao": "Apartamentos com áreas maiores, alto valor, boa localização",
        "perfil": "Imóveis de alto padrão para famílias maiores"
    },
    4: {
        "nome": "💎 Apartamentos Luxury",
        "descricao": "Poucos imóveis, valores muito altos, áreas grandes",
        "perfil": "Segmento de luxo do mercado"
    }
}

for cluster_id in range(5):
    with st.expander(f"Cluster {cluster_id}: {interpretacoes[cluster_id]['nome']}"):
        cluster_stats = clusters_info.loc[cluster_id]
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"""
            **📊 Estatísticas:**
            - Quantidade: {cluster_stats['count']:,} imóveis ({cluster_stats['count']/len(df_clustered)*100:.1f}%)
            - Valor médio m²: R$ {cluster_stats['valor_m2']:,.0f}
            - Área média: {cluster_stats['area_construida']:.0f} m²
            - Ano médio: {cluster_stats['ano_construcao']:.0f}
            """)
        
        with col2:
            st.markdown(f"""
            **🏘️ Características:**
            - {interpretacoes[cluster_id]['descricao']}
            - {interpretacoes[cluster_id]['perfil']}
            - Bairro principal: {cluster_stats['bairro_principal']}
            - Tipo predominante: {cluster_stats['tipo_imovel']}
            """)

# Conclusões
st.header("🎯 Conclusões da Análise")

col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **🔍 Insights Principais:**
    
    1. **Segmentação Clara**: 5 clusters distintos no mercado residencial
    2. **Apartamentos Dominam**: 91.4% dos imóveis residenciais  
    3. **Boa Viagem**: Aparece como bairro principal em múltiplos clusters
    4. **Polarização**: Clusters de entrada vs. premium bem definidos
    """)

with col2:
    st.markdown("""
    **📈 Aplicações Práticas:**
    
    1. **Precificação**: Benchmarking por cluster
    2. **Segmentação de Marketing**: Campanhas direcionadas
    3. **Análise de Investimento**: Identificar oportunidades
    4. **Planejamento Urbano**: Entender padrões residenciais
    """)

# Filtros interativos
st.header("🔎 Exploração Interativa")

selected_clusters = st.multiselect(
    "Selecione clusters para análise detalhada:",
    options=list(range(5)),
    default=[0, 1],
    format_func=lambda x: f"Cluster {x}: {interpretacoes[x]['nome']}"
)

if selected_clusters:
    filtered_data = df_clustered[df_clustered['cluster'].isin(selected_clusters)]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Imóveis Selecionados", f"{len(filtered_data):,}")
    
    with col2:
        avg_price = filtered_data['valor_m2'].mean()
        st.metric("Valor Médio m²", f"R$ {avg_price:,.0f}")
    
    with col3:
        avg_area = filtered_data['area_construida'].mean()
        st.metric("Área Média", f"{avg_area:.0f} m²")
    
    # Mostrar amostra dos dados
    if st.checkbox("Mostrar dados detalhados"):
        st.dataframe(
            filtered_data[['bairro', 'tipo_imovel', 'valor_m2', 'area_construida', 'ano_construcao', 'cluster']]
            .head(100)
        )