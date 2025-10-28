import streamlit as st
import pandas as pd
import os
from data_processing import load_and_preprocess_data
from charts.charts import (
    plot_valor_m2_por_bairro,
    plot_valor_transacao_por_acabamento,
    plot_valor_m2_por_ano,
    plot_tipo_imovel_distribuicao,
    plot_qtd_transacoes_por_bairro
)

st.set_page_config(layout="wide", page_title="Análise ITBI Recife")

@st.cache_data
def get_data():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    return load_and_preprocess_data(data_dir=data_dir)

df = get_data()

st.title("📊 Análise Completa do Mercado Imobiliário de Recife (ITBI 2015-2023)")
st.markdown("""
**Evolução da Análise:** De exploração geral para segmentação inteligente do mercado residencial!

Esta aplicação apresenta uma análise completa dos dados do ITBI de Recife:
- **Seções 1-5**: Análise exploratória geral de todos os tipos de imóveis
- **Seção 6**: Clusterização focada em imóveis residenciais (apartamentos e casas)
""")

st.subheader("1. Valor Médio do Metro Quadrado por Bairro")
st.plotly_chart(plot_valor_m2_por_bairro(df), use_container_width=True)

st.subheader("2. Quantidade de Transações por Bairro")
st.plotly_chart(plot_qtd_transacoes_por_bairro(df), use_container_width=True)

st.subheader("3. Valor Médio da Transação por Padrão de Acabamento")
st.plotly_chart(plot_valor_transacao_por_acabamento(df), use_container_width=True)

st.subheader("4. Evolução do Valor Médio do Metro Quadrado por Ano")
st.plotly_chart(plot_valor_m2_por_ano(df), use_container_width=True)

st.subheader("5. Distribuição de Tipos de Imóveis")
st.plotly_chart(plot_tipo_imovel_distribuicao(df), use_container_width=True)

# Nova seção de Clusterização
st.markdown("---")
st.header("🏠 6. Análise de Clusterização - Dados Residenciais")
st.markdown("*Nova funcionalidade: Segmentação inteligente do mercado residencial*")

# Importar funções de clusterização
try:
    from clustering_analysis import (
        get_clustering_data_optimized,
        create_cluster_visualizations
    )
    
    @st.cache_data
    def get_clustering_data():
        """Carrega dados de clusterização usando cache otimizado."""
        return get_clustering_data_optimized()

    # Carregar dados de clusterização (agora ultra-rápido com cache!)
    with st.spinner("⚡ Carregando análise de clusterização..."):
        df_clustered, silhouette_score, features = get_clustering_data()
    
    # Informações da análise
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Imóveis Residenciais", f"{len(df_clustered):,}")
    
    with col2:
        st.metric("Silhouette Score", f"{silhouette_score:.3f}")
    
    with col3:
        st.metric("Clusters Identificados", "5")
    
    with col4:
        reducao = (1 - len(df_clustered)/len(df)) * 100
        st.metric("Filtragem Aplicada", f"-{reducao:.1f}%")

    # Justificativa da filtragem
    with st.expander("❓ Por que apenas dados residenciais?"):
        st.markdown("""
        **Nosso progresso: Evoluímos de análise geral para análise focada!**
        
        **🔄 Evolução da Análise:**
        1. **Análise Exploratória** (seções 1-5): Visão geral de todos os tipos de imóveis
        2. **Análise Focada** (seção 6): Clusterização apenas de imóveis residenciais
        
        **🎯 Motivos para filtrar apenas residências:**
        - 🏥 **Hospitais e instituições**: Valores extremos que distorcem análise
        - 🏢 **Edifícios comerciais**: Padrões completamente diferentes  
        - 🏪 **Lojas e salas**: Valores/m² não comparáveis com residências
        - 📈 **Resultado**: Análise mais precisa do mercado habitacional
        
        **✅ Dados incluídos:** Apartamentos (91.4%) + Casas (8.6%)  
        **❌ Dados excluídos:** {:.1f}% de imóveis comerciais/institucionais
        """.format(reducao))

    # Análise dos clusters
    st.subheader("📋 Segmentos Identificados no Mercado Residencial")
    
    # Calcular estatísticas dos clusters
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

    # Definir nomes dos clusters
    cluster_names = {
        0: "🌟 Apartamentos Premium Novos",
        1: "🏠 Apartamentos Econômicos Novos", 
        2: "🕰️ Imóveis Antigos Diversos",
        3: "🏛️ Apartamentos Grandes Premium",
        4: "💎 Apartamentos Luxury"
    }

    # Mostrar clusters em colunas
    col1, col2, col3, col4, col5 = st.columns(5)
    columns = [col1, col2, col3, col4, col5]

    for i, (cluster_id, col) in enumerate(zip(range(5), columns)):
        with col:
            cluster_data = clusters_info.loc[cluster_id]
            
            st.markdown(f"**{cluster_names[cluster_id]}**")
            st.metric(
                label=f"Cluster {cluster_id}",
                value=f"R$ {cluster_data['valor_m2']:,.0f}/m²"
            )
            
            st.write(f"📊 **{cluster_data['count']:,} imóveis**")
            st.write(f"📐 Área: {cluster_data['area_construida']:.0f} m²")
            st.write(f"📅 Ano: {cluster_data['ano_construcao']:.0f}")
            st.write(f"🏘️ {cluster_data['bairro_principal']}")

    # Visualizações dos clusters
    st.subheader("📈 Visualizações da Clusterização")
    
    # Criar as visualizações
    fig1, fig2, fig3 = create_cluster_visualizations(df_clustered)
    
    # Tabs para organizar gráficos
    tab1, tab2, tab3 = st.tabs(["🔍 Clusters no Espaço", "📊 Distribuição Valores", "🏠 Composição Clusters"])

    with tab1:
        st.plotly_chart(fig1, use_container_width=True)
        st.info("💡 **Nosso Progresso**: Cada ponto é um imóvel residencial. As cores mostram os 5 segmentos identificados automaticamente!")

    with tab2:
        st.plotly_chart(fig2, use_container_width=True) 
        st.info("📋 **Evolução**: De análise geral para segmentação detalhada do mercado residencial.")

    with tab3:
        st.plotly_chart(fig3, use_container_width=True)
        st.info("🏘️ **Insight**: Apartamentos dominam, mas cada cluster tem características únicas.")

    # Interpretação dos resultados
    st.subheader("🎯 Principais Descobertas")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        **🔍 Segmentação Identificada:**
        
        **Mercado Econômico (42%):**
        - Cluster 1: Apartamentos novos acessíveis
        - Cluster 2: Imóveis antigos diversos
        
        **Mercado Premium (58%):**
        - Cluster 0: Apartamentos premium novos (maior grupo)
        - Cluster 3: Apartamentos grandes de alto padrão  
        - Cluster 4: Segmento luxury exclusivo
        """)
    
    with col2:
        st.markdown("""
        **📈 Nosso Progresso Metodológico:**
        
        1. ✅ **Análise Exploratória**: Entendemos o dataset completo
        2. ✅ **Migração para Parquet**: Otimizamos performance  
        3. ✅ **Filtragem Inteligente**: Focamos em dados residenciais
        4. ✅ **Clusterização**: Identificamos padrões ocultos
        5. ✅ **Integração**: Combinamos tudo em uma única aplicação
        """)

    # Aplicações práticas
    with st.expander("🚀 Como Usar Esta Análise"):
        st.markdown("""
        **Para Compradores:**
        - Compare seu imóvel de interesse com outros do mesmo cluster
        - Identifique qual cluster se adequa ao seu orçamento
        - Use como base para negociação
        
        **Para Investidores:**
        - Cluster 1: Maior potencial de valorização (novos + acessíveis)
        - Cluster 2: Oportunidades em áreas consolidadas  
        - Cluster 4: Investimento de luxo com menor liquidez
        
        **Para Construtoras:**
        - Posicione novos projetos baseado nos clusters existentes
        - Use como referência de precificação por segmento
        - Identifique lacunas no mercado
        """)

except ImportError:
    st.error("⚠️ Módulo de clusterização não encontrado. Execute 'python clustering_analysis.py' primeiro.")
