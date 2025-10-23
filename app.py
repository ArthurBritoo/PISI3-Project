"""
App Unificado - Análise ITBI Recife
Integra análise exploratória, clustering de perfis e dashboard regional.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_processing import load_and_preprocess_data
from charts.charts import (
    plot_valor_m2_por_bairro,
    plot_valor_transacao_por_acabamento,
    plot_valor_m2_por_ano,
    plot_tipo_imovel_distribuicao,
    plot_qtd_transacoes_por_bairro
)
from clustering_analysis import get_clustering_data_optimized, create_cluster_visualizations
from geo_clustering import build_regions_for_recife

st.set_page_config(
    page_title="ITBI Recife - Análise Completa",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CACHE E DADOS ====================

@st.cache_data(show_spinner=False)
def get_data():
    """Carrega dados gerais do ITBI."""
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    return load_and_preprocess_data(data_dir=data_dir)

@st.cache_data(show_spinner=False)
def get_clustering_data():
    """Carrega dados de clustering de perfis (K-means)."""
    return get_clustering_data_optimized()

@st.cache_data(show_spinner=True)
def build_regions_cached(df_in: pd.DataFrame, min_tx: int):
    """Constrói regiões geográficas IBGE (com cache)."""
    df_out, regions = build_regions_for_recife(df_in, min_tx_per_region=min_tx)
    return df_out, regions

@st.cache_data(show_spinner=False)
def get_integrated_data(min_tx: int):
    """
    Carrega dados integrados: clustering de perfis + regiões geográficas.
    Cruza os 5 perfis de mercado com as regiões IBGE.
    
    IMPORTANTE: Usa os dados de clustering como base (que já têm cluster)
    e apenas adiciona as regiões geográficas.
    """
    # 1. Dados com clustering de perfis (já têm cluster!)
    df_clustered, silhouette_score, features = get_clustering_data()
    
    # 2. Adicionar regiões geográficas ao DataFrame de clustering
    df_integrated, regions_dict = build_regions_cached(df_clustered, min_tx)
    
    # 3. Para análises que precisam de valor_avaliacao, calcular a partir de valor_m2 * area
    if 'valor_avaliacao' not in df_integrated.columns and 'valor_m2' in df_integrated.columns:
        df_integrated['valor_avaliacao'] = df_integrated['valor_m2'] * df_integrated['area_construida']
    
    return df_integrated, regions_dict, silhouette_score, features

# ==================== HELPER FUNCTIONS ====================

def moeda(v):
    """Formata valor como moeda brasileira."""
    try:
        return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except:
        return str(v)

def get_cluster_name(cluster_id):
    """Retorna nome descritivo do cluster."""
    names = {
        0: "🏠 Popular",
        1: "🏡 Entrada",
        2: "🏘️ Intermediário",
        3: "🏙️ Alto Padrão",
        4: "💎 Premium"
    }
    return names.get(cluster_id, f"Cluster {cluster_id}")

# ==================== NAVEGAÇÃO PRINCIPAL ====================

st.title("📊 Análise Completa do Mercado Imobiliário de Recife")
st.caption("ITBI 2015-2023 • Dados Residenciais (Apartamentos e Casas)")

# Tabs principais
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 EDA Exploratória",
    "🎯 Clustering de Perfis",
    "🗺️ Dashboard Regional",
    "🔥 Análise Integrada"
])

# ==================== TAB 1: EDA EXPLORATÓRIA ====================

with tab1:
    st.header("Análise Exploratória de Dados")
    st.markdown("Visão geral do mercado imobiliário de Recife (todos os tipos de imóveis)")
    
    df = get_data()
    
    # Sidebar para essa tab
    with st.sidebar:
        st.subheader("🔍 Filtros - EDA")
        bairros_disponiveis = sorted(df["bairro"].unique().tolist())
        selected_bairro = st.selectbox(
            "Bairro (para referência)",
            bairros_disponiveis,
            index=bairros_disponiveis.index("BOA VIAGEM") if "BOA VIAGEM" in bairros_disponiveis else 0
        )
        
        if st.checkbox("Mostrar Dados Brutos"):
            st.dataframe(df.head(100), use_container_width=True)
    
    # KPIs gerais
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de Transações", f"{len(df):,}".replace(",", "."))
    col2.metric("Valor Médio", moeda(df["valor_avaliacao"].mean()))
    col3.metric("Valor m² Mediano", moeda(df["valor_m2"].median()))
    col4.metric("Período", "2015-2023")
    
    st.divider()
    
    # Gráficos principais
    st.subheader("1. Valor Médio do Metro Quadrado por Bairro")
    st.plotly_chart(plot_valor_m2_por_bairro(df), use_container_width=True)
    
    st.subheader("2. Quantidade de Transações por Bairro")
    st.plotly_chart(plot_qtd_transacoes_por_bairro(df), use_container_width=True)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("3. Valor por Padrão de Acabamento")
        st.plotly_chart(plot_valor_transacao_por_acabamento(df), use_container_width=True)
    
    with col_b:
        st.subheader("4. Distribuição de Tipos")
        st.plotly_chart(plot_tipo_imovel_distribuicao(df), use_container_width=True)
    
    st.subheader("5. Evolução do Valor do m² por Ano")
    st.plotly_chart(plot_valor_m2_por_ano(df), use_container_width=True)
    
    st.divider()
    
    # Download de dados da EDA
    with st.expander("📥 Exportar dados para análise"):
        st.markdown("**Dados completos do ITBI Recife:**")
        csv_eda = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar dados completos (CSV)",
            data=csv_eda,
            file_name="itbi_recife_completo.csv",
            mime="text/csv"
        )
        
        st.markdown("**Resumo estatístico:**")
        csv_stats = df.describe().to_csv().encode("utf-8")
        st.download_button(
            "Baixar resumo estatístico (CSV)",
            data=csv_stats,
            file_name="itbi_recife_resumo.csv",
            mime="text/csv"
        )

# ==================== TAB 2: CLUSTERING DE PERFIS ====================

with tab2:
    st.header("🎯 Clustering de Perfis de Mercado")
    st.markdown("Segmentação inteligente em 5 perfis usando K-means (dados residenciais)")
    
    with st.spinner("Carregando clustering de perfis..."):
        df_clustered, silhouette_score, features = get_clustering_data()
    
    # Métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Imóveis Analisados", f"{len(df_clustered):,}".replace(",", "."))
    col2.metric("Silhouette Score", f"{silhouette_score:.3f}")
    col3.metric("Clusters", "5 perfis")
    col4.metric("Features", len(features))
    
    # Explicação dos perfis com estatísticas reais
    with st.expander("ℹ️ Entenda os 5 Perfis de Mercado (com Parâmetros Exatos)"):
        # Calcula estatísticas reais de cada cluster
        cluster_details = df_clustered.groupby("cluster").agg({
            "valor_m2": ["min", "max", "median", "mean"],
            "area_construida": ["min", "max", "median"],
            "area_terreno": ["min", "max", "median"],
            "ano_construcao": ["min", "max"]
        }).round(2)
        
        st.markdown("""
        **Metodologia:** Clustering K-means com 5 grupos, usando 4 features normalizadas:
        - `valor_m2` (Valor por metro quadrado)
        - `area_construida` (Área construída total)
        - `area_terreno` (Área do terreno)
        - `ano_construcao` (Ano de construção do imóvel)
        
        **Silhouette Score:** {:.3f} (qualidade da separação dos clusters)
        
        ---
        
        ### 🏠 **Cluster 0 - Popular**
        - **Valor m²:** R$ {:.0f} - R$ {:.0f} (mediana: R$ {:.0f})
        - **Área Construída:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Área Terreno:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Perfil:** Imóveis de entrada, menor valor por m², áreas compactas, construções mais antigas
        
        ### 🏡 **Cluster 1 - Entrada**
        - **Valor m²:** R$ {:.0f} - R$ {:.0f} (mediana: R$ {:.0f})
        - **Área Construída:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Área Terreno:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Perfil:** Bom custo-benefício, intermediário inferior, adequado para primeiro imóvel
        
        ### 🏘️ **Cluster 2 - Intermediário**
        - **Valor m²:** R$ {:.0f} - R$ {:.0f} (mediana: R$ {:.0f})
        - **Área Construída:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Área Terreno:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Perfil:** Classe média, equilíbrio entre área e valor, padrão de acabamento médio
        
        ### 🏙️ **Cluster 3 - Alto Padrão**
        - **Valor m²:** R$ {:.0f} - R$ {:.0f} (mediana: R$ {:.0f})
        - **Área Construída:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Área Terreno:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Perfil:** Imóveis maiores, acabamento superior, valor m² elevado
        
        ### 💎 **Cluster 4 - Premium**
        - **Valor m²:** R$ {:.0f} - R$ {:.0f} (mediana: R$ {:.0f})
        - **Área Construída:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Área Terreno:** {:.0f}m² - {:.0f}m² (mediana: {:.0f}m²)
        - **Perfil:** Elite do mercado, máximo valor m², grandes áreas, construções recentes
        """.format(
            silhouette_score,
            # Cluster 0
            cluster_details.loc[0, ("valor_m2", "min")],
            cluster_details.loc[0, ("valor_m2", "max")],
            cluster_details.loc[0, ("valor_m2", "median")],
            cluster_details.loc[0, ("area_construida", "min")],
            cluster_details.loc[0, ("area_construida", "max")],
            cluster_details.loc[0, ("area_construida", "median")],
            cluster_details.loc[0, ("area_terreno", "min")],
            cluster_details.loc[0, ("area_terreno", "max")],
            cluster_details.loc[0, ("area_terreno", "median")],
            # Cluster 1
            cluster_details.loc[1, ("valor_m2", "min")],
            cluster_details.loc[1, ("valor_m2", "max")],
            cluster_details.loc[1, ("valor_m2", "median")],
            cluster_details.loc[1, ("area_construida", "min")],
            cluster_details.loc[1, ("area_construida", "max")],
            cluster_details.loc[1, ("area_construida", "median")],
            cluster_details.loc[1, ("area_terreno", "min")],
            cluster_details.loc[1, ("area_terreno", "max")],
            cluster_details.loc[1, ("area_terreno", "median")],
            # Cluster 2
            cluster_details.loc[2, ("valor_m2", "min")],
            cluster_details.loc[2, ("valor_m2", "max")],
            cluster_details.loc[2, ("valor_m2", "median")],
            cluster_details.loc[2, ("area_construida", "min")],
            cluster_details.loc[2, ("area_construida", "max")],
            cluster_details.loc[2, ("area_construida", "median")],
            cluster_details.loc[2, ("area_terreno", "min")],
            cluster_details.loc[2, ("area_terreno", "max")],
            cluster_details.loc[2, ("area_terreno", "median")],
            # Cluster 3
            cluster_details.loc[3, ("valor_m2", "min")],
            cluster_details.loc[3, ("valor_m2", "max")],
            cluster_details.loc[3, ("valor_m2", "median")],
            cluster_details.loc[3, ("area_construida", "min")],
            cluster_details.loc[3, ("area_construida", "max")],
            cluster_details.loc[3, ("area_construida", "median")],
            cluster_details.loc[3, ("area_terreno", "min")],
            cluster_details.loc[3, ("area_terreno", "max")],
            cluster_details.loc[3, ("area_terreno", "median")],
            # Cluster 4
            cluster_details.loc[4, ("valor_m2", "min")],
            cluster_details.loc[4, ("valor_m2", "max")],
            cluster_details.loc[4, ("valor_m2", "median")],
            cluster_details.loc[4, ("area_construida", "min")],
            cluster_details.loc[4, ("area_construida", "max")],
            cluster_details.loc[4, ("area_construida", "median")],
            cluster_details.loc[4, ("area_terreno", "min")],
            cluster_details.loc[4, ("area_terreno", "max")],
            cluster_details.loc[4, ("area_terreno", "median")]
        ))
    
    st.divider()
    
    # Análise por cluster
    st.subheader("📊 Características dos Clusters")
    
    cluster_stats = df_clustered.groupby("cluster").agg({
        "valor_m2": "median",
        "area_construida": "median",
        "area_terreno": "median",
        "bairro": "count"
    }).reset_index()
    cluster_stats.columns = ["Cluster", "Valor m² Mediano", "Área Construída Mediana", "Área Terreno Mediana", "Qtd Imóveis"]
    cluster_stats["Perfil"] = cluster_stats["Cluster"].apply(get_cluster_name)
    cluster_stats = cluster_stats[["Perfil", "Qtd Imóveis", "Valor m² Mediano", "Área Construída Mediana", "Área Terreno Mediana"]]
    
    st.dataframe(
        cluster_stats.style.format({
            "Qtd Imóveis": "{:,.0f}",
            "Valor m² Mediano": "R$ {:,.0f}",
            "Área Construída Mediana": "{:.0f} m²",
            "Área Terreno Mediana": "{:.0f} m²"
        }),
        use_container_width=True,
        hide_index=True
    )
    
    st.divider()
    
    # Visualizações do clustering
    st.subheader("📈 Visualizações dos Perfis")
    
    figs = create_cluster_visualizations(df_clustered)
    
    col_left, col_right = st.columns(2)
    with col_left:
        st.plotly_chart(figs[0], use_container_width=True)  # PCA
    with col_right:
        st.plotly_chart(figs[1], use_container_width=True)  # Distribuição
    
    st.plotly_chart(figs[2], use_container_width=True)  # Características
    
    # Distribuição geográfica dos clusters
    st.subheader("🗺️ Distribuição Geográfica dos Perfis")
    
    geo_cluster = df_clustered.groupby(["bairro", "cluster"]).size().reset_index(name="count")
    geo_cluster["perfil"] = geo_cluster["cluster"].apply(get_cluster_name)
    
    top_bairros = df_clustered["bairro"].value_counts().head(20).index
    geo_cluster_top = geo_cluster[geo_cluster["bairro"].isin(top_bairros)]
    
    fig_geo = px.bar(
        geo_cluster_top,
        x="bairro",
        y="count",
        color="perfil",
        title="Distribuição de Perfis nos Top 20 Bairros",
        labels={"count": "Quantidade", "bairro": "Bairro", "perfil": "Perfil"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_geo.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_geo, use_container_width=True)
    
    st.divider()
    
    # Download de dados de clustering
    with st.expander("📥 Exportar análise de clusters"):
        st.markdown("**Resumo dos clusters:**")
        csv_cluster_summary = cluster_stats.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar resumo de clusters (CSV)",
            data=csv_cluster_summary,
            file_name="clusters_resumo.csv",
            mime="text/csv"
        )
        
        st.markdown("**Dados completos com clustering:**")
        csv_clustered = df_clustered.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar dados clustering completo (CSV)",
            data=csv_clustered,
            file_name="dados_clustering_completo.csv",
            mime="text/csv"
        )

# ==================== TAB 3: DASHBOARD REGIONAL ====================

with tab3:
    st.header("🗺️ Dashboard Regional (IBGE)")
    st.markdown("Análise por regiões geográficas com agrupamento de subdistritos")
    
    df_res = get_data()
    residential_types = [t for t in ["Apartamento", "Casa"] if t in df_res["tipo_imovel"].unique()]
    df_res = df_res[df_res["tipo_imovel"].isin(residential_types)].copy()
    
    # Sidebar para filtros regionais
    with st.sidebar:
        st.subheader("🗺️ Filtros - Regional")
        
        min_tx_per_region = st.number_input(
            "Mínimo tx/região",
            min_value=50, max_value=5000, value=2000, step=50,
            help="Agrupa subdistritos vizinhos até atingir este mínimo de transações"
        )
        
        anos = sorted(df_res["data_transacao"].dt.year.dropna().unique().tolist())
        min_ano, max_ano = (anos[0], anos[-1]) if anos else (2015, 2023)
        ano_range = st.slider("Período", min_ano, max_ano, (min_ano, max_ano))
        
        aggr_label = st.radio("Agregação", ["Média", "Mediana"], index=0)
        aggr_func = "mean" if aggr_label == "Média" else "median"
        
        top_n = st.selectbox("Top N", [10, 20, 30], index=0)
    
    # Construir regiões
    df_reg, regions_dict = build_regions_cached(df_res, int(min_tx_per_region))
    
    # Aplicar filtros
    mask = df_reg["data_transacao"].dt.year.between(ano_range[0], ano_range[1])
    dff = df_reg[mask].copy()
    
    if dff.empty:
        st.warning("Nenhum dado encontrado com os filtros.")
        st.stop()
    
    # Mostrar composição das regiões
    with st.expander("Ver composição das regiões IBGE"):
        reg_view = pd.DataFrame([
            {"Região": k, "Subdistritos": ", ".join(sorted(v))}
            for k, v in regions_dict.items()
        ]).sort_values("Região")
        st.dataframe(reg_view, use_container_width=True, hide_index=True)
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transações", f"{len(dff):,}".replace(",", "."))
    col2.metric("Ticket Médio", moeda(dff["valor_avaliacao"].mean()))
    col3.metric(f"Valor m² ({aggr_label})", moeda(dff["valor_m2"].agg(aggr_func)))
    col4.metric("Regiões", len(regions_dict))
    
    st.divider()
    
    # Gráficos regionais
    st.subheader(f"1. Valor do Metro Quadrado por Região — Top {top_n} ({aggr_label})")
    g1 = (
        dff.groupby("regiao")["valor_m2"]
        .agg(aggr_func)
        .reset_index()
        .sort_values("valor_m2", ascending=False)
        .head(top_n)
    )
    fig1 = px.bar(
        g1, x="regiao", y="valor_m2",
        title=f"Top {top_n} Regiões por Valor do m² ({aggr_label})",
        labels={"regiao": "Região", "valor_m2": "Valor do m² (R$)"},
        color="valor_m2",
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
    
    st.subheader(f"2. Quantidade de Transações por Região — Top {top_n}")
    g2 = (
        dff.groupby("regiao")
        .size()
        .reset_index(name="qtd_transacoes")
        .sort_values("qtd_transacoes", ascending=False)
        .head(top_n)
    )
    fig2 = px.bar(
        g2, x="regiao", y="qtd_transacoes",
        title=f"Transações por Região (Top {top_n})",
        labels={"regiao": "Região", "qtd_transacoes": "Quantidade de Transações"},
        color="qtd_transacoes",
        color_continuous_scale=px.colors.sequential.Blues,
    )
    fig2.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader(f"3. Valor da Transação por Padrão de Acabamento ({aggr_label})")
    g3 = (
        dff.groupby("padrao_acabamento")["valor_avaliacao"]
        .agg(aggr_func)
        .reset_index()
    )
    cat_order = ["Simples", "Médio", "Superior"]
    g3["padrao_acabamento"] = pd.Categorical(g3["padrao_acabamento"], categories=cat_order, ordered=True)
    g3 = g3.sort_values("padrao_acabamento")
    fig3 = px.bar(
        g3, x="padrao_acabamento", y="valor_avaliacao",
        title=f"Valor da Transação por Padrão de Acabamento ({aggr_label})",
        labels={"padrao_acabamento": "Padrão de Acabamento", "valor_avaliacao": "Valor da Transação (R$)"},
        color="valor_avaliacao",
        color_continuous_scale=px.colors.sequential.Plasma,
    )
    st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader(f"4. Evolução do Valor do m² por Ano ({aggr_label})")
    g4 = (
        dff.groupby(dff["data_transacao"].dt.year)["valor_m2"]
        .agg(aggr_func)
        .reset_index()
    )
    g4.columns = ["ano", "valor_m2"]
    fig4 = px.line(
        g4, x="ano", y="valor_m2",
        title=f"Evolução do Valor do m² ({aggr_label})",
        labels={"ano": "Ano", "valor_m2": "Valor do m² (R$)"},
        markers=True,
    )
    fig4.update_xaxes(type="category")
    st.plotly_chart(fig4, use_container_width=True)
    
    st.subheader("5. Distribuição de Tipos de Imóveis (Residencial)")
    g5 = (
        dff.groupby("tipo_imovel")
           .size()
           .reset_index(name="count")
           .sort_values("count", ascending=False)
    )
    fig5 = px.pie(
        g5,
        values="count",
        names="tipo_imovel",
        hole=0.3,
        title="Distribuição por Tipo"
    )
    st.plotly_chart(fig5, use_container_width=True)
    
    st.subheader("6. Dispersão Valor vs Área (Scatter)")
    with st.expander("Ver Dispersão Valor vs Área (R$ x m²)"):
        fig6 = px.scatter(
            dff, x="area_construida", y="valor_avaliacao",
            color="tipo_imovel",
            hover_data=["regiao", "bairro"],
            labels={"area_construida": "Área construída (m²)", "valor_avaliacao": "Valor da Transação (R$)"},
            title="Dispersão Valor x Área",
            opacity=0.6,
        )
        st.plotly_chart(fig6, use_container_width=True)
    
    st.divider()
    
    # Tabela e download dos dados filtrados
    with st.expander("Ver dados filtrados"):
        st.dataframe(dff.sort_values("data_transacao", ascending=False), use_container_width=True, hide_index=True)
    
    csv = dff.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Baixar CSV (filtro atual)", data=csv, file_name="itbi_residencial_filtrado.csv", mime="text/csv")

# ==================== TAB 4: ANÁLISE INTEGRADA ====================

with tab4:
    st.header("🔥 Análise Integrada: Perfis × Regiões")
    st.markdown("**Inovação:** Cruzamento dos clusters de mercado com regiões geográficas IBGE")
    
    with st.sidebar:
        st.subheader("🔥 Filtros - Integrado")
        min_tx_integrated = st.number_input(
            "Mínimo tx/região (integrado)",
            min_value=50, max_value=5000, value=2000, step=50,
            key="integrated_min_tx",
            help="Agrupa subdistritos vizinhos até atingir este mínimo de transações"
        )
    
    with st.spinner("Carregando dados integrados..."):
        df_int, regions_dict_int, silh_int, feat_int = get_integrated_data(int(min_tx_integrated))
    
    # Debug: verificar se coluna cluster existe
    if 'cluster' not in df_int.columns:
        st.error(f"⚠️ Coluna 'cluster' não encontrada. Colunas disponíveis: {df_int.columns.tolist()}")
        st.stop()
    
    # Adicionar nome dos perfis
    df_int["perfil"] = df_int["cluster"].apply(get_cluster_name)
    
    # KPIs integrados
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Imóveis Integrados", f"{len(df_int):,}".replace(",", "."))
    col2.metric("Regiões IBGE", len(regions_dict_int))
    col3.metric("Perfis de Mercado", "5 clusters")
    col4.metric("Silhouette Score", f"{silh_int:.3f}")
    
    st.divider()
    
    # 1. HEATMAP: Perfis × Regiões
    st.subheader("🔥 Heatmap: Distribuição de Perfis por Região")
    
    cross_tab = pd.crosstab(df_int["regiao"], df_int["perfil"], normalize="index") * 100
    cross_tab = cross_tab.round(1)
    
    # Ordenar regiões por volume total
    regiao_volume = df_int["regiao"].value_counts()
    top_regions = regiao_volume.head(15).index
    cross_tab_top = cross_tab.loc[top_regions]
    
    fig_heatmap = px.imshow(
        cross_tab_top,
        labels=dict(x="Perfil de Mercado", y="Região IBGE", color="% Imóveis"),
        title="Concentração de Perfis por Região (Top 15 regiões)",
        color_continuous_scale="YlOrRd",
        aspect="auto"
    )
    fig_heatmap.update_xaxes(side="bottom")
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # 2. Barras empilhadas: Perfis em cada região
    st.subheader("📊 Composição de Perfis por Região (Top 10)")
    
    cross_counts = pd.crosstab(df_int["regiao"], df_int["perfil"])
    cross_counts["total"] = cross_counts.sum(axis=1)
    cross_counts = cross_counts.sort_values("total", ascending=False).head(10)
    cross_counts = cross_counts.drop("total", axis=1)
    
    fig_stacked = go.Figure()
    for perfil in cross_counts.columns:
        fig_stacked.add_trace(go.Bar(
            name=perfil,
            x=cross_counts.index,
            y=cross_counts[perfil],
        ))
    
    fig_stacked.update_layout(
        barmode='stack',
        title="Quantidade de Imóveis por Perfil e Região",
        xaxis_title="Região",
        yaxis_title="Quantidade de Imóveis",
        xaxis_tickangle=-45
    )
    st.plotly_chart(fig_stacked, use_container_width=True)
    
    # 3. Análise de dominância: qual perfil domina cada região?
    st.subheader("🎯 Perfil Dominante por Região")
    
    dominant_profile = df_int.groupby("regiao")["perfil"].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else "N/A")
    profile_counts = df_int.groupby("regiao").size()
    
    dominant_df = pd.DataFrame({
        "Região": dominant_profile.index,
        "Perfil Dominante": dominant_profile.values,
        "Total Imóveis": profile_counts.values
    }).sort_values("Total Imóveis", ascending=False).head(15)
    
    fig_dominant = px.bar(
        dominant_df,
        x="Região",
        y="Total Imóveis",
        color="Perfil Dominante",
        title="Perfil de Mercado Dominante nas Top 15 Regiões",
        labels={"Total Imóveis": "Quantidade de Imóveis"},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig_dominant.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_dominant, use_container_width=True)
    
    # 4. Scatter: Valor médio × Volume por Região, colorido por perfil dominante
    st.subheader("💰 Valor Médio vs Volume por Região")
    
    region_stats = df_int.groupby("regiao").agg({
        "valor_avaliacao": "mean",
        "valor_m2": "median",
        "perfil": lambda x: x.mode()[0] if len(x.mode()) > 0 else "Misto",
        "bairro": "count"
    }).reset_index()
    region_stats.columns = ["regiao", "valor_medio", "valor_m2_mediano", "perfil_dominante", "quantidade"]
    
    fig_scatter = px.scatter(
        region_stats,
        x="quantidade",
        y="valor_medio",
        size="valor_m2_mediano",
        color="perfil_dominante",
        hover_name="regiao",
        title="Valor Médio × Volume (tamanho = valor m²)",
        labels={
            "quantidade": "Quantidade de Transações",
            "valor_medio": "Valor Médio (R$)",
            "perfil_dominante": "Perfil Dominante"
        },
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # 5. Tabela interativa com filtro
    st.subheader("🔍 Explorador Interativo: Perfis × Regiões")
    
    col_filter1, col_filter2 = st.columns(2)
    
    with col_filter1:
        perfis_sel = st.multiselect(
            "Filtrar por Perfil",
            options=sorted(df_int["perfil"].unique()),
            default=sorted(df_int["perfil"].unique())
        )
    
    with col_filter2:
        regioes_sel = st.multiselect(
            "Filtrar por Região (deixe vazio = todas)",
            options=sorted(df_int["regiao"].unique()),
            default=[]
        )
    
    df_filtered = df_int[df_int["perfil"].isin(perfis_sel)]
    if regioes_sel:
        df_filtered = df_filtered[df_filtered["regiao"].isin(regioes_sel)]
    
    summary_table = df_filtered.groupby(["regiao", "perfil"]).agg({
        "valor_avaliacao": ["mean", "count"],
        "valor_m2": "median",
        "area_construida": "median"
    }).reset_index()
    
    summary_table.columns = ["Região", "Perfil", "Valor Médio", "Quantidade", "Valor m² Mediano", "Área Mediana"]
    summary_table = summary_table.sort_values(["Região", "Quantidade"], ascending=[True, False])
    
    st.dataframe(
        summary_table.style.format({
            "Valor Médio": "R$ {:,.0f}",
            "Quantidade": "{:,.0f}",
            "Valor m² Mediano": "R$ {:,.0f}",
            "Área Mediana": "{:.0f} m²"
        }),
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Download
    csv_integrated = summary_table.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Baixar Análise Integrada (CSV)",
        data=csv_integrated,
        file_name="analise_integrada_perfis_regioes.csv",
        mime="text/csv"
    )
    
    # Insights automáticos
    with st.expander("💡 Insights Automáticos"):
        st.markdown(f"""
        ### Principais Descobertas:
        
        **📊 Distribuição Geral:**
        - Total de {len(df_int):,} imóveis residenciais analisados
        - Agrupados em {len(regions_dict_int)} regiões geográficas (IBGE)
        - Segmentados em 5 perfis de mercado distintos
        
        **🎯 Perfil Mais Comum:**
        - Perfil dominante no dataset: **{df_int['perfil'].mode()[0]}**
        - Representa {(df_int['perfil'].value_counts().iloc[0] / len(df_int) * 100):.1f}% do total
        
        **🗺️ Região com Maior Volume:**
        - Região líder: **{df_int['regiao'].value_counts().index[0]}**
        - Concentra {df_int['regiao'].value_counts().iloc[0]:,} transações
        
        **💎 Mercado Premium:**
        - Regiões com mais imóveis Premium: {', '.join(df_int[df_int['perfil'] == '💎 Premium']['regiao'].value_counts().head(3).index.tolist())}
        
        **🏠 Mercado Popular:**
        - Regiões com mais imóveis Populares: {', '.join(df_int[df_int['perfil'] == '🏠 Popular']['regiao'].value_counts().head(3).index.tolist())}
        """)

# ==================== FOOTER ====================

st.divider()
st.caption("💻 Desenvolvido com Streamlit | 📊 Dados: ITBI Recife 2015-2023")
st.caption("🔬 Técnicas: K-means Clustering, Geo-clustering IBGE, Análise Integrada")
