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
    
    # Explicação dos perfis
    with st.expander("ℹ️ Entenda os 5 Perfis de Mercado"):
        st.markdown("""
        **Clustering K-means** agrupa imóveis por características similares:
        
        - 🏠 **Popular**: Menor valor, área compacta, padrão simples
        - 🏡 **Entrada**: Intermediário inferior, bom custo-benefício
        - 🏘️ **Intermediário**: Padrão médio, áreas moderadas
        - 🏙️ **Alto Padrão**: Imóveis maiores, acabamento superior
        - 💎 **Premium**: Elite do mercado, máximo valor e área
        """)
    
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
            min_value=50, max_value=2000, value=200, step=50
        )
        
        anos = sorted(df_res["data_transacao"].dt.year.dropna().unique().tolist())
        min_ano, max_ano = (anos[0], anos[-1]) if anos else (2015, 2023)
        ano_range = st.slider("Período", min_ano, max_ano, (min_ano, max_ano))
        
        aggr_label = st.radio("Agregação", ["Mediana", "Média"], index=0)
        aggr_func = "median" if aggr_label == "Mediana" else "mean"
        
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
    st.subheader(f"Top {top_n} Regiões por Valor m²")
    g1 = (
        dff.groupby("regiao")["valor_m2"]
        .agg(aggr_func)
        .reset_index()
        .sort_values("valor_m2", ascending=False)
        .head(top_n)
    )
    fig1 = px.bar(
        g1, x="regiao", y="valor_m2",
        title=f"Valor m² por Região ({aggr_label})",
        labels={"regiao": "Região", "valor_m2": "Valor m² (R$)"},
        color="valor_m2",
        color_continuous_scale=px.colors.sequential.Viridis
    )
    fig1.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig1, use_container_width=True)
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader(f"Top {top_n} por Quantidade")
        g2 = (
            dff.groupby("regiao")
            .size()
            .reset_index(name="qtd")
            .sort_values("qtd", ascending=False)
            .head(top_n)
        )
        fig2 = px.bar(
            g2, x="regiao", y="qtd",
            labels={"regiao": "Região", "qtd": "Transações"},
            color="qtd",
            color_continuous_scale=px.colors.sequential.Blues
        )
        fig2.update_layout(xaxis_tickangle=-45, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    with col_b:
        st.subheader("Distribuição por Padrão")
        g3 = (
            dff.groupby("padrao_acabamento")["valor_avaliacao"]
            .agg(aggr_func)
            .reset_index()
        )
        fig3 = px.bar(
            g3, x="padrao_acabamento", y="valor_avaliacao",
            labels={"padrao_acabamento": "Padrão", "valor_avaliacao": "Valor"},
            color="valor_avaliacao",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        st.plotly_chart(fig3, use_container_width=True)

# ==================== TAB 4: ANÁLISE INTEGRADA ====================

with tab4:
    st.header("🔥 Análise Integrada: Perfis × Regiões")
    st.markdown("**Inovação:** Cruzamento dos clusters de mercado com regiões geográficas IBGE")
    
    with st.sidebar:
        st.subheader("🔥 Filtros - Integrado")
        min_tx_integrated = st.number_input(
            "Mínimo tx/região (integrado)",
            min_value=50, max_value=2000, value=200, step=50,
            key="integrated_min_tx"
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
