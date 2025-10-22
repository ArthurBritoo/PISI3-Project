import os
import streamlit as st
import pandas as pd
import plotly.express as px

from data_processing import load_and_preprocess_data
from geo_clustering import build_regions_for_recife

st.set_page_config(page_title="Dashboard ITBI Recife - Análises Complementares", layout="wide")

# ------------------ Cache de dados ------------------
@st.cache_data(show_spinner=False)
def get_data():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    return load_and_preprocess_data(data_dir=data_dir)

df = get_data().copy()

# Filtrar apenas residenciais
df = df[df["tipo_imovel"].isin(["Apartamento", "Casa"])]

st.title("📊 Dashboard ITBI Recife - Análises Complementares")
st.caption("Explorando relações adicionais entre características e valores de imóveis residenciais (2015–2023).")

# ------------------ Sidebar: Filtros ------------------
st.sidebar.header("Filtros")

anos = sorted(df["data_transacao"].dt.year.dropna().unique().tolist())
ano_range = st.sidebar.slider("Período", min(anos), max(anos), (min(anos), max(anos)))

mask = df["data_transacao"].dt.year.between(ano_range[0], ano_range[1])
dff = df[mask].copy()

# ------------------ Gráfico 1 ------------------
st.subheader("1️⃣ Relação entre Ano de Construção e Valor do m²")

if "ano_construcao" in dff.columns:
    g1 = (
        dff.groupby("ano_construcao")["valor_m2"]
        .median()
        .reset_index()
        .dropna()
        .sort_values("ano_construcao")
    )
    fig1 = px.line(
        g1,
        x="ano_construcao",
        y="valor_m2",
        markers=True,
        title="Valor mediano do m² por ano de construção",
        labels={"ano_construcao": "Ano de construção", "valor_m2": "Valor do m² (R$)"},
    )
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.info("Dados de ano de construção não disponíveis.")

# ------------------ Gráfico 2 ------------------
st.subheader("2️⃣ Distribuição de Valores por Tipo de Imóvel")

fig2 = px.box(
    dff,
    x="tipo_imovel",
    y="valor_m2",
    color="tipo_imovel",
    points="all",
    title="Distribuição do valor do m² por tipo de imóvel",
    labels={"tipo_imovel": "Tipo de imóvel", "valor_m2": "Valor do m² (R$)"},
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------ Gráfico 3 ------------------
st.subheader("3️⃣ Correlação entre Área e Valor por Padrão de Acabamento")

if "padrao_acabamento" in dff.columns:
    fig3 = px.scatter(
        dff,
        x="area_construida",
        y="valor_avaliacao",
        color="padrao_acabamento",
        trendline="ols",
        hover_data=["regiao", "bairro"],
        title="Correlação entre área construída e valor de avaliação por padrão de acabamento",
        labels={
            "area_construida": "Área construída (m²)",
            "valor_avaliacao": "Valor de avaliação (R$)",
            "padrao_acabamento": "Padrão de acabamento",
        },
    )
    st.plotly_chart(fig3, use_container_width=True)
else:
    st.info("Dados de padrão de acabamento não disponíveis.")

# ------------------ Gráfico 4 ------------------
st.subheader("4️⃣ Evolução da Quantidade de Transações ao Longo do Tempo")

g4 = (
    dff.groupby(dff["data_transacao"].dt.to_period("Y"))
    .size()
    .reset_index(name="qtd_transacoes")
)
g4["data_transacao"] = g4["data_transacao"].astype(str)

fig4 = px.bar(
    g4,
    x="data_transacao",
    y="qtd_transacoes",
    title="Quantidade de transações por ano",
    labels={"data_transacao": "Ano", "qtd_transacoes": "Quantidade de transações"},
)
st.plotly_chart(fig4, use_container_width=True)

# ------------------ Gráfico 5 ------------------
st.subheader("5️⃣ Relação entre Região e Padrão de Acabamento")

if "padrao_acabamento" in dff.columns:
    g5 = (
        dff.groupby(["regiao", "padrao_acabamento"])
        .size()
        .reset_index(name="qtd")
    )
    fig5 = px.bar(
        g5,
        x="regiao",
        y="qtd",
        color="padrao_acabamento",
        title="Distribuição de padrão de acabamento por região",
        labels={"regiao": "Região", "qtd": "Quantidade"},
    )
    fig5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)
else:
    st.info("Não há dados de padrão de acabamento para essa análise.")

# ------------------ Download dos dados filtrados ------------------
st.divider()
csv = dff.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Baixar dados filtrados (CSV)",
    data=csv,
    file_name="itbi_residencial_analises_complementares.csv",
    mime="text/csv",
)
