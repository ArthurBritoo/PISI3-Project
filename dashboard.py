import os
import streamlit as st
import pandas as pd
import plotly.express as px

from data_processing import load_and_preprocess_data
from geo_clustering import build_regions_for_recife  # <- novo

st.set_page_config(page_title="Dashboard ITBI Recife - Residencial", layout="wide")

@st.cache_data(show_spinner=False)
def get_data():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    return load_and_preprocess_data(data_dir=data_dir)

# Cache para construção das regiões (evita chamar a API do IBGE a cada interação)
@st.cache_data(show_spinner=True)
def build_regions_cached(df_in: pd.DataFrame, min_tx: int):
    df_out, regions = build_regions_for_recife(df_in, min_tx_per_region=min_tx)
    return df_out, regions

df = get_data().copy()

# Apenas tipos residenciais
residential_types = [t for t in ["Apartamento", "Casa"] if t in df["tipo_imovel"].unique()]
df = df[df["tipo_imovel"].isin(residential_types)].copy()

st.title("Dashboard ITBI Recife - Residencial (2015-2023)")
st.caption("Dados filtrados para imóveis residenciais (Apartamento, Casa)")

# ---------------- Sidebar: Filtros ----------------
st.sidebar.header("Filtros")

# Controle de regiões (mínimo de transações por região) e construção das regiões
min_tx_per_region = st.sidebar.number_input(
    "Mínimo de transações por região (IBGE)",
    min_value=50, max_value=2000, value=200, step=50,
    help="Agrupa subdistritos vizinhos até alcançar este mínimo por região."
)
df_reg, regions_dict = build_regions_cached(df, int(min_tx_per_region))

# Período (anos)
anos = sorted(df_reg["data_transacao"].dt.year.dropna().unique().tolist())
min_ano, max_ano = (anos[0], anos[-1]) if anos else (2015, 2023)
ano_range = st.sidebar.slider("Período (Ano)", min_ano, max_ano, (min_ano, max_ano), step=1)

# Tipos residenciais
tipos_sel = st.sidebar.multiselect(
    "Tipos de imóvel (Residencial)",
    residential_types,
    default=residential_types,
)

# Filtro por Regiões (substitui filtro por bairro)
regioes = sorted(df_reg["regiao"].dropna().unique().tolist())
regioes_sel = st.sidebar.multiselect(
    "Regiões (IBGE agrupadas)",
    regioes,
    default=[],
    help="Deixe vazio para considerar todas as regiões"
)

# Faixas (área e valor)
area_min = float(df_reg["area_construida"].quantile(0.01))
area_max = float(df_reg["area_construida"].quantile(0.99))
valor_min = float(df_reg["valor_avaliacao"].quantile(0.01))
valor_max = float(df_reg["valor_avaliacao"].quantile(0.99))

area_range = st.sidebar.slider(
    "Área construída (m²)",
    min_value=float(max(0.0, area_min)),
    max_value=float(area_max),
    value=(float(area_min), float(area_max)),
)
valor_range = st.sidebar.slider(
    "Valor de avaliação (R$)",
    min_value=float(max(0.0, valor_min)),
    max_value=float(valor_max),
    value=(float(valor_min), float(valor_max)),
)

# Filtros: Tipo e Ano de Construção (se existirem nas colunas)
tipos_constr_sel = None
if "tipo_construcao" in df_reg.columns:
    tipos_constr = sorted(df_reg["tipo_construcao"].dropna().unique().tolist())
    if tipos_constr:
        tipos_constr_sel = st.sidebar.multiselect(
            "Tipo de construção",
            tipos_constr,
            default=tipos_constr,
        )

ano_constr_range = None
if "ano_construcao" in df_reg.columns:
    anos_constr = pd.to_numeric(df_reg["ano_construcao"], errors="coerce").dropna().astype(int)
    if not anos_constr.empty:
        min_ano_c, max_ano_c = int(anos_constr.min()), int(anos_constr.max())
        ano_constr_range = st.sidebar.slider(
            "Ano de construção",
            min_value=min_ano_c,
            max_value=max_ano_c,
            value=(min_ano_c, max_ano_c),
            step=1,
        )

# Controles do gráfico: agregação e Top N
col_aggr, col_top = st.sidebar.columns(2)
with col_aggr:
    aggr_label = st.radio("Agregação m²", ["Mediana", "Média"], index=0, horizontal=True)
aggr_func = "median" if aggr_label == "Mediana" else "mean"
with col_top:
    top_n = st.selectbox("Top N", [10, 20, 30], index=0)

# ---------------- Aplicação de filtros ----------------
agrupador = "regiao"  # sempre por região
mask = (
    df_reg["data_transacao"].dt.year.between(ano_range[0], ano_range[1]) &
    df_reg["tipo_imovel"].isin(tipos_sel) &
    df_reg["area_construida"].between(area_range[0], area_range[1]) &
    df_reg["valor_avaliacao"].between(valor_range[0], valor_range[1])
)

if regioes_sel:
    mask &= df_reg["regiao"].isin(regioes_sel)

if tipos_constr_sel is not None and len(tipos_constr_sel) > 0 and "tipo_construcao" in df_reg.columns:
    mask &= df_reg["tipo_construcao"].isin(tipos_constr_sel)

if ano_constr_range is not None and "ano_construcao" in df_reg.columns:
    ano_col = pd.to_numeric(df_reg["ano_construcao"], errors="coerce")
    mask &= ano_col.between(ano_constr_range[0], ano_constr_range[1])

dff = df_reg[mask].copy()

if dff.empty:
    st.warning("Nenhum dado encontrado com os filtros selecionados.")
    st.stop()

# Mostrar composição das regiões
with st.expander("Ver composição das regiões (subdistritos IBGE)"):
    reg_view = pd.DataFrame([
        {"regiao": k, "subdistritos": ", ".join(sorted(v))}
        for k, v in regions_dict.items()
    ]).sort_values("regiao")
    st.dataframe(reg_view, use_container_width=True, hide_index=True)

# ---------------- KPIs ----------------
def moeda(v):
    try:
        return f"R$ {v:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    except Exception:
        return str(v)

kpi1 = len(dff)
kpi2 = dff["valor_avaliacao"].mean()
kpi3 = dff["valor_m2"].median() if aggr_func == "median" else dff["valor_m2"].mean()
kpi4 = dff["area_construida"].median()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transações", f"{kpi1:,}".replace(",", "."))
c2.metric("Ticket médio", moeda(kpi2))
c3.metric(f"Valor m² ({aggr_label})", moeda(kpi3))
c4.metric("Área construída (mediana)", f"{kpi4:,.0f} m²".replace(",", ".").replace(".", ",").replace(",", ".", 1))

st.divider()

# ---------------- Gráficos ----------------

# 1) Top Regiões por valor m² (agregação escolhida)
st.subheader(f"1. Valor do Metro Quadrado por Região — Top {top_n} ({aggr_label})")
g1 = (
    dff.groupby(agrupador)["valor_m2"]
    .agg(aggr_func)
    .reset_index()
    .sort_values("valor_m2", ascending=False)
    .head(top_n)
)
fig1 = px.bar(
    g1, x=agrupador, y="valor_m2",
    title=f"Top {top_n} Regiões por Valor do m² ({aggr_label})",
    labels={agrupador: "Região", "valor_m2": "Valor do m² (R$)"},
    color="valor_m2",
    color_continuous_scale=px.colors.sequential.Viridis,
)
fig1.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig1, use_container_width=True)

# 2) Quantidade de Transações por Região (Top N)
st.subheader(f"2. Quantidade de Transações por Região — Top {top_n}")
g2 = (
    dff.groupby(agrupador)
      .size()
      .reset_index(name="qtd_transacoes")
      .sort_values("qtd_transacoes", ascending=False)
      .head(top_n)
)
fig2 = px.bar(
    g2, x=agrupador, y="qtd_transacoes",
    title=f"Transações por Região (Top {top_n})",
    labels={agrupador: "Região", "qtd_transacoes": "Quantidade de Transações"},
    color="qtd_transacoes",
    color_continuous_scale=px.colors.sequential.Blues,
)
fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

# 3) Valor da Transação por Padrão de Acabamento (mesma agregação)
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

# 4) Evolução do Valor do m² por Ano (agregação)
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

# 5) Distribuição de Tipos (residenciais)
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

# 6) Dispersão Valor vs Área (opcional)
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
st.download_button("Baixar CSV (filtro atual)", data=csv, file_name="itbi_residencial_filtrado.csv", mime="text/csv")