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

st.title("Análise Exploratória do Mercado Imobiliário de Recife (ITBI 2015-2023)")
st.markdown("Esta aplicação apresenta uma análise exploratória dos dados do Imposto sobre a Transmissão de Bens Imóveis (ITBI) de Recife, abrangendo o período de 2015 a 2023.")

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