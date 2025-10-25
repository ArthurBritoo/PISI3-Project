import streamlit as st
import pandas as pd
import plotly.express as px

# Carregar o dataset
df = pd.read_parquet("data/itbi_2023.parquet")

# Converter colunas para numérico
df['valor_avaliacao'] = pd.to_numeric(df['valor_avaliacao'], errors='coerce')
df['area_construida'] = pd.to_numeric(df['area_construida'], errors='coerce')

# Filtrar apenas linhas válidas
df = df[(df['area_construida'] > 0) & (df['valor_avaliacao'] > 0)]

# Calcular preço do m²
df['preco_m2'] = df['valor_avaliacao'] / df['area_construida']

# Dicionário com médias FipeZap (R$/m²)
fipezap = {
    "SANTO AMARO": 8459,
    "BOA VIAGEM": 8241,
    "PARNAMIRIM": 7982,
    "MADALENA": 7922,
    "GRAÇAS": 7348,
    "TAMARINEIRA": 7293,
    "CASA AMARELA": 7038,
    "IMBIRIBEIRA": 6925,
    "ESPINHEIRO": 6435,
    "DERBY": 4876
}

# Calcular média e mediana do seu dataset por bairro
resultado = []
for bairro, media_fipe in fipezap.items():
    df_bairro = df[df['bairro'] == bairro]
    
    if not df_bairro.empty:
        # Remover outliers (5%-95%)
        q_low = df_bairro['preco_m2'].quantile(0.05)
        q_high = df_bairro['preco_m2'].quantile(0.95)
        df_bairro_filtrado = df_bairro[(df_bairro['preco_m2'] >= q_low) & (df_bairro['preco_m2'] <= q_high)]
        
        media_dataset = df_bairro_filtrado['preco_m2'].mean()
        mediana_dataset = df_bairro_filtrado['preco_m2'].median()
        discrepancia = media_dataset - media_fipe
        
        resultado.append({
            "Bairro": bairro,
            "Média Dataset (R$/m²)": media_dataset,
            "Mediana Dataset (R$/m²)": mediana_dataset,
            "Média FipeZap (R$/m²)": media_fipe,
            "Diferença (Dataset - FipeZap)": discrepancia
        })

# Criar DataFrame final
df_comparacao = pd.DataFrame(resultado)

st.title("Comparação do preço do m² por bairro: Dataset vs FipeZap")
st.dataframe(df_comparacao.style.format({
    "Média Dataset (R$/m²)": "R$ {:,.2f}",
    "Mediana Dataset (R$/m²)": "R$ {:,.2f}",
    "Média FipeZap (R$/m²)": "R$ {:,.2f}",
    "Diferença (Dataset - FipeZap)": "R$ {:,.2f}"
}))

# Opcional: gráfico comparando médias
fig = px.bar(
    df_comparacao,
    x="Bairro",
    y=["Média Dataset (R$/m²)", "Média FipeZap (R$/m²)"],
    barmode='group',
    title="Comparação de preço médio do m²: Dataset x FipeZap"
)
st.plotly_chart(fig)
