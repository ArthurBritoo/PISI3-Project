import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import t, norm
import altair as alt # Importar Altair para visualizações

st.set_page_config(page_title="Análise de Preços ITBI", layout="wide") # Usar layout wide para mais espaço

st.title("Intervalo de Confiança para Preço por m² • ITBI Boa Viagem")

# Função para carregar dados de um arquivo Parquet com cache
@st.cache_data
def load_data():
    """
    Carrega dados ITBI de um arquivo Parquet.
    Armazena os dados em cache para evitar recarregamento a cada execução.
    """
    try:
        df = pd.read_parquet("PISI3-Project/data/itbi_2023.parquet")
        return df
    except FileNotFoundError:
        st.error("Erro: Arquivo 'itbi_2023.parquet' não encontrado. Certifique-se de que o arquivo de dados esteja em 'PISI3-Project/data/'.")
        return None
    except Exception as e:
        st.error(f"Ocorreu um erro inesperado ao carregar o arquivo: {e}")
        return None

df = load_data()

if df is not None:
    # --- Limpeza Inicial de Dados e Conversão de Tipo ---
    required_columns = ['valor_avaliacao', 'area_construida', 'area_terreno', 'tipo_imovel', 'bairro']
    if not all(col in df.columns for col in required_columns):
        st.error(f"Erro: Colunas essenciais faltando no dataset. Esperadas: {required_columns}")
        st.stop()

    # Converter colunas para numérico, convertendo erros para NaN
    df['valor_avaliacao'] = pd.to_numeric(df['valor_avaliacao'], errors='coerce')
    df['area_construida'] = pd.to_numeric(df['area_construida'], errors='coerce')
    df['area_terreno'] = pd.to_numeric(df['area_terreno'], errors='coerce')

    # Remover linhas onde colunas numéricas essenciais são NaN após a conversão
    df.dropna(subset=['valor_avaliacao', 'area_construida', 'area_terreno'], inplace=True)

    # --- NOVA VERIFICAÇÃO: Parar se o DataFrame estiver vazio após o dropna inicial ---
    if df.empty:
        st.error("Erro: Nenhum dado válido restante após a limpeza inicial de valores ausentes/inválidos nas colunas essenciais (valor_avaliacao, area_construida, area_terreno). Por favor, verifique seus dados.")
        st.stop()

    # --- Filtro de Bairro (Aplicado a todo o DataFrame no início) ---
    st.sidebar.header("Filtros Globais")
    all_bairros = df['bairro'].unique().tolist()
    selected_bairros = st.sidebar.multiselect(
        "Selecionar Bairro(s)",
        options=all_bairros,
        default=[] # Alterado de all_bairros para []
    )

    if not selected_bairros:
        st.warning("Por favor, selecione pelo menos um bairro para análise.")
        st.stop()

    df = df[df['bairro'].isin(selected_bairros)].copy() # Aplicar filtro de bairro ao df principal

    if df.empty:
        st.warning("Nenhum dado restante após aplicar o filtro de bairro. Por favor, ajuste suas seleções.")
        st.stop()

    # --- Categorizar Tipos de Propriedade ---
    df['tipo_agrupado'] = 'Outros' # Categoria padrão

    # Classificar 'Terrenos' primeiro com base nos critérios de área
    # Garantir que area_terreno não seja zero para evitar divisão por zero na razão
    is_terrain_by_area = (
        (df['area_construida'] <= 10) |
        ((df['area_terreno'] > 0) & (df['area_construida'] / df['area_terreno'] <= 0.05))
    )
    df.loc[is_terrain_by_area, 'tipo_agrupado'] = 'Terreno'

    # Classificar 'Apartamento' e 'Casa'
    df.loc[df['tipo_imovel'] == 'Apartamento', 'tipo_agrupado'] = 'Apartamento'
    df.loc[df['tipo_imovel'] == 'Casa', 'tipo_agrupado'] = 'Casa'

    # --- Calcular 'value_m2' com base no tipo agrupado ---
    df['value_m2'] = np.nan # Inicializar com NaN

    # Máscara para 'Apartamento' válido com base nos limites de area_construida
    valid_apartment_area_mask = (
        (df['tipo_agrupado'] == 'Apartamento') &
        (df['area_construida'] >= 25) &
        (df['area_construida'] <= 350)
    )

    # Máscara para 'Casa' (sem limites específicos de area_construida ainda)
    valid_casa_mask = (df['tipo_agrupado'] == 'Casa')

    # Combinar máscaras para cálculo residencial
    residential_mask = (
        (valid_apartment_area_mask | valid_casa_mask) &
        (df['area_construida'] > 0) &
        (df['valor_avaliacao'].notna())
    )
    df.loc[residential_mask, 'value_m2'] = df['valor_avaliacao'] / df['area_construida']

    # Para Terrenos: valor_avaliacao / area_terreno
    terrain_mask = (
        (df['tipo_agrupado'] == 'Terreno') &
        (df['area_terreno'] > 0) &
        (df['valor_avaliacao'].notna())
    )
    df.loc[terrain_mask, 'value_m2'] = df['valor_avaliacao'] / df['area_terreno']

    st.success("Imóveis categorizados e 'value_m2' calculado por tipo, com filtragem de área para apartamentos.")

    st.subheader("Pré-visualização de Dados Brutos (com novas colunas)")
    st.write(df.head())
    st.write(f"Contagem por tipo agrupado: {df['tipo_agrupado'].value_counts().to_dict()}")

    st.divider()

    # --- UI do Streamlit para Filtragem e Análise ---
    st.sidebar.header("Opções de Análise por Tipo de Imóvel") # Renomeado para clareza
    selected_type = st.sidebar.selectbox(
        "Selecionar Tipo de Imóvel para Análise",
        ('Todos', 'Apartamento', 'Casa', 'Terreno'),
        index=0
    )

    # Filtrar dados com base no tipo selecionado para análise posterior
    if selected_type != 'Todos':
        df_filtered_by_type = df[df['tipo_agrupado'] == selected_type].copy()
    else:
        df_filtered_by_type = df[df['tipo_agrupado'].isin(['Apartamento', 'Casa', 'Terreno'])].copy()

    # --- Scatter Plot: Área Construída vs. Preço Total ---
    # Preparar dados para o scatter plot: garantir que area_construida e valor_avaliacao sejam válidos e positivos
    # Agora, também aplicar filtros de área específicos para apartamentos se 'Apartamento' for selecionado
    df_plot_scatter = df_filtered_by_type.copy()

    if selected_type == 'Apartamento':
        df_plot_scatter = df_plot_scatter[
            (df_plot_scatter['area_construida'] >= 25) &
            (df_plot_scatter['area_construida'] <= 350)
        ]

    df_plot_scatter = df_plot_scatter.dropna(subset=['area_construida', 'valor_avaliacao']).copy()
    df_plot_scatter = df_plot_scatter[
        (df_plot_scatter['area_construida'] > 0) &
        (df_plot_scatter['valor_avaliacao'] > 0)
    ]

    if not df_plot_scatter.empty:
        st.subheader(f"Gráfico de Dispersão: Área Construída vs. Preço Total para {selected_type}s em {', '.join(selected_bairros)}")
        scatter_chart = alt.Chart(df_plot_scatter).mark_point().encode(
            x=alt.X('area_construida', title='Área Construída (m²)'),
            y=alt.Y('valor_avaliacao', title='Preço Total (R$)'),
            tooltip=['tipo_imovel', 'bairro', 'area_construida', 'valor_avaliacao', 'value_m2'] # Adicionado value_m2 ao tooltip
        ).properties(
            title=f"Área Construída vs. Preço Total para {selected_type}s em {', '.join(selected_bairros)}"
        ).interactive() # Tornar gráfico interativo para zoom/pan
        st.altair_chart(scatter_chart, use_container_width=True)
        st.divider()
    else:
        st.info(f"Nenhum dado válido para o gráfico de dispersão (Área Construída vs. Preço Total) para {selected_type}s em {', '.join(selected_bairros)} após a limpeza inicial.")

    # Limpar dados para análise de value_m2 (esta parte permanece como está, agindo na coluna value_m2)
    data_for_analysis = df_filtered_by_type['value_m2'].replace([np.inf, -np.inf], np.nan).dropna()
    data_for_analysis = data_for_analysis[data_for_analysis > 0]

    if data_for_analysis.empty and selected_type != 'Todos': # Verificar se os dados estão vazios *após* a seleção de tipo
        st.warning(f"Nenhum dado válido para '{selected_type}' em {', '.join(selected_bairros)} após a limpeza inicial. Não é possível prosseguir com a análise.")
        st.stop()
    elif data_for_analysis.empty and selected_type == 'Todos':
        st.warning(f"Nenhum dado válido para nenhum tipo de propriedade selecionado em {', '.join(selected_bairros)} após a limpeza inicial. Não é possível prosseguir com a análise.")
        st.stop()


    # --- Nova Seção: Análise de Área Construída para Apartamentos (somente se 'Apartamento' for selecionado) ---
    if selected_type == 'Apartamento':
        st.subheader(f"Análise de Área Construída de Apartamentos (Antes da Filtragem de Preço/m²) em {', '.join(selected_bairros)}")
        # Filtrar df original para apartamentos antes do cálculo de value_m2, mas após os limites de área iniciais
        apartment_area = df[
            (df['tipo_agrupado'] == 'Apartamento') &
            (df['area_construida'] >= 25) &
            (df['area_construida'] <= 350)
        ]['area_construida'].copy()

        apartment_area = apartment_area.dropna()[apartment_area > 0] # Limpar dados de área

        if not apartment_area.empty:
            st.write("Estatísticas Descritivas para Área Construída de Apartamentos (m²):")
            st.write(apartment_area.describe())

            # Histograma da Área Construída para Apartamentos
            chart_area = alt.Chart(pd.DataFrame({'area_construida': apartment_area})).mark_bar().encode(
                alt.X('area_construida', bin=alt.Bin(maxbins=50), title="Área Construída (m²)"),
                alt.Y('count()', title="Número de Apartamentos")
            ).properties(title=f"Histograma da Área Construída de Apartamentos (Após filtragem inicial) em {', '.join(selected_bairros)}")
            st.altair_chart(chart_area, use_container_width=True)
        else:
            st.info(f"Nenhum dado válido de área construída encontrado para apartamentos após a limpeza inicial baseada em área em {', '.join(selected_bairros)}.")
        st.divider()

    # --- NOVO GRÁFICO: Barras Empilhadas de Preço por m² por Faixa de Área Construída (APARTAMENTOS) ---
    if selected_type == 'Apartamento' and not df_plot_scatter.empty:
        st.subheader(f"Gráfico de Barras Empilhadas: Preço por m² por Faixa de Área Construída para {selected_type}s em {', '.join(selected_bairros)}")

        # Criar faixas para Preço por m²
        price_m2_bins = [0, 2000, 4000, 6000, 8000, 10000, 12000, 15000, 20000, np.inf] # Ajustar conforme necessário
        price_m2_labels = ['0-2k', '2k-4k', '4k-6k', '6k-8k', '8k-10k', '10k-12k', '12k-15k', '15k-20k', '>20k']
        df_plot_scatter['price_m2_band'] = pd.cut(df_plot_scatter['value_m2'], bins=price_m2_bins, labels=price_m2_labels, right=False)

        # Criar faixas para Área Construída
        area_bins = [0, 50, 80, 120, 180, 250, 350, np.inf] # Ajustar conforme necessário
        area_labels = ['<50m²', '50-80m²', '80-120m²', '120-180m²', '180-250m²', '250-350m²', '>350m²']
        df_plot_scatter['area_band'] = pd.cut(df_plot_scatter['area_construida'], bins=area_bins, labels=area_labels, right=False)

        # Garantir que as colunas de faixas sejam categóricas para ordenação
        df_plot_scatter['price_m2_band'] = pd.Categorical(df_plot_scatter['price_m2_band'], categories=price_m2_labels, ordered=True)
        df_plot_scatter['area_band'] = pd.Categorical(df_plot_scatter['area_band'], categories=area_labels, ordered=True)


        stacked_bar_chart = alt.Chart(df_plot_scatter).mark_bar().encode(
            x=alt.X('price_m2_band', title='Preço por m² (R$)', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('count()', title='Número de Apartamentos'),
            color=alt.Color('area_band', title='Faixa de Área Construída', sort=area_labels), # Empilhar por faixa de área
            tooltip=[
                alt.Tooltip('price_m2_band', title='Faixa Preço/m²'),
                alt.Tooltip('area_band', title='Faixa Área Const.'),
                alt.Tooltip('count()', title='Qtd. Apartamentos')
            ]
        ).properties(
            title=f"Distribuição de Apartamentos por Preço/m² e Área Construída em {', '.join(selected_bairros)}"
        ).interactive()
        st.altair_chart(stacked_bar_chart, use_container_width=True)
        st.divider()
    elif selected_type == 'Apartamento':
        st.info(f"Nenhum dado válido para gerar o gráfico de barras empilhadas para apartamentos em {', '.join(selected_bairros)}.")
        st.divider()


    # Definir faixas de filtro padrão com base no tipo selecionado
    if selected_type == 'Apartamento':
        default_min, default_max = 2500.0, 12000.0
    elif selected_type == 'Casa':
        default_min, default_max = 1200.0, 8000.0
    elif selected_type == 'Terreno':
        default_min, default_max = 500.0, 5000.0 # Sugestão inicial para terrenos
    else: # Todos
        default_min, default_max = float(data_for_analysis.min()), float(data_for_analysis.max())

    st.subheader(f"Limpeza e Filtragem de Dados para {selected_type}s em {', '.join(selected_bairros)}")

    # --- Visualização da distribuição de 'value_m2' ---
    st.subheader(f"Distribuição de Preço por m² para {selected_type}s em {', '.join(selected_bairros)}")
    chart = alt.Chart(pd.DataFrame({'value_m2': data_for_analysis})).mark_bar().encode(
        alt.X('value_m2', bin=alt.Bin(maxbins=50), title="Preço por m² (R$)"),
        alt.Y('count()', title="Número de Propriedades")
    ).properties(title=f"Histograma de Preço por m² para {selected_type}s (Antes da Filtragem Personalizada) em {', '.join(selected_bairros)}")
    st.altair_chart(chart, use_container_width=True)

    st.markdown("--- ")
    st.subheader("Filtragem Personalizada")
    st.write("Você pode refinar os dados definindo faixas de preço personalizadas ou ajustando a remoção de outliers.")

    # Filtro de faixa de preço personalizado
    min_price_m2 = st.sidebar.number_input("Preço Mínimo por m² (R$)", min_value=0.0, value=default_min, format="%.2f")
    max_price_m2 = st.sidebar.number_input("Preço Máximo por m² (R$)", min_value=0.0, value=default_max, format="%.2f")

    data_filtered_by_range = data_for_analysis[
        (data_for_analysis >= min_price_m2) & (data_for_analysis <= max_price_m2)
    ]

    # Opção de remoção de outliers IQR (agora aplicada *após* a faixa personalizada, se houver)
    remove_outliers_iqr = st.sidebar.checkbox("Remover outliers usando IQR (Intervalo Interquartil)", value=True)
    iqr_factor = st.sidebar.slider("Fator IQR (para remoção de outliers IQR)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)

    data = data_filtered_by_range.copy() # Começar com dados potencialmente filtrados por faixa personalizada

    if remove_outliers_iqr and not data.empty:
        Q1 = data.quantile(0.25)
        Q3 = data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - iqr_factor * IQR
        upper_bound = Q3 + iqr_factor * IQR
        data = data[(data >= lower_bound) & (data <= upper_bound)]
        st.info(f"Outliers removidos usando fator IQR {iqr_factor}: Dados filtrados entre {lower_bound:,.2f} e {upper_bound:,.2f} R$/m².")
    elif remove_outliers_iqr and data.empty:
         st.warning("Nenhum dado restante para aplicar a remoção de outliers IQR após a filtragem de faixa personalizada.")

    if data.empty:
        st.warning(f"Nenhum dado válido restante após a filtragem para calcular o intervalo de confiança em {', '.join(selected_bairros)}.")
        st.stop()

    # --- Cálculos Estatísticos ---
    n = len(data)
    mean = np.mean(data)
    median = np.median(data) # Calcular mediana
    std = np.std(data, ddof=1)
    se = std / np.sqrt(n) # Erro Padrão

    st.subheader(f"Resultados Estatísticos para {selected_type}s em {', '.join(selected_bairros)} (Após Filtragem)")
    st.write(f"Tamanho da amostra (após limpeza e filtragem): **{n}**")
    st.write(f"Preço Médio (R$/m²): **{mean:,.2f}**")
    st.write(f"Preço Mediano (R$/m²): **{median:,.2f}**") # Exibir mediana
    st.write(f"Desvio Padrão: **{std:,.2f}**")
    st.write(f"Erro Padrão da Média: **{se:,.2f}**")

    # --- Cálculo do Intervalo de Confiança ---
    confidence_level = 0.95
    alpha = 1 - confidence_level

    if n >= 30: # Heurística para usar a distribuição Z para grandes amostras
        # Z-score para um intervalo de duas caudas
        z_critical = norm.ppf(1 - alpha / 2)
        margin_of_error = z_critical * se
        distribution_used = "Distribuição Z"
    else: # Usar a distribuição t de Student para amostras menores
        # Valor t para um intervalo de duas caudas com n-1 graus de liberdade
        t_critical = t.ppf(1 - alpha / 2, df=n - 1)
        margin_of_error = t_critical * se
        distribution_used = "Distribuição t de Student"

    lower_ci = mean - margin_of_error
    upper_ci = mean + margin_of_error

    st.markdown(f"\n**{int(confidence_level*100)}% Intervalo de Confiança ({distribution_used}) para {selected_type}s em {', '.join(selected_bairros)}**")
    st.success(f"**{lower_ci:,.2f} R$/m²**  até  **{upper_ci:,.2f} R$/m²**")

    st.divider()

    # --- SEÇÃO DE COMPARAÇÃO: Filtros Fixos para Boa Viagem ---
    st.header("Comparação: Filtros Fixos para Boa Viagem (Apartamentos)")
    st.write("Esta seção mostra os resultados para um conjunto predefinido de filtros para oferecer um ponto de comparação.")

    df_bv_comparison = df.copy() # Começar com uma cópia fresca do df principal após a limpeza inicial e o cálculo de value_m2

    # Aplicar os filtros fixos
    df_bv_comparison = df_bv_comparison[
        (df_bv_comparison['bairro'] == 'BOA VIAGEM') &
        (df_bv_comparison['area_construida'] >= 60) &
        (df_bv_comparison['area_construida'] <= 200) &
        (df_bv_comparison['valor_avaliacao'] >= 350000) &
        (df_bv_comparison['value_m2'] >= 3500) &
        (df_bv_comparison['value_m2'] <= 15000)
    ]

    # Limpar a coluna 'value_m2' para este subconjunto de comparação
    data_bv_comparison = df_bv_comparison['value_m2'].replace([np.inf, -np.inf], np.nan).dropna()
    data_bv_comparison = data_bv_comparison[data_bv_comparison > 0]

    # Preparar strings formatadas para report_content para evitar SyntaxError
    n_bv_str = str(len(data_bv_comparison)) if not data_bv_comparison.empty else 'N/A'
    mean_bv_str = f"{np.mean(data_bv_comparison):,.2f}" if not data_bv_comparison.empty else 'N/A'
    median_bv_str = f"{np.median(data_bv_comparison):,.2f}" if not data_bv_comparison.empty else 'N/A'
    std_bv_str = f"{np.std(data_bv_comparison, ddof=1):,.2f}" if not data_bv_comparison.empty else 'N/A'
    se_bv_str = f"{np.std(data_bv_comparison, ddof=1) / np.sqrt(len(data_bv_comparison)):,.2f}" if not data_bv_comparison.empty else 'N/A'

    lower_ci_bv_str = 'N/A'
    upper_ci_bv_str = 'N/A'
    dist_used_bv_str = 'N/A'

    if not data_bv_comparison.empty:
        n_bv = len(data_bv_comparison)
        mean_bv = np.mean(data_bv_comparison)
        median_bv = np.median(data_bv_comparison)
        std_bv = np.std(data_bv_comparison, ddof=1)
        se_bv = std_bv / np.sqrt(n_bv)

        lower_ci_bv = mean_bv - (norm.ppf((1 + confidence_level) / 2) * se_bv if n_bv >= 30 else t.ppf((1 + confidence_level) / 2, df=n_bv - 1) * se_bv)
        upper_ci_bv = mean_bv + (norm.ppf((1 + confidence_level) / 2) * se_bv if n_bv >= 30 else t.ppf((1 + confidence_level) / 2, df=n_bv - 1) * se_bv)
        dist_used_bv = "Distribuição Z" if n_bv >= 30 else "Distribuição t de Student"

        lower_ci_bv_str = f"{lower_ci_bv:,.2f}"
        upper_ci_bv_str = f"{upper_ci_bv:,.2f}"
        dist_used_bv_str = dist_used_bv

        st.write(f"Pontos de Dados Filtrados: **{n_bv}**")
        st.write(f"Preço Médio (R$/m²): **{mean_bv:,.2f}**")
        st.write(f"Preço Mediano (R$/m²): **{median_bv:,.2f}**")
        st.write(f"Desvio Padrão: **{std_bv:,.2f}**")
        st.write(f"Erro Padrão: **{se_bv:,.2f}**")

        st.markdown(f"\n**{int(confidence_level*100)}% Intervalo de Confiança ({dist_used_bv})**")
        st.success(f"**{lower_ci_bv:,.2f} R$/m²**  até  **{upper_ci_bv:,.2f} R$/m²**")
    else:
        st.warning("Nenhum dado restante para os filtros fixos de comparação de Boa Viagem.")

    st.divider()

    # --- Download do Relatório ---
    report_content = (
        f"Relatório de Intervalo de Confiança para Preço por m² ({selected_type}s em {', '.join(selected_bairros)})\n"
        f"---------------------------------------------\n"
        f"Tamanho da amostra (após limpeza e filtragem): {n}\n"
        f"Preço Médio (R$/m²): {mean:,.2f}\n"
        f"Preço Mediano (R$/m²): {median:,.2f}\n"
        f"Desvio Padrão: {std:,.2f}\n"
        f"Erro Padrão da Média: {se:,.2f}\n"
        f"{int(confidence_level*100)}% Intervalo de Confiança: {lower_ci:,.2f} até {upper_ci:,.2f} R$/m²\n"
        f"Distribuição utilizada: {distribution_used}\n"
        f"Outliers removidos (Fator IQR {iqr_factor}): {'Sim' if remove_outliers_iqr else 'Não'}\n"
        f"Faixa de preço personalizada aplicada: {min_price_m2:,.2f} até {max_price_m2:,.2f} R$/m²\n"
        f"Bairros: {', '.join(selected_bairros)}\n"
        f"\n--- Seção de Comparação (Filtros Fixos para Boa Viagem) ---\n"
        f"Pontos de Dados Filtrados: {n_bv_str}\n"
        f"Preço Médio (R$/m²): {mean_bv_str}\n"
        f"Preço Mediano (R$/m²): {median_bv_str}\n"
        f"Desvio Padrão: {std_bv_str}\n"
        f"Erro Padrão: {se_bv_str}\n"
        f"{int(confidence_level*100)}% Intervalo de Confiança: {lower_ci_bv_str} até {upper_ci_bv_str} R$/m²\n"
        f"Distribuição utilizada: {dist_used_bv_str}\n"
    )

    st.download_button(
        label="Baixar Relatório do IC",
        data=report_content,
        file_name=f"relatorio_intervalo_confianca_{selected_type.lower()}_{'_'.join(selected_bairros)}.txt",
        mime="text/plain"
    )
else:
    st.warning("Aguardando o carregamento do dataset ou ocorreu um erro. Por favor, verifique o console para detalhes.")
