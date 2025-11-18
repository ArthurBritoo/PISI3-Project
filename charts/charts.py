import plotly.express as px
import pandas as pd

def plot_valor_m2_por_bairro(
    df: pd.DataFrame,
    tipo_agregacao: str = "median",
    top_n: int = 20
):
    """
    Retorna um gráfico de barras com a média ou mediana do valor por m²
    por bairro, permitindo escolher o tipo de agregação e o número de bairros.

    Parâmetros:
        df: DataFrame já pré-processado contendo colunas `tipo_imovel`,
            `bairro` e `valor_m2`.
        tipo_agregacao: 'mean' ou 'median' para definir a agregação.
        top_n: Número de bairros a mostrar (ex: 10 ou 20).

    Retorno:
        plotly.graph_objs.Figure pronto para exibir.
    """
    # Filtra apenas apartamentos e casas
    df_filtered = df[df["tipo_imovel"].isin(["Apartamento", "Casa"])]

    # Escolhe agregação
    if tipo_agregacao == "mean":
        df_grouped = df_filtered.groupby("bairro")["valor_m2"].mean().reset_index()
        titulo = f"Média do Valor do Metro Quadrado por Bairro (Top {top_n})"
        ylabel = "Valor do m² (R$) - Média"
    else:
        df_grouped = df_filtered.groupby("bairro")["valor_m2"].median().reset_index()
        titulo = f"Mediana do Valor do Metro Quadrado por Bairro (Top {top_n})"
        ylabel = "Valor do m² (R$) - Mediana"

    # Ordena e pega os Top N bairros
    df_grouped = df_grouped.sort_values(by="valor_m2", ascending=False).head(top_n)

    fig = px.bar(
        df_grouped,
        x="bairro",
        y="valor_m2",
        title=titulo,
        labels={"bairro": "Bairro", "valor_m2": ylabel},
        color="valor_m2",
        color_continuous_scale=px.colors.sequential.Viridis,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_qtd_transacoes_por_bairro(df: pd.DataFrame):
    """Gera um gráfico de barras com a quantidade de transações por bairro.

    Detalhes:
    - Filtra imóveis comparáveis (Apartamento, Casa).
    - Usa `value_counts()` em `bairro` para contar transações por bairro.=
    """
    # Filtra para manter comparabilidade entre tipos
    df_filtered = df[df["tipo_imovel"].isin(["Apartamento", "Casa"])]
    # Conta as ocorrências por bairro e transforma em DataFrame
    df_grouped = df_filtered["bairro"].value_counts().reset_index()
    df_grouped.columns = ["bairro", "qtd_transacoes"]
    # Calcula o total de transações e os tipos presentes
    total_transacoes = df_grouped["qtd_transacoes"].sum()
    tipos_presentes = ", ".join(sorted(df_filtered["tipo_imovel"].unique()))
    # Adiciona legenda personalizada ao gráfico
    fig = px.bar(
        df_grouped,
        x="bairro",
        y="qtd_transacoes",
        title="Quantidade de Transações por Bairro",
        labels={"bairro": "Bairro", "qtd_transacoes": "Quantidade de Transações"},
        color="qtd_transacoes",
        color_continuous_scale=px.colors.sequential.Blues,
    )
    fig.update_layout(
        xaxis_tickangle=-45,
        legend_title_text=f"Total: {total_transacoes} transações | Tipos: {tipos_presentes}"
    )
    # Monta o gráfico com escala de cor para o count
    fig = px.bar(
        df_grouped,
        x="bairro",
        y="qtd_transacoes",
        title="Quantidade de Transações por Bairro (Top 20)",
        labels={"bairro": "Bairro", "qtd_transacoes": "Quantidade de Transações"},
        color="qtd_transacoes",
        color_continuous_scale=px.colors.sequential.Blues,
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def plot_valor_transacao_por_acabamento(df: pd.DataFrame):
    """Plota a mediana do valor da transação por padrão de acabamento.

    Observações:
    - Converte `padrao_acabamento` para Categorical com ordem semântica
      (Simples < Médio < Superior) para que a ordenação do gráfico faça
      sentido.
    - Calcula a mediana do valor de avaliação para cada categoria.
    """
    df_filtered = df[df["tipo_imovel"].isin(["Apartamento", "Casa"])]

    # Agrupa por padrão de acabamento calculando a média
    df_grouped = df_filtered.groupby("padrao_acabamento")["valor_avaliacao"].mean().reset_index()

    # Define a ordem desejada para as categorias de acabamento
    df_grouped["padrao_acabamento"] = pd.Categorical(
        df_grouped["padrao_acabamento"],
        categories=["Simples", "Médio", "Superior"],
        ordered=True,
    )
    df_grouped = df_grouped.sort_values(by="padrao_acabamento")

    fig = px.bar(
        df_grouped,
        x="padrao_acabamento",
        y="valor_avaliacao",
        title="Mediana do Valor da Transação por Padrão de Acabamento",
        labels={"padrao_acabamento": "Padrão de Acabamento", "valor_avaliacao": "Valor da Transação (R$)"},
        color="valor_avaliacao",
        color_continuous_scale=px.colors.sequential.Plasma,
    )
    return fig

def plot_valor_m2_por_ano(df: pd.DataFrame):
    """Desenha a evolução temporal (por ano) da mediana do valor do m².

    - Agrupa pela coluna `data_transacao` extraindo o ano via `.dt.year`.
    - Calcula a mediana anual de `valor_m2`.
    - Retorna um gráfico de linhas com marcadores.
    """
    df_filtered = df[df["tipo_imovel"].isin(["Apartamento", "Casa"])]

    # Agrupa por ano extraído da data de transação
    df_grouped = df_filtered.groupby(df["data_transacao"].dt.year)["valor_m2"].median().reset_index()
    df_grouped.columns = ["ano", "valor_m2"]

    fig = px.line(
        df_grouped,
        x="ano",
        y="valor_m2",
        title="Mediana do Valor do Metro Quadrado por Ano",
        labels={"ano": "Ano da Transação", "valor_m2": "Valor do m² (R$)"},
        markers=True,
    )

    # Força o eixo x para tratar os anos como categorias (evita valores decimais)
    fig.update_xaxes(type="category")
    return fig

def plot_tipo_imovel_distribuicao(df: pd.DataFrame):
    """Gera um gráfico de pizza com a distribuição de tipos de imóveis.

    Filtra um subconjunto de tipos (Apartamento, Casa, Sala, Loja) para
    evitar categorias esparsas e tornar o gráfico mais interpretável.
    """
    df_filtered = df[df["tipo_imovel"].isin(["Apartamento", "Casa", "Sala", "Loja"])]

    # Contagem por tipo de imóvel
    df_counts = df_filtered["tipo_imovel"].value_counts().reset_index()
    df_counts.columns = ["tipo_imovel", "count"]

    fig = px.pie(
        df_counts,
        values="count",
        names="tipo_imovel",
        title="Distribuição de Tipos de Imóveis",
        hole=0.3,
    )
    return fig