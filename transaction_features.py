import pandas as pd

def add_transaction_volume_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona uma feature de volume de transações por bairro e ano ao DataFrame.

    Esta função calcula o número de transações imobiliárias para cada combinação
    de bairro e ano de venda e adiciona essa informação ao DataFrame original
    como uma nova coluna 'volume_transacoes_bairro_ano'.

    Args:
        df (pd.DataFrame): O DataFrame de entrada contendo as colunas 'bairro'
                           e 'data_transacao'.

    Returns:
        pd.DataFrame: O DataFrame atualizado com a coluna 'volume_transacoes_bairro_ano'.

    Raises:
        ValueError: Se as colunas 'bairro' ou 'data_transacao' estiverem ausentes
                    ou contiverem valores nulos.
    """
    print("\n=== Volume Transacional (Bairro x Ano) ===")

    # 3. Boas práticas: Tratamento leve de ausências
    if 'bairro' not in df.columns or df['bairro'].isnull().any():
        raise ValueError("A coluna 'bairro' está faltando ou contém valores nulos. Por favor, verifique seus dados.")
    if 'data_transacao' not in df.columns or df['data_transacao'].isnull().any():
        raise ValueError("A coluna 'data_transacao' está faltando ou contém valores nulos. Por favor, verifique seus dados.")
    
    # 3. Boas práticas: Não duplicar colunas (idempotente)
    if 'volume_transacoes_bairro_ano' in df.columns:
        print("A coluna 'volume_transacoes_bairro_ano' já existe. Removendo para reprocessamento idempotente.")
        df = df.drop(columns=['volume_transacoes_bairro_ano'])

    df_copy = df.copy() # Trabalhar com uma cópia para evitar SettingWithCopyWarning

    # 1. Se não existir ano_venda, deve extrair do campo data_venda
    if 'ano_transacao' not in df_copy.columns:
        df_copy['ano_transacao'] = pd.to_datetime(df_copy['data_transacao']).dt.year
        print("Coluna 'ano_transacao' extraída de 'data_transacao'.")
    else:
        print("Coluna 'ano_transacao' já existe.")

    # 1. Calcula quantas transações ocorreram por combinação (bairro, ano_transacao)
    # 3. Boas práticas: Não remover registros
    transaction_volume = df_copy.groupby(['bairro', 'ano_transacao']).size().reset_index(name='volume_transacoes_bairro_ano')
    print("Volume de transações por bairro e ano calculado.")

    # 1. Faz o merge dessa informação para cada linha do DataFrame original
    df_merged = pd.merge(df_copy, transaction_volume, on=['bairro', 'ano_transacao'], how='left')
    
    # Garantir que não estamos alterando outras features, apenas adicionando
    # E que o merge não removeu registros
    if len(df_merged) != len(df):
        print("Atenção: O número de registros mudou após o merge. Verifique a chave de merge.")

    # 1. Prints amigáveis
    print("Feature 'volume_transacoes_bairro_ano' adicionada com sucesso.")
    print(f"Registros após merge: {len(df_merged):,}")
    
    unique_bairros = df_merged['bairro'].nunique()
    unique_anos = df_merged['ano_transacao'].nunique()
    min_ano = df_merged['ano_transacao'].min()
    max_ano = df_merged['ano_transacao'].max()
    print(f"Bairros únicos: {unique_bairros} | Anos mapeados: {min_ano}-{max_ano}")
    
    # Limpeza da coluna temporária
    if 'ano_transacao' in df_merged.columns and 'ano_transacao' not in df.columns:
        df_merged = df_merged.drop(columns=['ano_transacao'])

    return df_merged

def get_transaction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera uma tabela agregada com o volume de transações por ano e por bairro.

    Esta função requer que a coluna 'volume_transacoes_bairro_ano' já esteja presente
    no DataFrame, idealmente adicionada pela função add_transaction_volume_feature.

    Args:
        df (pd.DataFrame): O DataFrame de entrada, contendo 'bairro', 'ano_transacao'
                           e 'volume_transacoes_bairro_ano'.

    Returns:
        pd.DataFrame: Uma tabela resumida com o volume de transações por bairro e ano.

    Raises:
        ValueError: Se as colunas 'bairro', 'ano_transacao' ou 'volume_transacoes_bairro_ano'
                    estiverem ausentes.
    """
    print("\n=== Resumo de Transações (Bairro x Ano) ===")

    if 'bairro' not in df.columns:
        raise ValueError("A coluna 'bairro' está faltando no DataFrame.")
    if 'data_transacao' not in df.columns and 'ano_transacao' not in df.columns:
         raise ValueError("As colunas 'data_transacao' ou 'ano_transacao' estão faltando no DataFrame.")
    
    # Assegurar que 'ano_transacao' existe para o resumo
    df_summary = df.copy()
    if 'ano_transacao' not in df_summary.columns:
        df_summary['ano_transacao'] = pd.to_datetime(df_summary['data_transacao']).dt.year

    # Gera a tabela agregada
    # Usamos .drop_duplicates() para somar o volume de transacoes que já foi mergeado
    # evitando contar várias vezes o mesmo volume para cada linha do DF original
    summary_table = df_summary[['bairro', 'ano_transacao', 'volume_transacoes_bairro_ano']].drop_duplicates()
    summary_table = summary_table.groupby(['bairro', 'ano_transacao'])['volume_transacoes_bairro_ano'].sum().reset_index()

    # Printa estatísticas básicas
    unique_bairros = summary_table['bairro'].nunique()
    unique_anos = summary_table['ano_transacao'].nunique()
    min_ano = summary_table['ano_transacao'].min()
    max_ano = summary_table['ano_transacao'].max()

    print(f"Tabela resumo gerada com sucesso. ({len(summary_table):,} registros)")
    print(f"Bairros únicos no resumo: {unique_bairros} | Anos: {min_ano}-{max_ano}")

    return summary_table

if __name__ == '__main__':
    # Exemplo de uso
    # Criar um DataFrame de exemplo
    data = {
        'bairro': ['A', 'A', 'B', 'A', 'B', 'A', 'C', 'C', 'A'],
        'data_transacao': pd.to_datetime([
            '2020-01-15', '2020-02-20', '2020-03-10', '2021-01-05', 
            '2021-02-18', '2021-03-22', '2022-04-01', '2022-05-10', '2022-06-15'
        ]),
        'outra_feature': [1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    df_sample = pd.DataFrame(data)

    print("DataFrame Original:")
    print(df_sample)

    # Testar add_transaction_volume_feature
    df_with_volume = add_transaction_volume_feature(df_sample.copy())
    print("\nDataFrame com volume de transações:")
    print(df_with_volume)

    # Testar reprocessamento (idempotência)
    df_with_volume_reprocessed = add_transaction_volume_feature(df_with_volume.copy())
    print("\nDataFrame com volume de transações (reprocessado):")
    print(df_with_volume_reprocessed)

    # Testar get_transaction_summary
    transaction_summary = get_transaction_summary(df_with_volume)
    print("\nResumo de transações por bairro e ano:")
    print(transaction_summary)

    # Exemplo de erro (comente para rodar o resto)
    # df_error = df_sample.drop(columns=['bairro'])
    # add_transaction_volume_feature(df_error)

    # df_error_null = df_sample.copy()
    # df_error_null.loc[0, 'bairro'] = None
    # add_transaction_volume_feature(df_error_null)
