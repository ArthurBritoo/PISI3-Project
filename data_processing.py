
import pandas as pd
import os

def load_and_preprocess_data(data_dir='data'):
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.parquet')]
    
    li = []

    for filename in all_files:
        df = pd.read_parquet(filename, engine='pyarrow')
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    
    # Para arquivos Parquet, os tipos já estão preservados, mas vamos garantir conversões seguras
    # Conversões numéricas com tratamento de erros
    numeric_columns = ['valor_avaliacao', 'area_construida', 'area_terreno', 'sfh']
    
    for col in numeric_columns:
        if col in df.columns:
            # Converte para string primeiro para tratar vírgulas/pontos se necessário
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preencher valores nulos em 'sfh' com 0.0, pois parece ser um valor monetário ou indicador
    df['sfh'] = df['sfh'].fillna(0.0)

    # Remover linhas com valores nulos em colunas críticas para a análise
    df.dropna(subset=['bairro', 'valor_avaliacao', 'area_construida', 'padrao_acabamento'], inplace=True)

    # Filtrar valores de área construída e valor de avaliação que fazem sentido
    df = df[(df['area_construida'] > 0) & (df['valor_avaliacao'] > 0)]

    # Criar a coluna valor_m2
    df['valor_m2'] = df['valor_avaliacao'] / df['area_construida']
    
    # Remover valores infinitos ou muito grandes que podem surgir da divisão
    df = df[df['valor_m2'] < df['valor_m2'].quantile(0.99)] # Remove outliers superiores
    df = df[df['valor_m2'] > df['valor_m2'].quantile(0.01)] # Remove outliers inferiores

    # Converter 'data_transacao' para datetime
    df['data_transacao'] = pd.to_datetime(df['data_transacao'])
    
    # Padronizar nomes de bairros (ex: remover espaços extras, converter para maiúsculas)
    df['bairro'] = df['bairro'].str.strip().str.upper()
    
    # Filtrar apenas imóveis de Recife, se houver outros
    df = df[df['cidade'].str.upper() == 'RECIFE']

    return df

if __name__ == '__main__':
    # Exemplo de uso para testar a função localmente. Use um caminho de
    # dados relativo ao diretório deste arquivo para evitar FileNotFoundError
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data')

    if not os.path.isdir(data_dir):
        raise SystemExit(f"Diretório de dados não encontrado: {data_dir}. Verifique se a pasta 'data' existe no projeto.")

    df_processed = load_and_preprocess_data(data_dir=data_dir)
    print(df_processed.head())
    print(df_processed.info())
    print(df_processed.describe())

