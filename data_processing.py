
import pandas as pd
import os

def load_and_preprocess_data():
    # Diretório deste arquivo (raiz do projeto)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # A pasta 'data' fica na raiz do projeto
    data_dir = os.path.join(script_dir, 'data')

    # Fallback: tentar um nível acima (em caso de execução a partir de subpastas)
    if not os.path.isdir(data_dir):
        candidate = os.path.join(os.path.dirname(script_dir), 'data')
        if os.path.isdir(candidate):
            data_dir = candidate
        else:
            raise FileNotFoundError(f"Diretório de dados não encontrado: {data_dir}. Verifique se a pasta 'data' existe no projeto.")

    # OTIMIZAÇÃO: Carregar apenas dados de 2020-2023 para reduzir tamanho do modelo
    # Isso permite que o modelo fique abaixo do limite de 50MB do Supabase
    years_to_load = ['2020', '2021', '2022', '2023']
    all_files = [
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.parquet') and any(year in f for year in years_to_load)
    ]
    
    print(f"📊 Carregando dados de {', '.join(years_to_load)}...")
    print(f"   Arquivos encontrados: {len(all_files)}")
    
    li = []

    for filename in all_files:
        df = pd.read_parquet(filename, engine='pyarrow')
        year = [y for y in years_to_load if y in os.path.basename(filename)][0]
        print(f"   • {year}: {len(df):,} registros")
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    print(f"   ✅ Total: {len(df):,} registros carregados")
    
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
    # Exemplo de uso para testar a função localmente.
    # A função load_and_preprocess_data agora determina o caminho de dados internamente.
    df_processed = load_and_preprocess_data()
    print(df_processed.head())
    print(df_processed.info())
    print(df_processed.describe())
