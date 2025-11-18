import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.utils.data_processing import load_and_preprocess_data

def filter_residential_data(df):
    """
    Filtra apenas dados residenciais (Apartamento e Casa).
    
    JUSTIFICATIVA: Excluímos imóveis comerciais/institucionais para evitar:
    - Hospitais, shoppings e prédios comerciais gigantes
    - Terrenos vazios com valores distorcidos  
    - Salas comerciais com padrões de preço diferentes
    - Garantir homogeneidade na análise residencial
    """
    print("=== FILTRAGEM PARA DADOS RESIDENCIAIS ===")
    
    residential_types = ['Apartamento', 'Casa']
    df_residential = df[df['tipo_imovel'].isin(residential_types)].copy()
    
    print(f"Dataset original: {len(df):,} registros")
    print(f"Dataset filtrado: {len(df_residential):,} registros")
    reduction = (1 - len(df_residential)/len(df)) * 100
    print(f"Redução: {reduction:.1f}% (foco em dados residenciais)")
    
    print(f"\nTipos incluídos:")
    for tipo in residential_types:
        count = (df_residential['tipo_imovel'] == tipo).sum()
        pct = (count / len(df_residential)) * 100
        print(f"  • {tipo}: {count:,} ({pct:.1f}%)")
    
    return df_residential

def prepare_clustering_features(df_residential):
    """
    Prepara features para clusterização focando em características residenciais.
    
    ATENÇÃO: 'valor_m2' foi removido das features de clusterização para que os clusters
    refletem o padrão construtivo e não o efeito de mercado (preço/localização).
    'padrao_acabamento' é incluído via One-Hot Encoding.
    """
    print(f"\n=== PREPARAÇÃO DAS FEATURES ===")
    
    # Features numéricas relevantes para residências
    numerical_features = ['area_construida', 'area_terreno', 'ano_construcao']
    
    # Feature categórica a ser usada e encodificada
    categorical_features = ['padrao_acabamento']
    
    # Colunas para manter no DataFrame final para análise (mesmo que não usadas para clustering)
    analysis_cols = ['bairro', 'tipo_imovel', 'valor_m2']
    
    # Criar DataFrame com todas as features necessárias para o processo
    df_features = df_residential[numerical_features + categorical_features + analysis_cols].copy()
    
    # Remover outliers extremos (além do 99º percentil) para as features numéricas
    for feature in numerical_features:
        q99 = df_features[feature].quantile(0.99)
        q01 = df_features[feature].quantile(0.01)
        df_features = df_features[
            (df_features[feature] >= q01) & (df_features[feature] <= q99)
        ]
    
    # Aplicar One-Hot Encoding para 'padrao_acabamento'
    df_features = pd.get_dummies(df_features, columns=categorical_features, prefix=categorical_features, drop_first=False)
    
    # Atualizar a lista de features para incluir as colunas one-hot encoded
    # As novas colunas terão o formato 'padrao_acabamento_VALOR'.
    # Primeiro, identificamos as colunas criadas pelo get_dummies
    encoded_feature_names = [col for col in df_features.columns if col.startswith('padrao_acabamento_')]
    
    # A lista final de features para clustering será numérica + encoded
    final_clustering_features = numerical_features + encoded_feature_names
    
    print(f"Após remoção de outliers e encoding: {len(df_features):,} registros")
    print(f"Features selecionadas para clusterização (incluindo One-Hot Encoding): {final_clustering_features}")
    
    return df_features, final_clustering_features

def perform_clustering(df_features, features, n_clusters=5):
    """
    Executa clusterização K-means nos dados residenciais.
    """
    print(f"\n=== CLUSTERIZAÇÃO K-MEANS ===")
    
    # Preparar dados para clustering
    X = df_features[features].values
    
    # Normalizar features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Executar K-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calcular score de silhueta
    silhouette = silhouette_score(X_scaled, clusters)
    
    # Adicionar clusters ao DataFrame
    df_clustered = df_features.copy()
    df_clustered['cluster'] = clusters
    
    print(f"Número de clusters: {n_clusters}")
    print(f"Silhouette Score: {silhouette:.3f}")
    print(f"Registros clusterizados: {len(df_clustered):,}")
    
    # Analisar clusters
    print(f"\nDistribuição dos clusters:")
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        pct = (count / len(clusters)) * 100
        print(f"  Cluster {cluster_id}: {count:,} ({pct:.1f}%)")
    
    return df_clustered, kmeans, scaler, silhouette

def analyze_clusters(df_clustered, features):
    """
    Analisa características de cada cluster.
    """
    print(f"\n=== ANÁLISE DOS CLUSTERS ===")
    
    # As features para o summary devem ser as originais numéricas e o valor_m2
    # Nomes das colunas one-hot encoded não são ideais para o summary 'mean'/'median'
    summary_features_for_agg = [col for col in df_clustered.columns if col in ['area_construida', 'area_terreno', 'ano_construcao', 'valor_m2']]
    cluster_summary = df_clustered.groupby('cluster')[summary_features_for_agg].agg(['mean', 'median']).round(2)
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        print(f"\nCluster {cluster_id}:")
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        # Características principais
        valor_m2_med = cluster_data['valor_m2'].median()
        area_med = cluster_data['area_construida'].median()
        ano_med = cluster_data['ano_construcao'].median()
        
        print(f"  • Valor m²: R$ {valor_m2_med:,.0f} (mediana)")
        print(f"  • Área construída: {area_med:.0f} m² (mediana)")
        print(f"  • Ano construção: {ano_med:.0f} (mediana)")
        
        # Tipo de imóvel predominante
        tipo_predominante = cluster_data['tipo_imovel'].value_counts().index[0]
        pct_tipo = (cluster_data['tipo_imovel'].value_counts().iloc[0] / len(cluster_data)) * 100
        print(f"  • Tipo predominante: {tipo_predominante} ({pct_tipo:.1f}%)")
        
        # Padrão de acabamento predominante
        if 'padrao_acabamento' in cluster_data.columns:
            acabamento_predominante = cluster_data['padrao_acabamento'].value_counts().index[0]
            pct_acabamento = (cluster_data['padrao_acabamento'].value_counts().iloc[0] / len(cluster_data)) * 100
            print(f"  • Padrão Acabamento predominante: {acabamento_predominante} ({pct_acabamento:.1f}%)")
        
        # Bairros mais comuns
        top_bairros = cluster_data['bairro'].value_counts().head(3)
        print(f"  • Bairros principais: {', '.join(top_bairros.index[:3])}")
    
    return cluster_summary

def create_cluster_visualizations(df_clustered):
    """
    Cria visualizações dos clusters.
    """
    print(f"\n=== CRIANDO VISUALIZAÇÕES ===")
    
    # Gráfico 1: Scatter plot Valor m² vs Área construída
    fig1 = px.scatter(
        df_clustered, 
        x='area_construida', 
        y='valor_m2',
        color='cluster',
        hover_data=['bairro', 'tipo_imovel', 'ano_construcao', 'padrao_acabamento'],
        title='Clusters: Valor m² vs Área Construída (Dados Residenciais)',
        labels={
            'area_construida': 'Área Construída (m²)',
            'valor_m2': 'Valor por m² (R$)',
            'cluster': 'Cluster'
        }
    )
    
    # Gráfico 2: Box plot do valor m² por cluster
    fig2 = px.box(
        df_clustered,
        x='cluster',
        y='valor_m2',
        title='Distribuição do Valor m² por Cluster (Dados Residenciais)',
        labels={
            'cluster': 'Cluster',
            'valor_m2': 'Valor por m² (R$)'
        }
    )
    
    # Gráfico 3: Contagem por tipo de imóvel e cluster
    cluster_type_counts = df_clustered.groupby(['cluster', 'tipo_imovel']).size().reset_index(name='count')
    fig3 = px.bar(
        cluster_type_counts,
        x='cluster',
        y='count',
        color='tipo_imovel',
        title='Distribuição de Tipos de Imóveis por Cluster',
        labels={
            'cluster': 'Cluster',
            'count': 'Número de Imóveis',
            'tipo_imovel': 'Tipo de Imóvel'
        }
    )

    # Gráfico 4: Contagem por Padrão de Acabamento e Cluster
    if 'padrao_acabamento' in df_clustered.columns:
        cluster_acabamento_counts = df_clustered.groupby(['cluster', 'padrao_acabamento']).size().reset_index(name='count')
        fig4 = px.bar(
            cluster_acabamento_counts,
            x='cluster',
            y='count',
            color='padrao_acabamento',
            title='Distribuição de Padrão de Acabamento por Cluster',
            labels={
                'cluster': 'Cluster',
                'count': 'Número de Imóveis',
                'padrao_acabamento': 'Padrão de Acabamento'
            }
        )
        return fig1, fig2, fig3, fig4
    else:
        return fig1, fig2, fig3

def main():
    """
    Função principal que executa toda a análise de clusterização.
    """
    print("🏠 CLUSTERIZAÇÃO DE DADOS RESIDENCIAIS - ITBI RECIFE")
    print("=" * 60)
    
    # 1. Carregar dados
    print("1. Carregando dados...")
    df = load_and_preprocess_data()
    
    # 2. Filtrar dados residenciais
    print(f"\n2. Filtrando dados residenciais...")
    df_residential = filter_residential_data(df)
    
    # 3. Preparar features
    print(f"\n3. Preparando features...")
    df_features, features = prepare_clustering_features(df_residential)
    
    # 4. Executar clusterização
    print(f"\n4. Executando clusterização...")
    df_clustered, kmeans, scaler, silhouette = perform_clustering(df_features, features)
    
    # 5. Analisar clusters
    print(f"\n5. Analisando clusters...")
    cluster_summary = analyze_clusters(df_clustered, features)
    
    # 6. Criar visualizações
    print(f"\n6. Criando visualizações...")
    figures = create_cluster_visualizations(df_clustered)
    
    print(f"\n✅ Análise de clusterização concluída!")
    print(f"📊 {len(df_clustered):,} imóveis residenciais clusterizados")
    print(f"🎯 Silhouette Score: {silhouette:.3f}")
    
    return df_clustered, figures, cluster_summary

def get_cache_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, 'data') # Agora aponta para PISI3-Project/data
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, 'clustering_cache.parquet')
    metadata_file = os.path.join(cache_dir, 'clustering_metadata.json')
    return cache_file, metadata_file, cache_dir

def save_clustering_cache(df_clustered, silhouette_score, features):
    """
    Salva os resultados da clusterização em cache para carregamento rápido.
    """
    cache_file, metadata_file, _ = get_cache_paths()
    
    # Salvar DataFrame com clusters
    df_clustered.to_parquet(cache_file, engine='pyarrow', compression='snappy')
    
    # Salvar metadados em arquivo separado
    metadata = {
        'silhouette_score': silhouette_score,
        'features': features,
        'n_clusters': 5,
        'total_records': len(df_clustered)
    }
    
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)
    
    print(f"✅ Cache da clusterização salvo:")
    print(f"   • Dados: {cache_file}")
    print(f"   • Metadata: {metadata_file}")
    print(f"   • Registros: {len(df_clustered):,}")

def load_clustering_cache():
    """
    Carrega os resultados da clusterização do cache se disponível.
    
    Returns:
        tuple: (df_clustered, silhouette_score, features) ou None se cache não existir
    """
    cache_file, metadata_file, _ = get_cache_paths()
    
    if not os.path.exists(cache_file) or not os.path.exists(metadata_file):
        return None
    
    try:
        # Carregar dados
        df_clustered = pd.read_parquet(cache_file, engine='pyarrow')
        
        # Carregar metadata
        import json
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        silhouette_score = metadata['silhouette_score']
        features = metadata['features']
        
        print(f"✅ Cache da clusterização carregado:")
        print(f"   • Registros: {len(df_clustered):,}")
        print(f"   • Silhouette Score: {silhouette_score:.3f}")
        
        return df_clustered, silhouette_score, features
        
    except Exception as e:
        print(f"❌ Erro ao carregar cache: {e}")
        return None

def get_clustering_data_optimized():
    """
    Função otimizada que usa cache quando possível, senão processa do zero.
    """
    print("🚀 Carregando dados de clusterização...")
    
    # Tentar carregar do cache primeiro
    cached_result = load_clustering_cache()
    
    if cached_result is not None:
        print("⚡ Dados carregados do cache (ultra-rápido)!")
        # O cache pode não ter a coluna 'padrao_acabamento' one-hot encoded se foi salvo antes da mudança
        # Então, precisamos verificar e recriar as features para o K-Means, se necessário.
        # Para simplificar agora, vou forçar o reprocessamento se as features mudaram muito.
        # Em um sistema real, você teria uma lógica de versão para o cache.
        df_clustered, silhouette, cached_features_list = cached_result
        
        # Verificar se as features do cache correspondem às features esperadas após a atualização
        # Esta é uma verificação simplificada; uma abordagem robusta compararia conjuntos de features
        expected_numerical_features = ['area_construida', 'area_terreno', 'ano_construcao']
        # Verificação se o cache foi gerado *antes* do one-hot encoding de padrao_acabamento
        if not any(f.startswith('padrao_acabamento_') for f in cached_features_list):
            print("⚠️ Cache antigo detectado (sem One-Hot Encoding de padrão de acabamento). Reprocessando...")
            # Força o reprocessamento para garantir que o One-Hot Encoding seja aplicado
            return process_and_save_new_clustering_data()
        
        # Se o cache é válido e já tem as features corretas, retorna-o
        return df_clustered, silhouette, cached_features_list
    
    print("🔄 Cache não encontrado ou inválido, processando dados...")
    return process_and_save_new_clustering_data()

def process_and_save_new_clustering_data():
    """
    Processa os dados de clusterização do zero e salva no cache.
    """
    df = load_and_preprocess_data()
    df_residential = filter_residential_data(df)
    df_features, features = prepare_clustering_features(df_residential)
    df_clustered, kmeans, scaler, silhouette = perform_clustering(df_features, features)
    
    # Salvar no cache para próximas vezes
    save_clustering_cache(df_clustered, silhouette, features)
    
    return df_clustered, silhouette, features

if __name__ == "__main__":
    # Modificando a chamada para main para refletir as novas features retornadas
    df_clustered, figures, summary = main()

    # Exibir as figuras se o ambiente permitir (apenas para teste local)
    # if len(figures) == 4:
    #     fig1, fig2, fig3, fig4 = figures
    #     fig1.show()
    #     fig2.show()
    #     fig3.show()
    #     fig4.show()
    # elif len(figures) == 3:
    #     fig1, fig2, fig3 = figures
    #     fig1.show()
    #     fig2.show()
    #     fig3.show()