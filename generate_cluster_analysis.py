"""
Script para gerar análises adicionais para o dashboard:
- Análise de silhueta para diferentes valores de K
- Método do cotovelo (Elbow Method)
- Nomes descritivos para clusters
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import json
import os
from clustering_analysis import get_clustering_data_optimized, prepare_clustering_features, filter_residential_data
from data_processing import load_and_preprocess_data

def get_cluster_names():
    """
    Define nomes descritivos para cada cluster baseado nas características
    
    Cluster 0: Novos Premium (37.8%) - R$ 3.888/m², 98m², ano 2015, Boa Viagem
    Cluster 1: Econômicos Antigos (12.4%) - R$ 2.383/m², 93m², ano 1981
    Cluster 2: Amplos Terreno Grande (1.3%) - R$ 4.076/m², 178m², terreno 52.411m²
    Cluster 3: Padrão Intermediário (35.3%) - R$ 3.022/m², 95m², ano 2005
    Cluster 4: Grandes Alto Padrão (13.2%) - R$ 4.049/m², 249m², ano 2008
    """
    return {
        0: "Novos Premium",
        1: "Econômicos Antigos", 
        2: "Amplos Terreno Grande",
        3: "Padrão Intermediário",
        4: "Grandes Alto Padrão"
    }

def get_cluster_descriptions():
    """Descrições detalhadas de cada cluster"""
    return {
        0: {
            "name": "Novos Premium",
            "description": "Apartamentos recentes (2015) em bairros nobres como Boa Viagem, com valor médio-alto e área padrão",
            "characteristics": [
                "Construções recentes (mediana: 2015)",
                "Valor/m² alto (R$ 3.888)",
                "Área padrão (~98m²)",
                "Predominância em Boa Viagem",
                "Maior cluster (37.8%)"
            ]
        },
        1: {
            "name": "Econômicos Antigos",
            "description": "Imóveis mais antigos (1981) com menor valor/m², distribuídos em diversos bairros",
            "characteristics": [
                "Construções antigas (mediana: 1981)",
                "Menor valor/m² (R$ 2.383)",
                "Área média (~93m²)",
                "Distribuição variada de bairros",
                "12.4% do total"
            ]
        },
        2: {
            "name": "Amplos Terreno Grande",
            "description": "Imóveis diferenciados com grandes terrenos (52.411m²), possivelmente condomínios ou áreas especiais",
            "characteristics": [
                "Terreno excepcional (52.411m²)",
                "Área construída ampla (178m²)",
                "Valor/m² alto (R$ 4.076)",
                "Cluster exclusivo (1.3%)",
                "Predominância Imbiribeira/Cordeiro"
            ]
        },
        3: {
            "name": "Padrão Intermediário",
            "description": "Imóveis de padrão médio (2005) com valor e área intermediários, bem distribuídos",
            "characteristics": [
                "Ano construção médio (2005)",
                "Valor/m² intermediário (R$ 3.022)",
                "Área padrão (~95m²)",
                "Segundo maior cluster (35.3%)",
                "Bem distribuído geograficamente"
            ]
        },
        4: {
            "name": "Grandes Alto Padrão",
            "description": "Imóveis amplos (249m²) de alto padrão em bairros nobres, com maior área construída",
            "characteristics": [
                "Maior área construída (249m²)",
                "Valor/m² alto (R$ 4.049)",
                "Construções relativamente novas (2008)",
                "Forte presença em Boa Viagem",
                "13.2% do total"
            ]
        }
    }

def analyze_silhouette_scores():
    """
    Calcula silhouette scores para diferentes valores de K
    """
    print("\n=== ANÁLISE DE SILHUETA PARA DIFERENTES VALORES DE K ===")
    
    # Carregar e preparar dados
    df = load_and_preprocess_data()
    df_residential = filter_residential_data(df)
    df_features, features = prepare_clustering_features(df_residential)
    
    X = df_features[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Testar diferentes valores de K
    k_range = range(2, 11)
    silhouette_scores = []
    inertias = []
    
    print("\nTestando diferentes valores de K:")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        silhouette_avg = silhouette_score(X_scaled, labels)
        inertia = kmeans.inertia_
        
        silhouette_scores.append(silhouette_avg)
        inertias.append(inertia)
        
        print(f"  K={k}: Silhouette Score = {silhouette_avg:.4f}, Inertia = {inertia:,.0f}")
    
    # Salvar resultados
    results = {
        'k_values': list(k_range),
        'silhouette_scores': silhouette_scores,
        'inertias': inertias
    }
    
    output_file = 'silhouette_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Resultados salvos em: {output_file}")
    
    # Criar gráficos
    create_silhouette_plots(k_range, silhouette_scores, inertias, X_scaled)
    
    return results

def create_silhouette_plots(k_range, silhouette_scores, inertias, X_scaled):
    """Cria gráficos de silhueta e método do cotovelo"""
    
    docs_dir = 'docs'
    os.makedirs(docs_dir, exist_ok=True)
    
    # Gráfico 1: Silhouette Score vs K
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'b-o', linewidth=2, markersize=8)
    plt.axvline(x=5, color='r', linestyle='--', label='K=5 (escolhido)')
    plt.xlabel('Número de Clusters (K)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Análise de Silhueta: Escolha do Número de Clusters', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Adicionar anotações
    max_idx = silhouette_scores.index(max(silhouette_scores))
    plt.annotate(f'Máximo: {max(silhouette_scores):.4f}',
                xy=(k_range[max_idx], silhouette_scores[max_idx]),
                xytext=(k_range[max_idx]+0.5, silhouette_scores[max_idx]-0.02),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'silhouette_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de silhueta salvo: {os.path.join(docs_dir, 'silhouette_analysis.png')}")
    
    # Gráfico 2: Método do Cotovelo (Elbow Method)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertias, 'g-o', linewidth=2, markersize=8)
    plt.axvline(x=5, color='r', linestyle='--', label='K=5 (escolhido)')
    plt.xlabel('Número de Clusters (K)', fontsize=12)
    plt.ylabel('Inertia (Soma das Distâncias Quadráticas)', fontsize=12)
    plt.title('Método do Cotovelo: Determinação do Número Ótimo de Clusters', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(docs_dir, 'elbow_method.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico do cotovelo salvo: {os.path.join(docs_dir, 'elbow_method.png')}")
    
    # Gráfico 3: Silhouette Plot Detalhado para K=5
    create_detailed_silhouette_plot(X_scaled)

def create_detailed_silhouette_plot(X_scaled):
    """Cria gráfico de silhueta detalhado mostrando cada cluster"""
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    sample_silhouette_values = silhouette_samples(X_scaled, cluster_labels)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_lower = 10
    cluster_names = get_cluster_names()
    colors = plt.cm.Set3(np.linspace(0, 1, 5))
    
    for i in range(5):
        # Pegar valores de silhueta para o cluster i
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, ith_cluster_silhouette_values,
                        facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        # Label do cluster
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, 
                f'{cluster_names[i]}',
                fontsize=10, fontweight='bold')
        
        y_lower = y_upper + 10
    
    ax.set_title('Gráfico de Silhueta para K=5 Clusters', fontsize=14, fontweight='bold')
    ax.set_xlabel('Coeficiente de Silhueta', fontsize=12)
    ax.set_ylabel('Cluster', fontsize=12)
    
    # Linha vertical para silhouette score médio
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", linewidth=2,
              label=f'Média: {silhouette_avg:.3f}')
    ax.legend(loc='best')
    
    ax.set_yticks([])
    ax.set_xlim([-0.1, 1])
    
    plt.tight_layout()
    docs_dir = 'docs'
    plt.savefig(os.path.join(docs_dir, 'silhouette_detailed_k5.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Gráfico de silhueta detalhado salvo: {os.path.join(docs_dir, 'silhouette_detailed_k5.png')}")

def update_dashboard_stats_with_names():
    """Adiciona nomes descritivos ao dashboard_stats.json"""
    
    cluster_names = get_cluster_names()
    cluster_descriptions = get_cluster_descriptions()
    
    # Carregar stats existente
    with open('dashboard_stats.json', 'r', encoding='utf-8') as f:
        stats = json.load(f)
    
    # Adicionar nomes aos clusters
    for cluster in stats['clustering']['cluster_stats']:
        cluster_id = cluster['cluster_id']
        cluster['cluster_name'] = cluster_names[cluster_id]
        cluster['cluster_description'] = cluster_descriptions[cluster_id]['description']
        cluster['characteristics'] = cluster_descriptions[cluster_id]['characteristics']
    
    # Adicionar informações sobre os nomes
    stats['clustering']['cluster_names'] = cluster_names
    stats['clustering']['cluster_descriptions'] = cluster_descriptions
    
    # Salvar atualizado
    with open('dashboard_stats.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print("✅ dashboard_stats.json atualizado com nomes de clusters")

def main():
    print("=" * 70)
    print("GERANDO ANÁLISES COMPLEMENTARES PARA O DASHBOARD")
    print("=" * 70)
    
    # 1. Analisar silhouette scores
    print("\n1. Análise de Silhueta...")
    analyze_silhouette_scores()
    
    # 2. Atualizar stats com nomes
    print("\n2. Atualizando dashboard_stats.json com nomes de clusters...")
    update_dashboard_stats_with_names()
    
    # 3. Exibir resumo dos clusters
    print("\n" + "=" * 70)
    print("RESUMO DOS CLUSTERS")
    print("=" * 70)
    
    cluster_names = get_cluster_names()
    cluster_descriptions = get_cluster_descriptions()
    
    for cluster_id in range(5):
        print(f"\n{'='*70}")
        print(f"CLUSTER {cluster_id}: {cluster_names[cluster_id]}")
        print(f"{'='*70}")
        print(f"Descrição: {cluster_descriptions[cluster_id]['description']}\n")
        print("Características:")
        for char in cluster_descriptions[cluster_id]['characteristics']:
            print(f"  • {char}")
    
    print("\n" + "=" * 70)
    print("✅ ANÁLISES CONCLUÍDAS COM SUCESSO!")
    print("=" * 70)
    print("\nArquivos gerados:")
    print("  • silhouette_analysis.json")
    print("  • docs/silhouette_analysis.png")
    print("  • docs/elbow_method.png")
    print("  • docs/silhouette_detailed_k5.png")
    print("  • dashboard_stats.json (atualizado)")

if __name__ == "__main__":
    main()
