"""
Script para gerar estatísticas reais dos clusters e modelo para o dashboard.
"""
import pandas as pd
import json
from clustering_analysis import get_clustering_data_optimized
from classification_model import create_classification_target
from sklearn.metrics import classification_report
import joblib

def analyze_clusters():
    """Analisa os clusters e retorna estatísticas detalhadas"""
    df_clustered, silhouette, features = get_clustering_data_optimized()
    
    if df_clustered is None:
        print("Erro ao carregar dados")
        return None
    
    # Estatísticas por cluster
    cluster_stats = []
    
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
        
        stats = {
            'cluster_id': int(cluster_id),
            'total_imoveis': len(cluster_data),
            'percentual': (len(cluster_data) / len(df_clustered)) * 100,
            'valor_m2_mediano': cluster_data['valor_m2'].median(),
            'valor_m2_medio': cluster_data['valor_m2'].mean(),
            'area_construida_mediana': cluster_data['area_construida'].median(),
            'area_construida_media': cluster_data['area_construida'].mean(),
            'area_terreno_mediana': cluster_data['area_terreno'].median(),
            'ano_construcao_mediano': cluster_data['ano_construcao'].median(),
            'tipo_imovel_predominante': cluster_data['tipo_imovel'].mode()[0] if len(cluster_data['tipo_imovel'].mode()) > 0 else 'N/A',
            'top_3_bairros': cluster_data['bairro'].value_counts().head(3).to_dict()
        }
        
        cluster_stats.append(stats)
    
    # Estatísticas gerais
    general_stats = {
        'total_imoveis': len(df_clustered),
        'silhouette_score': silhouette,
        'n_clusters': len(df_clustered['cluster'].unique()),
        'features_used': features,
        'anos_range': f"{int(df_clustered['ano_construcao'].min())} - {int(df_clustered['ano_construcao'].max())}",
        'valor_m2_min': df_clustered['valor_m2'].min(),
        'valor_m2_max': df_clustered['valor_m2'].max(),
        'valor_m2_medio': df_clustered['valor_m2'].mean(),
        'area_media': df_clustered['area_construida'].mean()
    }
    
    return {
        'cluster_stats': cluster_stats,
        'general_stats': general_stats
    }

def analyze_classification_model():
    """Analisa o modelo de classificação e retorna métricas"""
    try:
        model = joblib.load('property_classifier_model_optimized.joblib')
        
        # Carregar dados de teste para obter métricas reais
        df_clustered, _, _ = get_clustering_data_optimized()
        df_classification = create_classification_target(df_clustered)
        
        from sklearn.model_selection import train_test_split
        
        features_to_use = ['area_construida', 'area_terreno', 'ano_construcao', 'cluster', 'bairro', 'tipo_imovel']
        existing_features = [f for f in features_to_use if f in df_classification.columns]
        X = df_classification[existing_features]
        y = df_classification['categoria_valor']
        
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        y_pred = model.predict(X_test)
        
        # Gerar relatório de classificação
        report = classification_report(y_test, y_pred, output_dict=True)
        
        # Extrair feature importance
        if hasattr(model.named_steps['classifier'], 'feature_importances_'):
            # Obter nomes das features após preprocessamento
            preprocessor = model.named_steps['preprocessor']
            feature_names = []
            
            # Features numéricas
            numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
            feature_names.extend(numeric_features)
            
            # Features categóricas (one-hot encoded)
            cat_features = preprocessor.named_transformers_['cat']
            if hasattr(cat_features, 'get_feature_names_out'):
                categorical_names = cat_features.get_feature_names_out().tolist()
                feature_names.extend(categorical_names)
            
            importances = model.named_steps['classifier'].feature_importances_
            
            # Criar lista de features com importâncias
            feature_importance = []
            for i, (name, importance) in enumerate(zip(feature_names, importances[:len(feature_names)])):
                feature_importance.append({
                    'feature': name,
                    'importance': float(importance)
                })
            
            # Ordenar por importância
            feature_importance = sorted(feature_importance, key=lambda x: x['importance'], reverse=True)[:10]
        else:
            feature_importance = []
        
        model_stats = {
            'accuracy': report['accuracy'],
            'precision_macro': report['macro avg']['precision'],
            'recall_macro': report['macro avg']['recall'],
            'f1_macro': report['macro avg']['f1-score'],
            'class_metrics': {
                'Alto Valor': {
                    'precision': report['Alto Valor']['precision'],
                    'recall': report['Alto Valor']['recall'],
                    'f1-score': report['Alto Valor']['f1-score'],
                    'support': int(report['Alto Valor']['support'])
                },
                'Econômico': {
                    'precision': report['Econômico']['precision'],
                    'recall': report['Econômico']['recall'],
                    'f1-score': report['Econômico']['f1-score'],
                    'support': int(report['Econômico']['support'])
                },
                'Médio': {
                    'precision': report['Médio']['precision'],
                    'recall': report['Médio']['recall'],
                    'f1-score': report['Médio']['f1-score'],
                    'support': int(report['Médio']['support'])
                }
            },
            'feature_importance': feature_importance,
            'best_params': model.named_steps['classifier'].get_params()
        }
        
        return model_stats
        
    except Exception as e:
        print(f"Erro ao analisar modelo: {e}")
        return None

def main():
    print("=" * 60)
    print("GERANDO ESTATÍSTICAS PARA O DASHBOARD")
    print("=" * 60)
    
    print("\n1. Analisando clusters...")
    cluster_analysis = analyze_clusters()
    
    print("\n2. Analisando modelo de classificação...")
    model_analysis = analyze_classification_model()
    
    # Combinar resultados
    dashboard_data = {
        'clustering': cluster_analysis,
        'classification': model_analysis
    }
    
    # Salvar em JSON
    output_file = 'dashboard_stats.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dashboard_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Estatísticas salvas em: {output_file}")
    
    # Exibir resumo
    print("\n" + "=" * 60)
    print("RESUMO DAS ESTATÍSTICAS")
    print("=" * 60)
    
    if cluster_analysis:
        print(f"\n📊 CLUSTERING:")
        print(f"   • Total de imóveis: {cluster_analysis['general_stats']['total_imoveis']:,}")
        print(f"   • Silhouette Score: {cluster_analysis['general_stats']['silhouette_score']:.3f}")
        print(f"   • Número de clusters: {cluster_analysis['general_stats']['n_clusters']}")
        
        print(f"\n   Distribuição por cluster:")
        for cluster in cluster_analysis['cluster_stats']:
            print(f"   • Cluster {cluster['cluster_id']}: {cluster['total_imoveis']:,} imóveis ({cluster['percentual']:.1f}%)")
    
    if model_analysis:
        print(f"\n🔮 CLASSIFICAÇÃO:")
        print(f"   • Acurácia: {model_analysis['accuracy']:.2%}")
        print(f"   • Precision (macro): {model_analysis['precision_macro']:.2%}")
        print(f"   • Recall (macro): {model_analysis['recall_macro']:.2%}")
        print(f"   • F1-Score (macro): {model_analysis['f1_macro']:.2%}")
        
        print(f"\n   Top 5 Features mais importantes:")
        for i, feat in enumerate(model_analysis['feature_importance'][:5], 1):
            print(f"   {i}. {feat['feature']}: {feat['importance']:.4f}")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
