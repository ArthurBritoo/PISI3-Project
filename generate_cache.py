#!/usr/bin/env python3
"""
Script para gerar cache da clusterização.
Execute uma vez para criar os arquivos de cache que aceleram o Streamlit.
"""

from clustering_analysis import get_clustering_data_optimized

if __name__ == "__main__":
    print("🚀 Gerando cache inicial da clusterização...")
    
    try:
        df_clustered, silhouette_score, features = get_clustering_data_optimized()
        
        print(f"\n✅ Cache gerado com sucesso!")
        print(f"📊 {len(df_clustered):,} imóveis clusterizados")
        print(f"🎯 Silhouette Score: {silhouette_score:.3f}")
        print(f"🔧 Features: {features}")
        
        # Verificar tamanho dos arquivos
        import os
        cache_file = 'data/clustering_cache.parquet'
        metadata_file = 'data/clustering_metadata.json'

        if os.path.exists(cache_file):
            cache_size = os.path.getsize(cache_file) / (1024 * 1024)
            print(f"💾 Arquivo cache: {cache_size:.2f} MB")

        print(f"\n⚡ Próximas execuções do Streamlit serão MUITO mais rápidas!")
        print(f"🔄 Para regenerar o cache, delete os arquivos e execute este script novamente")
        
    except Exception as e:
        print(f"❌ Erro ao gerar cache: {e}")
        import traceback
        traceback.print_exc()