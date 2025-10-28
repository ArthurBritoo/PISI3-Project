import pandas as pd
import os
from pathlib import Path

def convert_csv_to_parquet(csv_file_path, output_dir='data'):
    print(f"Convertendo: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path, sep=';', decimal=',')
    
    csv_filename = Path(csv_file_path).stem
    parquet_filename = f"{csv_filename}.parquet"
    parquet_path = os.path.join(output_dir, parquet_filename)
    
    df.to_parquet(parquet_path, engine='pyarrow', compression='snappy')
    
    csv_size = os.path.getsize(csv_file_path) / (1024 * 1024)
    parquet_size = os.path.getsize(parquet_path) / (1024 * 1024)
    compression_ratio = (1 - parquet_size / csv_size) * 100
    
    print(f"  ✓ {csv_filename}.csv ({csv_size:.2f} MB) → {parquet_filename} ({parquet_size:.2f} MB)")
    print(f"  📊 Compressão: {compression_ratio:.1f}% de redução")
    print(f"  📁 Salvo em: {parquet_path}")
    
    return parquet_path

def main():
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        print(f"❌ Diretório {data_dir} não encontrado!")
        return
    
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"❌ Nenhum arquivo CSV encontrado em {data_dir}/")
        return
    
    print(f"🚀 Iniciando conversão de {len(csv_files)} arquivos CSV para Parquet...")
    print("=" * 70)
    
    total_csv_size = 0
    total_parquet_size = 0
    
    for csv_file in sorted(csv_files):
        csv_path = os.path.join(data_dir, csv_file)
        
        try:
            parquet_path = convert_csv_to_parquet(csv_path, data_dir)
            
            total_csv_size += os.path.getsize(csv_path)
            total_parquet_size += os.path.getsize(parquet_path)
            
        except Exception as e:
            print(f"❌ Erro ao converter {csv_file}: {e}")
            continue
        
        print("-" * 50)
    
    total_csv_mb = total_csv_size / (1024 * 1024)
    total_parquet_mb = total_parquet_size / (1024 * 1024)
    total_compression = (1 - total_parquet_mb / total_csv_mb) * 100
    
    print("=" * 70)
    print("📈 RESUMO DA CONVERSÃO:")
    print(f"📊 Total CSV: {total_csv_mb:.2f} MB")
    print(f"📊 Total Parquet: {total_parquet_mb:.2f} MB")
    print(f"🎯 Compressão total: {total_compression:.1f}% de redução")
    print(f"✅ Conversão concluída com sucesso!")
    
    parquet_files = [f for f in os.listdir(data_dir) if f.endswith('.parquet')]
    print(f"\n📁 Arquivos Parquet criados ({len(parquet_files)}):")
    for pf in sorted(parquet_files):
        print(f"  • {pf}")

if __name__ == "__main__":
    main()
