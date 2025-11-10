import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
import joblib

# Importar a função otimizada para carregar dados de clusterização
from clustering_analysis import get_clustering_data_optimized

def train_and_evaluate_model(df_clustered):
    """
    Treina e avalia um modelo de regressão para prever o valor_m2,
    utilizando os clusters como uma feature.
    """
    print("\n=== PREVISÃO DO VALOR DO IMÓVEL COM CLUSTERS ===")

    # Definir as features (X) e o target (y)
    # Excluímos 'valor_m2' do X, pois é o nosso target
    # Incluímos 'cluster' como uma feature categórica
    # 'bairro' e 'tipo_imovel' também são importantes
    features_to_use = [
        'area_construida',
        'area_terreno',
        'ano_construcao',
        'padrao_acabamento', # Será tratada como categórica
        'cluster',         # Categórica
        'bairro',          # Categórica
        'tipo_imovel'      # Categórica
    ]

    # Filtrar o DataFrame para incluir apenas as features e o target que existem
    # Isso lida com o caso de 'padrao_acabamento' ter sido one-hot encoded em 'clustering_analysis'
    # mas aqui queremos a coluna original para o ColumnTransformer
    df_model = df_clustered[['valor_m2'] + [f for f in features_to_use if f in df_clustered.columns]].copy()

    # Se 'padrao_acabamento' já estiver one-hot encoded, removemos a coluna original da lista features_to_use
    # e usamos as colunas one-hot encoded diretamente
    # Para o propósito deste exemplo, vamos assumir que 'padrao_acabamento' está como uma única coluna
    # e o ColumnTransformer fará o one-hot encoding.
    # Se o 'padrao_acabamento' original não existir, teremos que ajustar.
    if 'padrao_acabamento' not in df_model.columns:
        # Se 'padrao_acabamento' não está no df_model, é porque já foi one-hot encoded.
        # Precisamos então identificar as colunas one-hot e adicioná-las às features.
        # Por simplicidade, para este pipeline, vamos assumir que queremos a coluna original
        # para que o ColumnTransformer faça o encoding.
        # Se as colunas one-hot já estiverem lá, teríamos que adaptar o 'features_to_use'
        # e o ColumnTransformer.
        # Para este exercício, vamos re-filtrar o df_clustered para garantir 'padrao_acabamento' está presente
        # como uma coluna categórica antes do one-hot encoding.
        # Ou, alternativamente, ajustar `clustering_analysis.py` para que 'padrao_acabamento' não seja one-hot encoded na saída
        # do df_clustered se quisermos que o ColumnTransformer faça isso.
        
        # Dada a estrutura atual de clustering_analysis, 'padrao_acabamento' é one-hot encoded
        # e as colunas originais são perdidas para as features de CLUSTERING.
        # No entanto, 'df_clustered' retém 'padrao_acabamento' se ela era uma coluna antes do encoding.
        # Vamos verificar se as colunas one-hot de 'padrao_acabamento' estão presentes e usá-las.
        
        # Identificar colunas de padrao_acabamento one-hot encoded
        encoded_padrao_cols = [col for col in df_clustered.columns if col.startswith('padrao_acabamento_')]
        if encoded_padrao_cols:
            print(f"Detectadas colunas one-hot encoded para 'padrao_acabamento': {encoded_padrao_cols}")
            # Remover 'padrao_acabamento' da lista de features_to_use se estiver lá
            features_to_use = [f for f in features_to_use if f != 'padrao_acabamento']
            # Adicionar as colunas one-hot encoded
            features_to_use.extend(encoded_padrao_cols)
            df_model = df_clustered[['valor_m2'] + features_to_use].copy()
        else:
            print("AVISO: 'padrao_acabamento' não encontrada ou não one-hot encoded. Removendo da lista de features.")
            features_to_use = [f for f in features_to_use if f != 'padrao_acabamento']
            df_model = df_clustered[['valor_m2'] + features_to_use].copy()
    else:
        # 'padrao_acabamento' está presente como uma única coluna, o que é bom para o ColumnTransformer
        pass

    # Garantir que todas as features_to_use realmente existem no df_model
    features_to_use = [f for f in features_to_use if f in df_model.columns]

    X = df_model[features_to_use]
    y = df_model['valor_m2']

    print(f"Features selecionadas para o modelo: {X.columns.tolist()}")
    print(f"Total de registros para modelagem: {len(X):,}")

    # Separar dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Dados de treino: {len(X_train):,} registros")
    print(f"Dados de teste: {len(X_test):,} registros")

    # Identificar colunas numéricas e categóricas para pré-processamento
    numerical_features = ['area_construida', 'area_terreno', 'ano_construcao']
    # Filtrar para apenas as que estão realmente em X.columns
    numerical_features = [f for f in numerical_features if f in X.columns]

    categorical_features = ['cluster', 'bairro', 'tipo_imovel']
    # Se 'padrao_acabamento' foi mantido como categórico e não one-hot encoded, adicioná-lo
    if 'padrao_acabamento' in X.columns:
        categorical_features.append('padrao_acabamento')
    
    # Adicionar as colunas one-hot de padrao_acabamento se elas foram detectadas
    encoded_padrao_cols = [col for col in X.columns if col.startswith('padrao_acabamento_')]
    if encoded_padrao_cols:
        print(f"Usando colunas de 'padrao_acabamento' já one-hot encoded.")
        # Estas não precisam ser re-encoded, mas devem ser tratadas como numéricas (já que são 0/1)
        numerical_features.extend(encoded_padrao_cols)
        # Remover de categorical_features se por acaso foi adicionado
        categorical_features = [f for f in categorical_features if f not in encoded_padrao_cols]
        categorical_features = [f for f in categorical_features if not f.startswith('padrao_acabamento_')] # Garante remoção da original se presente

    # Filtrar colunas categóricas para garantir que existam em X
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Criar um pré-processador usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Manter outras colunas (se houver)
    )

    # Criar o pipeline do modelo
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("\nIniciando treinamento do modelo...")
    model.fit(X_train, y_train)
    print("Treinamento concluído!")

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\n=== Métricas de Avaliação do Modelo ===")
    print(f"Mean Absolute Error (MAE): R$ {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): R$ {rmse:,.2f}")
    print(f"R-squared (R²): {r2:.3f}")
    
    # Salvar o modelo treinado
    model_filename = 'PISI3-Project/property_value_prediction_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Modelo salvo em: {model_filename}")

    return model, mae, rmse, r2

def main():
    """
    Função principal para carregar dados, treinar e avaliar o modelo.
    """
    # Carregar dados clusterizados (otimizado para usar cache)
    df_clustered, silhouette_score, features_used_for_clustering = get_clustering_data_optimized()

    if df_clustered is None:
        print("Erro: Não foi possível carregar os dados clusterizados. Verifique 'clustering_analysis.py'.")
        return

    # Iniciar o treinamento e avaliação do modelo
    model, mae, rmse, r2 = train_and_evaluate_model(df_clustered)

if __name__ == "__main__":
    main()
