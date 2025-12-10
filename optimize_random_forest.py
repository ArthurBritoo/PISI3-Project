import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import time
import pandas as pd
from clustering_analysis import get_clustering_data_optimized # Para carregar os dados
from sklearn.model_selection import train_test_split # Para a divisão


# --- 1. FUNÇÃO DE PREPARAÇÃO DE DADOS (Extraída do seu código) ---
def prepare_data_and_preprocessor(df_clustered):
    # Definir features e target (Lógica copiada e simplificada do seu código)
    features_to_use = ['area_construida', 'area_terreno', 'ano_construcao', 'cluster', 'bairro', 'tipo_imovel']
    encoded_padrao_cols = [col for col in df_clustered.columns if col.startswith('padrao_acabamento_')]
    if encoded_padrao_cols:
        features_to_use.extend(encoded_padrao_cols)
    elif 'padrao_acabamento' in df_clustered.columns:
        features_to_use.append('padrao_acabamento')
    
    features_to_use = [f for f in features_to_use if f in df_clustered.columns]
    
    X = df_clustered[features_to_use].copy()
    y = df_clustered['valor_m2'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Definir colunas para o pré-processador
    numerical_features = ['area_construida', 'area_terreno', 'ano_construcao']
    categorical_features = ['cluster', 'bairro', 'tipo_imovel']

    # Ajuste para as colunas de padrao_acabamento já encodificadas
    numerical_features.extend([f for f in X.columns if f.startswith('padrao_acabamento_')])
    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Criar o pré-processador (ColumnTransformer)
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), [f for f in numerical_features if f in X.columns]),
            ('cat', OneHotEncoder(handle_unknown='ignore'), [f for f in categorical_features if f in X.columns])
        ],
        remainder='passthrough'
    )
    
    return X_train, X_test, y_train, y_test, preprocessor

# --- 2. EXECUÇÃO PRINCIPAL DA OTIMIZAÇÃO ---
def optimize_random_forest():
    df_clustered, _, _ = get_clustering_data_optimized()
    if df_clustered is None:
        print("Erro ao carregar dados.")
        return

    X_train, X_test, y_train, y_test, preprocessor = prepare_data_and_preprocessor(df_clustered)
    
    print("Iniciando a otimização do Pipeline com RandomizedSearchCV...")
    start_time = time.time()

    # 2.1. Definir o Espaço de Busca (prefixo 'regressor__')
    # ESPAÇO REDUZIDO para execução mais rápida (tempo estimado: 5-10 min)
    param_dist = {
        # O prefixo 'regressor__' é obrigatório ao otimizar um componente do Pipeline
        'regressor__n_estimators': [100, 200, 300],  # Reduzido de 5 para 3 opções
        'regressor__max_depth': [10, 30, None],      # Reduzido de 4 para 3 opções
        'regressor__min_samples_split': [2, 5],      # Reduzido de 3 para 2 opções
        'regressor__min_samples_leaf': [1, 2],       # Reduzido de 3 para 2 opções
        'regressor__max_features': ['sqrt'],         # Reduzido de 2 para 1 opção
    }

    # 2.2. Criar o Pipeline completo
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1)) # O n_jobs do RF deve ser -1
    ])

    # 2.3. Inicializar e Treinar o RandomizedSearchCV
    rf_random = RandomizedSearchCV(
        estimator=rf_pipeline,
        param_distributions=param_dist,
        n_iter=10,  # Reduzido para 10 (10 iter × 3 CV = 30 modelos treinados)
        cv=3,       # Reduzido de 5 para 3-fold (mais rápido, ainda confiável)
        verbose=2,  # Aumentado para ver progresso detalhado
        random_state=42,
        n_jobs=-1, # O n_jobs do RS CV também pode ser -1 (para a busca de hiperparâmetros)
        scoring='neg_mean_absolute_error'
    )

    rf_random.fit(X_train, y_train)
    
    end_time = time.time()
    
    # --- 3. AVALIAÇÃO FINAL E RESULTADOS ---
    best_pipeline = rf_random.best_estimator_
    best_params = rf_random.best_params_
    
    # Previsões no conjunto de TESTE (dados não vistos)
    y_pred_opt = best_pipeline.predict(X_test)

    # Avaliação
    mae_opt = mean_absolute_error(y_test, y_pred_opt)
    
    print("\n--- Otimização Concluída ---")
    print(f"Tempo de execução total: {end_time - start_time:.2f} segundos")
    print("\n## Resultados Finais (Modelo Otimizado - Test Set)")
    print(f"Melhores Hiperparâmetros Encontrados: {best_params}")
    print(f"Mean Absolute Error (MAE) no Teste: R$ {mae_opt:,.2f}")
    
    return best_pipeline

if __name__ == "__main__":
    optimized_model = optimize_random_forest()
    # Próximo passo: Salvar 'optimized_model' usando joblib

