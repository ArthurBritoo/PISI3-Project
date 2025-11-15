import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os
import joblib

# Importar a função otimizada para carregar dados de clusterização
from clustering_analysis import get_clustering_data_optimized

def train_and_evaluate_model(df_clustered, regressor, model_name="model"):
    """
    Treina e avalia um modelo de regressão para prever o valor_m2,
    utilizando os clusters como uma feature.
    """
    print(f"\n=== PREVISÃO DO VALOR DO IMÓVEL COM CLUSTERS USANDO {model_name.upper()} ===")

    # Definir as features (X) e o target (y)
    features_to_use = [
        'area_construida',
        'area_terreno',
        'ano_construcao',
        'padrao_acabamento', # Será tratada como categórica
        'cluster',         # Categórica
        'bairro',          # Categórica
        'tipo_imovel'      # Categórica
    ]

    df_model = df_clustered[['valor_m2'] + [f for f in features_to_use if f in df_clustered.columns]].copy()

    encoded_padrao_cols = [col for col in df_clustered.columns if col.startswith('padrao_acabamento_')]
    if encoded_padrao_cols:
        print(f"Detectadas colunas one-hot encoded para 'padrao_acabamento': {encoded_padrao_cols}")
        features_to_use = [f for f in features_to_use if f != 'padrao_acabamento']
        features_to_use.extend(encoded_padrao_cols)
        df_model = df_clustered[['valor_m2'] + features_to_use].copy()
    else:
        print("AVISO: 'padrao_acabamento' não encontrada ou não one-hot encoded. Removendo da lista de features.")
        features_to_use = [f for f in features_to_use if f != 'padrao_acabamento']
        df_model = df_clustered[['valor_m2'] + features_to_use].copy()

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
    numerical_features = [f for f in numerical_features if f in X.columns]

    categorical_features = ['cluster', 'bairro', 'tipo_imovel']
    if 'padrao_acabamento' in X.columns:
        categorical_features.append('padrao_acabamento')
    
    encoded_padrao_cols = [col for col in X.columns if col.startswith('padrao_acabamento_')]
    if encoded_padrao_cols:
        print(f"Usando colunas de 'padrao_acabamento' já one-hot encoded.")
        numerical_features.extend(encoded_padrao_cols)
        categorical_features = [f for f in categorical_features if f not in encoded_padrao_cols]
        categorical_features = [f for f in categorical_features if not f.startswith('padrao_acabamento_')]

    categorical_features = [f for f in categorical_features if f in X.columns]
    
    # Criar um pré-processador usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Criar o pipeline do modelo
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', regressor)
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

    print(f"\n=== Métricas de Avaliação do Modelo ({model_name.upper()}) ===")
    print(f"Mean Absolute Error (MAE): R$ {mae:,.2f}")
    print(f"Root Mean Squared Error (RMSE): R$ {rmse:,.2f}")
    print(f"R-squared (R²): {r2:.3f}")
    
    # Salvar o modelo treinado
    model_filename = f'PISI3-Project/property_value_prediction_{model_name}.joblib'
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

    # === Treinar e avaliar diferentes modelos ===

    # 1. Random Forest Regressor
    print("\n--- Avaliando Random Forest ---")
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model, rf_mae, rf_rmse, rf_r2 = train_and_evaluate_model(df_clustered, rf_regressor, "random_forest")

    # 2. Gradient Boosting Regressor
    print("\n--- Avaliando Gradient Boosting ---")
    gb_regressor = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb_model, gb_mae, gb_rmse, gb_r2 = train_and_evaluate_model(df_clustered, gb_regressor, "gradient_boosting")

    # 3. Linear Regression
    print("\n--- Avaliando Regressão Linear ---")
    lr_regressor = LinearRegression()
    lr_model, lr_mae, lr_rmse, lr_r2 = train_and_evaluate_model(df_clustered, lr_regressor, "linear_regression")

    print("\n=== Comparativo de Modelos ===")
    print(f"{'Modelo':<20} | {'MAE':<10} | {'RMSE':<10} | {'R²':<10}")
    print(f"{'-'*20} | {'-'*10} | {'-'*10} | {'-'*10}")
    print(f"{'Random Forest':<20} | {rf_mae:<10.2f} | {rf_rmse:<10.2f} | {rf_r2:<10.3f}")
    print(f"{'Gradient Boosting':<20} | {gb_mae:<10.2f} | {gb_rmse:<10.2f} | {gb_r2:<10.3f}")
    print(f"{'Linear Regression':<20} | {lr_mae:<10.2f} | {lr_rmse:<10.2f} | {lr_r2:<10.3f}")


if __name__ == "__main__":
    main()
