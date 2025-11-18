import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import plotly.figure_factory as ff
import plotly.io as pio
import time

# Importar a função otimizada para carregar dados de clusterização
from clustering_analysis import get_clustering_data_optimized

def create_classification_target(df):
    # ... (código existente para criar o alvo de classificação - sem alterações)
    print("\n=== CRIANDO VARIÁVEL ALVO PARA CLASSIFICAÇÃO ===")
    quantiles = df['valor_m2'].quantile([0.33, 0.66]).values
    q1, q2 = quantiles[0], quantiles[1]
    print(f"Definindo categorias com base nos quantis de 'valor_m2':")
    print(f"  - Econômico: <= R$ {q1:,.2f}")
    print(f"  - Médio: > R$ {q1:,.2f} e <= R$ {q2:,.2f}")
    print(f"  - Alto Valor: > R$ {q2:,.2f}")
    df['categoria_valor'] = pd.cut(df['valor_m2'], bins=[-float('inf'), q1, q2, float('inf')], labels=['Econômico', 'Médio', 'Alto Valor'])
    print("\nDistribuição das classes criadas:")
    print(df['categoria_valor'].value_counts(normalize=True))
    return df

def train_classification_model(df):
    """
    Treina, otimiza e avalia um modelo de classificação.
    """
    print("\n=== OTIMIZAÇÃO E TREINAMENTO DO MODELO DE CLASSIFICAÇÃO ===")

    features_to_use = ['area_construida', 'area_terreno', 'ano_construcao', 'padrao_acabamento', 'cluster', 'bairro', 'tipo_imovel']
    existing_features = [f for f in features_to_use if f in df.columns]
    X = df[existing_features]
    y = df['categoria_valor']

    print(f"\nFeatures selecionadas: {X.columns.tolist()}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dados de treino: {len(X_train):,} registros")
    print(f"Dados de teste: {len(X_test):,} registros")

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Pipeline sem o classificador final
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))])

    # 2. DEFINIR GRADE DE HIPERPARÂMETROS PARA O GRIDSEARCH
    # Nota: A grade está pequena para uma execução mais rápida.
    # Para uma busca exaustiva, aumente o número de opções.
    param_grid = {
        'classifier__n_estimators': [100, 200],         # Número de árvores
        'classifier__max_depth': [10, 20, None],       # Profundidade máxima
        'classifier__min_samples_split': [2, 5]        # Mínimo de amostras para dividir
    }

    # 3. CONFIGURAR E EXECUTAR O GRIDSEARCHCV
    # cv=3 significa 3-fold cross-validation
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=2)
    
    print("\nIniciando otimização com GridSearchCV... (Isso pode levar alguns minutos)")
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    end_time = time.time()
    print(f"Otimização concluída em { (end_time - start_time) / 60:.2f} minutos.")

    # Exibir os melhores parâmetros encontrados
    print("\nMelhores parâmetros encontrados pelo GridSearchCV:")
    print(grid_search.best_params_)

    # O grid_search já retém o melhor modelo treinado com todos os dados de treino
    best_model = grid_search.best_estimator_

    # Fazer previsões no conjunto de teste com o modelo otimizado
    y_pred = best_model.predict(X_test)

    # Avaliar o modelo otimizado
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n=== Métricas de Avaliação do Classificador OTIMIZADO ===")
    print(f"Melhor acurácia da validação cruzada (CV): {grid_search.best_score_:.2%}")
    print(f"Acurácia no conjunto de teste: {accuracy:.2%}")
    print("\nRelatório de Classificação:")
    print(report)

    # Matriz de Confusão com Plotly
    cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
    fig = ff.create_annotated_heatmap(z=cm[::-1], x=list(best_model.classes_), y=list(best_model.classes_)[::-1], colorscale='Blues', showscale=True)
    fig.update_layout(title_text='Matriz de Confusão (Modelo Otimizado)', xaxis_title='Previsto', yaxis_title='Verdadeiro')
    pio.write_html(fig, 'PISI3-Project/confusion_matrix_optimized.html')
    print("\nMatriz de confusão interativa salva em 'PISI3-Project/confusion_matrix_optimized.html'")

    # Salvar o modelo otimizado
    model_filename = 'PISI3-Project/property_classifier_model_optimized.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Modelo OTIMIZADO salvo em: {model_filename}")

    return best_model

def main():
    """
    Função principal para orquestrar o processo.
    """
    df_clustered, _, _ = get_clustering_data_optimized()
    if df_clustered is None:
        print("Erro: Não foi possível carregar os dados clusterizados.")
        return

    df_classification = create_classification_target(df_clustered)
    train_classification_model(df_classification)


if __name__ == "__main__":
    main()
