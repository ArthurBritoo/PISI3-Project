
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import plotly.figure_factory as ff
import plotly.io as pio

# Importar a função otimizada para carregar dados de clusterização
from clustering_analysis import get_clustering_data_optimized

def create_classification_target(df):
    """
    Cria uma variável alvo categórica a partir do 'valor_m2'.
    """
    print("\n=== CRIANDO VARIÁVEL ALVO PARA CLASSIFICAÇÃO ===")
    
    # Usar quantis para criar 3 categorias com distribuição de dados semelhante
    quantiles = df['valor_m2'].quantile([0.33, 0.66]).values
    q1 = quantiles[0]
    q2 = quantiles[1]
    
    print(f"Definindo categorias com base nos quantis de 'valor_m2':")
    print(f"  - Econômico: <= R$ {q1:,.2f}")
    print(f"  - Médio: > R$ {q1:,.2f} e <= R$ {q2:,.2f}")
    print(f"  - Alto Valor: > R$ {q2:,.2f}")

    # Criar a coluna 'categoria_valor'
    df['categoria_valor'] = pd.cut(df['valor_m2'],
                                   bins=[-float('inf'), q1, q2, float('inf')],
                                   labels=['Econômico', 'Médio', 'Alto Valor'])
    
    # Verificar a distribuição das classes
    print("\nDistribuição das classes criadas:")
    print(df['categoria_valor'].value_counts(normalize=True))
    
    return df

def train_classification_model(df):
    """
    Treina e avalia um modelo de classificação para prever a 'categoria_valor'.
    """
    print("\n=== TREINAMENTO DO MODELO DE CLASSIFICAÇÃO ===")

    # Definir as features (X) e o target (y)
    # Excluímos 'valor_m2' pois a nossa variável alvo é derivada dele
    features_to_use = [
        'area_construida', 'area_terreno', 'ano_construcao', 'padrao_acabamento',
        'cluster', 'bairro', 'tipo_imovel'
    ]
    
    # Garantir que todas as features existem no DataFrame
    existing_features = [f for f in features_to_use if f in df.columns]
    X = df[existing_features]
    y = df['categoria_valor']

    print(f"\nFeatures selecionadas: {X.columns.tolist()}")

    # Separar dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Dados de treino: {len(X_train):,} registros")
    print(f"Dados de teste: {len(X_test):,} registros")

    # Identificar colunas numéricas e categóricas para pré-processamento
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Criar um pré-processador usando ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )

    # Criar o pipeline do modelo
    # **PRÓXIMO PASSO**: Adicionar SMOTEN ao pipeline aqui, se necessário.
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        # **PRÓXIMO PASSO**: Integrar GridSearchCV ou RandomizedSearchCV aqui.
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print("\nIniciando treinamento do modelo de classificação...")
    model.fit(X_train, y_train)
    print("Treinamento concluído!")

    # Fazer previsões no conjunto de teste
    y_pred = model.predict(X_test)

    # Avaliar o modelo
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"\n=== Métricas de Avaliação do Classificador ===")
    print(f"Acurácia: {accuracy:.2%}")
    print("\nRelatório de Classificação:")
    print(report)

    # Matriz de Confusão com Plotly
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    
    # Plotly espera que os dados da matriz de confusão sejam invertidos verticalmente para a exibição correta
    fig = ff.create_annotated_heatmap(
        z=cm[::-1], # inverte a matriz
        x=list(model.classes_),
        y=list(model.classes_)[::-1], # inverte as labels do eixo y
        colorscale='Blues',
        showscale=True
    )
    
    fig.update_layout(
        title_text='Matriz de Confusão',
        xaxis_title='Previsto',
        yaxis_title='Verdadeiro'
    )
    
    # Salvar como um arquivo HTML interativo
    pio.write_html(fig, 'PISI3-Project/confusion_matrix.html')
    print("\nMatriz de confusão interativa salva em 'PISI3-Project/confusion_matrix.html'")


    # Salvar o modelo treinado
    # **PRÓXIMO PASSO**: Implementar explicabilidade com SHAP usando este modelo.
    model_filename = 'PISI3-Project/property_classifier_model.joblib'
    joblib.dump(model, model_filename)
    print(f"Modelo de classificação salvo em: {model_filename}")

    return model

def main():
    """
    Função principal para orquestrar o processo.
    """
    # Carregar dados clusterizados
    df_clustered, _, _ = get_clustering_data_optimized()
    if df_clustered is None:
        print("Erro: Não foi possível carregar os dados clusterizados.")
        return

    # 1. Transformar em problema de classificação
    df_classification = create_classification_target(df_clustered)
    
    # 2. Treinar e avaliar o modelo de classificação
    train_classification_model(df_classification)


if __name__ == "__main__":
    main()
