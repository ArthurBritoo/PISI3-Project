import pandas as pd
import shap
import joblib
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Importar funções de preparação de dados dos scripts anteriores
from clustering_analysis import get_clustering_data_optimized
from classification_model import create_classification_target

def generate_shap_explanations():
    """
    Carrega o modelo otimizado, calcula os valores SHAP e gera gráficos de explicabilidade.
    """
    print("\n=== GERANDO EXPLICAÇÕES SHAP PARA O MODELO OTIMIZADO ===")

    # 1. CARREGAR DADOS E PREPARÁ-LOS
    print("Carregando e preparando dados...")
    df_clustered, _, _ = get_clustering_data_optimized()
    if df_clustered is None:
        print("Erro ao carregar dados. Abortando.")
        return

    df_classification = create_classification_target(df_clustered)
    features_to_use = ['area_construida', 'area_terreno', 'ano_construcao', 'padrao_acabamento', 'cluster', 'bairro', 'tipo_imovel']
    X = df_classification[[f for f in features_to_use if f in df_classification.columns]]
    y = df_classification['categoria_valor']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 2. CARREGAR O MODELO OTIMIZADO
    model_filename = 'PISI3-Project/property_classifier_model_optimized.joblib'
    print(f"Carregando modelo otimizado de: {model_filename}")
    try:
        best_model = joblib.load(model_filename)
    except FileNotFoundError:
        print(f"Erro: Arquivo do modelo não encontrado. Execute o script de classificação primeiro.")
        return

    preprocessor = best_model.named_steps['preprocessor']
    classifier = best_model.named_steps['classifier']

    # 3. PREPARAR DADOS PARA O SHAP
    print("Aplicando pré-processamento aos dados...")
    X_test_transformed = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()
    X_test_transformed_df = pd.DataFrame(X_test_transformed.toarray(), columns=feature_names)
    sample_size = 800
    X_test_sample = X_test_transformed_df.sample(min(sample_size, len(X_test_transformed_df)), random_state=42)

    # 4. CALCULAR VALORES SHAP (API MODERNA E OTIMIZADA)
    print("Calculando valores SHAP com a API moderna... (Muito mais rápido!)")
    explainer = shap.TreeExplainer(classifier)
    # Desativar check_additivity para um grande ganho de performance
    shap_values_obj = explainer(X_test_sample, check_additivity=False)

    class_names = best_model.classes_

    # 5. GERAR E SALVAR GRÁFICOS DE EXPLICAÇÃO GLOBAL
    # Gráfico de Barras (Feature Importance Global)
    print("Gerando gráfico de barras de importância das features...")
    plt.figure()
    # A nova API usa o objeto shap_values_obj diretamente
    shap.summary_plot(shap_values_obj, plot_type="bar", feature_names=X_test_sample.columns, class_names=class_names, show=False)
    plt.title("Importância Global das Features (SHAP)")
    plt.savefig('PISI3-Project/shap_summary_bar.png', bbox_inches='tight')
    plt.close()
    print("Gráfico salvo em: PISI3-Project/shap_summary_bar.png")

    # Gráfico Beeswarm (Distribuição do Impacto das Features)
    print("Gerando gráficos beeswarm...")
    # O objeto Explanation contém os valores para todas as classes
    # Acessamos os valores de cada classe com a sintaxe [:, :, i]
    for i, class_name in enumerate(class_names):
        plt.figure()
        shap.summary_plot(shap_values_obj[:, :, i], features=X_test_sample, show=False)
        plt.title(f'Impacto das Features na Classe: {class_name}')
        plt.savefig(f'PISI3-Project/shap_summary_beeswarm_{class_name}.png', bbox_inches='tight')
        plt.close()
    print("Gráficos beeswarm salvos em: PISI3-Project/")

    # Gráfico SHAP: Importância por classe (Multiclasse Bar Plot)
    print("Gerando Gráfico de Importância por Classe (Barra Multiclasse)...")
    try:
        plt.figure(figsize=(12, 8)) 
        # A função summary_plot gera o gráfico de barras multiclasse automaticamente
        # quando recebe um objeto de explicação (shap_values_obj) multiclasse.
        shap.summary_plot(
            shap_values=shap_values_obj,
            features=X_test_sample, # Passamos features para nomes e contexto
            plot_type="bar",        # Especifica o tipo de gráfico como barra
            class_names=class_names,# Nomes das classes
            show=False              # Não exibe, apenas salva
        )
        plt.title("Importância das Features Segmentada por Classe (SHAP)")
        # Salva o gráfico com o novo nome
        plt.savefig(
            os.path.join('PISI3-Project/shap_summary_bar_multiclass.png'),
            bbox_inches='tight'
        )
        plt.close()
        print(f"NOVO Gráfico 'shap_summary_bar_multiclass.png' gerado e salvo em: PISI3-Project/shap_summary_bar_multiclass.png")

    except Exception as e:
        print(f"Erro ao gerar o gráfico de barras multiclasse SHAP: {e}")

    # 6. GERAR E SALVAR GRÁFICO DE EXPLICAÇÃO LOCAL
    print("Gerando gráfico de força para uma predição local...")
    prediction_array = classifier.predict(X_test_sample.iloc[[0]])
    predicted_class_index = list(class_names).index(prediction_array[0])

    # A nova API simplifica muito a chamada do force_plot
    # Passamos apenas a explicação para a amostra 0 e a classe predita
    force_plot = shap.force_plot(shap_values_obj[0, :, predicted_class_index], matplotlib=False)
    
    # Salvar o gráfico de força como um arquivo HTML
    shap.save_html('PISI3-Project/shap_force_plot_local.html', force_plot)
    print("Gráfico de força local salvo em: PISI3-Project/shap_force_plot_local.html")

if __name__ == "__main__":
    generate_shap_explanations()
