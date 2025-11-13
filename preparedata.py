def prepare_data(df_clustered):
    """
    Prepara as features e o target, define colunas, e divide os dados.
    Retorna X_train, y_train, X_test, y_test e as features brutas (X).
    """
    features_to_use = [
        'area_construida',
        'area_terreno',
        'ano_construcao',
        'cluster',
        'bairro',
        'tipo_imovel'
    ]
    # Adicionando a lógica para 'padrao_acabamento' já codificada
    encoded_padrao_cols = [col for col in df_clustered.columns if col.startswith('padrao_acabamento_')]
    if encoded_padrao_cols:
        features_to_use.extend(encoded_padrao_cols)
    else:
        # Se 'padrao_acabamento' for categórica, garanta que está na lista se for usar OHE
        if 'padrao_acabamento' in df_clustered.columns:
            features_to_use.append('padrao_acabamento')

    features_to_use = [f for f in features_to_use if f in df_clustered.columns]
    
    df_model = df_clustered[['valor_m2'] + features_to_use].copy()
    X = df_model[features_to_use]
    y = df_model['valor_m2']

    # Separar dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, X.columns.tolist()
