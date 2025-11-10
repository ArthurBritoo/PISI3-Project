# PISI3-Project

Análise completa do mercado imobiliário de Recife utilizando dados do ITBI (2015-2023).

## 🚀 Aplicação Principal: `app.py`

App unificado com **4 modos de análise**:

### 📈 1. EDA Exploratória
Análise exploratória geral do mercado imobiliário de Recife
- Valor m² por bairro
- Quantidade de transações
- Padrão de acabamento
- Evolução temporal
- Distribuição de tipos de imóveis

### 🎯 2. Clustering de Perfis
Segmentação inteligente do mercado residencial em **5 perfis** usando K-means:
- 🏠 **Popular**: Menor valor, área compacta, padrão simples
- 🏡 **Entrada**: Intermediário inferior, bom custo-benefício
- 🏘️ **Intermediário**: Padrão médio, áreas moderadas
- 🏙️ **Alto Padrão**: Imóveis maiores, acabamento superior
- 💎 **Premium**: Elite do mercado, máximo valor e área

**Técnicas:** K-means, StandardScaler, Silhouette Score (~0.29)

### 🗺️ 3. Dashboard Regional
Análise por regiões geográficas com agrupamento de subdistritos IBGE
- Consulta API do IBGE para geometrias de subdistritos de Recife
- Agrupa subdistritos vizinhos por mínimo de transações
- Filtros avançados: período, agregação, top N
- Visualizações regionais interativas

**Técnicas:** Geo-clustering IBGE, grafos de adjacência espacial, fuzzy matching

### 🔥 4. Análise Integrada ⭐
**Inovação:** Cruzamento único de **Perfis de Mercado × Regiões Geográficas**
- Heatmap de concentração de perfis por região
- Perfil dominante em cada região
- Scatter valor médio × volume
- Explorador interativo com filtros
- Download de dados integrados

## 📁 Estrutura do Projeto

```
PISI3-Project/
├── app.py                          # 🌟 App principal unificado (USE ESTE!)
├── eda.py                          # [Legado] EDA + Clustering
├── dashboard.py                    # [Legado] Dashboard regional
├── clustering_app.py               # [Legado] App standalone de clustering
├── data_processing.py              # Carregamento e pré-processamento (Parquet)
├── clustering_analysis.py          # Pipeline de clustering K-means + cache
├── geo_clustering.py               # Módulo de geo-clustering IBGE
├── generate_cache.py               # Gerador de cache de clustering
├── convert_to_parquet.py           # Utilitário de conversão CSV → Parquet
├── charts/
│   └── charts.py                   # Funções de visualização Plotly
├── data/
│   ├── itbi_20*.parquet            # Dados ITBI em Parquet (otimizado)
│   ├── clustering_cache.parquet    # Cache de clustering (ignorado no git)
│   └── clustering_metadata.json    # Metadados do clustering (ignorado no git)
├── requirements.txt                # Dependências Python
├── .gitignore                      # Ignora cache, venv, __pycache__
├── .gitattributes                  # Marca *.parquet como binário
├── CLUSTERING_ANALYSIS_REPORT.md   # Documentação do clustering
└── PARQUET_MIGRATION_SUMMARY.md    # Documentação da migração Parquet
```

## 🛠️ Instalação e Uso

### 1. Clonar o repositório
```bash
git clone https://github.com/ArthurBritoo/PISI3-Project.git
cd PISI3-Project
```

### 2. Criar ambiente virtual e instalar dependências
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# ou: source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3. Gerar cache de clustering (primeira vez)
```bash
python generate_cache.py
```

### 4. Executar aplicação principal
```bash
streamlit run app.py
```

Acesse: http://localhost:8501

## 📊 Dados

- **Fonte:** ITBI Recife 2015-2023
- **Formato:** Parquet (migrado de CSV para otimização)
- **Redução de tamanho:** ~78% vs CSV
- **Tipos residenciais:** Apartamentos e Casas
- **Total de registros:** ~86.000 imóveis residenciais após filtragem

## 🧪 Técnicas e Tecnologias

### Análise de Dados
- **Pandas** + **PyArrow**: Processamento eficiente com Parquet
- **NumPy**: Operações numéricas
- **Scikit-learn**: K-means, StandardScaler, Silhouette Score

### Geo-Análise
- **Shapely**: Operações geométricas e cálculo de adjacência
- **Requests**: Consulta à API do IBGE
- **difflib**: Fuzzy matching de nomes de bairros/subdistritos

### Visualização
- **Streamlit**: Interface web interativa
- **Plotly**: Gráficos interativos (scatter, bar, heatmap, pie)

### Performance
- Cache de clustering em Parquet (~0.15s vs ~10s)
- Cache de regiões IBGE
- Decoradores `@st.cache_data` para otimização

## 📝 Documentação Adicional

- [CLUSTERING_ANALYSIS_REPORT.md](CLUSTERING_ANALYSIS_REPORT.md): Detalhes da análise de clustering
- [PARQUET_MIGRATION_SUMMARY.md](PARQUET_MIGRATION_SUMMARY.md): Migração CSV → Parquet

## 👥 Contribuidores

- **Arthur Brito** (@ArthurBritoo): Clustering, migração Parquet, integração
- **Gustavo Macena**: Dashboard regional, geo-clustering IBGE

## 📜 Licença

MIT License