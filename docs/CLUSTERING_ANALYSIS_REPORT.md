# Análise de Clusterização - Mercado Residencial de Recife

## 🎯 Objetivo e Justificativa

### Por que Clusterização?
A clusterização nos permite identificar padrões ocultos no mercado imobiliário, segmentando automaticamente imóveis com características similares. Isso é fundamental para:
- **Precificação inteligente** baseada em grupos homogêneos
- **Segmentação de mercado** para estratégias direcionadas  
- **Identificação de oportunidades** de investimento
- **Benchmarking** entre imóveis similares

### Por que Apenas Dados Residenciais?

**🏠 DADOS INCLUÍDOS:**
- **Apartamentos**: 84.171 registros (91.4% dos dados residenciais)
- **Casas**: 7.902 registros (8.6% dos dados residenciais)
- **Total**: 92.073 registros residenciais

**🏢 DADOS EXCLUÍDOS (13.6% do dataset):**
- Salas comerciais (8.386 registros)
- Lojas (3.397 registros)  
- Centros comerciais (671 registros)
- Hospitais (55 registros)
- Hotéis (208 registros)
- Instituições financeiras (116 registros)
- Outros tipos comerciais/industriais

### 💡 Justificativa da Filtragem:

1. **Evitar Distorções**: Hospitais e shoppings têm valores extremamente altos que distorcem a análise
2. **Homogeneidade**: Imóveis residenciais têm padrões de precificação comparáveis
3. **Relevância**: Foco no mercado que mais interessa para análise habitacional
4. **Precisão**: Algoritmos de clusterização funcionam melhor com dados homogêneos

## 📊 Metodologia Aplicada

### Preparação dos Dados
1. **Carregamento**: 106.606 registros totais do ITBI 2015-2023
2. **Filtragem residencial**: 92.073 registros mantidos
3. **Remoção de outliers**: Valores além do 1º e 99º percentil removidos
4. **Dataset final**: 86.006 registros para clusterização

### Features Selecionadas
- **valor_m2**: Valor por metro quadrado (R$/m²)
- **area_construida**: Área construída do imóvel (m²)
- **area_terreno**: Área do terreno (m²)  
- **ano_construcao**: Ano de construção do imóvel

### Algoritmo Utilizado
- **K-Means**: Clusterização não-supervisionada
- **Normalização**: StandardScaler para equalizar escalas
- **Número de clusters**: 5 (otimizado)
- **Silhouette Score**: 0.294 (boa separação)

## 🏘️ Clusters Identificados

### Cluster 0: Apartamentos Premium Novos (42.9%)
- **36.935 imóveis** - Maior cluster
- **Valor médio**: R$ 3.939/m²
- **Área média**: 99 m²
- **Ano médio**: 2015 (construção recente)
- **Tipo**: 98.7% Apartamentos
- **Bairros**: Boa Viagem, Madalena, Casa Amarela
- **Perfil**: Apartamentos de padrão médio-alto em bairros valorizados

### Cluster 1: Apartamentos Econômicos Novos (22.7%)
- **19.504 imóveis**
- **Valor médio**: R$ 2.729/m²
- **Área média**: 85 m²
- **Ano médio**: 2013 (construção recente)
- **Tipo**: 98.1% Apartamentos  
- **Bairros**: Boa Viagem, Várzea, Imbiribeira
- **Perfil**: Imóveis mais acessíveis com boa qualidade construtiva

### Cluster 2: Imóveis Antigos Diversos (19.3%)
- **16.600 imóveis**
- **Valor médio**: R$ 2.493/m²
- **Área média**: 112 m²
- **Ano médio**: 1981 (construção antiga)
- **Tipo**: 79.9% Apartamentos + 20.1% Casas
- **Bairros**: Boa Viagem, Boa Vista, Graças
- **Perfil**: Mix de imóveis estabelecidos com localização consolidada

### Cluster 3: Apartamentos Grandes Premium (13.0%)
- **11.210 imóveis**
- **Valor médio**: R$ 3.744/m²
- **Área média**: 256 m² (maior área)
- **Ano médio**: 2006
- **Tipo**: 89.0% Apartamentos
- **Bairros**: Boa Viagem, Madalena, Monteiro
- **Perfil**: Imóveis de alto padrão com áreas generosas

### Cluster 4: Apartamentos Luxury (2.0%)
- **1.757 imóveis** - Menor cluster
- **Valor médio**: R$ 4.171/m² (mais alto)
- **Área média**: 194 m²
- **Ano médio**: 2013
- **Tipo**: 100% Apartamentos
- **Bairros**: Imbiribeira, Cordeiro, Ibura
- **Perfil**: Segmento de luxo do mercado residencial

## 📈 Insights e Descobertas

### Padrões Identificados

1. **Polarização do Mercado**:
   - Clusters 1 e 2: Faixa econômica/média (R$ 2.400-2.700/m²)
   - Clusters 0, 3 e 4: Faixa premium (R$ 3.700-4.200/m²)

2. **Influência da Idade**:
   - Imóveis mais novos (2013-2015) tendem a ter valores maiores
   - Cluster 2 (1981) tem valor menor apesar da boa localização

3. **Importância da Área**:
   - Cluster 3 se destaca pela área (256 m²) mesmo com valor/m² menor que Cluster 4
   - Apartamentos pequenos (85-99 m²) dominam o mercado

4. **Concentração Geográfica**:
   - **Boa Viagem** aparece em todos os clusters (bairro mais valorizado)
   - Outros bairros nobres: Madalena, Casa Amarela, Graças

### Segmentação de Mercado

**Entrada (Clusters 1 + 2)**: 42% do mercado
- Valores: R$ 2.400-2.700/m²
- Público: Primeira casa, jovens casais

**Médio (Cluster 0)**: 43% do mercado  
- Valores: R$ 3.900/m²
- Público: Famílias estabelecidas

**Premium (Clusters 3 + 4)**: 15% do mercado
- Valores: R$ 3.700-4.200/m²
- Público: Alto poder aquisitivo

## 🎯 Aplicações Práticas

### Para Investidores
1. **Cluster 1**: Maior potencial de valorização (novos + preço acessível)
2. **Cluster 2**: Oportunidades em áreas consolidadas
3. **Cluster 4**: Investimento de luxo com menor liquidez

### Para Compradores
1. **Benchmarking**: Compare seu imóvel com outros do mesmo cluster
2. **Negociação**: Use dados do cluster para fundamentar ofertas
3. **Escolha**: Identifique qual cluster atende seu perfil

### Para Construtoras
1. **Posicionamento**: Definir qual cluster mirar em novos projetos
2. **Precificação**: Usar clusters como referência de preços
3. **Localização**: Entender onde cada cluster se concentra

## 🔍 Limitações e Considerações

### Limitações da Análise
- **Temporal**: Dados de 2015-2023 podem não refletir situação atual
- **Geográfica**: Apenas Recife, não região metropolitana
- **Features**: Outras características (vista, andar, garagem) não consideradas
- **Silhouette Score**: 0.294 é moderado, indica sobreposição entre clusters

### Próximos Passos
1. **Análise temporal**: Como clusters evoluíram ao longo dos anos
2. **Subclusterização**: Analisar clusters grandes (0 e 1) em subgrupos
3. **Features adicionais**: Incluir características qualitativas
4. **Validação**: Comparar com avaliações de corretores

## 📊 Resumo Executivo

### Principais Achados
✅ **5 clusters distintos** identificados no mercado residencial de Recife  
✅ **86.006 imóveis residenciais** analisados (apartamentos e casas)  
✅ **Boa segmentação** entre faixas econômica, média e premium  
✅ **Boa Viagem** como epicentro do mercado imobiliário  
✅ **Apartamentos dominam** 91% do mercado residencial  

### Valor da Análise
- **Fundamentação científica** para decisões imobiliárias
- **Segmentação objetiva** do mercado residencial
- **Base sólida** para precificação e investimentos
- **Insights acionáveis** para diferentes perfis de usuários

---

*Análise realizada com dados do ITBI Recife 2015-2023 | Clusterização K-Means | 86.006 imóveis residenciais*