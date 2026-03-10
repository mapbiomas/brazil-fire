[LEMBRETE - REVISAR E TRADUZIR PARA O INGLES]

# Coleção 04 - MapBiomas Fogo

Esta pasta contém a estrutura completa de scripts e processamentos referentes à **Coleção 04** e seu desdobramento, a **Coleção 04.1**, do projeto MapBiomas Fogo.

## 📌 Visão Geral
A Coleção 04 representa o esforço de mapeamento de cicatrizes de fogo no Brasil, utilizando imagens Landsat e Sentinel. A versão 4.1 inclui refinamentos de reclassificação e ajustes espaciais baseados em dados de LULC (Uso e Cobertura da Terra) mais recentes.

## 📂 Estrutura de Pastas
- **[01-pre_classification](./01-pre_classification/)**: Ferramentas para geração de mosaicos, coleta de amostras e investigação de cenas.
- **[02-classification_algorithms](./02-classification_algorithms/)**: Algoritmos e modelos de classificação (Random Forest, etc).
- **[03-post_classification](./03-post_classification/)**: Filtros temporais, aplicação de máscaras de exclusão e geração de subprodutos.
- **[04-statistics](./04-statistics/)**: Scripts para cálculo de áreas e exportação de estatísticas tabulares para o Google Drive.

## 🚀 Fluxo de Trabalho
1. **Pré-processamento**: Preparação dos dados e ferramentas de suporte.
2. **Classificação**: Execução do mapeamento primário.
3. **Pós-processamento**: Refino dos resultados brutos e geração de máscaras anuais.
4. **Estatísticas**: Consolidação dos dados para relatórios e dashboards.

---
**Equipe MapBiomas Fogo**
