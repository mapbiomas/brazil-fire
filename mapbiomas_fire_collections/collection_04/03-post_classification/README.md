[LEMBRETE - REVISAR E TRADUZIR PARA O INGLES]

# 03. Pós-Classificação

Esta etapa consolida os resultados da classificação bruta, aplicando filtros de qualidade e integrando com mapas de uso e cobertura da terra (MapBiomas).

## 📂 Subpastas e Funções

### [00-annual_classification](./00-annual_classification/)
Foca na geração do produto anual e aplicação de máscaras:
- **merge_collections_pampa.js**: Consolida diferentes versões de classificação para o bioma Pampa.
- **export_col4_masks-regions_lulc9/10.js**: Gera máscaras de exclusão baseadas nos dados de LULC (Coleções 9 e 10) para remover falsos positivos em áreas que não podem queimar.

### [01-subproducts_collection_04](./01-subproducts_collection_04/)
Geração de subprodutos para a **versão original** da Coleção 04:
- Scripts numerados de 01 a 07 (Exportação, Filtros Temporais, Diferença de Cicatrizes, etc).

### [02-subproducts_collection_04_1](./02-subproducts_collection_04_1/)
Geração de subprodutos para a **versão 4.1**:
- Mesma lógica de processamento da pasta anterior, porém utilizando os inputs corrigidos e refinados da 4.1.

## 🔄 Ordem Recomendada de Processamento
1. Realizar a mesclagem e máscaras na pasta `00`.
2. Executar sequencialmente os scripts de 01 a 07 nas pastas de subprodutos para gerar os Assets finais de exportação.
