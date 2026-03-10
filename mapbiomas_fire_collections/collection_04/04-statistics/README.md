[LEMBRETE - REVISAR E TRADUZIR PARA O INGLES]

# 04. Estatísticas

Etapa final voltada para a extração de dados tabulares (CSV) a partir das cicatrizes de fogo mapeadas e refinadas.

## 📊 O que é processado
Os scripts nesta pasta calculam áreas queimadas cruzando com diversas bases geográficas:
- Municípios e Estados.
- Unidades de Conservação e Terras Indígenas.
- Categorias Fundiárias.
- Tipos de Cobertura e Uso (LULC).

## 📂 Divisão de Versões
- **[01-subproducts_collection_04](./01-subproducts_collection_04/)**: Estatísticas baseadas nos dados originais da Coleção 4.
- **[02-subproducts_collection_04_1](./02-subproducts_collection_04_1/)**: Estatísticas baseadas nos dados refinados da Coleção 4.1.

## 📤 Saídas (Output)
Os arquivos são exportados diretamente para o Google Drive do usuário configurado, facilitando a criação de gráficos e tabelas para os relatórios anuais.
- `toDrive-area-cobertura_queimada_mensal`: Área queimada por mês.
- `toDrive-area-frequencia`: Quantas vezes a mesma área queimou no período.
- `toDrive-area-tamanho_de_cicatrizes`: Distribuição por classe de tamanho.
