[LEMBRETE - REVISAR E TRADUZIR PARA O INGLES]

# 01. Pré-Classificação

Responsável pela preparação dos conjuntos de dados e ferramentas necessárias para o início do mapeamento de cicatrizes de fogo.

## 🛠️ Scripts Principais
- **01-toolkit_for_collection_samples_and_export_mosaics_to_google_cloud.js**: Interface principal para navegação, inspeção de mosaicos Landsat/Sentinel e exportação de amostras de treinamento.

## 📁 Pasta Auxiliar (`./auxiliar/`)
Contém módulos de suporte que são chamados pelo Toolkit ou usados de forma independente:
- **module-blockList.js**: Lista de cenas/imagens que devem ser ignoradas no processo por problemas de qualidade ou ruído.
- **toolkit-investigate-scenes.js**: Ferramenta para inspeção detalhada de cenas individuais e metadados.

## 📖 Como Usar
1. Configure os parâmetros de bioma e ano no `Toolkit`.
2. Utilize as camadas de visualização para identificar áreas de queima.
3. Exporte os mosaicos ou amostras para o Google Cloud Storage ou Assets do GEE para a etapa de classificação.
