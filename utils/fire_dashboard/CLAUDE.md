
- **Severidade:** o gráfico de barras da distribuição usa `rotulos: "texto"` + `rotulosTexto` (valor em Mha/ha escrito no fim da barra, na cor da classe).
- **Rede:** `fetchTerritoryTile` repete até 4 vezes e `apiGet` também repete "Falha de rede" (antes um "Failed to fetch" derrubava o ranking de municípios).
- Uso e cobertura, boxplot por período: horizontal (períodos no eixo Y), com p do Kruskal-Wallis desenhado no canto do canvas e quadro de destaque ao lado (`boxplotFixo.lado`: selo Mudou/Indício/Estável, manchete, Kruskal-Wallis, Mann-Kendall com Sen por década, Mann-Whitney último período × anteriores). O card tem seus próprios controles: nível de legenda e depois a classe do nível. O PNG exporta só o gráfico (com o p).
