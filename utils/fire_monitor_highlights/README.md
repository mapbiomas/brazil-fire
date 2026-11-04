# Boletim do Monitor do Fogo 🔥

**Protótipo**: Teste de automação do boletim mensal do [Monitor do Fogo](https://plataforma.monitorfogo.mapbiomas.org/).

Relatório visual com dados ao vivo sobre área queimada no Brasil, exportável em PDF.

## 📄 O que tem

- **Boletim do Mês** (5 folhas A4 paisagem) ✅
  - Brasil: o mês e o acumulado do ano lado a lado, com comparação com o ano anterior e a média, série por ano e uso e cobertura
  - Ranking de biomas e ranking de estados, com mapa e participação de cada um no total do Brasil
  - Bioma e estado mais afetados: mapa (cicatrizes do ano no bioma; municípios no estado) e o mesmo comparativo mês × acumulado

- **Destaque Dinâmico** (painel interativo, versão de teste)
  - Qualquer recorte da API: Brasil, bioma, estado, município, terra indígena, UC, quilombo, bacias e outros
  - Período livre (ano, mês inicial e final) e filtro por classe de cobertura
  - Sazonalidade, acumulado contra a média, série anual, calendário do fogo e uso e cobertura (níveis 1 a 4)
  - Mapa único por subdivisão (área queimada ou variação % contra o ano anterior ou a média histórica), cicatrizes de fogo e mapa de calor
  - Top 15 com microbarras, comparado com o ano anterior ou com a média histórica, com filtro para ver apenas os territórios acima da média, ordem por maior área ou maior aumento, e clique para abrir o território
  - Download dos dados em CSV, dos gráficos em PNG, PDF do painel e link com os filtros
  - "Fora do normal?": o território e o período escolhidos comparados à variação típica dos anos anteriores, no território e em cada subdivisão

## 🚀 Acesso

**Online**: https://mapbiomas.github.io/brazil-fire/fire_monitor_highlights/

**Localmente**: abra [`src/boletim-fogo.html`](src/boletim-fogo.html) no navegador (mantenha `Monitor_PT.png` na mesma pasta). Não há build nem dependências para instalar.

## 🗂️ Estrutura desta pasta

```
utils/fire_monitor_highlights/
├── README.md
├── CLAUDE.md                 # documentação técnica
├── .gitignore
└── src/
    ├── boletim-fogo.html     # aplicação (única fonte de verdade)
    └── Monitor_PT.png        # logo usado nas folhas e na impressão
```

## 🌎 Outros países

Hoje o boletim foi construído para o Brasil e lê a API pública do Monitor do Fogo. A ideia é que o mesmo gerador sirva a outros países da rede MapBiomas Fogo; essa adaptação ainda não foi feita.

## 📚 Para desenvolvedores

Leia [CLAUDE.md](CLAUDE.md) para documentação técnica completa.

---

**Desenvolvido com dados do [MapBiomas Fogo](https://www.monitorfogo.mapbiomas.org/)**
