**Online**: https://mapbiomas.github.io/brazil-fire/fire_monitor_highlights/ (Brasil: `/brazil/` · Perú: `/peru/`)

**Localmente**: abra [`src/brazil/boletim-fogo.html`](src/brazil/boletim-fogo.html) ou [`src/peru/boletin-fuego.html`](src/peru/boletin-fuego.html) no navegador (mantenha o logo da pasta junto do HTML). Não há build nem dependências para instalar.

# Boletim do Monitor do Fogo 🔥

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

## 🗂️ Estrutura desta pasta

```
utils/fire_monitor_highlights/
├── README.md
├── CLAUDE.md                     # documentação técnica
├── .gitignore
├── logos/                        # logos oficiais: Monitor_EN.png, Monitor_ES.png, Monitor_PT.png
└── src/
    ├── brazil/                   # Brasil (português) · API plataforma.monitorfogo.mapbiomas.org
    │   ├── boletim-fogo.html     # boletim do mês + Destaque dinâmico
    │   └── Monitor_PT.png
    └── peru/                     # Perú (español) · API fogo-peru.geodatin.com
        ├── boletin-fuego.html    # só o boletim do mês (5 folhas A4)
        └── Monitor_ES.png
```

Cada país tem a sua pasta, com um HTML independente: as APIs, os tipos de território (estados e biomas no Brasil; departamentos, biomas e distritos no Peru), os dados disponíveis e o idioma são diferentes. Para um novo país, copie a pasta mais parecida e adapte a API e os textos.

## 🌎 Outros países

- **Brasil** (`src/brazil/`): boletim do mês com 5 folhas e o painel Destaque dinâmico.
- **Peru** (`src/peru/`): boletim do mês com 5 folhas, em espanhol, sem o painel dinâmico. Limites: a API do Peru só tem dados desde julho de 2025 (não há média histórica) e não tem dados de maio de 2026; o boletim avisa isso no rodapé e no gráfico acumulado.
- Outros países: ainda não adaptados.

## 📚 Para desenvolvedores

Leia [CLAUDE.md](CLAUDE.md) para documentação técnica completa.

---

**Desenvolvido com dados do [MapBiomas Fogo](https://www.monitorfogo.mapbiomas.org/)**
