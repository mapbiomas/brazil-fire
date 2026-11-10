# Boletim do Monitor do Fogo — Documentação

**Protótipo**: automação do boletim mensal do [Monitor do Fogo](https://plataforma.monitorfogo.mapbiomas.org/) — relatório visual sobre área queimada no Brasil, exportável em PDF.

> **Local no repositório**: `utils/fire_monitor_highlights/`, uma pasta por país em `src/` (`src/brazil/` e `src/peru/`). Esta documentação descreve o boletim do **Brasil** (`src/brazil/boletim-fogo.html`); a tabela abaixo foi escrita para o repositório original `veralaisa/boletim-monitor-fogo`, onde os arquivos ficam na raiz. Publicado em https://mapbiomas.github.io/brazil-fire/fire_monitor_highlights/brazil/ (workflow `.github/workflows/pages.yml`).

> **Peru** (`src/peru/boletin-fuego.html`): derivado do boletim do Brasil, em espanhol, só com o boletim do mês. Usa a API `https://fogo-peru.geodatin.com` (mesmo formato; território `country` com código 1; subdivisões `department`, `province`, `district`, `biome`; tiles em `/api/territories/grouping/tiles/{grouping}`; `/bounds` devolve só a caixa). A API do Peru tem dados só desde julho de 2025 e o mês de maio de 2026 não existe.

## 📁 Arquivos

| Arquivo | Papel |
|---|---|
| `boletim-fogo.html` | **Versão consolidada e publicada.** É a única fonte de verdade; vira o `index.html` do repositório. |
| `boletim-fogo-versao-2.html` | Backup local da revisão anterior. Não é publicado e não deve ser editado. |
| `boletim-fogo-backup-2026-10-27.html` | Backup local antes da aba Destaques (boletim antigo de 7 páginas retrato). |
| `Monitor_PT.png` | Logo usado no cabeçalho das folhas e no cabeçalho de impressão do painel. |
| `README.md` | Visão geral para o público. |
| `CLAUDE.md` | Este documento técnico. |

> **Regra do projeto**: não fazer commit nem push sem autorização explícita (política do repositório: `utils/_base/docs/politica_commit_push.md`). Padrões compartilhados (logos, textos de interface, README em inglês): `utils/_base/`. A pasta `logos/` desta ferramenta foi removida; as logos oficiais estão em `utils/_base/logos/` e cada país mantém uma cópia ao lado do HTML.

---

## 🏗️ Arquitetura

### Stack
- **HTML5 + CSS3 + JavaScript vanilla** (sem frameworks, sem build)
- **Bibliotecas via CDN**:
  - [Chart.js 4.4.1](https://www.chartjs.org/) — gráficos
  - [D3.js 7.9.0](https://d3js.org/) — mapas de estados e biomas (SVG)
  - [Leaflet 1.9.4](https://leafletjs.com/) — mapas
  - [Leaflet.VectorGrid 1.3.0](https://github.com/Leaflet/Leaflet.VectorGrid) — tiles vetoriais (MVT)
  - [Montserrat](https://fonts.google.com/specimen/Montserrat) — fonte institucional
- **Dados**: API pública do Monitor do Fogo (sem autenticação)

### Estrutura do arquivo

```
boletim-fogo.html
├─ <head> — metadados, CSS, imports de CDN
├─ <body>
│  ├─ .top — barra do app (logo, abas, botão PDF)
│  ├─ #tab-boletim — "Boletim do mês" (aba inicial): 5 folhas A4 paisagem
│  │  ├─ #d1 — Brasil: título + layout base Mês × Acumulado (.mxg) com cards de contexto
│  │  ├─ #d2 — Ranking de biomas: mapa D3 (GEO_BIO), selo circular, KPI e barras com ha e %
│  │  ├─ #d3 — Bioma mais afetado: cicatrizes do ano (Leaflet) + layout base em 3 colunas (.c3)
│  │  ├─ #d4 — Ranking de estados: mapa D3, selo com os 3 primeiros, KPI, resumo e barras com ha e %
│  │  └─ #d5 — Estado mais afetado: mapa municipal (MVT) com localizador circular + layout base (.c3)
│  └─ #tab-explorer — Destaque dinâmico (painel interativo; filtros no hash #explorer?…)
│     └─ … no fim: "Fora do normal?" (#x-sec-fn, #x-fn-z, #x-fn-sub, #x-fn-como), que segue os filtros
└─ <script> — lógica JavaScript
```

O antigo "Boletim do mês" (7 páginas A4 retrato, `#pg1`…`#pg7`) foi removido em 2026-10-28. Dele, só o "Fora do normal?" continua, agora no fim do Destaque dinâmico e (desde 2026-10-29) acompanhando o território e o período dos filtros.

### Folhas do boletim

| Folha | Conteúdo |
|---|---|
| 1 | Brasil: "BRASIL" + "Destaques do fogo no Brasil em {mês} de {ano}"; colunas Mês e Acumulado, cada uma com card (total, variações e card de contexto: "Vegetação nativa" no mês, "Pico do ano" no acumulado), barras por ano e barra LULC |
| 2 | Ranking de biomas: mapa, selo "O Cerrado foi o bioma mais afetado", KPI do Brasil, barras "1.559.913 ha · 60%" |
| 3 | Bioma mais afetado: selo + título, cicatrizes de jan–mês sobre o limite do bioma, colunas Mês e Acumulado do bioma |
| 4 | Ranking de estados: mapa, selo com os 3 primeiros, KPI, resumo ("Os 10 primeiros estados somam…"), barras com ha e % |
| 5 | Estado mais afetado: selo + título, mapa municipal com localizador circular e legenda sem fundo abaixo, colunas Mês e Acumulado do estado |

**Layout A4 paisagem**: cada folha tem 297 × 210 mm. Ao mexer em alturas, gerar o PDF e conferir que continuam 5 páginas e que nada encosta no rodapé.

---

## 🔑 Constantes principais

### APIs e dados
```javascript
const API = 'https://plataforma.monitorfogo.mapbiomas.org';
const GEO_URL = '...brasil-states.geojson'; // Estados (click_that_hood)
const GEO_BIO = {...};                      // Biomas (IBGE 1:250 mil, simplificado e embutido)
```

### Referências geográficas
```javascript
const UF = [...]           // 27 estados com código IBGE (UF_BY_NAME: nome → estado)
const BIOMAS = [...]       // 6 biomas (Amazônia=1, Caatinga=2, ...); o código é o mesmo do ranking da API
const PREF = { country: ['state'], biome: ['biome'], state: ['city'] }
```

### Cobertura do solo
```javascript
const PALETA_LULC = [...]  // Paleta oficial: código, nome, cor, grupo e id da API
const COR_LULC = {...}     // id → cor (sobrepõe as cores da API no painel)
const ANTROPICAS = new Set([...])  // IDs de classes antrópicas (painel)
const NATIVAS = new Set([...])     // IDs de vegetação nativa (painel)
const RAMP5 = [...]        // 5 tons de vermelho para mapas de fogo
const SEM_DADO = '#EFEFEF'
```

Paleta oficial de uso e cobertura (`PALETA_LULC`, obrigatória em todos os gráficos de cobertura):

| Grupo | Classes (cor) |
|---|---|
| 1. Floresta `#1f8d49` | Formação Florestal `#1f8d49`, Floresta Alagável `#007785`, Formação Savânica `#7dc975`, Savana Alagada `#228c70`, Mangue `#04381d`, Restinga Arbórea `#02d659` |
| 2. Vegetação Herbácea e Arbustiva `#d6bc74` | Formação Campestre `#d6bc74`, Formação Herbáceo Arbustiva `#86b074`, Campo Alagado e Área Pantanosa `#519799`, Marisma `#81dbbf`, Restinga Herbácea `#ffaa5f`, Apicum `#fc8114`, Afloramento Rochoso `#ad5100` |
| 3. Agropecuária `#ffefc3` | Pastagem `#edde8e`, Agricultura `#E974ED` (Lavoura Temporária `#C27BA0`: Soja `#f5b3c8`, Cana `#db7093`, Arroz `#c71585`, Algodão `#ff69b4`, Outras `#f54ca9`; Lavoura Perene `#d082de`: Café `#d68fe2`, Citrus `#9932cc`, Dendê `#9065d0`, Outras `#e6ccff`), Silvicultura `#7a5900`, Mosaico de Usos `#ffefc3` |
| 4. Área não Vegetada `#d4271e` | Praia, Duna e Areal `#ffa07a`, Área Urbanizada `#d4271e`, Mineração `#9c0027`, Usina Fotovoltaica `#757272`, Usina Eólica `#403d3e`, Outras `#db4d4f` |
| 5. Corpo D'água `#2532e4` | Rio, Lago e Oceano `#2532e4`, Aquicultura `#091077` |

Savana Alagada, Formação Herbáceo Arbustiva, Marisma e Usina Eólica não têm id no nível 2 da API e são reconhecidas pelo nome. Na barra LULC, água, praias e outras áreas não vegetadas naturais ficam fora da conta, sem nota.

### Cores e formatação
```javascript
const C = {
  red: '#E94030',     // vermelho institucional (logo, ênfases)
  redD: '#B42A1C',
  fire: '#D32F2F',    // ano corrente nas séries
  gray: '#BDBDBD',    // anos anteriores nas séries
  dark: '#212121',    // linha da média
  ink: '#1F1F1F', ink2: '#4F4F4F', muted: '#6B6B6B',
  line: '#E6E6E6', good: '#2E7D4F'
};
```

### Variação no mapa (divergente)
```javascript
const DIV_LIM = [-50, -10, 10, 50];   // limites das 5 classes
// DIV_CORES: queda forte → neutro → alta forte (verde, cinza, vermelho)
classeVar(p)  // converte a variação % em índice 0..4 (-1 = sem dado)
legVar()      // legenda HTML das 5 classes
```

### Helpers de formatação
```javascript
nf0(v)          // Número inteiro formatado (pt-BR)
cmp(v, d)       // textos: "566 mil", "1,4 milhão", "2,1 milhões" (nunca "mi")
haC(v, d)       // "566 mil ha", "1,4 milhão de ha"
curto(v)        // mapas e legendas: "1.435 mil" (nunca milhão)
unid(max)       // gráficos: uma unidade por gráfico. Com algum valor ≥ 1 milhão de ha: eixo
                // "Área queimada (milhões de hectares)" em duas linhas (eixoHa), marcas 2/4/6 (t),
                // rótulos "3,6 Mha" (f) só no ano em destaque e na média (abaixo de 1 milhão, "631 mil ha"); demais só "3,6" (n) e subtítulo "…, em milhões de hectares" (ext).
                // Abaixo de 1 milhão: tudo em hectares inteiros ("566.000", eixo "(ha)")
pctTxt(p)       // "+5%" / "−3%"
fmtP(p)         // "5%" / "<1%"
pct1(p)         // "77,5%"
esc(s)          // HTML escape
```

---

## 📊 Funções principais

### API e dados
| Função | O que faz |
|---|---|
| `getJSON(path)` | Fetch + JSON da API |
| `groupings(type, code)` | Lista agrupamentos válidos (com cache) |
| `pickGroup(type, code, want)` | Valida o agrupamento em `/groupings` antes de qualquer estatística (evita HTTP 500) |
| `st(kind, type, code, params, want)` | GET genérico de estatística |
| `rankingAll(type, code, params, want)` | Percorre todas as páginas de um ranking `/raw` |
| `loadGeo()` | Carrega GeoJSON dos estados |
| `lcTrans()` | Tradução de classes de cobertura |
| `detectarPeriodo()` | Encontra o mês mais recente com dados |
| `dadosTerritorio(type, code, want)` | Mês, acumulado, série e cobertura de um território |

### Gráficos
| Função | O que faz |
|---|---|
| `mkChart(key, canvasId, cfg)` | Cria Chart.js com `devicePixelRatio: 3` e sem animação |
| `yearChart(id, anos, vals, Y)` | Barras por ano: anos anteriores em cinza, ano corrente em vermelho-fogo e linha da média tracejada |
| `classes(values, n=5)` | Classes para colorir mapas: quebras naturais (`jenks(v, k)`, Fisher-Jenks) sobre a raiz cúbica dos valores; com até `n` valores distintos (ex.: 6 biomas) usa quantis |
| `topLabels`, `linhaMedia`, `faixasNormal`, `rotulos` | Plugins de rótulos, média, faixas de normalidade e valores no fim das barras |
| `opLinha(yT, xT, st, extra)` | Opções padrão dos gráficos de linha do painel |
| `pintarCalendario(cv, M, ano, ini, fim, o)` | Calendário do fogo em canvas (painel) |

### Mapas
| Função | O que faz |
|---|---|
| `mapaEstatico(id, bounds)` | Leaflet estático com fundo branco. Em aba escondida (tamanho zero) só centraliza; `reajustarMapas()` enquadra quando a aba aparece |
| `reajustarMapas()` | Refita mapas se o tamanho mudou |
| `mvt(path, style, zIndex)` | Cria tile layer vetorial (MVT) |
| `drawStates(el, geo, vals, total, cls)` | Mapa D3 dos estados com cores, rótulos e divisas cinza-claro |
| `drawBiomes(el, r, total, cls)` | Mapa D3 dos biomas (GEO_BIO) com rótulos e linhas-guia |
| `mapaFogo(id, type, code, ini, fim)` | Cicatrizes de fogo (tiles do Monitor) sobre o limite do território |
| `mapaMun(id, legId, u, mun)` | Coroplético municipal de um estado |
| `localizador(el, geo, nome)` | Brasil em cinza, estado em vermelho e círculo em volta dele |
| `novoMapa`, `limparMapa`, `camadas`, `xMapa` | Mapa interativo do painel |
| `classeVar`, `legVar` | Classes e legenda do mapa divergente de variação |

### Boletim do mês
| Função | O que faz |
|---|---|
| `cabecalho(per)` | Logo + "DESTAQUES" + mês |
| `initBoletim()` | Busca os dados (`D`) e renderiza as 5 folhas em paralelo |
| `colunaHTML(id)`, `colunaMx(id, tipo, d, br, tile)` | Layout base Mês × Acumulado: os elementos `[data-col]` recebem a estrutura; `d` vem de `dadosBR(D)`, `D.dBio` ou `D.dUF`; `br` (`totaisBR(D)`) mostra a % do Brasil; `tile` acrescenta o card de contexto |
| `kpiCard(el, rótulo, d, per, sub, tile)` | Card minimalista com número grande, pílulas de variação e card de contexto opcional |
| `tilePico(ts)`, `tileNativa(lc)` | Cards de contexto da folha 1: "Pico do ano" e "Vegetação nativa" |
| `kpiSlide`, `caixaVar(v, ref, pilula)` | Número grande; variação em caixa ou em pílula "▼ −55% vs. … (−3,2 mi ha)" |
| `anosSlide(id, ts, acum)` | Barras por ano: só o mês ou, com `acum = true`, janeiro até o mês |
| `lulcResumo(lc)`, `lulcHTML(lc, titulo, min)` | Classes da paleta com área e grupo; barra em dois níveis (natural × antrópico e classes de nível 2). Classes < 0,5% viram "Outras" (cinza, dentro de cada lado). Texto na fatia: % inteira, só se a fatia tiver ≥ `min`% (6 nas colunas largas, 9 nas estreitas). Toda fatia tem `title` com classe, % com uma casa e área em ha |
| `barrasH(id, nomes, vals, cls, total)` | Barras horizontais coloridas pela classe do mapa; com `total`, rótulo "515.254 ha · 20%" |
| `d2Mapa`, `d2Rank`, `d3Mapa`, `d4Mapa`, `d4Rank`, `d5Mapa` | Blocos das folhas 2 a 5 |

### Destaque dinâmico
| Função | O que faz |
|---|---|
| `initPainel()` | Inicializa o painel e lê o hash |
| `ligarEventos()` | Liga filtros, chips, selects, toggles e botões |
| `definirTerritorio(tipo, code)` | Troca o recorte e recarrega os dados |
| `atualizarDiv`, `atualizarControles` | Atualiza subdivisões e controles |
| `xRun()` | Orquestra os jobs do painel |
| `xKpi`, `xSaz`, `xAcu`, `xAnual`, `xCal` | Destaques e histórico |
| `xMapa`, `xRank`, `xUso`, `xUsoAno` | Mapa, Top 15 e uso/cobertura |
| `xForaNormal(M, f)`, `xForaTabela(porAno, anos, f)` | "Fora do normal?": anomalia (z-score) do território e do período escolhidos, contra os anos anteriores ao ano escolhido; tabela das subdivisões (todas até 12, senão os 5 extremos de cada lado; lista completa no CSV). São jobs do `xRun` (`x-fn-z`, `x-fn-sub`) e reaproveitam o cache das consultas do Top 15 |
| `foraNormalAviso(f)`, `perTxt(f)` | Aviso quando não há normal (período passa do último mês com dados ou menos de 3 anos anteriores); meses do período sem o ano |
| `zInfo`, `zClasse`, `linhasZ(porAno, anos, Y)`, `Z_CORES` | Cálculo e classes da anomalia |
| `sincronizarOrdem()` | Estado visual do seletor "Maior área / Maior aumento" do Top 15 |
| `lerHash`, `salvarHash` | Estado dos filtros na URL (`#explorer?t=&c=&a=&i=&f=&m=&ref=`) |
| `baixarCSV`, `baixarPNG`, `copiarLink` | Exportações e link compartilhável |
| `modoPainel`, `imprimirPainel` | Layout A4 paisagem para PDF do painel |

### Abas, impressão e início
| Função | O que faz |
|---|---|
| `abrirAba(nome)` | Alterna entre "Boletim do mês" (`boletim`) e "Destaque dinâmico" (`explorer`) |
| `modoImpressao(on)` | Ativa o CSS de folha A4 paisagem (classe `html.pm`) |
| `beforeprint` / `afterprint` | Ajusta mapas e gráficos antes/depois da impressão |

---

## 🔄 Fluxo de dados

```
User abre boletim-fogo.html
  ↓
detectarPeriodo() encontra o mês mais recente com dados
  ↓
initBoletim() busca dados da API em paralelo (objeto D)
  ├─ totais do Brasil (grouping=biome), séries mensais e cobertura do mês e do acumulado
  ├─ rankings de estados, biomas e municípios do estado líder
  ├─ dados do bioma e do estado líderes (dadosTerritorio)
  └─ GeoJSON dos estados
  ↓
Renderiza as 5 folhas (cada bloco tem carregamento/erro via secao())
  ↓
Ao abrir o Destaque dinâmico: initPainel() → xRun() busca os dados do filtro (inclusive o "Fora do normal?")
  ↓
"Exportar PDF" → modoImpressao(true) → window.print() → PDF → modoImpressao(false)
```

---

## 🎯 Como usar

### Acesso online
```
https://veralaisa.github.io/boletim-monitor-fogo/
```

### Rodar localmente
```powershell
cd "C:\Users\vera.arruda\OneDrive - IPAM-Amazonia\Backup\Projetos\Boletim-Monitor-Fogo"
start boletim-fogo.html
```

### Testar o PDF
```powershell
# Boletim (5 folhas A4 paisagem)
& "C:\Program Files\Google\Chrome\Application\chrome.exe" `
  --headless --disable-gpu --disable-extensions --disable-background-networking `
  --no-sandbox --no-pdf-header-footer --run-all-compositor-stages-before-draw `
  --virtual-time-budget=70000 `
  --print-to-pdf="$env:TEMP\boletim.pdf" `
  "file:///C:/Users/vera.arruda/OneDrive%20-%20IPAM-Amazonia/Backup/Projetos/Boletim-Monitor-Fogo/boletim-fogo.html#pdf"

# Painel (A4 paisagem, com os filtros do hash)
# ...#explorer?t=country&c=0&a=2026&i=1&f=9&px=1
```

Para checar erros sem abrir o navegador: `--dump-dom` e procurar `class="st err"` no HTML gerado (deve ser 0), com e sem `#explorer`.

Fonte dos dados (query string, antes do hash):
- Sem parâmetro, usa a API pública (`https://plataforma.monitorfogo.mapbiomas.org`).
- `?api=dev` usa a API dev (`https://dev-fogo.geodatin.com`, mesmo formato). O seletor "API pública / API dev" na barra do topo troca o parâmetro e recarrega mantendo o hash. No modo dev aparecem uma faixa de aviso (fora do PDF) e a fonte trocada no PNG.

Hashtags:
- Sem hash, abre na aba **Boletim do mês**.
- `#pdf` — abre o boletim já no layout de impressão (5 folhas A4 paisagem). `#dpdf` é o nome antigo e continua funcionando.
- `#explorer?…&px=1` — abre o painel já no layout de impressão.
- `&m=var` — abre o mapa já no modo de variação percentual; combina com `&ref=media` para variar contra a média histórica.
- `&ref=media` — usa a média histórica como base de comparação (padrão: ano anterior).
- `&anom=1` — abre o Top 15 só com os territórios acima da média histórica (implica `ref=media`).
- `&o=alta` — ordena o Top 15 pelo maior aumento em ha contra a referência (padrão: maior área).

---

## 🔗 API do Monitor do Fogo

Documentação: https://plataforma.monitorfogo.mapbiomas.org/api/docs/

### Endpoints principais

**Dados de área queimada**
```
GET /api/territories/{type}/{code}/statistics
  ?year=2026&monthStart=1&monthEnd=9&grouping=biome&landCover=true
```
Retorna: `monthly[{month, year, areaHa, ...}]`, `landCover[{name, hectares, ...}]`

**Validar agrupamentos**
```
GET /api/territories/{type}/{code}/groupings
```
Resposta: `groupings: ['biome', 'state', ...]` (o que é válido para esse tipo)

**Ranking bruto (todas as páginas)**
```
GET /api/statistics/ranking/{type}/{code}/{grouping}/raw
  ?year=2026&monthStart=9&monthEnd=9
```

**Séries mensais**
```
GET /api/statistics/time-series/{type}/{code}/{grouping}?monthStart=1&monthEnd=12
```

**Tiles de fogo (raster)**
```
GET /api/maps/fire/monthly?territoryType=biome&territoryCode=3&year=2026&monthStart=1&monthEnd=9
Retorna: { url } — template de tiles para o L.tileLayer
```

**Tiles vetoriais (MVT)**
```
GET /api/territories/grouping/tiles/country/0/state/{z}/{x}/{y}.mvt
GET /api/territories/grouping/tiles/country/0/biome/{z}/{x}/{y}.mvt
GET /api/territories/grouping/tiles/state/{code}/city/{z}/{x}/{y}.mvt
GET /api/territories/grouping/tiles/biome/{code}/biome/{z}/{x}/{y}.mvt
```

**Limites**
```
GET /api/territories/{type}/{code}/bounds
Retorna: [minLat, minLng, maxLat, maxLng]
```

**Cobertura do solo**
```
GET /api/statistics/land-cover/{type}/{code}/{grouping}
  ?year=2026&monthStart=9&monthEnd=9&level=2
Retorna: [{ id, namePt, nameEs, nameEn, areaHa }]
```

### Codificações
- **Países**: 0 = Brasil
- **Biomas**: 1=Amazônia, 2=Caatinga, 3=Cerrado, 4=Mata Atlântica, 5=Pampa, 6=Pantanal
- **Estados**: código IBGE (15=Amazonas, 17=Bahia, ..., 51=Mato Grosso, ...)
- **Classes de cobertura**: ids de `PALETA_LULC` (3=Formação Florestal, 4=Savânica, 12=Campestre, 15=Pastagem, …)

---

## 🛠️ Desenvolvimento

### Editar localmente
1. Abra o HTML em um editor (VS Code, Notepad++, etc.).
2. Teste no navegador (`file:///...`).
3. Abra o DevTools (F12) para ver erros.
4. Se for publicar, faça commit/push **somente com autorização**.
5. O arquivo usa final de linha LF e UTF-8 sem BOM; preserve ao editar por script.

### Estrutura CSS
```css
:root {
  --red: #E94030;        /* Cor institucional */
  --red-d: #B42A1C;      /* Escuro */
  --fire: #D32F2F;       /* Ano corrente */
  --gray: #BDBDBD;       /* Anos anteriores */
  --dark: #212121;       /* Linha da média */
  --ink: #1F1F1F;        /* Texto */
  --good: #2E7D4F;       /* Verde (positivo) */
}

/* Folhas A4 paisagem */
.sheet.slide { width: 297mm; min-height: 210mm; }

/* Layout base Mês × Acumulado e variante com mapa */
.sl-g.mxg { grid-template-columns: 1fr 1fr; }
.sl-g.mxg.c3 { grid-template-columns: 1.1fr 1fr 1fr; }   /* folhas 3 e 5 */

/* Cards */
.mx-card { border-radius: 16px; box-shadow: …; }          /* card principal */
.mx-tile { border-radius: 14px; background: #F4F4F4; }    /* card de contexto */
.selo { border-radius: 50%; }                             /* selo circular (substitui os losangos) */
```

### Modo impressão
```css
@media print {
  @page { size: A4 landscape; margin: 0; }
  .top, .no-print { display: none !important; }
  .sheet.slide { width: 297mm; height: 210mm; break-after: page; }
}

html.pm { /* Ativado por modoImpressao(true) */ }
```

---

## 🎨 Sistema visual

### Tipografia
- **Família**: Montserrat (400/500/600/700/800), com fallback para `system-ui`.
- **Números dos cards**: 46 px (folha 1) e 34 px (folhas 3 e 5), peso 800, unidade acoplada.
- **Apoio e legendas**: 9–11 px, `line-height: 1.2–1.4`.
- **Chamadas**: frases curtas e técnicas, sem transições redundantes.

### Séries históricas
- Anos anteriores: barras cinza `#BDBDBD`.
- Ano corrente: barra vermelho-fogo `#D32F2F`.
- Média histórica: linha tracejada `#212121`.

### Mapas
- Fundo branco absoluto (`#FFFFFF`), sem basemap Esri.
- Divisas estaduais finas em cinza-claro (`#D9D9D9`).
- Limite de biomas em preto (`#212121`).
- Escala de fogo em 5 classes (`RAMP5`) de quebras naturais (Jenks) sobre a raiz cúbica da área. Trocado de quantis em 2026-11-09: com quantis, os municípios grandes da última classe dominavam o mapa; com Jenks direto, quase todos caíam na primeira classe. Pará, out/2026, municípios por classe: quantis 29·28·28·28·27; Jenks 118·11·6·4·1; Jenks na raiz cúbica 45·45·28·15·7.
- Legendas dos mapas coropléticos sem fundo nem borda.

### Destaque dinâmico (explorer)
- **Mapa único**: ao comparar, um só mapa coroplético divergente de variação percentual (verde = queda, cinza = estável, vermelho = alta).
- **Camadas**: "Por subdivisão" (área ou variação), "Cicatrizes de fogo" e "Mapa de calor".
- **Medida**: `Área queimada (ha)`, `Variação % vs. ano anterior` ou `Variação % vs. média histórica`; o seletor grava `f.metrica` e `f.ref` e fica sincronizado com o toggle do Top 15 por `sincronizarCmp()`.
- **Top 15**: lista com posição, nome, microbarras, área e etiqueta (`▼ -X%` / `▲ +X%`).
- **Filtro de anomalias**: o pill `Todos` / `Apenas acima da média` grava `f.anom`; ativá-lo muda `f.ref` para `'media'`.
- **Ordem do Top 15**: `Maior área` / `Maior aumento` grava `f.ordem` (`'area'`/`'alta'`). "Maior aumento" usa a diferença em ha contra a referência (não a %, que favoreceria territórios pequenos) e só lista quem aumentou.
- **Top 15 clicável**: quando a subdivisão é um tipo navegável (mesma regra do clique no mapa), cada linha (`li.nav[data-code]`) abre o território com clique, Enter ou espaço; o `title` da linha traz área e diferença.
- **Nível da legenda**: `sincronizarNivel()` mantém a classe `.on` e o `aria-pressed` no botão clicado, e o `nv` no hash acompanha a escolha.
- **Fora do normal?**: no fim do painel, segue território, período e filtro de cobertura: gráfico de faixas do território (PNG/CSV), tabela das subdivisões (`.ztab`, CSV) e o quadro do método (`.como`). Não carrega nada a mais no boletim.

---

## 📝 Notas técnicas

### Por que um arquivo só?
- Fácil de publicar no GitHub Pages.
- Sem build, sem dependências locais.
- Tudo em um único arquivo para o MapBiomas integrar.

### Gráficos em alta resolução
```javascript
mkChart(key, canvasId, cfg) {
  // ...
  devicePixelRatio: Math.max(3, window.devicePixelRatio || 1)
  // Garante que gráficos saem nítidos no PDF (3x a tela).
}
```

### VectorGrid (tiles vetoriais)
- O VectorGrid desenha polígonos via Canvas/SVG.
- Às vezes o Leaflet não marca o tile como "loaded" e o tile fica oculto.
- Fix: CSS `.lmap svg.leaflet-tile { visibility: visible; }`.

### Texto em canvas no PDF
- Canvas é rasterizado na impressão; em prévias de baixa resolução os números pequenos parecem "duplicados".
- Conferir sempre em 300 dpi antes de considerar um problema real.

### Diferenças na API
- `areaHa` dentro de `previousYearVariation` é uma **diferença**, não um valor.
  - Exemplo: se o mês anterior = 100 ha e agora = 80 ha → `areaHa: -20` (diminuiu 20).
- Totais do Brasil usam o agrupamento `biome` — é o recorte que reproduz os números do boletim oficial.
- `pickGroup()` valida o agrupamento em `/groupings` antes de qualquer estatística; sem isso a API pode retornar HTTP 500.
- `pct(a, b)` devolve `null` quando `b <= 0`; o Top 15 mostra "sem fogo na base" nesses casos.

### Versionamento
- O repositório publica `index.html` (cópia de `boletim-fogo.html`).
- `boletim-fogo-versao-2.html` e `boletim-fogo-backup-2026-10-27.html` são apenas backups locais.
- Edite, teste e só publique quando estiver pronto.

---

## 🚀 Deploy

### GitHub Pages (automático, somente com autorização)
```
Edite localmente
  ↓
Copie boletim-fogo.html → index.html no repositório
  ↓
Commit + Push (GitHub Desktop, ou o git.exe que vem com ele)
  ↓
GitHub Pages atualiza automaticamente (~30s)
  ↓
https://veralaisa.github.io/boletim-monitor-fogo/
```

Repositório local: `C:\Users\vera.arruda\OneDrive - IPAM-Amazonia\Documentos\GitHub\boletim-monitor-fogo` (branch `main`).

O `git` não está no PATH. Use `C:\Users\vera.arruda\AppData\Local\GitHubDesktop\app-*\resources\app\git\cmd\git.exe`.

> **Não fazer commit nem push** sem pedido explícito.

---

## ❓ Troubleshooting

| Problema | Solução |
|----------|---------|
| "L is not defined" | Leaflet não carregou; checar conexão com a internet |
| Mapa em branco | API fora do ar; testar https://plataforma.monitorfogo.mapbiomas.org/api/docs/ |
| "Invalid LatLng (NaN, NaN)" | Mapa criado em aba escondida; `mapaEstatico` já trata, conferir se o contêiner tem altura |
| Erro no Console (F12) | Ler o erro, localizar a linha no HTML e conferir |
| PDF vazio | Esperar os dados carregarem antes de exportar |
| Página a mais no PDF | Alguma folha passou de 210 mm; reduzir alturas (`.cv-sl`, mapas) |
| Gráfico cortado no PDF | Conferir a altura do contêiner (`.cv-*`) e o `layout.padding` do Chart.js |
| GitHub Pages mostra 404 | Checar Settings → Pages (branch main, folder /) |

---

## 📞 Próximas etapas

- [x] **Destaque dinâmico**: filtros por qualquer território, sazonalidade, acumulado, calendário, mapa único, Top 15, cobertura, CSV/PNG/PDF.
- [x] **Boletim do mês em paisagem**: 5 folhas no layout Mês × Acumulado, paleta LULC oficial, selos circulares.
- [x] **Fora do normal?** movido para o Destaque dinâmico.
- [ ] **Fonte customizada**: quando o arquivo .ttf/.woff2 chegar, atualizar `@font-face`.
- [ ] **Mais integrações**: considerar copiar para a produção do MapBiomas.

---

**Última atualização**: 2026-10-28  
**Versão**: arquivo único (Boletim do mês em 5 folhas A4 paisagem + Destaque dinâmico)
