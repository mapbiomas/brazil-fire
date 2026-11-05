# CLAUDE.md · fire_dashboard

Notas técnicas para quem mantém o dashboard. O que a ferramenta faz e como abrir está no `README.md`.

## O que é

Página estática (HTML + CSS + JS, sem build) que consulta a API pública da Plataforma MapBiomas e desenha gráficos (Chart.js 4.4.1) e mapas (Leaflet 1.9.4). Turf.js 6.5.0 recorta municípios pelo bioma. Tudo vem do cdnjs; a fonte é a Montserrat (Google Fonts).

## Como rodar e testar

- Abrir `app/index.html` no navegador. Não há testes automatizados.
- Checagem de sintaxe: `node --check app/app.js`.

## Fonte da verdade

- API de produção `https://plataforma.mapbiomas.org/api/v1` (staging só como reserva). Estritamente **Coleção 5.1** (`CONFIG.colecaoExigida`).
- Estatísticas: `/statistics/area` (tarefas assíncronas; interseção com `spatialMethod=intersection` demora de 25 a 255 s na primeira vez e fica em cache no servidor). `territoryAreaHa` da resposta já é a área da parte dentro do recorte.
- Geometrias: `/territories/geometry`. Ranking: `/statistics/ranking/area` (só municípios inteiros, não recorta por bioma).
- Categorias de território: 230 municípios, 214 estados, 4 biomas (Cerrado = `1-4-3`).

## Decisões

- **Município × bioma:** municípios totalmente dentro do bioma usam o ranking; os da divisa são recortados com Turf.js e têm a área exata pedida à API em fila (8 em paralelo). Enquanto calculam, aparecem em cinza e ficam fora do ranking. Corte de 1% (`RECORTE_MIN`) e 98% (`RECORTE_MAX`).
- **Alternância de tipo de gráfico:** o card declara `alterna` (`linhas`, `rosca` ou `anual`). Barras ↔ linhas reaproveitam a instância do Chart.js; rosca e "Por bioma" recriam o corpo do card.
- **Cores:** biomas oficiais em `BIOMA_CORES`; paletas da vista "Por bioma" em `PALETAS_BIOMA` (terrosa, vermelhos); degradês de frequência e de tempo decorrido em `FREQ_RAMP` e `DECORRIDO_RAMP`.

## Problemas conhecidos

- Só português e só Brasil (ver README, seção de adaptação).
- A primeira consulta de um estado ou município com bioma pode levar minutos; a página mostra o progresso.
- O recorte exato dos 524 municípios na divisa do Cerrado leva dezenas de minutos na primeira vez.

## Mapas Interativos (blocos)

- **Ranking por território** (`renderMapaDinamico`): tudo roda no navegador com o ranking anual já carregado (sem novas consultas). Passo 1 indicador (área · % do território · mudança entre períodos), passo 2 período todo / um ano com ▶ (escala de cores fixa entre anos) ou ano de corte, passo 3 cores ou bolhas e CSV (`;`, vírgula decimal). Clique abre série anual; em estados, atalho para filtrar o painel.
- A área do território para "% do território" vem do contorno (`turf.area`), portanto é aproximada; em municípios os pedaços dos tiles se sobrepõem um pouco na borda.
- **Dados brutos**: sobrepor uma 2ª camada (com transparência), cortina entre dois anos (só camadas de um ano; recorte com `style.clip`, como o leaflet-side-by-side) e "Destacar o recorte" (`turf.mask`).
- Não feito: limites de TIs/UCs sobre o mapa (a API só entrega geometria por território, não por categoria inteira) e hotspots (estatística espacial).

## Fábrica de GIFs (aba `gif`)

- Padrão do `utils/mapbiomas_gif_factory` (um quadro por ano, título, território, ano, legenda, fonte, logo), mas feito no navegador: tiles de `maps/map` da Plataforma (EE devolve CORS) + mapa base Esri, desenhados em canvas e codificados com gifenc (import dinâmico do jsDelivr).
- Países em `GIF_PAISES` (região da API, idioma do texto no GIF, logo). Só entram países com tema de fogo na API (verificado em 2026-11: brazil, argentina, bolivia, chile, colombia, paraguay, peru, indonesia). Paraguai e Indonésia não informam o período na nota informativa: os anos vão de 1985 até o ano passado e o usuário confere.
- Produtos: folhas do tema de fogo, sem mensal, `last_year` e `mean_interval`. Acumulado e frequência (`year_interval`) usam `year=[ini, ano]` em cada quadro.
- Não usa `state.conn` nem os filtros do topo; conexão própria por país em `gifRegiao` (cache `gif:<região>`).
- Aberto como `file://`, o logo pode "sujar" o canvas; `carregarLogo` testa e omite o logo nesse caso.
