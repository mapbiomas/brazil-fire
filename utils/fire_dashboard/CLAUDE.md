# CLAUDE.md · fire_dashboard

Notas técnicas para quem mantém o dashboard. O que a ferramenta faz e como abrir está no `README.md`.

## O que é

Página estática (HTML + CSS + JS, sem build) que consulta a API pública da Plataforma MapBiomas e desenha gráficos (Chart.js 4.4.1) e mapas (Leaflet 1.9.4). Turf.js 6.5.0 recorta municípios pelo bioma. Tudo vem do cdnjs; a fonte é a Montserrat (Google Fonts).

## Como rodar e testar

- Abrir `app/index.html` no navegador (ou o site publicado). Não há testes automatizados.
- Checagem de sintaxe: `node --check app/app.js` e `node --check app/i18n.js`.
- Teste sem interface: Edge com `--headless=new --virtual-time-budget=120000 --dump-dom "file:///…/app/index.html#aba=anual"` e ler KPIs e títulos no HTML gerado. Para testar cliques, copiar o `index.html` para uma pasta temporária com `<base href>` apontando para `app/` e um script de teste no fim do `body`.

## Fonte da verdade

- API de produção `https://plataforma.mapbiomas.org/api/v1/{região}` (staging só como reserva, e só no Brasil). Brasil: estritamente **Coleção 5.1** (`colecaoExigida`). Outros países: a coleção que a Plataforma serve, lida da nota informativa (`colecaoDaNota`; `colecaoPadrao` quando a nota não cita).
- Estatísticas: `/statistics/area` (tarefas assíncronas; interseção com `spatialMethod=intersection` demora de 25 a 255 s na primeira vez e fica em cache no servidor). `territoryAreaHa` da resposta já é a área da parte dentro do recorte.
- Geometrias: `/territories/geometry`. Ranking: `/statistics/ranking/area` (só municípios inteiros, não recorta por bioma).
- Categorias de território no Brasil: 230 municípios, 214 estados, 4 biomas (Cerrado = `1-4-3`). Nos outros países, as categorias são achadas por regex na chave (`PAISES[].categorias`).

## Decisões

- **Município × bioma:** municípios totalmente dentro do bioma usam o ranking; os da divisa são recortados com Turf.js e têm a área exata pedida à API em fila (8 em paralelo). Enquanto calculam, aparecem em cinza e ficam fora do ranking. Corte de 1% (`RECORTE_MIN`) e 98% (`RECORTE_MAX`).
- **Alternância de tipo de gráfico:** o card declara `alterna` (`linhas`, `rosca` ou `anual`). Barras ↔ linhas reaproveitam a instância do Chart.js; rosca e "Por bioma" recriam o corpo do card.
- **Cores:** biomas oficiais em `BIOMA_CORES`; paletas da vista "Por bioma" em `PALETAS_BIOMA` (terrosa, vermelhos); degradês de frequência e de tempo decorrido em `FREQ_RAMP` e `DECORRIDO_RAMP`.

## Problemas conhecidos

- A primeira consulta de um estado ou município com bioma pode levar minutos; a página mostra o progresso.
- O recorte exato dos 524 municípios na divisa do Cerrado leva dezenas de minutos na primeira vez.
- Primeiros cálculos no servidor podem passar de 10 minutos (ex.: Frequência 2020–2025 e Cicatriz no Brasil, rankings fora do Brasil); recarregar mais tarde resolve, porque o resultado fica em cache.
- Fora do Brasil: sem cores oficiais de bioma nem categorias fundiárias; Paraguai e Indonésia não informam o período nem a coleção na nota informativa.
- Traduções (es, en, id) sem revisão de falante nativo.

## Mapas Interativos (blocos)

- **Ranking por território** (`renderMapaDinamico`): tudo roda no navegador com o ranking anual já carregado (sem novas consultas). Passo 1 indicador (área · % do território · comparar anos), passo 2 período todo / um ano com ▶ (escala de cores fixa entre anos) ou a comparação, passo 3 busca por nome (só com mais de 20 territórios), passo seguinte PNG e CSV (`;`, vírgula decimal). Clique (ou busca) abre a série anual; em estados e biomas, atalho para filtrar o painel.
- A área do território para "% do território" vem do contorno (`turf.area`), portanto é aproximada; em municípios os pedaços dos tiles se sobrepõem um pouco na borda.
- **Dados brutos**: sobrepor uma 2ª camada (com transparência), cortina entre dois anos (só camadas de um ano; recorte com `style.clip`, como o leaflet-side-by-side) e "Destacar o recorte" (`turf.mask`).
- Não feito: limites de TIs/UCs sobre o mapa (a API só entrega geometria por território, não por categoria inteira) e hotspots (estatística espacial).

## Fábrica de GIFs (aba `gif`)

- Padrão do `utils/mapbiomas_gif_factory` (um quadro por ano, título, território, ano, legenda, fonte, logo), mas feito no navegador: tiles de `maps/map` da Plataforma (EE devolve CORS) + mapa base Esri, desenhados em canvas e codificados com gifenc (import dinâmico do jsDelivr).
- Países em `GIF_PAISES` (região da API, idioma do texto no GIF, logo). Só entram países com tema de fogo na API (verificado em 2026-11: brazil, argentina, bolivia, chile, colombia, paraguay, peru, indonesia). Paraguai e Indonésia não informam o período na nota informativa: os anos vão de 1985 até o ano passado e o usuário confere.
- Produtos: folhas do tema de fogo, sem mensal, `last_year` e `mean_interval`. Acumulado e frequência (`year_interval`) usam `year=[ini, ano]` em cada quadro.
- Não usa `state.conn` nem os filtros do topo; conexão própria por país em `gifRegiao` (cache `gif:<região>`).
- Aberto como `file://`, o logo pode "sujar" o canvas; `carregarLogo` testa e omite o logo nesse caso.
- Testado (navegador sem interface): Chile 2018–2025, 8 quadros, 0,3 MB.

## Vários países

- `PAISES` (topo de `app.js`): região da API, idioma, logo, marca, coleção exigida (só Brasil, 5.1), staging (só Brasil), categorias dos filtros (regex na chave) e rótulos em português (`rot("estado")` → "Departamento"). O país vem de `#pais=<região>`; trocar no seletor recarrega a página.
- Subtemas por regex nos nomes pt/es/en ou pela análise (`SUBTHEMES`); só "anual" é obrigatório. `TABS[].requer` esconde as abas sem dados; `soBrasil` (fundiário).
- Municípios de um estado: Brasil pela UF no nome; outros países pelo centróide do município (tiles MVT, caixa do país pela geometria) dentro do contorno do estado (`getMunicipiosDoEstado`).
- Ranking: a API aceita um só `pixelValue` e, fora do Brasil, muitas vezes devolve `taskID` (1º cálculo leva minutos); `getRankingAno` espera e repete (até 2 falhas).
- Anos: da nota informativa; Paraguai e Indonésia não informam, então `anosComDados` lê os anos com área no país.
- Coleção: `colecaoDaNota` ignora a coleção de uso e cobertura citada na mesma nota.
- Colômbia: a legenda anual traz "Quemado MODIS MCD64A1" (referência): fica fora do total e do GIF.
- Testado (navegador sem interface, 2026-11-05): aba Anual nos 8 países; Territórios em Peru, Chile, Indonésia e Paraguai; Acumulado, Mensal, Frequência e Último fogo no Peru; Mapas e GIF no Chile. Colômbia, Argentina e Bolívia: Territórios ainda em cálculo no servidor na 1ª vez.

## Idiomas (i18n.js)

- `I18N_LINHAS`: [pt, es, en, id] por texto; o pt é a chave. `tr()` troca trechos (regex única, maiores primeiro, palavra inteira quando a chave começa/termina com letra). `I18N_ROTULOS` vale também em minúsculas.
- A tradução é aplicada no que aparece: MutationObserver no `body` (textos e title/aria-label/placeholder/alt) e `traducaoChartPlugin` (rótulos, séries, títulos de eixo, tooltips do Chart.js). O código continua escrevendo em português.
- Idioma: `#lang=pt|es|en|id`; padrão Brasil pt, Indonésia en, demais es; `id` só na Indonésia. Números em `LOCALE`; `ha()` usa `PALAVRAS_HA`. Nomes da API no idioma (`IDIOMA.api`). Marca e logo seguem o idioma (Fogo/Fuego/Fire), também no GIF.
- Texto novo na interface: acrescentar a linha em `i18n.js`, senão ele aparece em português. Traduções sem revisão de falante nativo.

## Ajustes de mapas e GIF (2026-11-05)

- Seletores de país mostram a coleção de cada país (`COLECOES`, lida em segundo plano por `gifRegiao`); Paraguai e Indonésia: "coleção não informada".
- Mapa por território: sem bolhas; "Comparar anos" = ano B − ano A ou ano − média do período. Clique: contorno em destaque (halo) e zoom. "Baixar mapa (PNG)" (`baixarMapaPng`): copia tiles e canvas do Leaflet (`preferCanvas`, tiles com `crossOrigin`) e acrescenta título, legenda, escala, seta do norte e fonte. `L.control.scale` em todos os mapas.
- Largura do PNG (`pngMenuHtml`, `larguraEscolhida`): "⬇ PNG" abre um menu (como na tela · 480 · 720 · 1.100 px · outra de 320 a 2.400 px), guardado em `localStorage` "fogo:pngLargura". `baixarGraficoPng(root, card, largura)` põe essa largura no `.card-body`, chama `Chart.resize()`, espera 200 ms (por tempo, não `requestAnimationFrame`, que não vem com a aba em segundo plano), copia e restaura.
- Gráficos anual e acumulado: vista "Horizontal" (barras deitadas, anos de cima para baixo) no seletor Barras · Horizontal · Linhas · Por bioma. `specNaVista` vira `kind: "hbar"` com altura de 17 px por ano; está em `VISTAS_DE_CORPO` porque a caixa muda de altura. A linha da média (dataset de linha) fica vertical sozinha com `indexAxis: "y"`.
- Mapa por território (2026-11-11):
  - Cores por quebras naturais (`classesPorQuebras` + `jenksCortes`, Jenks sobre a raiz cúbica, amostra de até 1.200 valores); os quantis deixavam quase tudo na última classe.
  - "% do território" usa classes fixas (`classesPct`: até 10 · 25 · 50 · 75 · 90 · acima de 90%, `PCT_RAMP` creme → laranja → vermelho → vinho); o vinho é só acima de 90%. No norte do Cerrado muitos municípios passam de 90% de verdade (Tocantins inteiro: 70% em 1985–2025).
  - "% do território" no período todo = acumulado ÷ área (`getRankingAcumulado`: ranking do subtema acumulado com `year: [ini, fim]`, igual à estatística — conferido MT 1985–2025 = 45.120.667 ha); no modo "um ano", área do ano ÷ área. Recortados por bioma usam o acumulado da interseção (`calcularRecorte` devolve `acum`).
  - Estados com bioma no filtro: só a parte de cada estado no bioma (geometria `getGeometry([estado, bioma], "intersection")`, descarta < `RECORTE_MIN` do estado; valores pela mesma fila da divisa, `valoresDaDivisa`). Antes eram estados inteiros com aviso.
- Dados brutos: "Subproduto" com grupos Anuais / Por período (`opcoesSelectHtml` com `grupo`); ano escolhível nos anuais; cortina só nos anuais, com ano da esquerda e da direita; contornos clicáveis (biomas, estados ou municípios do estado).
- GIF: velocidades 700/350/180 ms; passo 6 "Limites no mapa" (biomas/ecorregiões ou 1º nível administrativo do país), desenhados em cinza sob o contorno do território.

## Cards: PNG e tabela; Mensal com temporada

- `cardHtml` põe "⬇ PNG" e "▦ Tabela" em todo card com dados; `acaoDoCard` (delegado em `el.panel`). A tabela vem de `tabelaDoCard`: dos gráficos Chart.js desenhados (`Chart.getChart`; `options.plugins.tabela.percent`), ou de `ranking`/`propbars`/`topcards` (cards lazy guardam `root._res`). CSV com `;`.
- A tabela fixa da aba Anual saiu (agora é o botão Tabela do gráfico anual).
- PNG do card (`baixarGraficoPng`): fundo branco; copia os canvas e redesenha as partes em HTML (barras 100%, listas/ranking, cards de destaque, legendas com valores, texto no meio das roscas) com `elementosHtmlDoPng` + `desenharHtmlNoPng` (fundos e bordas antes dos canvas, textos depois, palavra por palavra na posição que o navegador dá; números `tabular-nums` caractere por caractere). No tema escuro, `eixosEscurosParaPng` escurece os eixos do Chart.js só durante a cópia. Para testar todos os botões: script headless que troca cada vista, clica PNG/Tabela/CSV e guarda as imagens.
- Mensal: `state.mesIni`/`mesFim` definem a temporada; atravessando o ano, a temporada Y = meses ≥ início em Y + meses < início em Y+1 (rótulo "2020/21"; a última sem o ano seguinte fica de fora). `state.paletaMes`: legenda oficial da API ou `PALETA_MESES_FOGO`. Novo card empilhado por ano e mês. A legenda mensal usa só os 12 primeiros itens (Colômbia traz 24).

## Revisão das abas (2026-11-05, tarde)

- Controles dentro de um card: item de `toolbar` com `card: <índice>` (renderizado no card, não no topo).
- Frequência: KPIs "frequência predominante" e "queimou 2 vezes ou mais"; card "Área queimada por classe de frequência" com classes e "Mostrar" ha/% (`state.freqUnid`).
- Cicatriz: faixa da proporção por ano escolhível (`state.cicatrizClasse`). Último fogo, Cicatriz, Severidade: KPIs de área total removidos.
- Roscas: anel mais largo (`cutout` 56%), maiores (260 px) e % dentro das fatias ≥ 4% (`rotulosRoscaPlugin`).
- Retorno: duas abas, "Retorno do Último Fogo" (`loadRetornoUltimo`) e "Intervalo Médio de Retorno" (`loadMediaRetorno`).
- Mapa por território: também por biomas; subtítulo "Ranking de ini–fim".
- Coleção padrão quando a nota não cita: Paraguai 1, Indonésia 2 (`colecaoPadrao`).
- GIF: ordem país, território, produto.
- Mensal reorganizado: 1) média mensal com chips de anos no card; 2) distribuição mensal (soma de todos os anos, barras com % ou rosca); 3) soma e média por mês; 4) área por ano e mês; 5) proporção mensal por território com escolha biomas/estados (`state.mensalNivel`; estados: os 8 com mais área).
- Frequência: cards em largura total; `rotulos: "pct"` no spec escreve a % sobre as barras (`rotulosBarrasPlugin`).

## Explicações, link, busca e unidade dos eixos (2026-11-05)

- Botão ⓘ ao lado do título de cada card: abre uma explicação curta. Os textos ficam em `INFO_CARDS` (regex no título em português; `card.info` tem prioridade) e precisam de linha em `i18n.js`.
- Recordes ("posição do ano na série") e tendência (Sen + Mann-Kendall, média por década) foram feitos e retirados a pedido (commit 16d4d4f, se precisar recuperar).
- Eixos em `unit: "Mha"`: `chartConfig` passa para mil ha (`unit: "mil"`, título "(mil ha)") quando `maiorValorDoGrafico` (nos empilhados, o maior total) é menor que 1 milhão de ha. A nota "Mha = milhões de hectares" só aparece quando há milhões.
- Frases com números no meio são montadas por idioma (`porIdioma`), porque o `i18n.js` só troca trechos fixos.
- "🔗 Copiar link desta análise" (`copiarLink`): copia o endereço com o hash. O hash agora guarda também temporada, paleta e "por" do Mensal, unidade da Frequência e faixa da Cicatriz.
- Mapa por território: passo "Buscar" com lista de nomes (`datalist`), só com mais de 20 territórios; aceita nome sem acento ou só o começo; seleciona, contorna e dá zoom.
- Testado sem interface: Anual no Brasil (Mha) e no Peru (mil ha); Acumulado no Peru; mapa por estados (com busca, "sao paulo" abre o detalhe) e por biomas (sem busca).

## Mensal: mapa de calor e boxplot (2026-11-05)

- Card "Soma do período e média por mês" com `alterna: "mensal"` (Mapa de calor | Boxplot | Lista, padrão `vistaPadrao: "calor"`). `card.legendaVista` troca a legenda por vista.
- Mapa de calor (`desenharCalor`): canvas próprio, sem Chart.js; cor pela raiz quadrada da área em `CALOR_RAMP` (0 = cinza), escala desenhada no canvas (entra no PNG), dica no `mousemove`, redesenha com `ResizeObserver` quando a largura muda.
- Boxplot (`boxplotConfig`, `estatBox`): barra flutuante [Q1, Q3] + plugin para mediana, hastes (Tukey, 1,5 × IQR) e pontos dos anos fora do comum; quartis por interpolação linear. Sem animação, para as hastes não descolarem das caixas.
- Tabela sob demanda: ano × mês no mapa de calor; mínimo, quartis, máximo e média no boxplot.
- Testado sem interface: Brasil 1985–2025 (pt) e Peru com temporada jul–jun (en), nas três vistas, com tabela.

## Cor das séries anuais (2026-11-09)

- Cards com `corTrocavel: "<cor original>"` (Área queimada anual, Área queimada acumulada, Composição do acumulado) mostram um círculo discreto no topo; a paleta tem a cor original, `CORES_ANUAIS` (vinho, marrom, preto), um seletor livre (`input type=color`) e um campo HEX (`normalizarHex` aceita `#abc`, `#aabbcc`, com ou sem `#`; aplica no Enter ou ao sair do campo).
- `comCorEscolhida` troca a cor original pela escolhida no `chart` e na `legend` só na hora de desenhar (`renderView`), sem mudar `state.lastView`. Trocar a cor redesenha a aba com os dados já carregados.
- `state.corAnual` fica no `localStorage` (`fogo:corAnual`) e no hash (`cor=<hex sem #>`, abas Anual e Acumulado). Na vista "Por bioma" o círculo some (paletas próprias).
