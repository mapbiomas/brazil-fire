/* ==========================================================================
   MapBiomas Fogo · Dashboard estático (HTML/CSS/JS + Chart.js + Leaflet)

   O que foi mapeado (plugin QGIS V05 + testes diretos na API, out/2026):

   - Endpoints (mapbiomas_api.py do plugin + frontend da plataforma):
       GET /api/v1/brazil/themes                      -> árvore de temas/subtemas + nota informativa
       GET /api/v1/brazil/territories/categories      -> categorias de território
       GET /api/v1/brazil/territories?categoryId=N    -> territórios (máx. 1000 por página)
       GET /api/v1/brazil/territories/geometry        -> contorno GeoJSON do território
       GET /api/v1/brazil/legends?subthemeKey=K       -> legendas do subtema
       GET /api/v1/brazil/legends/{id}                -> itens: pixelValue, nome, cor (hex oficial)
       GET /api/v1/brazil/statistics/area             -> estatística de área (ha)
       GET /api/v1/brazil/statistics/task/{taskID}    -> status de cálculo assíncrono
       GET /api/v1/brazil/maps/map                    -> URL de tiles XYZ (Earth Engine)
       GET /api/v1/brazil/statistics/ranking/area     -> ranking de territórios de uma categoria

   - Ranking (/statistics/ranking/area), testado:
       * funciona com o subtema Anual e UM ano por chamada; pageSize=6000 traz
         todos os municípios com fogo (≈4.770) numa resposta;
       * uma lista de anos NÃO devolve a soma (MT 2020–2024 < MT 2024), então
         somamos ano a ano no cliente;
       * ignora territoryId (não recorta por bioma) e volta vazio para o Acumulado.
     Por isso o ranking de estados do Acumulado usa statistics/area por estado
     (27 consultas; o 1º cálculo levou ~4 min, depois fica em cache no servidor).

   - Coleção: a nota informativa do tema de fogo na PRODUÇÃO diz "A Coleção 5.1
     do MapBiomas Fogo [...] entre 1985 e 2025". O STAGING serve a Coleção 5.
     A coleção e o intervalo de anos são lidos dessa nota em tempo de execução;
     se só o staging responder, o painel mostra um aviso explícito.

   - Produção é multi-organização: sem tenant responde 400 ("No database
     configured for organization 'undefined'"). O frontend oficial envia o header
     `tenant-id: mapbiomas`; usamos `?tenantId=mapbiomas` na query, que mantém o
     GET "simples" (sem preflight CORS). CORS: `access-control-allow-origin: *`.

   - As chaves NÃO são iguais entre ambientes (tema "fogo-copia-1h7mac" na
     produção x "fire" no staging; Brasil = UUID x "1-1-1"; categorias de
     estado/município 214/230 x 3/95). Tudo é descoberto em tempo de execução.

   - Municípios não têm parentId na produção; o nome vem como "Palmas - TO"
     (staging: nameFormatted "Palmas - TO - 1721000"). Filtramos pela UF.

   - Bioma + Estado/Município selecionados juntos = interseção
     (territoryId=A&territoryId=B&spatialMethod=intersection).

   - Combinações ainda não pré-calculadas viram tarefa assíncrona
     ({taskID, strategy: "batch"}): acompanhamos /statistics/task/{id} até
     "success" e repetimos a consulta, que então vem do cache. A tarefa é por
     subtema + território (todos os anos de uma vez): depois de pronta, qualquer
     período daquele recorte é instantâneo. Tempos medidos na 1ª consulta:
     ~25 s (Pantanal, cicatriz), ~113 s (Cerrado ∩ Goiás), >10 min (Brasil, cicatriz).
   ========================================================================== */

"use strict";

// ---------------------------------------------------------------------------
// Configuração
// ---------------------------------------------------------------------------

// Países com o tema de fogo na API da Plataforma MapBiomas (verificado em 2026-11).
// Por país: região da API, idioma dos nomes, logo, coleção exigida (só Brasil), se pode
// cair para o staging (só Brasil) e as categorias de território dos filtros (regex na
// chave da categoria) com os rótulos em português. bioma: null = país sem esse filtro.
const LOGO_ES = "logos/MapBiomas_Fuego_ES.png";
const PAISES = [
  {
    region: "brazil", nome: "Brasil", de: "do Brasil", em: "no Brasil", lang: "pt-BR", logo: "logos/MapBiomas_Fogo_PT.png", marca: "MapBiomas Fogo",
    colecaoExigida: "5.1", staging: true,
    cats: { pais: /^BRASIL|^POLITICAL_LEVEL_1$/, bioma: /^BIOMES$/, estado: /^ESTADOS|^POLITICAL_LEVEL_3$/, municipio: /^MUNICIPIOS_IBGE|^POLITICAL_LEVEL_4$/ },
    rotulos: { bioma: ["Bioma", "Biomas"], estado: ["Estado", "Estados"], municipio: ["Município", "Municípios"] },
  },
  {
    region: "argentina", nome: "Argentina", de: "da Argentina", em: "na Argentina", lang: "es-ES", logo: LOGO_ES, marca: "MapBiomas Fuego",
    cats: { pais: /^PAIS_/, bioma: /^ECORREGIONES_ADMINISTRACION/, estado: /^PROVINCIAS_/, municipio: /^DEPARTAMENTO_/ },
    rotulos: { bioma: ["Ecorregião", "Ecorregiões"], estado: ["Província", "Províncias"], municipio: ["Departamento", "Departamentos"] },
  },
  {
    region: "bolivia", nome: "Bolívia", de: "da Bolívia", em: "na Bolívia", lang: "es-ES", logo: LOGO_ES, marca: "MapBiomas Fuego",
    cats: { pais: /^PAIS_/, bioma: /^ECORREGIONES_/, estado: /^DEPARTAMENTO_/, municipio: /^MUNICIPIO_/ },
    rotulos: { bioma: ["Ecorregião", "Ecorregiões"], estado: ["Departamento", "Departamentos"], municipio: ["Município", "Municípios"] },
  },
  {
    region: "chile", nome: "Chile", de: "do Chile", em: "no Chile", lang: "es-ES", logo: LOGO_ES, marca: "MapBiomas Fuego",
    cats: { pais: /^POLITICAL_LEVEL_1$/, bioma: /^ECORREGION$/, estado: /^POLITICAL_LEVEL_2$/, municipio: /^POLITICAL_LEVEL_4$/ },
    rotulos: { bioma: ["Ecorregião", "Ecorregiões"], estado: ["Região", "Regiões"], municipio: ["Comuna", "Comunas"] },
  },
  {
    region: "colombia", nome: "Colômbia", de: "da Colômbia", em: "na Colômbia", lang: "es-ES", logo: LOGO_ES, marca: "MapBiomas Fuego",
    cats: { pais: /^NIVEL_POLITICO_1/, bioma: /^BIOMES$/, estado: /^POLITICAL_LEVEL_2$/, municipio: /^POLITICAL_LEVEL_3$/ },
    rotulos: { bioma: ["Bioma", "Biomas"], estado: ["Departamento", "Departamentos"], municipio: ["Município", "Municípios"] },
  },
  {
    region: "paraguay", nome: "Paraguai", colecaoPadrao: "1", de: "do Paraguai", em: "no Paraguai", lang: "es-ES", logo: LOGO_ES, marca: "MapBiomas Fuego",
    cats: { pais: /^NIVEL_POLITICO_1/, bioma: /^ECORREGIONES_DINERSTEIN/, estado: /^NIVEL_POLITICO_2/, municipio: /^NIVEL_POLITICO_3/ },
    rotulos: { bioma: ["Ecorregião", "Ecorregiões"], estado: ["Departamento", "Departamentos"], municipio: ["Distrito", "Distritos"] },
  },
  {
    region: "peru", nome: "Peru", de: "do Peru", em: "no Peru", lang: "es-ES", logo: LOGO_ES, marca: "MapBiomas Fuego",
    cats: { pais: /^POLITICAL_LEVEL_1$/, bioma: /^BIOMES$/, estado: /^POLITICAL_LEVEL_2$/, municipio: /^POLITICAL_LEVEL_4$/ },
    rotulos: { bioma: ["Bioma", "Biomas"], estado: ["Departamento", "Departamentos"], municipio: ["Distrito", "Distritos"] },
  },
  {
    region: "indonesia", nome: "Indonésia", colecaoPadrao: "2", de: "da Indonésia", em: "na Indonésia", lang: "en-US", logo: "logos/MapBiomas_Fire_EN.png", marca: "MapBiomas Fire",
    cats: { pais: /^POLITICAL_LEVEL_1$/, bioma: /^POLITICAL_LEVEL_2$/, estado: /^POLITICAL_LEVEL_3$/, municipio: /^POLITICAL_LEVEL_4$/ },
    rotulos: { bioma: ["Grupo de ilhas", "Grupos de ilhas"], estado: ["Província", "Províncias"], municipio: ["Distrito", "Distritos"] },
  },
];

// País da página: vem do link (#pais=peru); trocar de país recarrega a página.
const PAIS = (() => {
  try {
    const r = new URLSearchParams(location.hash.slice(1)).get("pais");
    return PAISES.find((p) => p.region === r) || PAISES[0];
  } catch {
    return PAISES[0];
  }
})();
const ehBrasil = PAIS.region === "brazil";

// Idioma da interface (i18n.js): #lang=pt|es|en|id; padrão pelo país (Brasil pt, Indonésia en,
// demais es). Bahasa indonésio só aparece para a Indonésia.
const IDIOMAS_DO_PAIS = IDIOMAS.filter((i) => !i.soPais || i.soPais === PAIS.region);
const IDIOMA = (() => {
  let pedido = null;
  try { pedido = new URLSearchParams(location.hash.slice(1)).get("lang"); } catch { /* sem hash */ }
  const padrao = ehBrasil ? "pt" : PAIS.region === "indonesia" ? "en" : "es";
  return IDIOMAS_DO_PAIS.find((i) => i.id === pedido) || IDIOMAS.find((i) => i.id === padrao);
})();
const IDIOMA_PADRAO = ehBrasil ? "pt" : PAIS.region === "indonesia" ? "en" : "es";
const LOCALE = IDIOMA.locale;
// Marca e logo seguem o idioma: MapBiomas Fogo (pt), Fuego (es), Fire (en e bahasa).
const MARCA = { pt: ["MapBiomas Fogo", "logos/MapBiomas_Fogo_PT.png"], es: ["MapBiomas Fuego", "logos/MapBiomas_Fuego_ES.png"],
  en: ["MapBiomas Fire", "logos/MapBiomas_Fire_EN.png"], id: ["MapBiomas Fire", "logos/MapBiomas_Fire_EN.png"] }[IDIOMA.id];
PAIS.marca = MARCA[0];
PAIS.logo = MARCA[1];

// Tradução por trechos: o texto em português é a chave (I18N_LINHAS). Uma regex única com
// todas as chaves (as maiores primeiro), palavra inteira quando a chave começa/termina com letra.
const TR = (() => {
  const col = { es: 1, en: 2, id: 3 }[IDIOMA.id];
  if (!col) return null;
  const mapa = new Map();
  for (const linha of I18N_LINHAS) if (linha[col]) mapa.set(linha[0], linha[col]);
  for (const r of I18N_ROTULOS) {
    const t = mapa.get(r);
    if (t && !mapa.has(r.toLowerCase())) mapa.set(r.toLowerCase(), t.toLowerCase());
  }
  const esc = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const partes = [...mapa.keys()].sort((a, b) => b.length - a.length)
    .map((k) => `${/^\p{L}/u.test(k) ? "(?<!\\p{L})" : ""}${esc(k)}${/\p{L}$/u.test(k) ? "(?!\\p{L})" : ""}`);
  return { mapa, re: new RegExp(partes.join("|"), "gu") };
})();
const tr = (s) => (TR && typeof s === "string" && s ? s.replace(TR.re, (m) => TR.mapa.get(m) ?? m) : s);

// Traduz os textos e atributos visíveis de um trecho da página.
const ATRIBUTOS_TRADUZIDOS = ["title", "aria-label", "placeholder", "alt", "label"];
function traduzirNo(raiz) {
  if (!TR || !raiz) return;
  if (raiz.nodeType === Node.TEXT_NODE) {
    const novo = tr(raiz.nodeValue);
    if (novo !== raiz.nodeValue) raiz.nodeValue = novo;
    return;
  }
  if (raiz.nodeType !== Node.ELEMENT_NODE || /^(SCRIPT|STYLE|CANVAS|svg)$/i.test(raiz.nodeName)) return;
  const andar = document.createTreeWalker(raiz, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  for (let n = andar.currentNode; n; n = andar.nextNode()) {
    if (n.nodeType === Node.TEXT_NODE) {
      if (/^(SCRIPT|STYLE)$/.test(n.parentNode?.nodeName)) continue;
      const novo = tr(n.nodeValue);
      if (novo !== n.nodeValue) n.nodeValue = novo;
    } else {
      for (const a of ATRIBUTOS_TRADUZIDOS) {
        const v = n.getAttribute?.(a);
        if (v) { const novo = tr(v); if (novo !== v) n.setAttribute(a, novo); }
      }
    }
  }
}

// Tudo o que entra na página passa pela tradução (abas, cards, tooltips do mapa, selects).
function iniciarTraducao() {
  if (!TR) return;
  traduzirNo(document.body);
  document.title = tr(document.title);
  new MutationObserver((mudancas) => {
    for (const m of mudancas) {
      if (m.type === "childList") m.addedNodes.forEach(traduzirNo);
      else if (m.type === "characterData") traduzirNo(m.target);
      else if (m.type === "attributes") {
        const v = m.target.getAttribute(m.attributeName);
        const novo = tr(v);
        if (v && novo !== v) m.target.setAttribute(m.attributeName, novo);
      }
    }
  }).observe(document.body, { childList: true, subtree: true, characterData: true, attributes: true, attributeFilter: ATRIBUTOS_TRADUZIDOS });
}

// Textos desenhados no canvas do Chart.js (rótulos, séries, títulos de eixo, tooltips).
const traducaoChartPlugin = {
  id: "traducao",
  beforeInit(chart) {
    if (!TR) return;
    const cb = chart.config.options?.plugins?.tooltip?.callbacks;
    if (cb) {
      for (const [k, fn] of Object.entries(cb)) {
        if (typeof fn !== "function") continue;
        cb[k] = function (...args) {
          const r = fn.apply(this, args);
          return Array.isArray(r) ? r.map(tr) : tr(r);
        };
      }
    }
  },
  beforeUpdate(chart) {
    if (!TR) return;
    const d = chart.data;
    if (Array.isArray(d.labels)) d.labels = d.labels.map((x) => (typeof x === "string" ? tr(x) : x));
    for (const ds of d.datasets || []) if (typeof ds.label === "string") ds.label = tr(ds.label);
    for (const sc of Object.values(chart.config.options?.scales || {})) {
      if (sc?.title?.text) sc.title.text = tr(sc.title.text);
    }
  },
};
// Rótulos dos filtros: rot("estado") = "Departamento"; rot("estado", true) = "Departamentos".
const rot = (k, plural = false) => PAIS.rotulos[k][plural ? 1 : 0];

const CONFIG = {
  colecaoExigida: PAIS.colecaoExigida || null, // null = aceita a coleção que a produção servir
  region: PAIS.region,
  anoMinPadrao: 1985, // usados só se a nota informativa não trouxer o intervalo
  anoMaxPadrao: 2025,
  requestTimeoutMs: 60000,
  taskPollMs: 4000,
  taskTimeoutMs: 600000,
  territoriosPorPagina: 1000,
};

// Ordem = prioridade. O staging (Coleção 5) só é usado se a produção falhar.
const ENVIRONMENTS = [
  { id: "prd", label: "Produção", baseUrl: "https://plataforma.mapbiomas.org/api/v1", query: { tenantId: "mapbiomas" } },
  { id: "prd2", label: "Produção (prd)", baseUrl: "https://prd.plataforma.mapbiomas.org/api/v1", query: { tenantId: "mapbiomas" } },
  { id: "stg", label: "Staging", baseUrl: "https://stg.plataforma.mapbiomas.org/api/v1", query: {} },
];

// Como achar cada subtema na árvore do tema de fogo (válido em prd e stg e nos outros países):
// pela análise ou por regex nos nomes (pt, es, en, sem acento). Só "anual" é obrigatório;
// as abas dos outros subtemas só aparecem quando o país os tem.
const SUBTHEMES = {
  anual: { re: /^(queimado |quemado )?anual$|^annual( burned)?$/ },
  mensal: { re: /^(queimado |quemado )?(mensal|mensual)$|^monthly( burned)?$/ },
  cicatriz: { re: /cicatriz|scar size/ },
  retorno: { re: /retorno do ultimo fogo|fire return/ },
  acumulado: { analysis: "accumulated" },
  frequencia: { analysis: "frequency" },
  mediaRetorno: { analysis: "mean_interval" },
  ultimoFogo: { analysis: "last_year" },
  severidade: { re: /^severidad[e]?$|^severity$/ },
};

// Legendas oficiais usadas (cores + pixelValues). Sem `match`, usa a primeira por `order`.
const LEGENDS = {
  anualTotal: { subtheme: "anual", match: (l) => /total/i.test(l.key) },
  anualCobertura: { subtheme: "anual", match: (l) => /coverage|cobertura/i.test(l.key) && !/natural|antrop/i.test(l.key) },
  anualNatural: { subtheme: "anual", match: (l) => /natural|antrop/i.test(l.key) },
  mensal: { subtheme: "mensal" },
  cicatriz: { subtheme: "cicatriz" },
  acumulado: { subtheme: "acumulado" },
  frequencia: { subtheme: "frequencia" },
  retorno: { subtheme: "retorno" },
  mediaRetorno: { subtheme: "mediaRetorno" },
  ultimoFogo: { subtheme: "ultimoFogo" },
  severidade: { subtheme: "severidade" },
};

// Cores para séries sem cor oficial (anos sobrepostos no Mensal, territórios
// comparados). Conjunto validado (daltonismo e visão normal) junto com o
// vermelho oficial "Total queimado"; a 3ª/4ª série repete a cor com traço.
const SERIES_COLORS = [
  { color: "#ff5340" }, // vermelho oficial "Total queimado" (o RAF destaca o ano em vermelho)
  { color: "#2a78d6" },
  { color: "#1baf7a" },
  { color: "#2a78d6", dash: [6, 4] },
];

// Cores dos biomas como nas figuras do RAF 2025 (identidade visual MapBiomas).
const BIOMA_CORES = {
  "amazonia": "#0b4d2c",
  "caatinga": "#c9a27e",
  "cerrado": "#f7931e",
  "mata atlantica": "#13a35f",
  "pantanal": "#6b6b16",
  "pampa": "#d4b23c",
};
// Outros países: cores categóricas na ordem da lista de biomas/ecorregiões do país.
const CORES_REGIOES = ["#3f5e3a", "#b5652b", "#d9a441", "#2a78d6", "#7a5c43", "#93b06f", "#c0392b", "#6b6b16", "#8e6bb8", "#1baf7a", "#c4b49a", "#5c0a14"];
const corBioma = (nome) => {
  if (BIOMA_CORES[norm(nome)]) return BIOMA_CORES[norm(nome)];
  const i = (state.conn?.biomas || []).findIndex((b) => b.name === nome);
  return i < 0 ? "#9aa0a6" : CORES_REGIOES[i % CORES_REGIOES.length];
};

// Paletas da vista "Por bioma" (anual e acumulado). Ordem de empilhamento, de baixo para cima:
const BIOMA_ORDEM = ["amazonia", "cerrado", "caatinga", "mata atlantica", "pantanal", "pampa"];
const PALETAS_BIOMA = {
  terrosa: { nome: "Terrosa", cores: ["#3f5e3a", "#b5652b", "#d9a441", "#93b06f", "#7a5c43", "#c4b49a"] }, // mata escura, terracota, ocre, folha, lama, trigo
  vermelhos: { nome: "Vermelhos", cores: ["#5c0a14", "#c0392b", "#ee8a50", "#f7c1a6", "#8f3b2e", "#e8a9a0"] }, // vinho, vermelho, laranja queimado, salmão, tijolo, rosado
};
const ordemBioma = (nome) => {
  const i = BIOMA_ORDEM.indexOf(norm(nome));
  return i < 0 ? 99 : i;
};
const corPaleta = (nome) => (ehBrasil && PALETAS_BIOMA[state.paletaBioma]?.cores[ordemBioma(nome)]) || corBioma(nome);

// Rampa sequencial (um matiz, claro -> escuro) para os mapas coropléticos.
const CHORO_RAMP = ["#fee5d9", "#fcae91", "#fb6a4a", "#de2d26", "#a50f15"];
const TERRITORY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]; // 3 primeiras cores validadas entre si

// Degradês contínuos (interpolados em RGB entre as paradas, ver rampColor).
// Frequência: como na plataforma MapBiomas Fogo, amarelo/laranja claro (1 vez) -> vermelho -> púrpura escuro.
const FREQ_RAMP = ["#ffd76e", "#fdae61", "#f46d43", "#d73027", "#a50026", "#5e0b4a"];
// Tempo desde o último fogo: quente = fogo recente; verde/azul = muito tempo sem fogo.
const DECORRIDO_RAMP = ["#c62828", "#f46d43", "#fdae61", "#d9ef8b", "#66bd63", "#1a9850", "#2c7fb8", "#253494"];

const UF_POR_ESTADO = {
  "acre": "AC", "alagoas": "AL", "amapa": "AP", "amazonas": "AM", "bahia": "BA", "ceara": "CE",
  "distrito federal": "DF", "espirito santo": "ES", "goias": "GO", "maranhao": "MA", "mato grosso": "MT",
  "mato grosso do sul": "MS", "minas gerais": "MG", "para": "PA", "paraiba": "PB", "parana": "PR",
  "pernambuco": "PE", "piaui": "PI", "rio de janeiro": "RJ", "rio grande do norte": "RN",
  "rio grande do sul": "RS", "rondonia": "RO", "roraima": "RR", "santa catarina": "SC",
  "sao paulo": "SP", "sergipe": "SE", "tocantins": "TO",
};

const MAP_LAYERS = [
  { id: "acumulado", label: "Área queimada acumulada (período)", subtheme: "acumulado", legend: "acumulado", periodo: true },
  { id: "frequencia", label: "Frequência do fogo (período)", subtheme: "frequencia", legend: "frequencia", periodo: true },
  { id: "mediaRetorno", label: "Intervalo médio de retorno (período)", subtheme: "mediaRetorno", legend: "mediaRetorno", periodo: true },
  { id: "anual", label: "Área queimada (ano fim)", subtheme: "anual", legend: "anualTotal" },
  { id: "cobertura", label: "Cobertura queimada (ano fim)", subtheme: "anual", legend: "anualCobertura" },
  { id: "cicatriz", label: "Tamanho da cicatriz (ano fim)", subtheme: "cicatriz", legend: "cicatriz" },
  { id: "severidade", label: "Severidade (ano fim)", subtheme: "severidade", legend: "severidade" },
  { id: "ultimoFogo", label: "Ano do último fogo (período)", subtheme: "ultimoFogo", legend: "ultimoFogo", periodo: true },
];

const MAP_VIEWS = [
  { id: "biomas", label: "Biomas" },
  { id: "estados", label: "Estados" },
  { id: "municipios", label: "Municípios" },
];

// requer: subtemas que o país precisa ter para a aba aparecer; soBrasil: categorias só do Brasil.
const TABS = [
  { id: "anual", label: "Fogo Anual", load: loadAnual, requer: ["anual"] },
  { id: "acumulado", label: "Fogo Acumulado", load: loadAcumulado, requer: ["acumulado"] },
  { id: "mensal", label: "Fogo Mensal", load: loadMensal, requer: ["mensal"] },
  { id: "frequencia", label: "Frequência do Fogo", load: loadFrequencia, requer: ["frequencia"] },
  { id: "ultimoFogo", label: "Ano do Último Fogo", load: loadUltimoFogo, requer: ["ultimoFogo"] },
  { id: "cicatriz", label: "Tamanho de Cicatriz", load: loadCicatriz, requer: ["cicatriz"] },
  { id: "severidade", label: "Severidade", load: loadSeveridade, requer: ["severidade"] },
  { id: "retorno", label: "Retorno do Último Fogo", load: loadRetornoUltimo, requer: ["retorno"] },
  { id: "mediaRetorno", label: "Intervalo Médio de Retorno", load: loadMediaRetorno, requer: ["mediaRetorno"] },
  { id: "fundiario", label: "Categorias Fundiárias", load: loadFundiario, requer: ["anual"], soBrasil: true },
  { id: "territorios", label: "Territórios", load: loadTerritorios, requer: ["anual"] },
  { id: "mapa", label: "Mapas Interativos", load: loadMapa, requer: ["anual"] },
  { id: "gif", label: "Fábrica de GIFs", load: loadGif, requer: [] },
];

// Abas que o país atual tem (antes de conectar, todas).
const abasDisponiveis = () => TABS.filter((t) => !state.conn
  || ((!t.soBrasil || ehBrasil) && t.requer.every((s) => state.conn.subthemes[s])));

const state = {
  conn: null,
  tab: "anual",
  biomaId: "",
  estadoId: "",
  municipioId: "",
  municipiosDoEstado: [],
  ini: CONFIG.anoMinPadrao,
  fim: CONFIG.anoMaxPadrao,
  nivel: 1,
  mensalAnos: [], // anos sobrepostos ao gráfico de média ± desvio padrão (máx. 4)
  mensalAnosTocados: false, // até o usuário escolher, o Mensal destaca o ano final
  mesIni: 1, // temporada do fogo no Mensal: mês de início e de fim (1 = janeiro)
  mesFim: 12,
  paletaMes: "oficial", // "oficial" (legenda da API) ou "fogo" (tons de fogo)
  mensalNivel: "bioma", // Mensal: proporção mensal por "bioma" ou "estado"
  // camada2: camada sobreposta; cortina: compara dois anos; anoA: ano da esquerda; destaque: escurece fora do recorte.
  mapa: { tipo: "bruto", unidade: "estados", camada: "acumulado", visualizar: "biomas", camada2: "", cortina: "off", anoA: null, destaque: "off" },
  freqGrupo: "todas", // Frequência: "todas" (1 a N vezes), "15" (1 a 15 e 16+), "9" (1 a 9 e 10+)
  freqUnid: "ha", // Frequência: barras em hectares ou em % da área queimada
  cicatrizClasse: "grandes", // Cicatriz: faixa da "proporção por ano" (índice da faixa ou "grandes")
  paletaBioma: "terrosa", // paleta da vista "Por bioma" (ver PALETAS_BIOMA)
  vistas: {}, // tipo escolhido em cada card com alternância ("aba:índice" -> "barras" | "linhas" | "rosca")
  hash: {},
  seq: 0,
  charts: [],
  map: null,
  lastView: null,
};

const el = {
  badge: document.getElementById("source-badge"),
  subtitle: document.getElementById("subtitle"),
  banner: document.getElementById("banner"),
  bioma: document.getElementById("f-bioma"),
  estado: document.getElementById("f-estado"),
  municipio: document.getElementById("f-municipio"),
  ini: document.getElementById("f-ini"),
  fim: document.getElementById("f-fim"),
  status: document.getElementById("status"),
  tabs: document.getElementById("tabs"),
  panel: document.getElementById("panel"),
  footerEnv: document.getElementById("footer-env"),
};

// ---------------------------------------------------------------------------
// Camada de API (uma função assíncrona por endpoint)
// ---------------------------------------------------------------------------

class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

function buildUrl(env, path, params = {}) {
  const url = new URL(env.baseUrl + path);
  for (const [key, value] of Object.entries({ ...params, ...env.query })) {
    if (value === undefined || value === null) continue;
    // Listas viram chaves repetidas (year=1985&year=1986...), como o `requests` do plugin.
    for (const v of [].concat(value)) url.searchParams.append(key, v);
  }
  return url;
}

// Erros passageiros do servidor (429/5xx, "upstream connect error") são
// repetidos com espera crescente, como o plugin faz.
async function apiGet(env, path, params, attempt = 1) {
  try {
    return await apiGetOnce(env, path, params);
  } catch (err) {
    const transitorio = err.status === 429 || err.status >= 500 || err.naoJson;
    if (!transitorio || attempt >= 3) throw err;
    await sleep(1500 * attempt);
    return apiGet(env, path, params, attempt + 1);
  }
}

async function apiGetOnce(env, path, params) {
  const url = buildUrl(env, path, params);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), CONFIG.requestTimeoutMs);
  try {
    const resp = await fetch(url, { signal: controller.signal });
    const body = await resp.text();
    if (!resp.ok) {
      let message = body;
      try { message = JSON.parse(body).message || body; } catch { /* corpo não-JSON */ }
      throw new ApiError(`HTTP ${resp.status}: ${message}`, resp.status);
    }
    try {
      return JSON.parse(body);
    } catch {
      const err = new ApiError(`Resposta inválida em ${path}: ${body.slice(0, 120)}`);
      err.naoJson = true;
      throw err;
    }
  } catch (err) {
    if (err.name === "AbortError") throw new ApiError(`Tempo esgotado (${CONFIG.requestTimeoutMs / 1000}s) em ${path}`);
    if (err instanceof ApiError) throw err;
    throw new ApiError(`Falha de rede em ${path}: ${err.message}`);
  } finally {
    clearTimeout(timer);
  }
}

const fetchThemes = (env) =>
  apiGet(env, `/${CONFIG.region}/themes`, { page: 1, pageSize: 100 });

const fetchTerritoryCategories = (env) =>
  apiGet(env, `/${CONFIG.region}/territories/categories`, { page: 1, pageSize: 500 });

const fetchTerritories = (env, categoryId, page = 1, pageSize = 100) =>
  apiGet(env, `/${CONFIG.region}/territories`, { page, pageSize, categoryId });

const fetchTerritoryGeometry = (env, territoryIds, spatialMethod) =>
  apiGet(env, `/${CONFIG.region}/territories/geometry`, { territoryId: territoryIds, spatialMethod, simplify: "true" });

const fetchLegendList = (env, subthemeKey) =>
  apiGet(env, `/${CONFIG.region}/legends`, { page: 1, pageSize: 100, subthemeKey });

const fetchLegendDetail = (env, legendId) =>
  apiGet(env, `/${CONFIG.region}/legends/${legendId}`);

const fetchMapTiles = (env, params) =>
  apiGet(env, `/${CONFIG.region}/maps/map`, { region: CONFIG.region, ...params });

const fetchTaskStatus = (env, taskId) =>
  apiGet(env, `/${CONFIG.region}/statistics/task/${taskId}`);

const fetchRanking = (env, params) =>
  apiGet(env, `/${CONFIG.region}/statistics/ranking/area`, { page: 1, ...params });

async function fetchAreaStatistics(env, params, onWaiting) {
  const query = { region: CONFIG.region, spatialMethod: "union", ...params };
  const started = Date.now();
  for (;;) {
    const payload = await apiGet(env, `/${CONFIG.region}/statistics/area`, query);
    if (Array.isArray(payload.statistic)) return payload;
    if (!payload.taskID) throw new ApiError("Resposta de estatística sem o campo 'statistic'.");
    if (Date.now() - started > CONFIG.taskTimeoutMs) {
      throw new ApiError(
        `O servidor ainda está calculando esta combinação (mais de ${CONFIG.taskTimeoutMs / 60000} min). ` +
        "O cálculo continua lá e o resultado fica em cache: tente novamente em alguns minutos, " +
        "ou reduza o período/recorte. Séries longas para o país inteiro podem levar mais de 10 min."
      );
    }
    await waitForTask(env, payload.taskID, started, onWaiting);
  }
}

async function waitForTask(env, taskId, started, onWaiting) {
  let falhasSeguidas = 0;
  while (Date.now() - started <= CONFIG.taskTimeoutMs) {
    await sleep(CONFIG.taskPollMs);
    let status;
    try {
      ({ status } = await fetchTaskStatus(env, taskId));
      falhasSeguidas = 0;
    } catch (err) {
      if (err.status === 404) return; // sem endpoint de tarefa: reconsulta statistics/area, como o plugin
      // Erros passageiros não devem derrubar uma espera longa.
      if ((err.status === 0 || err.status >= 500 || err.naoJson) && ++falhasSeguidas <= 3) continue;
      throw err;
    }
    onWaiting?.(status, Date.now() - started);
    if (status === "success" || status === "exported") return;
    if (status === "failed" || status === "aborted") {
      throw new ApiError(
        `O servidor interrompeu o cálculo desta estatística (tarefa ${status}). ` +
        "Isso costuma ser passageiro em recortes grandes: clique em \"Tentar novamente\"."
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Descoberta do ambiente e validação da coleção
// ---------------------------------------------------------------------------

// Coleção do MapBiomas Fogo citada na nota ("Coleção 5.1 do MapBiomas Fogo", "MapBiomas Fire Chile
// Collection 1"). Não pega a coleção de uso e cobertura que a nota também cita ("MapBiomas Chile Collection 2").
function colecaoDaNota(text) {
  const m = text.match(/(?:Colecci[oó]n|Cole[cç][aã]o|Collection)\s+(\d+(?:\.\d+)?)(?:\s*-\s*Beta)?\s+(?:de|do|da|-)\s+MapBiomas\s+(?:Fuego|Fogo|Fire)/i)
    || text.match(/MapBiomas\s+(?:Fire|Fuego|Fogo)(?:\s+[^\s,.;:]+)?\s+(?:Collection|Colecci[oó]n|Cole[cç][aã]o)\s+(\d+(?:\.\d+)?)/i);
  return m ? m[1] : null;
}

function readInformativeNote(fire) {
  const text = JSON.stringify([fire.description, fire.informativeNote]).replace(/<[^>]+>/g, " ");
  const colecao = colecaoDaNota(text);
  const anos = text.match(/(?:entre|between|from)\s+(\d{4})\s+(?:e|and|to|y|a)\s+(\d{4})/i)
    || text.match(/\b((?:19|20)\d{2})\s*[–-]\s*((?:19|20)\d{2})\b/);
  const ok = anos && Number(anos[1]) >= 1980 && Number(anos[2]) >= Number(anos[1]) && Number(anos[2]) <= new Date().getFullYear();
  return {
    colecao,
    anosDaNota: Boolean(ok),
    anoMin: ok ? Number(anos[1]) : CONFIG.anoMinPadrao,
    anoMax: ok ? Number(anos[2]) : CONFIG.anoMaxPadrao,
  };
}

async function discoverEnvironment(env) {
  const [themes, categories] = await Promise.all([fetchThemes(env), fetchTerritoryCategories(env)]);

  const fire = (themes.themes || []).find(
    (t) => /^(fire|fogo|fuego)/i.test(t.key) || /^(fogo|fuego|fire)$/.test(norm(t.name?.["pt-BR"] || t.name?.["es-ES"] || t.name?.["en-US"]))
  );
  if (!fire) throw new Error("tema de fogo não encontrado");

  // prd aninha subtemas em `children` (às vezes em dois níveis); stg em `subthemes`; alguns países os deixam soltos.
  const folhas = (arr) => (arr || []).flatMap((s) => (s.children?.length ? folhas(s.children)
    : s.subthemes?.length ? folhas(s.subthemes) : [s]));
  const leaves = folhas(fire.subthemes);
  const nomes = (s) => Object.values(s.name || {}).map(norm);
  const subthemes = {};
  for (const [id, rule] of Object.entries(SUBTHEMES)) {
    const found = leaves.find((s) => (rule.analysis ? s.analysis === rule.analysis
      : !s.analysis && nomes(s).some((n) => rule.re.test(n))));
    if (found) subthemes[id] = found.key;
  }
  if (!subthemes.anual) throw new Error("subtema anual de fogo não encontrado");

  const cats = categories.categories || [];
  const findCat = (re) => (re ? cats.find((c) => re.test(c.key)) : null);
  const catPais = findCat(PAIS.cats.pais);
  const catBioma = findCat(PAIS.cats.bioma);
  const catEstado = findCat(PAIS.cats.estado);
  const catMunicipio = findCat(PAIS.cats.municipio);
  if (!catPais || !catEstado) throw new Error("categorias de território não encontradas");

  const todas = async (catId) => {
    if (!catId) return [];
    const first = await fetchTerritories(env, catId, 1, CONFIG.territoriosPorPagina);
    const rest = await Promise.all(range(2, first.numberOfPages || 1).map((p) => fetchTerritories(env, catId, p, CONFIG.territoriosPorPagina)));
    return [first, ...rest].flatMap((r) => r.territories || []);
  };
  const [paises, biomas, estados] = await Promise.all([todas(catPais.id), todas(catBioma?.id), todas(catEstado.id)]);
  // Colômbia tem "Colombia" e "Colombia - Costa": prefere o nome sem sufixo.
  const territorioPais = paises.find((t) => !/\s-\s/.test(t.name)) || paises[0];
  if (!territorioPais) throw new Error(`território ${PAIS.nome} não encontrado`);

  const nota = readInformativeNote(fire);
  if (!nota.colecao) nota.colecao = PAIS.colecaoPadrao || null; // Paraguai (1) e Indonésia (2) não citam a coleção na nota
  const byName = (a, b) => a.name.localeCompare(b.name, "pt-BR");
  const conn = {
    env,
    subthemes,
    ...nota,
    brasil: { id: territorioPais.id, name: PAIS.nome }, // território do país inteiro (nome mantido do Brasil)
    biomas: biomas.map((t) => ({ id: t.id, name: nomeLimpo(t.name) })).sort(byName),
    estados: estados
      .filter((t) => !/\s-\s(costa|mar)$/i.test(t.name)) // partes marítimas (Colômbia)
      .map((t) => ({ id: t.id, name: nomeLimpo(t.name), uf: ehBrasil ? UF_POR_ESTADO[norm(t.name)] : undefined }))
      .sort(byName),
    catPais: catPais.id,
    catBioma: catBioma?.id || null,
    catEstado: catEstado.id,
    catMunicipio: catMunicipio?.id || null,
  };
  if (!nota.anosDaNota) Object.assign(conn, await anosComDados(env, conn));
  return conn;
}

// Nomes de território sem sufixos de código ("Buenos Aires - 06", "Amazonas - Perú", "Aceh - Sumatera - 11").
function nomeLimpo(nome) {
  let n = String(nome || "").replace(/\s-\s\d+$/, "");
  n = n.replace(new RegExp(`\\s-\\s(${PAIS.nome}|Perú|Peru)$`, "i"), "");
  return n;
}

// Países sem o período na nota informativa: lê os anos com área queimada no país inteiro.
async function anosComDados(env, conn) {
  try {
    const lista = await fetchLegendList(env, conn.subthemes.anual);
    const lg = [...(lista.legends || [])].sort((a, b) => (a.order ?? 0) - (b.order ?? 0))[0];
    const det = await fetchLegendDetail(env, lg.id);
    const itens = flattenLegend((det.legend || det).items);
    const res = await fetchAreaStatistics(env, {
      subthemeKey: conn.subthemes.anual, legendKey: (det.legend || det).key, pixelValue: pixelValues(levelItems({ items: itens }, 0)),
      year: range(1985, new Date().getFullYear()), territoryId: [conn.brasil.id],
    });
    const anos = byYear(res).filter((a) => a.area > 0).map((a) => a.ano);
    if (anos.length) return { anoMin: Math.min(...anos), anoMax: Math.max(...anos) };
  } catch (err) {
    console.warn("[API] não foi possível descobrir os anos com dados:", err);
  }
  return {};
}

async function connect() {
  const failures = [];
  let alternativa = null;
  // O staging só serve o Brasil; nos outros países vale a coleção que a produção servir.
  for (const env of ENVIRONMENTS.filter((e) => PAIS.staging || e.id !== "stg")) {
    try {
      const conn = await discoverEnvironment(env);
      if (!CONFIG.colecaoExigida || conn.colecao === CONFIG.colecaoExigida) return { ...conn, failures, colecaoOk: true };
      failures.push(`${env.label}: serve a Coleção ${conn.colecao || "?"}`);
      alternativa = alternativa || conn;
    } catch (err) {
      console.warn(`[API] ${env.label} indisponível:`, err);
      failures.push(`${env.label}: ${err.message}`);
    }
  }
  if (alternativa) return { ...alternativa, failures, colecaoOk: false };
  throw new Error(`Nenhum ambiente da API respondeu.\n${failures.join("\n")}`);
}

// ---------------------------------------------------------------------------
// Dados com cache (legendas, estatísticas, territórios, geometrias, tiles)
// ---------------------------------------------------------------------------

const cache = new Map();

function cached(key, factory) {
  if (!cache.has(key)) {
    const promise = factory();
    cache.set(key, promise);
    promise.catch(() => cache.delete(key)); // falhas não ficam em cache
  }
  return cache.get(key);
}

const envId = () => state.conn.env.id;
const sub = (id) => state.conn.subthemes[id];

function getLegend(id) {
  const def = LEGENDS[id];
  const env = state.conn.env;
  return cached(`${env.id}:legend:${id}`, async () => {
    const subthemeKey = sub(def.subtheme);
    const list = await cached(`${env.id}:legends:${subthemeKey}`, () => fetchLegendList(env, subthemeKey));
    const ordered = [...(list.legends || [])].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
    // A legenda "total" de outros países nem sempre tem "total" na chave: usa a primeira.
    const chosen = def.match ? (ordered.find(def.match) || (id === "anualTotal" ? ordered[0] : null)) : ordered[0];
    if (!chosen) throw new Error(`Legenda "${id}" não encontrada no subtema ${subthemeKey}.`);
    const detail = await fetchLegendDetail(env, chosen.id);
    const legend = detail.legend || detail;
    let items = flattenLegend(legend.items);
    // A legenda anual da Colômbia traz também "Quemado MODIS MCD64A1" (referência externa): fica de fora do total.
    if (id === "anualTotal") items = items.filter((i) => !/modis/i.test(i.name));
    return { key: legend.key, items };
  });
}

function getStats(params) {
  const env = state.conn.env;
  return cached(`${env.id}:stats:${JSON.stringify(params)}`, () =>
    fetchAreaStatistics(env, params, (status, elapsedMs) => {
      const fase = { pending: "na fila", running: "em execução" }[status] || status;
      setStatus(`Calculando no servidor (${fase}) · ${Math.round(elapsedMs / 1000)} s`, true);
    })
  );
}

function getGeometry(ids, method = "union") {
  return cached(`${envId()}:geom:${ids.join(",")}:${method}`, async () =>
    (await fetchTerritoryGeometry(state.conn.env, ids, method)).geometry
  );
}

function getMapUrl(params) {
  return cached(`${envId()}:map:${JSON.stringify(params)}`, async () => {
    const { url } = await fetchMapTiles(state.conn.env, params);
    if (!url) throw new Error("A API não devolveu URL de tiles para esta camada.");
    return url;
  });
}

// Todos os municípios (≈5.570, em páginas de 1000), agrupados por UF.
function getMunicipiosPorUf() {
  const { env, catMunicipio } = state.conn;
  return cached(`${env.id}:municipios`, async () => {
    const size = CONFIG.territoriosPorPagina;
    const first = await fetchTerritories(env, catMunicipio, 1, size);
    const rest = await Promise.all(
      range(2, first.numberOfPages || 1).map((p) => fetchTerritories(env, catMunicipio, p, size))
    );
    const porUf = new Map();
    for (const t of [first, ...rest].flatMap((r) => r.territories || [])) {
      const match = (t.nameFormatted || t.name).match(/\s-\s([A-Z]{2})(?:\s-\s\d+)?$/);
      if (!match) continue;
      const nome = t.name.replace(/\s-\s[A-Z]{2}(?:\s-\s\d+)?$/, "");
      const lista = porUf.get(match[1]) || [];
      lista.push({ id: t.id, name: `${nome} (${match[1]})`, nome });
      porUf.set(match[1], lista);
    }
    for (const lista of porUf.values()) lista.sort((a, b) => a.nome.localeCompare(b.nome, "pt-BR"));
    return porUf;
  });
}

// Ranking oficial de UM ano para uma categoria (estados, municípios, biomas),
// guardado já reduzido a Map(id -> {name, value}).
function getRankingAno(categoryId, ano) {
  return cached(`${envId()}:ranking:${categoryId}:${ano}`, async () => {
    const total = await getLegend("anualTotal");
    const params = {
      subthemeKey: sub("anual"), legendKey: total.key, pixelValue: levelItems(total, 0)[0].pixelValue, // o ranking aceita um valor só
      year: ano, territoryCategoryId: categoryId, pageSize: 6000,
    };
    // Rankings ainda não calculados (comuns fora do Brasil) voltam como tarefa: espera e pede de novo.
    const inicio = Date.now();
    let resp = await fetchRanking(state.conn.env, params);
    let falhas = 0;
    while (!Array.isArray(resp.ranking) && resp.taskID && Date.now() - inicio < CONFIG.taskTimeoutMs) {
      try {
        await waitForTask(state.conn.env, resp.taskID, inicio, (status, ms) =>
          setStatus(`Calculando o ranking no servidor (${status}) · ${Math.round(ms / 1000)} s`, true));
      } catch (err) {
        if (++falhas > 2) throw err; // tarefa "failed" costuma ser passageira: pede de novo até 2 vezes
      }
      resp = await fetchRanking(state.conn.env, params);
    }
    if (!Array.isArray(resp.ranking) && resp.taskID) throw new ApiError("O servidor ainda está calculando este ranking. Tente de novo em alguns minutos.");
    return new Map((resp.ranking || []).map((r) => [r.territory.id, { name: ehBrasil ? r.territory.name : nomeLimpo(r.territory.name), value: r.value || 0 }]));
  });
}

// Soma do ranking anual em [ini, fim]: { linhas: [{id, name, value, porAno}], anos }.
async function rankingPeriodo(categoryId, ini, fim, onProgress) {
  const anos = range(ini, fim);
  let feitos = 0;
  const porAno = await mapLimit(anos, 6, async (ano) => {
    const m = await getRankingAno(categoryId, ano);
    onProgress?.(++feitos, anos.length);
    return m;
  });
  const acc = new Map();
  porAno.forEach((m, i) => {
    for (const [id, { name, value }] of m) {
      const row = acc.get(id) || { id, name, value: 0, porAno: new Array(anos.length).fill(0) };
      row.value += value;
      row.porAno[i] = value;
      acc.set(id, row);
    }
  });
  return { anos, linhas: [...acc.values()].sort((a, b) => b.value - a.value) };
}

async function mapLimit(items, limit, fn) {
  const out = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      out[i] = await fn(items[i], i);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

// ---------------------------------------------------------------------------
// Geometria de territórios a partir dos tiles vetoriais da API
//   GET /{region}/territories/tiles/{z}/{x}/{y}.mvt?categoryId=N
//   (mesma fonte que a plataforma usa para desenhar territórios; cada feição traz id e name)
// ---------------------------------------------------------------------------

const BRASIL_BBOX = [-74.5, -34.5, -28.5, 5.5]; // [oeste, sul, leste, norte]

// Caixa [oeste, sul, leste, norte] do país, pelo contorno da API (outros países).
async function caixaDoPais() {
  const g = await getGeometry([state.conn.brasil.id]);
  const polys = g.type === "Polygon" ? [g.coordinates] : g.coordinates;
  const c = [Infinity, Infinity, -Infinity, -Infinity];
  for (const rings of polys) for (const [x, y] of rings[0]) {
    if (x < c[0]) c[0] = x; if (y < c[1]) c[1] = y; if (x > c[2]) c[2] = x; if (y > c[3]) c[3] = y;
  }
  return c;
}

// Municípios (ou o 2º nível do país) dentro de um estado. Brasil: pela UF no nome;
// outros países: centróide do município (tiles vetoriais) dentro do contorno do estado.
function getMunicipiosDoEstado(estado) {
  if (estado.uf) return getMunicipiosPorUf().then((m) => m.get(estado.uf) || []);
  return cached(`${envId()}:munsDe:${estado.id}`, async () => {
    if (!state.conn.catMunicipio) return [];
    const [shapes, geom] = await Promise.all([getMunicipiosShapes(), getGeometry([estado.id])]);
    const lista = [];
    for (const m of shapes.values()) {
      if (m.centroide && pointInGeometry(m.centroide, geom)) lista.push({ id: m.id, name: nomeMunicipio(m.name), nome: nomeMunicipio(m.name) });
    }
    return lista.sort((a, b) => a.nome.localeCompare(b.nome, "pt-BR"));
  });
}

// "Abancay - Abancay - Apurimac - Perú" -> "Abancay": só a primeira parte, sem códigos.
const nomeMunicipio = (nome) => (ehBrasil ? nome : String(nome || "").split(" - ")[0]);
const MVT_ZOOM_MUNICIPIOS = 5;

const lon2tile = (lon, z) => Math.floor(((lon + 180) / 360) * 2 ** z);
const lat2tile = (lat, z) => {
  const r = (lat * Math.PI) / 180;
  return Math.floor(((1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2) * 2 ** z);
};
const tile2lon = (x, z) => (x / 2 ** z) * 360 - 180;
const tile2lat = (y, z) => (180 / Math.PI) * Math.atan(Math.sinh(Math.PI - (2 * Math.PI * y) / 2 ** z));

async function fetchTerritoryTile(env, categoryId, z, x, y) {
  const url = buildUrl(env, `/${CONFIG.region}/territories/tiles/${z}/${x}/${y}.mvt`, { categoryId });
  const resp = await fetch(url);
  if (!resp.ok) throw new ApiError(`HTTP ${resp.status} no tile ${z}/${x}/${y}`, resp.status);
  return new Uint8Array(await resp.arrayBuffer());
}

// Decodificador mínimo de Mapbox Vector Tile (protobuf) -> polígonos em lon/lat.
function decodeMvtPolygons(bytes, z, tx, ty) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const utf8 = new TextDecoder();
  let pos = 0;
  const varint = () => {
    let r = 0, s = 0, b;
    do { b = bytes[pos++]; r += (b & 0x7f) * 2 ** s; s += 7; } while (b & 0x80);
    return r;
  };
  const fields = (start, end, fn) => {
    const salvo = pos;
    pos = start;
    while (pos < end) {
      const key = varint(), tag = Math.floor(key / 8), wt = key & 7;
      if (wt === 0) fn(tag, varint());
      else if (wt === 2) { const len = varint(), s = pos; pos += len; fn(tag, [s, s + len]); }
      else if (wt === 5) { fn(tag, view.getFloat32(pos, true)); pos += 4; }
      else if (wt === 1) { fn(tag, view.getFloat64(pos, true)); pos += 8; }
      else throw new Error("Tile vetorial inválido");
    }
    pos = salvo;
  };
  const packed = ([s, e]) => { const out = []; const salvo = pos; pos = s; while (pos < e) out.push(varint()); pos = salvo; return out; };
  const zigzag = (n) => (n % 2 ? -(n + 1) / 2 : n / 2);

  const features = [];
  fields(0, bytes.length, (tag, layerRange) => {
    if (tag !== 3) return;
    const keys = [], values = [], raw = [];
    let extent = 4096;
    fields(layerRange[0], layerRange[1], (t, v) => {
      if (t === 3) keys.push(utf8.decode(bytes.subarray(v[0], v[1])));
      if (t === 4) fields(v[0], v[1], (t2, v2) => values.push(t2 === 1 ? utf8.decode(bytes.subarray(v2[0], v2[1])) : v2));
      if (t === 5) extent = v;
      if (t === 2) raw.push(v);
    });
    for (const fr of raw) {
      let tags = [], geom = [], type = 0;
      fields(fr[0], fr[1], (t, v) => {
        if (t === 2) tags = packed(v);
        if (t === 3) type = v;
        if (t === 4) geom = packed(v);
      });
      if (type !== 3) continue; // só polígonos
      const props = {};
      for (let i = 0; i < tags.length; i += 2) props[keys[tags[i]]] = values[tags[i + 1]];
      // Comandos MoveTo(1)/LineTo(2)/ClosePath(7) em coordenadas do tile.
      const rings = [];
      let cx = 0, cy = 0, ring = null;
      for (let i = 0; i < geom.length;) {
        const cmd = geom[i] & 7, count = geom[i] >> 3;
        i++;
        if (cmd === 1 || cmd === 2) {
          for (let k = 0; k < count; k++) {
            cx += zigzag(geom[i++]);
            cy += zigzag(geom[i++]);
            if (cmd === 1) { ring = [[cx, cy]]; rings.push(ring); } else ring.push([cx, cy]);
          }
        } else if (cmd === 7 && ring) ring.push(ring[0]);
      }
      // Os tiles desta API não seguem com rigor o giro da especificação (pedaços cortados na
      // borda vêm invertidos). Por isso um anel só é buraco se estiver DENTRO do anel externo
      // anterior; senão é um novo polígono.
      const polygons = [];
      let maior = { area: 0, ring: null };
      let externo = null;
      for (const r of rings) {
        const a = Math.abs(ringArea(r));
        if (!a) continue;
        const lonlat = r.map(([px, py]) => [tile2lon(tx + px / extent, z), tile2lat(ty + py / extent, z)]);
        if (externo && a < externo.area && pointInGeometry(lonlat[0], { type: "Polygon", coordinates: [externo.ring] })) {
          polygons[polygons.length - 1].push(lonlat);
        } else {
          polygons.push([lonlat]);
          externo = { area: a, ring: lonlat };
          if (a > maior.area) maior = { area: a, ring: lonlat };
        }
      }
      if (polygons.length) features.push({ id: props.id, name: props.name, polygons, maior });
    }
  });
  return features;
}

function ringArea(r) {
  let a = 0;
  for (let i = 0; i < r.length - 1; i++) a += r[i][0] * r[i + 1][1] - r[i + 1][0] * r[i][1];
  return a / 2;
}

function ringCentroid(r) {
  let a = 0, x = 0, y = 0;
  for (let i = 0; i < r.length - 1; i++) {
    const f = r[i][0] * r[i + 1][1] - r[i + 1][0] * r[i][1];
    a += f; x += (r[i][0] + r[i + 1][0]) * f; y += (r[i][1] + r[i + 1][1]) * f;
  }
  return a ? [x / (3 * a), y / (3 * a)] : r[0];
}

// Ponto em polígono (GeoJSON Polygon/MultiPolygon, com buracos), por raycasting.
function pointInGeometry([px, py], geometry) {
  const inRing = (ring) => {
    let dentro = false;
    for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
      const [xi, yi] = ring[i], [xj, yj] = ring[j];
      if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) dentro = !dentro;
    }
    return dentro;
  };
  const inPoly = (rings) => inRing(rings[0]) && !rings.slice(1).some(inRing);
  if (!geometry) return false;
  if (geometry.type === "Polygon") return inPoly(geometry.coordinates);
  if (geometry.type === "MultiPolygon") return geometry.coordinates.some(inPoly);
  return false;
}

// Todos os municípios do Brasil a partir dos tiles: Map(id -> {name, polygons, centroide}).
function getMunicipiosShapes(onProgress) {
  const { env, catMunicipio } = state.conn;
  const z = MVT_ZOOM_MUNICIPIOS;
  return cached(`${env.id}:mvt:${catMunicipio}:${z}`, async () => {
    const [w, s, e, n] = ehBrasil ? BRASIL_BBOX : await caixaDoPais();
    const tiles = [];
    for (let x = lon2tile(w, z); x <= lon2tile(e, z); x++)
      for (let y = lat2tile(n, z); y <= lat2tile(s, z); y++) tiles.push([x, y]);
    let feitos = 0;
    const partes = await mapLimit(tiles, 8, async ([x, y]) => {
      const bytes = await fetchTerritoryTile(env, catMunicipio, z, x, y);
      onProgress?.(++feitos, tiles.length);
      return decodeMvtPolygons(bytes, z, x, y);
    });
    // Um município aparece cortado em vários tiles: junta os pedaços e usa o maior para o centróide.
    const porId = new Map();
    for (const f of partes.flat()) {
      const m = porId.get(f.id) || { id: f.id, name: f.name, polygons: [], maior: { area: 0, ring: null } };
      m.polygons.push(...f.polygons);
      if (f.maior.area > m.maior.area) m.maior = f.maior;
      porId.set(f.id, m);
    }
    for (const m of porId.values()) m.centroide = m.maior.ring ? ringCentroid(m.maior.ring) : null;
    return porId;
  });
}

// ---------------------------------------------------------------------------
// Recorte município × bioma
//   Mapa: cada município é recortado no cliente (Turf.js) pela geometria do bioma da API,
//   e só a parte dentro do bioma é desenhada.
//   Dados: o ranking da API só tem o município inteiro. Para os municípios na divisa, a área
//   exata vem de /statistics/area com territoryId=[município, bioma] e spatialMethod=
//   intersection (o servidor calcula a interseção: a 1ª vez leva minutos por município,
//   depois fica em cache lá). Os da divisa sem valor ainda aparecem como "em cálculo".
// ---------------------------------------------------------------------------

// Fração do município dentro do bioma abaixo da qual é sobra de simplificação das bordas,
// e acima da qual o município conta como inteiro no bioma.
const RECORTE_MIN = 0.01;
const RECORTE_MAX = 0.98;

// Map(id -> { id, name, situacao: "dentro" | "divisa", polygons, fracao }) dos municípios do bioma.
function getRecorteBioma(biomaId, onProgress) {
  return cached(`${envId()}:recorte:${biomaId}`, async () => {
    const [shapes, geom] = await Promise.all([getMunicipiosShapes(onProgress), getGeometry([biomaId])]);
    const borda = indiceDeBorda(geom);
    const biomaFeature = { type: "Feature", properties: {}, geometry: geom };
    const out = new Map();
    let n = 0;
    for (const m of shapes.values()) {
      // O recorte leva alguns segundos: devolve o controle ao navegador a cada lote.
      if (++n % 150 === 0) {
        setStatus(`Recortando os municípios pelo bioma… ${Math.round((100 * n) / shapes.size)}%`, true);
        await new Promise((r) => setTimeout(r));
      }
      const caixa = caixaDe(m.polygons);
      if (!caixasSeTocam(caixa, borda.caixa)) continue;
      if (!borda.cruza(caixa)) {
        // Nenhum trecho da borda do bioma passa pela caixa do município: está todo dentro ou todo fora.
        if (m.centroide && pointInGeometry(m.centroide, geom)) {
          out.set(m.id, { id: m.id, name: m.name, situacao: "dentro", polygons: m.polygons, fracao: 1 });
        }
        continue;
      }
      let areaTotal = 0, areaDentro = 0;
      const pedacos = [];
      for (const coords of m.polygons) {
        let poly;
        try { poly = turf.polygon(coords); } catch { continue; }
        areaTotal += turf.area(poly);
        // O bioma é cortado antes à caixa do pedaço, para o Turf trabalhar com poucos vértices.
        const local = limparPoligonos(turf.bboxClip(biomaFeature, turf.bbox(poly)).geometry);
        if (!local) continue;
        let inter = null;
        try { inter = turf.intersect(poly, { type: "Feature", properties: {}, geometry: local }); } catch { /* geometria degenerada */ }
        if (!inter) continue;
        areaDentro += turf.area(inter);
        if (inter.geometry.type === "Polygon") pedacos.push(inter.geometry.coordinates);
        else if (inter.geometry.type === "MultiPolygon") pedacos.push(...inter.geometry.coordinates);
      }
      const fracao = safeDiv(areaDentro, areaTotal);
      if (fracao < RECORTE_MIN) continue;
      out.set(m.id, fracao > RECORTE_MAX
        ? { id: m.id, name: m.name, situacao: "dentro", polygons: m.polygons, fracao: 1 }
        : { id: m.id, name: m.name, situacao: "divisa", polygons: pedacos, fracao });
    }
    return out;
  });
}

// Remove anéis degenerados (< 4 posições) que o bboxClip pode deixar; null se nada sobrar.
function limparPoligonos(geometry) {
  const polys = (geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates)
    .map((rings) => rings.filter((r) => r.length >= 4))
    .filter((rings) => rings.length);
  return polys.length ? { type: "MultiPolygon", coordinates: polys } : null;
}

function caixaDe(polygons) {
  const c = [Infinity, Infinity, -Infinity, -Infinity];
  for (const rings of polygons) for (const [x, y] of rings[0]) {
    if (x < c[0]) c[0] = x; if (y < c[1]) c[1] = y; if (x > c[2]) c[2] = x; if (y > c[3]) c[3] = y;
  }
  return c;
}

const caixasSeTocam = (a, b) => a[0] <= b[2] && b[0] <= a[2] && a[1] <= b[3] && b[1] <= a[3];

// Segmentos da borda do bioma numa grade regular, para saber rápido se a borda
// atravessa a caixa de um município.
function indiceDeBorda(geometry, celula = 0.5) {
  const polys = geometry.type === "Polygon" ? [geometry.coordinates] : geometry.coordinates;
  const grade = new Map();
  const caixa = [Infinity, Infinity, -Infinity, -Infinity];
  const chave = (i, j) => `${i},${j}`;
  for (const rings of polys) for (const r of rings) for (let k = 0; k < r.length - 1; k++) {
    const [a, b] = [r[k], r[k + 1]];
    const [x0, x1] = [Math.min(a[0], b[0]), Math.max(a[0], b[0])];
    const [y0, y1] = [Math.min(a[1], b[1]), Math.max(a[1], b[1])];
    caixa[0] = Math.min(caixa[0], x0); caixa[1] = Math.min(caixa[1], y0);
    caixa[2] = Math.max(caixa[2], x1); caixa[3] = Math.max(caixa[3], y1);
    for (let i = Math.floor(x0 / celula); i <= Math.floor(x1 / celula); i++)
      for (let j = Math.floor(y0 / celula); j <= Math.floor(y1 / celula); j++) {
        const lista = grade.get(chave(i, j)) || [];
        lista.push([a, b]);
        grade.set(chave(i, j), lista);
      }
  }
  const cruza = (c) => {
    for (let i = Math.floor(c[0] / celula); i <= Math.floor(c[2] / celula); i++)
      for (let j = Math.floor(c[1] / celula); j <= Math.floor(c[3] / celula); j++)
        for (const [a, b] of grade.get(chave(i, j)) || []) if (segmentoNaCaixa(a, b, c)) return true;
    return false;
  };
  return { caixa, cruza };
}

// Segmento ab toca o retângulo c = [oeste, sul, leste, norte]? (recorte de Liang–Barsky)
function segmentoNaCaixa(a, b, c) {
  let t0 = 0, t1 = 1;
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const testes = [[-dx, a[0] - c[0]], [dx, c[2] - a[0]], [-dy, a[1] - c[1]], [dy, c[3] - a[1]]];
  for (const [p, q] of testes) {
    if (p === 0) { if (q < 0) return false; continue; }
    const r = q / p;
    if (p < 0) { if (r > t1) return false; if (r > t0) t0 = r; } else { if (r < t0) return false; if (r < t1) t1 = r; }
  }
  return true;
}

// Área queimada exata (anos ini–fim) da parte do município dentro do bioma, calculada pelo
// servidor. Resultados prontos ficam em `recorteValores`; os que faltam entram numa fila.
const recorteValores = new Map(); // chave -> { value, porAno }
const recorteFalhas = new Set();
const recorteFila = { itens: [], ativos: 0, feitos: 0, total: 0 };
const RECORTE_CONCORRENCIA = 8; // o servidor processa várias interseções em paralelo (25–90 s cada, medido)
const chaveRecorte = (munId, biomaId, ini, fim) => `${envId()}:${munId}:${biomaId}:${ini}:${fim}`;

function valoresDaDivisa(divisa, biomaId, ini, fim) {
  const valores = new Map();
  let pendentes = 0, falhas = 0;
  for (const id of divisa) {
    const k = chaveRecorte(id, biomaId, ini, fim);
    if (recorteValores.has(k)) valores.set(id, recorteValores.get(k));
    else if (recorteFalhas.has(k)) falhas++;
    else { pendentes++; enfileirarRecorte(k, id, biomaId, ini, fim); }
  }
  return { valores, pendentes, falhas };
}

function enfileirarRecorte(k, munId, biomaId, ini, fim) {
  if (recorteFila.itens.some((it) => it.k === k) || recorteFila.emCurso?.has(k)) return;
  recorteFila.itens.push({ k, munId, biomaId, ini, fim });
  recorteFila.total++;
  processarFilaRecorte();
}

function processarFilaRecorte() {
  recorteFila.emCurso = recorteFila.emCurso || new Set();
  while (recorteFila.ativos < RECORTE_CONCORRENCIA && recorteFila.itens.length) {
    const it = recorteFila.itens.shift();
    recorteFila.ativos++;
    recorteFila.emCurso.add(it.k);
    calcularRecorte(it)
      .then((v) => recorteValores.set(it.k, v))
      .catch((err) => { console.warn("[recorte]", it.munId, err); recorteFalhas.add(it.k); })
      .finally(() => {
        recorteFila.ativos--;
        recorteFila.feitos++;
        recorteFila.emCurso.delete(it.k);
        const terminou = !recorteFila.ativos && !recorteFila.itens.length;
        if (terminou) {
          recorteFila.feitos = recorteFila.total = 0;
          setStatus("");
          // Tudo calculado: redesenha a aba que pediu (agora com os valores exatos, do cache).
          if (["mapa", "territorios"].includes(state.tab)) loadCurrentTab();
        } else {
          setStatus(`Recorte exato município × bioma nas divisas: ${recorteFila.feitos}/${recorteFila.total} municípios calculados no servidor…`, true);
          processarFilaRecorte();
        }
      });
  }
}

async function calcularRecorte({ munId, biomaId, ini, fim }) {
  const total = await getLegend("anualTotal");
  const anos = range(ini, fim);
  const res = await fetchAreaStatistics(state.conn.env, {
    subthemeKey: sub("anual"), legendKey: total.key, pixelValue: pixelValues(levelItems(total, 0)),
    year: anos, territoryId: [munId, biomaId], spatialMethod: "intersection",
  }, () => {});
  const porAnoMap = new Map(byYear(res).map((a) => [a.ano, a.area]));
  const porAno = anos.map((a) => porAnoMap.get(a) || 0);
  return { value: sum(porAno), porAno };
}

// Aplica o recorte às linhas do ranking de municípios: os "dentro" mantêm o valor do
// ranking (= município inteiro); os da divisa recebem a área exata da interseção ou,
// enquanto ela não chega, ficam fora das linhas (contados em `pendentes`).
function aplicarRecorte(linhas, escopo, ini, fim) {
  if (!escopo.recorte) return { linhas, pendentes: 0, falhas: 0, divisa: 0 };
  const membros = [...escopo.recorte.values()].filter((m) => escopo.filtro(m));
  const divisa = membros.filter((m) => m.situacao === "divisa").map((m) => m.id);
  const { valores, pendentes, falhas } = valoresDaDivisa(divisa, escopo.biomaId, ini, fim);
  const porId = new Map(linhas.map((l) => [l.id, l]));
  const saida = [];
  for (const m of membros) {
    if (m.situacao === "dentro") {
      if (porId.has(m.id)) saida.push(porId.get(m.id));
    } else if (valores.has(m.id) && valores.get(m.id).value > 0) {
      saida.push({ id: m.id, name: m.name, ...valores.get(m.id), recortado: true, fracao: m.fracao });
    }
  }
  saida.sort((a, b) => b.value - a.value);
  return { linhas: saida, pendentes, falhas, divisa: divisa.length, valores };
}

function flattenLegend(items, depth = 0, parent = null, out = []) {
  for (const item of items || []) {
    out.push({
      pixelValue: item.pixelValue,
      name: item.name?.[IDIOMA.api] || (IDIOMA.id === "id" && item.name?.["en-US"]) || item.name?.["pt-BR"] || item.name?.["es-ES"] || item.name?.["en-US"] || item.key,
      color: safeColor(item.color),
      depth,
      parent,
      hasChildren: Array.isArray(item.children) && item.children.length > 0,
    });
    flattenLegend(item.children, depth + 1, item.pixelValue, out);
  }
  return out;
}

const levelItems = (legend, depth) => legend.items.filter((i) => i.depth === depth);
const maxDepth = (legend) => Math.max(0, ...legend.items.map((i) => i.depth));
const pixelValues = (items) => items.map((i) => i.pixelValue);

// Corte da árvore no nível pedido: itens daquela profundidade + folhas mais rasas.
const cutLevel = (legend, depth) =>
  legend.items.filter((i) => i.depth === depth || (i.depth < depth && !i.hasChildren));

// Grupos da Frequência vêm rotulados em "anos" ("1 a 2 anos"), mas agrupam
// contagens de ocorrência (os filhos são "1 vez", "2 vezes"…).
const frequencyLabel = (name) => name.replace(/\banos?\b/i, "vezes");

// Somamos `items` em vez de usar `total`: na Frequência o `total` é a área do
// território inteiro (= territoryAreaHa), não a área queimada.
const sumItems = (entry) => (entry?.items || []).reduce((s, it) => s + (it.value || 0), 0);
const valueMap = (entry) => new Map((entry?.items || []).map((it) => [it.pixelValue, it.value || 0]));
const lastYearOf = (year) => (Array.isArray(year) ? year[year.length - 1] : year);

function byYear(payload) {
  return (payload.statistic || [])
    .map((s) => ({ ano: lastYearOf(s.year), area: sumItems(s), valores: valueMap(s) }))
    .sort((a, b) => a.ano - b.ano);
}

// Soma os valores de cada classe em todas as entradas (anos) da resposta.
function totalsByClass(classes, payload) {
  const anos = byYear(payload);
  return classes.map((c) => ({
    label: c.name,
    color: c.color,
    item: c,
    value: sum(anos.map((a) => a.valores.get(c.pixelValue) || 0)),
  }));
}

// ---------------------------------------------------------------------------
// Território ativo (filtros globais)
// ---------------------------------------------------------------------------

function activeTerritory() {
  const c = state.conn;
  const bioma = c.biomas.find((t) => t.id === state.biomaId);
  const estado = c.estados.find((t) => t.id === state.estadoId);
  const municipio = state.municipiosDoEstado.find((t) => t.id === state.municipioId);
  const partes = [bioma, municipio || estado].filter(Boolean);
  if (!partes.length) return { ids: [c.brasil.id], nome: PAIS.nome, metodo: "union" };
  return {
    ids: partes.map((p) => p.id),
    nome: partes.map((p) => p.name).join(" ∩ "),
    metodo: partes.length > 1 ? "intersection" : "union",
  };
}

const territoryParams = (t) => ({ territoryId: t.ids, spatialMethod: t.metodo });

// ---------------------------------------------------------------------------
// Abas: cada uma busca seus dados e devolve um "view model"
// ---------------------------------------------------------------------------

// Série anual de cada bioma para a vista "Por bioma" (barras empilhadas). Com estado/município
// no filtro, usa a interseção do recorte com cada bioma; com bioma no filtro, só ele.
// As cores saem da paleta escolhida na hora de desenhar (trocar de paleta não refaz as consultas).
async function porBiomaAnual(t, params, anos, axisTitle, progress) {
  const c = state.conn;
  const idsBioma = new Set(c.biomas.map((b) => b.id));
  const base = t.ids.filter((id) => !idsBioma.has(id) && id !== c.brasil.id);
  const biomas = c.biomas.filter((b) => !state.biomaId || b.id === state.biomaId)
    .sort((a, b) => ordemBioma(a.name) - ordemBioma(b.name));
  let feitos = 0;
  progress(`Calculando biomas… 0/${biomas.length}`);
  const series = await mapLimit(biomas, 3, async (b) => {
    const r = await getStats({ ...params, territoryId: [...base, b.id], spatialMethod: base.length ? "intersection" : "union" });
    progress(`Calculando biomas… ${++feitos}/${biomas.length}`);
    const porAno = new Map(byYear(r).map((a) => [a.ano, a.area]));
    return { label: b.name, values: anos.map((a) => porAno.get(a) || 0), areaHa: r.territoryAreaHa || 0 };
  });
  const chart = {
    kind: "stacked", height: 300, big: true, unit: "Mha", axisTitle,
    labels: anos.map(String), series: series.filter((s) => sum(s.values) > 0),
  };
  const notaMha = maiorValorDoGrafico(chart) >= 1e6 ? "Mha = milhões de hectares" : "";
  return {
    chart,
    nota: [base.length ? "Parte do recorte dentro de cada bioma" : "", notaMha].filter(Boolean).join(" · "),
    biomas: series, // inclui a área de cada bioma (no recorte), informada pela API
    recorte: base.length > 0,
  };
}

// Rosquinhas "quanto do bioma queimou": área queimada (vermelho) x restante do bioma (cinza).
const VERMELHO_PROPORCAO = "#e8452f";

function miniRoscasHtml(itens) {
  if (!itens.length) return `<p class="empty">A API não informou a área dos biomas neste recorte.</p>`;
  return `<ul class="mini-donuts">${itens.map((it) => {
    const p = safeDiv(it.queimado, it.area);
    return `
    <li>
      <div class="mini-donut">
        <canvas role="img" aria-label="${escapeHtml(`${it.label}: ${pct(p)} da área queimou ao menos uma vez`)}"></canvas>
        <strong>${pct(p)}</strong>
      </div>
      <p class="md-nome">${escapeHtml(it.label)}</p>
      <p class="md-detalhe">${ha(it.queimado)} de ${ha(it.area)}</p>
    </li>`;
  }).join("")}</ul>`;
}

// Card de proporção queimada que acompanha o filtro:
//  - Brasil: o país e uma rosca por bioma (área queimada ÷ área do bioma);
//  - bioma: só o bioma (ou a interseção, se houver estado/município);
//  - estado/município: o território e a parte dele dentro de cada bioma (a proporção "invertida").
function proporcaoQueimadaCard({ t, ini, fim, final, areaTerritorio, acumuladoPorBioma }) {
  const c = state.conn;
  const bioma = c.biomas.find((b) => b.id === state.biomaId);
  const local = state.estadoId || state.municipioId;
  const total = { label: t.nome, queimado: final.area, area: areaTerritorio || 0 };

  let title, subtitle;
  if (bioma && !local) {
    title = `Quanto do ${bioma.name} queimou ao menos uma vez · ${ini}–${fim}`;
    subtitle = "Área queimada acumulada no período ÷ área total do bioma";
  } else if (bioma) {
    title = `Quanto de ${t.nome} queimou ao menos uma vez · ${ini}–${fim}`;
    subtitle = "Área queimada acumulada ÷ área da interseção entre o território e o bioma";
  } else if (local) {
    title = `Quanto de ${t.nome} queimou ao menos uma vez, por bioma · ${ini}–${fim}`;
    subtitle = "Primeira rosca: o território todo · demais: parte do território dentro de cada bioma, sobre a área dessa parte";
  } else {
    title = `Quanto de cada bioma queimou ao menos uma vez · ${ini}–${fim}`;
    subtitle = `Área queimada acumulada no período ÷ área total do ${rot("bioma").toLowerCase()} · primeira rosca: o país inteiro`;
  }

  return {
    span: 12,
    title,
    subtitle,
    lazy: async (progress) => {
      const itens = total.area > 0 ? [total] : [];
      if (!bioma) {
        const r = await acumuladoPorBioma(progress);
        itens.push(...r.biomas
          .filter((b) => b.areaHa > 0)
          .map((b) => ({ label: b.label, queimado: b.values[b.values.length - 1] || 0, area: b.areaHa }))
          .sort((a, b) => safeDiv(b.queimado, b.area) - safeDiv(a.queimado, a.area)));
      }
      return {
        mini: itens,
        legend: [
          { label: "Queimou ao menos uma vez", color: VERMELHO_PROPORCAO },
          { label: "Não queimou no período", color: "var(--axis)" },
        ],
      };
    },
  };
}

function miniDonutConfig(it, t) {
  const options = baseOptions(t);
  options.interaction = { mode: "nearest", intersect: true };
  options.cutout = "62%";
  options.plugins.tooltip.callbacks = { label: (ctx) => ` ${fmtFull.format(ctx.parsed)} ha (${pct(safeDiv(ctx.parsed, it.area))})` };
  return {
    type: "doughnut",
    data: {
      labels: ["Queimou ao menos uma vez", "Não queimou no período"],
      datasets: [{
        data: [it.queimado, Math.max(0, it.area - it.queimado)],
        backgroundColor: [VERMELHO_PROPORCAO, t.axis],
        borderWidth: 0,
        hoverOffset: 3,
      }],
    },
    options,
  };
}

function paletaToggleHtml() {
  return `
    <div class="paleta-row">
      <span class="paleta-label">Paleta dos biomas</span>
      <div class="vista-toggle paleta-toggle" role="group" aria-label="Paleta dos biomas">
        ${Object.entries(PALETAS_BIOMA).map(([id, p]) => `
          <button type="button" data-paleta="${id}" aria-pressed="${id === state.paletaBioma}" title="Paleta ${p.nome.toLowerCase()}">
            <span class="dots" aria-hidden="true">${p.cores.map((cor) => `<i style="background:${cor}"></i>`).join("")}</span><span>${p.nome}</span>
          </button>`).join("")}
      </div>
    </div>`;
}

async function loadAnual({ t, ini, fim }) {
  // Alguns países não têm a legenda natural x antrópico (ou a de cobertura): as seções somem.
  const [total, cobertura, natural] = await Promise.all([
    getLegend("anualTotal"), getLegend("anualCobertura").catch(() => null), getLegend("anualNatural").catch(() => null),
  ]);
  const niveis = [...(natural ? [0] : []), ...(cobertura ? range(1, maxDepth(cobertura) + 1) : [])];
  if (!niveis.includes(state.nivel)) state.nivel = niveis.includes(1) ? 1 : (niveis[0] ?? 0);
  const semCobertura = !niveis.length;
  const legenda = state.nivel === 0 ? natural : cobertura;
  const classes = semCobertura ? [] : state.nivel === 0 ? levelItems(natural, 0) : cutLevel(cobertura, state.nivel - 1);
  const totalItems = levelItems(total, 0);
  const naturalItems = natural ? levelItems(natural, 0) : [];
  const anosPeriodo = range(ini, fim);
  const pedir = (lg, itens) => getStats({
    subthemeKey: sub("anual"), legendKey: lg.key, pixelValue: pixelValues(itens), year: anosPeriodo, ...territoryParams(t),
  });
  const vazio = Promise.resolve({ statistic: [] });

  const [resTotal, resClasses, resNatural] = await Promise.all([
    pedir(total, totalItems), semCobertura ? vazio : pedir(legenda, classes), natural ? pedir(natural, naturalItems) : vazio,
  ]);

  // Tom um pouco mais escuro que o vermelho oficial "Total queimado", para as barras anuais.
  const vermelhoEscuro = shade(totalItems[0].color, 0.18);
  const anos = byYear(resTotal);
  const anosClasses = byYear(resClasses);
  const totalPeriodo = sum(anos.map((a) => a.area));
  const media = safeDiv(totalPeriodo, anos.length);
  const ultimo = anos[anos.length - 1] || { ano: fim, area: 0 };
  const variacao = safeDiv(ultimo.area, media) - 1;
  const areaTerritorio = resTotal.territoryAreaHa;
  const porClasse = totalsByClass(classes, resClasses).filter((c) => c.value > 0);
  // Pizza 1: natural x antrópico, sem a classe "Indefinido".
  const naturalAntropico = ordenarPorGrupo(totalsByClass(naturalItems, resNatural)
    .filter((c) => !/indefinid/i.test(c.label) && c.value > 0));
  const serie = porClasse.map((c) => ({
    label: c.label,
    color: c.color,
    values: anosClasses.map((a) => a.valores.get(c.item.pixelValue) || 0),
  }));
  const rotuloNivel = state.nivel === 0 ? "Nível 0 · natural e antrópico" : `Nível ${state.nivel} da legenda oficial de uso e cobertura`;
  const comMha = Math.max(0, ...anos.map((a) => a.area)) >= 1e6;
  const view = {
    toolbar: [{
      type: "segmented",
      inSection: true,
      label: "Nível de legenda",
      value: state.nivel,
      options: niveis.map((n) => ({
        value: n, label: String(n),
        title: n === 0 ? "Nível 0 · Natural e antrópico" : `Nível ${n} da legenda de cobertura`,
      })),
      onChange: (v) => { state.nivel = Number(v); loadCurrentTab(); },
    }],
    kpis: [
      { label: `Área queimada em ${ultimo.ano}`, value: ha(ultimo.area), note: "Último ano do período selecionado" },
      {
        label: `Variação em relação à média ${ini}–${fim}`,
        value: `${variacao >= 0 ? "+" : "−"}${pct(Math.abs(variacao))}`,
        note: `${variacao >= 0 ? "Acima" : "Abaixo"} da média anual de ${ha(media)}`,
      },
      {
        label: `Território afetado em ${ultimo.ano}`,
        value: areaTerritorio ? pct(safeDiv(ultimo.area, areaTerritorio)) : "—",
        note: areaTerritorio
          ? porIdioma({ pt: `de ${ha(areaTerritorio)} do recorte`, es: `de ${ha(areaTerritorio)} del recorte`, en: `of ${ha(areaTerritorio)} in the selected area`, id: `dari ${ha(areaTerritorio)} wilayah terpilih` })
          : "Área do território não informada pela API",
      },
    ],
    cards: [
      {
        span: 12,
        base: true,
        title: "Área queimada anual",
        alterna: "anual",
        // Vista "Por bioma": barras empilhadas com a área de cada bioma no recorte (interseção
        // com estado/município quando houver filtro).
        porBioma: (progress) => porBiomaAnual(t, {
          subthemeKey: sub("anual"), legendKey: total.key, pixelValue: pixelValues(totalItems), year: anosPeriodo,
        }, anosPeriodo, "Área queimada (Mha)", progress),
        chart: {
          kind: "bar",
          height: 300,
          big: true,
          unit: "Mha",
          axisTitle: "Área queimada (Mha)",
          labels: anos.map((a) => String(a.ano)),
          values: anos.map((a) => a.area),
          color: vermelhoEscuro,
          meanLine: media,
        },
        legend: [
          { label: "Área queimada no ano", color: vermelhoEscuro },
          { label: `Média ${ini}–${fim}: ${haCurto(media)} por ano`, color: "var(--text-2)", line: true, dash: true },
          ...(comMha ? [{ label: "Mha = milhões de hectares", note: true }] : []),
        ],
      },
      {
        span: 12,
        base: true,
        title: "Anos que mais queimaram",
        subtitle: `Top 6 · ${ini}–${fim} · variação em relação à média do período`,
        topcards: [...anos].sort((a, b) => b.area - a.area).slice(0, 6).map((a) => {
          const v = safeDiv(a.area, media) - 1;
          return { title: String(a.ano), value: ha(a.area), note: `${v >= 0 ? "+" : "−"}${pct(Math.abs(v))} em relação à média` };
        }),
      },
      {
        span: 12,
        section: true,
        title: "Uso e cobertura da terra nas áreas queimadas",
        subtitle: `Soma das áreas anuais ${ini}–${fim}`,
      },
      {
        span: 12,
        title: "Vegetação natural x uso antrópico e detalhamento por classe",
        subtitle: `${rotuloNivel} · classe "Indefinido" removida da barra natural x antrópico`,
        propbars: {
          rows: [
            ...(naturalAntropico.length ? [{ label: "Natural x antrópico", segments: naturalAntropico, nomes: true }] : []),
            ...(state.nivel === 0 ? [] : [{ label: "Por classe", segments: ordenarPorGrupo(porClasse) }]),
          ],
          legend: state.nivel === 0 ? naturalAntropico : ordenarPorGrupo(porClasse),
        },
        alterna: "rosca",
        rosca: [
          ...(naturalAntropico.length ? [{ titulo: "Natural x antrópico", slices: naturalAntropico, centerLabel: "queimados" }] : []),
          ...(state.nivel === 0 ? [] : [{ titulo: "Por classe", slices: ordenarPorGrupo(porClasse), centerLabel: "queimados" }]),
        ],
      },
      {
        span: 12,
        title: "Área queimada por ano e classe",
        subtitle: rotuloNivel,
        alterna: "linhas",
        chart: { kind: "stacked", labels: anosClasses.map((a) => String(a.ano)), series: serie },
        legend: serie.map((s) => ({ label: s.label, color: s.color })),
      },
    ],
  };
  // Sem legenda de cobertura no país: só os cards da área anual (série e anos que mais queimaram).
  return semCobertura ? { ...view, toolbar: [], cards: view.cards.filter((c) => c.base) } : view;
}

async function loadAcumulado({ t, ini, fim }) {
  const legenda = await getLegend("acumulado");
  const itens = levelItems(legenda, 0);
  const res = await getStats({
    subthemeKey: sub("acumulado"), legendKey: legenda.key, pixelValue: pixelValues(itens),
    year: [ini, fim], yearSeriesAnchor: "start", ...territoryParams(t),
  });

  const cor = itens[0].color;
  const pontos = byYear(res);
  const final = pontos[pontos.length - 1] || { area: 0 };
  // Acumulado(ano) − acumulado(ano anterior) = área que queimou pela 1ª vez desde `ini`.
  const ineditas = pontos.map((p, i) => ({ ano: p.ano, area: Math.max(0, p.area - (i ? pontos[i - 1].area : 0)) }));
  const areaTerritorio = res.territoryAreaHa;
  const posteriores = ineditas.slice(1);
  const totalPosteriores = sum(posteriores.map((p) => p.area));
  const labels = pontos.map((p) => String(p.ano));
  // Mesmas consultas (em cache) para a vista "Por bioma" e para a proporção queimada de cada bioma.
  const paramsAcumulado = {
    subthemeKey: sub("acumulado"), legendKey: legenda.key, pixelValue: pixelValues(itens),
    year: [ini, fim], yearSeriesAnchor: "start",
  };
  const acumuladoPorBioma = (progress) =>
    porBiomaAnual(t, paramsAcumulado, range(ini, fim), "Área queimada acumulada (Mha)", progress);

  return {
    kpis: [
      {
        label: `Área queimada acumulada ${ini}–${fim}`,
        value: ha(final.area),
        note: areaTerritorio
          ? `${pct(safeDiv(final.area, areaTerritorio))} do território queimou ao menos uma vez`
          : "Queimou ao menos uma vez no período",
      },
      { label: `Área queimada pela 1ª vez em ${fim}`, value: ha(ineditas[ineditas.length - 1]?.area || 0), note: `Sem registro de fogo desde ${ini}` },
      {
        label: "Média de área inédita por ano",
        value: posteriores.length ? ha(mean(posteriores.map((p) => p.area))) : "—",
        note: posteriores.length ? `${ini + 1}–${fim}` : "Período de um ano",
      },
    ],
    cards: [
      {
        span: 12,
        title: "Área queimada acumulada",
        alterna: "anual",
        porBioma: acumuladoPorBioma,
        chart: {
          kind: "line",
          height: 300,
          big: true,
          unit: "Mha",
          axisTitle: "Área queimada acumulada (Mha)",
          labels,
          values: pontos.map((p) => p.area),
          color: shade(cor, 0.18),
        },
        legend: [
          { label: `Área que queimou ao menos uma vez desde ${ini}`, color: shade(cor, 0.18), line: true },
          ...(final.area >= 1e6 ? [{ label: "Mha = milhões de hectares", note: true }] : []),
        ],
      },
      proporcaoQueimadaCard({ t, ini, fim, final, areaTerritorio, acumuladoPorBioma }),
      {
        span: 12,
        title: "Composição do acumulado por ano",
        subtitle: "Barra inferior = acumulado até o ano anterior · barra superior = área queimada pela 1ª vez no ano · total = novo acumulado",
        chart: {
          kind: "stacked",
          labels,
          series: [
            { label: "Acumulado até o ano anterior", color: "var(--axis)", values: pontos.map((p, i) => p.area - ineditas[i].area) },
            { label: "Queimou pela 1ª vez no ano", color: cor, values: ineditas.map((p) => p.area) },
          ],
        },
        legend: [
          { label: "Acumulado até o ano anterior", color: "var(--axis)" },
          { label: "Queimou pela 1ª vez no ano", color: cor },
        ],
      },
      {
        span: 12,
        title: "Anos que mais adicionaram área inédita",
        subtitle: posteriores.length
          ? `Top 5 · ${ini + 1}–${fim} (o ano inicial é todo inédito por definição)`
          : "Escolha um período maior que um ano",
        topcards: [...posteriores].sort((a, b) => b.area - a.area).slice(0, 5).map((p) => ({
          title: String(p.ano),
          value: ha(p.area),
          note: `${pct(safeDiv(p.area, totalPosteriores))} da área inédita de ${ini + 1}–${fim}`,
        })),
      },
    ],
  };
}

// Paleta "tons de fogo" dos meses (como o gráfico mensal da Plataforma): jan–mar vermelhos,
// abr–out amarelos e laranjas, nov–dez carmim. A "oficial" usa as cores da legenda da API.
const PALETA_MESES_FOGO = ["#8e0b1d", "#a8131f", "#c4262e", "#ee7a22", "#f6b51e", "#f7a33a",
  "#f39130", "#ec7a24", "#e66a1d", "#d95f1a", "#b8173c", "#9f1238"];

async function loadMensal({ t, ini, fim }) {
  const [legenda, total] = await Promise.all([getLegend("mensal"), getLegend("anualTotal")]);
  const meses = levelItems(legenda, 0).slice(0, 12); // jan..dez, na ordem da legenda
  const res = await getStats({
    subthemeKey: sub("mensal"), legendKey: legenda.key, pixelValue: pixelValues(meses),
    year: range(ini, fim), ...territoryParams(t),
  });
  const anosCal = byYear(res);
  const valorCal = (ano, k) => anosCal.find((a) => a.ano === ano)?.valores.get(meses[k].pixelValue) || 0;

  // Temporada do fogo: do mês de início ao mês de fim (pode atravessar o ano, ex.: jul–jun).
  const mi = clamp(state.mesIni, 1, 12), mf = clamp(state.mesFim, 1, 12);
  const janela = [];
  for (let k = 0; k < 12; k++) {
    const mm = ((mi - 1 + k) % 12) + 1;
    janela.push(mm - 1);
    if (mm === mf) break;
  }
  const cruza = janela.some((k, j) => j > 0 && k < janela[j - 1]);
  const mesesJ = janela.map((k) => meses[k]);
  const corMes = (k) => (state.paletaMes === "fogo" ? PALETA_MESES_FOGO[k] : meses[k].color);
  // Uma temporada por ano de início; atravessando o ano, a última precisa do ano seguinte no período.
  const temporadas = [];
  for (let Y = ini; Y <= fim; Y++) {
    if (cruza && Y + 1 > fim) break;
    temporadas.push({
      ano: Y,
      rotulo: cruza ? `${Y}/${String(Y + 1).slice(2)}` : String(Y),
      vals: janela.map((k) => valorCal(cruza && k < mi - 1 ? Y + 1 : Y, k)),
    });
  }
  const n = temporadas.length;
  const rotuloJanela = janela.length === 12 && !cruza ? "jan–dez" : `${meses[janela[0]].name.slice(0, 3).toLowerCase()}–${meses[janela[janela.length - 1]].name.slice(0, 3).toLowerCase()}`;

  // Matriz temporada × mês -> soma, média e desvio padrão amostral (n − 1), como a Tabela 9 do RAF.
  const porMes = janela.map((_, j) => temporadas.map((s) => s.vals[j]));
  const somas = porMes.map(sum);
  const totalPeriodo = sum(somas);
  const medias = porMes.map(mean);
  const dps = porMes.map((vals, i) => (n > 1 ? Math.sqrt(sum(vals.map((v) => (v - medias[i]) ** 2)) / (n - 1)) : 0));
  const picoIdx = medias.indexOf(Math.max(...medias));
  const curtos = mesesJ.map((m) => m.name.slice(0, 3));
  // Trimestre consecutivo de maior soma dentro da temporada (o RAF destaca agosto–outubro).
  const trimestres = janela.length >= 3 ? janela.slice(0, -2).map((_, i) => ({ i, soma: somas[i] + somas[i + 1] + somas[i + 2] })) : [];
  const trimestre = maxBy(trimestres, (x) => x.soma);
  const nomeTri = trimestre ? `${curtos[trimestre.i]}–${curtos[trimestre.i + 2]}` : "—";

  // Por padrão destaca a última temporada em vermelho, como a Figura 13 do RAF.
  const disponiveis = temporadas.map((s) => s.ano);
  const ultimaTemp = disponiveis[disponiveis.length - 1];
  if (!state.mensalAnosTocados) state.mensalAnos = ultimaTemp ? [ultimaTemp] : [];
  state.mensalAnos = state.mensalAnos.filter((a) => disponiveis.includes(a));
  const selecionados = state.mensalAnos.map((ano, i) => {
    const s = temporadas.find((x) => x.ano === ano);
    return { label: s.rotulo, ...SERIES_COLORS[i], values: s.vals };
  });
  const tempFim = temporadas[n - 1];
  const picoFimJ = tempFim ? tempFim.vals.indexOf(Math.max(...tempFim.vals)) : -1;
  const semFiltroLocal = !state.estadoId && !state.municipioId;
  const opcoesMes = meses.map((m, k) => ({ value: String(k + 1), label: m.name }));
  const nivelMensal = state.mensalNivel === "estado" || !state.conn.biomas.length ? "estado" : "bioma";

  return {
    toolbar: [
      {
        type: "select", label: "Temporada do fogo: mês de início", value: String(mi), options: opcoesMes,
        onChange: (v) => { state.mesIni = Number(v); state.mensalAnosTocados = false; loadCurrentTab(); },
      },
      {
        type: "select", label: "Mês de fim", value: String(mf), options: opcoesMes,
        onChange: (v) => { state.mesFim = Number(v); state.mensalAnosTocados = false; loadCurrentTab(); },
      },
      {
        type: "segmented", label: "Paleta dos meses", value: state.paletaMes,
        options: [{ value: "oficial", label: "Legenda oficial" }, { value: "fogo", label: "Tons de fogo" }],
        onChange: (v) => { state.paletaMes = v; loadCurrentTab(); },
      },
      {
        type: "segmented",
        card: 4,
        label: "Territórios",
        value: nivelMensal,
        options: [...(state.conn.biomas.length ? [{ value: "bioma", label: rot("bioma", true) }] : []), { value: "estado", label: rot("estado", true) }],
        onChange: (v) => { state.mensalNivel = v; loadCurrentTab(); },
      },
      {
        type: "chips",
        card: 0,
        label: `Anos em destaque (até ${SERIES_COLORS.length})`,
        options: temporadas.map((s) => {
          const slot = SERIES_COLORS[state.mensalAnos.indexOf(s.ano)];
          return { value: s.ano, label: s.rotulo, color: slot?.color, dash: !!slot?.dash };
        }),
        selected: state.mensalAnos,
        onChange: (ano) => {
          ano = Number(ano);
          state.mensalAnosTocados = true;
          const sel = state.mensalAnos;
          if (sel.includes(ano)) state.mensalAnos = sel.filter((a) => a !== ano);
          else state.mensalAnos = [...sel, ano].slice(-SERIES_COLORS.length); // o mais antigo sai
          loadCurrentTab();
        },
      },
    ],
    kpis: [
      { label: "Mês de pico (média histórica)", value: mesesJ[picoIdx]?.name || "—", note: `${ha(medias[picoIdx])} ± ${ha(dps[picoIdx])} · ${pct(safeDiv(somas[picoIdx], totalPeriodo))} da área do período` },
      { label: "Trimestre de maior área queimada", value: nomeTri, note: `${pct(safeDiv(trimestre?.soma || 0, totalPeriodo))} da área queimada na temporada ${rotuloJanela}` },
      {
        label: `Mês de pico em ${tempFim?.rotulo || fim}`,
        value: mesesJ[picoFimJ]?.name || "—",
        note: picoFimJ >= 0 && picoFimJ !== picoIdx
          ? `Deslocado em relação à média (${mesesJ[picoIdx]?.name})`
          : "Igual ao pico da média histórica",
      },
    ],
    cards: [
      // 1. Média mensal com os anos em destaque (os chips ficam neste card).
      {
        span: 12,
        title: `Média mensal da área queimada ${ini}–${fim} · temporada ${rotuloJanela}`,
        subtitle: n > 1
          ? `Área cinza = média (n = ${n} anos) · faixa clara = ± 1 desvio padrão · linhas = anos em destaque${cruza ? " · temporadas atravessam o ano (ex.: 2020/21)" : ""}`
          : "Escolha um período com mais de um ano para a média e o desvio padrão",
        chart: {
          kind: "band",
          height: 320,
          labels: curtos,
          tooltipTitles: mesesJ.map((m) => m.name),
          mean: { label: `Média ${ini}–${fim}`, color: "var(--text-muted)", values: medias },
          upper: medias.map((m, i) => m + dps[i]),
          lower: medias.map((m, i) => Math.max(0, m - dps[i])),
          years: selecionados,
        },
        legend: [
          { label: `Área queimada média ${ini}–${fim}`, color: "var(--text-muted)" },
          { label: "± 1 desvio padrão", color: "var(--grid)" },
          ...selecionados.map((s) => ({ label: `Área queimada mensal em ${s.label}`, color: s.color, line: true, dash: !!s.dash })),
        ],
      },
      // 2. Distribuição mensal: soma de todos os anos, em barras ou rosca.
      {
        span: 12,
        title: `Distribuição mensal · soma ${ini}–${fim}`,
        subtitle: "Área queimada em cada mês somando todos os anos do período · % do total",
        alterna: "rosca",
        chart: {
          kind: "bar",
          axisTitle: "Área queimada (ha)",
          labels: mesesJ.map((m) => m.name),
          tooltipTitles: mesesJ.map((m, i) => `${m.name} · ${pct(safeDiv(somas[i], totalPeriodo))}`),
          values: somas,
          colors: janela.map((k) => corMes(k)),
          rotulos: "pct",
        },
      },
      // 3. Soma do período e média de cada mês.
      {
        span: 12,
        title: "Soma do período e média por mês",
        subtitle: "Soma no período · % do total · média ± desvio padrão por ano",
        ranking: {
          plain: true,
          total: totalPeriodo,
          rows: mesesJ.map((m, i) => ({
            label: m.name,
            value: somas[i],
            color: corMes(janela[i]),
            detail: `média ${fmtCompact.format(medias[i])} ± ${fmtCompact.format(dps[i])} ha por ano`,
          })),
        },
      },
      // 4. Área queimada por ano e mês.
      {
        span: 12,
        title: `Área queimada por ano e mês · temporada ${rotuloJanela}`,
        subtitle: `Cada barra soma os meses da temporada${cruza ? " (ex.: 2020/21 = meses de 2020 e de 2021)" : ""} · cores: ${state.paletaMes === "fogo" ? "tons de fogo" : "legenda oficial"}`,
        alterna: "linhas",
        chart: {
          kind: "stacked",
          labels: temporadas.map((s) => s.rotulo),
          series: mesesJ.map((m, j) => ({ label: m.name, color: corMes(janela[j]), values: temporadas.map((s) => s.vals[j]) })),
        },
        legend: mesesJ.map((m, j) => ({ label: m.name, color: corMes(janela[j]) })),
      },
      // 5. Proporção mensal por território (biomas ou estados, escolhido no card).
      {
        span: 12,
        title: `Proporção mensal da área queimada por ${rot(nivelMensal).toLowerCase()} · ${ini}–${fim}`,
        subtitle: nivelMensal === "bioma" && !semFiltroLocal
          ? `${rot("bioma", true)} inteiros (este gráfico não aplica o filtro de ${rot("estado").toLowerCase()}/${rot("municipio").toLowerCase()})`
          : `Para cada ${rot(nivelMensal).toLowerCase()}, % da sua área queimada no período que ocorreu em cada mês`,
        lazy: async (progress) => {
          let feitos = 0;
          const lista = nivelMensal === "bioma" ? state.conn.biomas : state.conn.estados;
          const series = await mapLimit(lista, 3, async (b, i) => {
            const r = await getStats({
              subthemeKey: sub("mensal"), legendKey: legenda.key, pixelValue: pixelValues(meses),
              year: range(ini, fim), territoryId: [b.id], spatialMethod: "union",
            });
            progress(`Calculando ${rot(nivelMensal, true).toLowerCase()}… ${++feitos}/${lista.length}`);
            const somaMes = janela.map((k) => sum(byYear(r).map((a) => a.valores.get(meses[k].pixelValue) || 0)));
            const tot = sum(somaMes);
            const cor = nivelMensal === "bioma" ? corBioma(b.name) : CORES_REGIOES[i % CORES_REGIOES.length];
            return { label: b.name, color: cor, values: somaMes.map((v) => safeDiv(v, tot)), tot };
          });
          // Estados: só os 8 com mais área queimada, para o gráfico continuar legível.
          const visiveis = nivelMensal === "bioma" ? series : [...series].sort((a, b) => b.tot - a.tot).slice(0, 8);
          return {
            chart: { kind: "grouped", percent: true, labels: mesesJ.map((m) => m.name.toLowerCase()), series: visiveis },
            legend: visiveis.map((s) => ({ label: s.label, color: s.color })),
            note: nivelMensal === "bioma" ? "" : `Os 8 ${rot("estado", true).toLowerCase()} com mais área queimada no período`,
          };
        },
      },
    ],
  };
}
async function loadFrequencia({ t, ini, fim }) {
  const legenda = await getLegend("frequencia");
  // Brasil: faixas ("1 a 2 anos") com as classes "1 vez", "2 vezes"… dentro; outros países: só as classes.
  const hierarquica = levelItems(legenda, 1).length > 0;
  const classes = hierarquica ? levelItems(legenda, 1) : levelItems(legenda, 0);
  const faixas = hierarquica ? levelItems(legenda, 0) : [];
  const res = await getStats({
    subthemeKey: sub("frequencia"), legendKey: legenda.key, pixelValue: pixelValues(classes),
    year: [ini, fim], ...territoryParams(t),
  });

  const porClasse = totalsByClass(classes, res).map((c) => ({ ...c, short: (c.label.match(/\d+/) || [c.label])[0] }));
  const ultimo = findLastIndex(porClasse, (c) => c.value > 0);
  const visiveis = porClasse.slice(0, ultimo + 1);
  const total = sum(porClasse.map((c) => c.value));
  const umaVez = porClasse[0]?.value || 0;
  const faixasComDados = faixas.filter((f) => porClasse.some((c) => c.item.parent === f.pixelValue));
  const porFaixa = hierarquica
    ? faixasComDados.map((f, i) => ({
      label: frequencyLabel(f.name),
      color: rampColor(FREQ_RAMP, safeDiv(i, faixasComDados.length - 1)),
      value: sum(porClasse.filter((c) => c.item.parent === f.pixelValue).map((c) => c.value)),
    }))
    : porClasse.slice(0, ultimo + 1).map((c, i, arr) => ({
      label: c.label, color: rampColor(FREQ_RAMP, safeDiv(i, arr.length - 1)), value: c.value,
    }));

  // Intervalo dinâmico: 1 a N vezes (N = maior frequência observada) ou classes agrupadas
  // com a última aberta (1 a 15 e 16+; 1 a 9 e 10+). O degradê cobre as barras exibidas.
  const nMax = visiveis.length;
  const corte = state.freqGrupo === "todas" ? Infinity : Number(state.freqGrupo);
  const barras = nMax > corte
    ? [
      ...visiveis.slice(0, corte).map((c) => ({ short: c.short, label: c.label, value: c.value })),
      {
        short: `${corte + 1}+`,
        label: `${corte + 1} a ${nMax} vezes`,
        value: sum(visiveis.slice(corte).map((c) => c.value)),
      },
    ]
    : visiveis.map((c) => ({ short: c.short, label: c.label, value: c.value }));
  barras.forEach((b, i) => { b.color = rampColor(FREQ_RAMP, safeDiv(i, barras.length - 1)); });
  const rotuloGrupo = { todas: `1 a ${nMax || "N"}`, 15: "1 a 15 e 16+", 9: "1 a 9 e 10+" };

  const predominante = maxBy(porClasse, (c) => c.value);
  const emPct = state.freqUnid === "pct";

  return {
    // Controles do card "Área queimada por classe de frequência" (índice 1), ao lado do gráfico.
    toolbar: [
      {
        type: "segmented",
        card: 1,
        label: "Classes de frequência",
        value: state.freqGrupo,
        options: ["todas", "15", "9"].map((v) => ({ value: v, label: rotuloGrupo[v] })),
        onChange: (v) => { state.freqGrupo = v; loadCurrentTab(); },
      },
      {
        type: "segmented",
        card: 1,
        label: "Mostrar",
        value: emPct ? "pct" : "ha",
        options: [{ value: "ha", label: "Área (ha)" }, { value: "pct", label: "% da área queimada" }],
        onChange: (v) => { state.freqUnid = v; loadCurrentTab(); },
      },
    ],
    kpis: [
      {
        label: "Frequência predominante",
        value: predominante?.value ? predominante.label : "—",
        note: predominante?.value ? `${pct(safeDiv(predominante.value, total))} da área queimada · ${ha(predominante.value)}` : "",
      },
      { label: "Queimou 2 vezes ou mais", value: ha(total - umaVez), note: `${pct(safeDiv(total - umaVez, total))} da área queimada` },
    ],
    cards: [
      {
        span: 12,
        title: "Faixas de frequência",
        subtitle: "Área e participação na área queimada do período",
        alterna: "rosca",
        chart: {
          kind: "hbar",
          labels: porFaixa.map((f) => `${f.label} · ${pct(safeDiv(f.value, total))}`),
          tooltipTitles: porFaixa.map((f) => f.label),
          values: porFaixa.map((f) => f.value),
          colors: porFaixa.map((f) => f.color),
        },
      },
      {
        span: 12,
        title: "Área queimada por classe de frequência",
        subtitle: `${ini}–${fim} · classes ${rotuloGrupo[state.freqGrupo]}`,
        alterna: "rosca",
        chart: {
          kind: "bar",
          percent: emPct,
          axisTitle: emPct ? "Participação na área queimada (%)" : "Área queimada (ha)",
          categoryTitle: "Número de vezes que queimou",
          labels: barras.map((c) => c.short),
          tooltipTitles: barras.map((c) => `${c.label} · ${ha(c.value)} · ${pct(safeDiv(c.value, total))}`),
          values: barras.map((c) => (emPct ? safeDiv(c.value, total) : c.value)),
          colors: barras.map((c) => c.color),
          rotulos: emPct ? "pct" : "",
          height: 340,
        },
        rosca: [{ slices: barras.filter((b) => b.value > 0).map((b) => ({ label: b.label, value: b.value, color: b.color })) }],
        legend: [
          { label: "Menos frequente", color: FREQ_RAMP[0] },
          { label: "Mais frequente", color: FREQ_RAMP[FREQ_RAMP.length - 1] },
          { label: "Degradê de frequência (amarelo → púrpura)", note: true },
        ],
      },
    ],
  };
}
async function loadCicatriz({ t, ini, fim }) {
  const legenda = await getLegend("cicatriz");
  const subclasses = levelItems(legenda, 1);
  const grupos = levelItems(legenda, 0);
  const res = await getStats({
    subthemeKey: sub("cicatriz"), legendKey: legenda.key, pixelValue: pixelValues(subclasses),
    year: range(ini, fim), ...territoryParams(t),
  });

  const anos = byYear(res);
  const porSubclasse = totalsByClass(subclasses, res);
  const total = sum(porSubclasse.map((c) => c.value));
  const somaGrupo = (g, valores) => sum(subclasses.filter((s) => s.parent === g.pixelValue).map((s) => valores.get(s.pixelValue) || 0));
  // As 5 classes agregadas da legenda oficial são as faixas do RAF 2025:
  // < 250 ha · 250 a 500 ha · 500 a 10 mil ha · 10 mil a 100 mil ha · > 100 mil ha.
  const faixas = grupos.map((g) => ({ ...g, label: rafScarLabel(g.name) }));
  const porFaixa = faixas.map((g) => ({ label: g.label, color: g.color, value: sum(anos.map((a) => somaGrupo(g, a.valores))) }));
  const predominante = maxBy(porFaixa, (g) => g.value);
  const grandes = faixas.slice(-2); // ≥ 10 mil ha
  // Faixa(s) da "proporção por ano": as duas maiores (padrão) ou uma faixa escolhida.
  const escolha = state.cicatrizClasse === "grandes" ? grandes : [faixas[Number(state.cicatrizClasse)]].filter(Boolean);
  const alvo = escolha.length ? escolha : grandes;
  const rotuloAlvo = state.cicatrizClasse === "grandes" ? `${grandes[0]?.label.replace(/ a .*/, "")} ha ou mais` : alvo.map((g) => g.label).join(" e ");
  const somaGrandes = (valores) => sum(alvo.map((g) => somaGrupo(g, valores)));
  const anosComFogo = anos.filter((a) => a.area > 0);

  return {
    kpis: [
      { label: "Faixa de tamanho predominante", value: predominante?.label || "—", note: predominante ? `${pct(safeDiv(predominante.value, total))} da área` : "" },
      {
        label: `Em cicatrizes de ${grandes[0]?.label.replace(/ a .*/, "") || "10 mil"} ha ou mais`,
        value: pct(safeDiv(sum(porFaixa.slice(-2).map((g) => g.value)), total)),
        note: grandes.map((g) => g.label).join(" · "),
      },
    ],
    cards: [
      {
        span: 12,
        title: "Área queimada por ano e tamanho de cicatriz",
        subtitle: "Faixas de tamanho do RAF 2025 · cores oficiais",
        alterna: "linhas",
        chart: {
          kind: "stacked",
          labels: anos.map((a) => String(a.ano)),
          series: faixas.map((g) => ({ label: g.label, color: g.color, values: anos.map((a) => somaGrupo(g, a.valores)) })),
        },
        legend: faixas.map((g) => ({ label: g.label, color: g.color })),
      },
      {
        span: 6,
        title: "Distribuição por faixa de tamanho",
        subtitle: `${ini}–${fim} · participação na área queimada`,
        alterna: "rosca",
        chart: {
          kind: "hbar",
          labels: porFaixa.map((f) => `${f.label} · ${pct(safeDiv(f.value, total))}`),
          tooltipTitles: porFaixa.map((f) => f.label),
          values: porFaixa.map((f) => f.value),
          colors: porFaixa.map((f) => f.color),
        },
      },
      {
        span: 6,
        title: `Proporção por ano em cicatrizes de ${rotuloAlvo}`,
        subtitle: `Área em cicatrizes de ${rotuloAlvo} ÷ área queimada no ano`,
        alterna: "linhas",
        chart: {
          kind: "line",
          percent: true,
          axisTitle: "Proporção da área queimada no ano (%)",
          labels: anosComFogo.map((a) => String(a.ano)),
          values: anosComFogo.map((a) => safeDiv(somaGrandes(a.valores), a.area)),
          color: alvo[alvo.length - 1]?.color,
        },
      },
    ],
    toolbar: [{
      type: "select",
      card: 2,
      label: "Faixa de tamanho",
      value: state.cicatrizClasse,
      options: [{ value: "grandes", label: `${grandes[0]?.label.replace(/ a .*/, "")} ha ou mais` }, ...faixas.map((g, i) => ({ value: String(i), label: g.label }))],
      onChange: (v) => { state.cicatrizClasse = v; loadCurrentTab(); },
    }],
  };
}

// Classes de cobertura: naturais primeiro, depois antrópicas, depois o resto (cada grupo por área),
// para a barra "por classe" ficar alinhada à barra natural x antrópico logo acima.
// O teste antrópico vem antes porque "Área não vegetada" também contém "vegeta".
function grupoCobertura(nome) {
  if (/antr[oó]p|agropec|pastagem|agricult|lavoura|soja|cana|caf[eé]|citrus|dend|algod|arroz|silvicult|mosaico|urbaniz|minera|aquicult|infraestrut|n[aã]o vegetad/i.test(nome)) return 1;
  if (/natural|floresta|vegeta|herb[aá]c|arbust|sav[aâ]n|campo|campestre|mangue|restinga|alagad|[aá]gua|rio|lago|praia|duna|afloramento|apicum|rochos|hipersal/i.test(nome)) return 0;
  return 2;
}

const ordenarPorGrupo = (classes) =>
  [...classes].sort((a, b) => grupoCobertura(a.label) - grupoCobertura(b.label) || b.value - a.value);

// "entre 10.000 e 100.000 ha" -> "10 mil a 100 mil ha"; "menor que 250 ha" -> "Menor que 250 ha".
function rafScarLabel(name) {
  const mil = (s) => s.replace(/(\d{1,3})\.000\b/g, "$1 mil");
  const entre = name.match(/entre\s+([\d.]+)\s+e\s+([\d.]+)/i);
  if (entre) return `${mil(entre[1])} a ${mil(entre[2])} ha`;
  const texto = mil(name.trim());
  return texto.charAt(0).toUpperCase() + texto.slice(1);
}

// Retorno do último fogo: nas áreas queimadas no ano final que já tinham queimado antes,
// quantos anos se passaram desde o fogo anterior.
async function loadRetornoUltimo({ t, fim }) {
  const legenda = await getLegend("retorno");
  const faixas = levelItems(legenda, 0);
  const res = await getStats({
    subthemeKey: sub("retorno"), legendKey: legenda.key, pixelValue: pixelValues(faixas), year: fim, ...territoryParams(t),
  });
  const dados = totalsByClass(faixas, res);
  const total = sum(dados.map((d) => d.value));
  const curto = dados[0];
  const predominante = maxBy(dados, (d) => d.value);
  return {
    kpis: [
      {
        label: `Retorno em ${curto ? curto.label : "—"} (${fim})`,
        value: pct(safeDiv(curto?.value || 0, total)),
        note: `${ha(curto?.value || 0)} de ${ha(total)} queimados com fogo anterior`,
      },
      { label: "Classe predominante", value: predominante?.value ? predominante.label : "—", note: predominante?.value ? `${pct(safeDiv(predominante.value, total))} · ${ha(predominante.value)}` : "" },
      { label: `Área queimada em ${fim} com fogo anterior`, value: ha(total), note: "Base das proporções desta aba" },
    ],
    cards: [
      {
        span: 7,
        title: `Tempo desde o fogo anterior · ${fim}`,
        subtitle: "Áreas queimadas no ano final que já tinham queimado antes",
        alterna: "rosca",
        chart: {
          kind: "bar",
          axisTitle: "Área queimada (ha)",
          labels: dados.map((d) => d.label),
          values: dados.map((d) => d.value),
          colors: dados.map((d) => d.color),
        },
      },
      {
        span: 5,
        title: "Distribuição por tempo de retorno",
        subtitle: `${fim} · % da área queimada com fogo anterior`,
        ranking: { rows: dados.filter((d) => d.value > 0), total, plain: true },
      },
    ],
  };
}

// Intervalo médio de retorno: média de anos entre queimadas sucessivas no período.
async function loadMediaRetorno({ t, ini, fim }) {
  const legenda = await getLegend("mediaRetorno");
  const faixas = levelItems(legenda, 0);
  const res = await getStats({
    subthemeKey: sub("mediaRetorno"), legendKey: legenda.key, pixelValue: pixelValues(faixas), year: [ini, fim], ...territoryParams(t),
  });
  const dados = totalsByClass(faixas, res);
  const total = sum(dados.map((d) => d.value));
  const predominante = maxBy(dados, (d) => d.value);
  return {
    kpis: [
      {
        label: `Intervalo médio predominante (${ini}–${fim})`,
        value: predominante?.value ? predominante.label : "—",
        note: predominante?.value ? `${pct(safeDiv(predominante.value, total))} da área com recorrência · ${ha(predominante.value)}` : "",
      },
      { label: "Área com recorrência no período", value: ha(total), note: "Áreas que queimaram mais de uma vez" },
    ],
    cards: [
      {
        span: 7,
        title: `Intervalo médio de retorno · ${ini}–${fim}`,
        subtitle: "Média de anos entre queimadas sucessivas",
        alterna: "rosca",
        chart: {
          kind: "bar",
          axisTitle: "Área queimada (ha)",
          labels: dados.map((d) => d.label),
          values: dados.map((d) => d.value),
          colors: dados.map((d) => d.color),
        },
      },
      {
        span: 5,
        title: "Distribuição por intervalo médio",
        subtitle: `${ini}–${fim} · % da área com recorrência`,
        ranking: { rows: dados.filter((d) => d.value > 0), total, plain: true },
      },
    ],
  };
}
async function loadUltimoFogo({ t, ini, fim }) {
  const legenda = await getLegend("ultimoFogo");
  // Folhas da legenda = um item por ano ("2025", "2024"…); no staging há grupos de décadas acima delas.
  const folhas = legenda.items
    .filter((i) => !i.hasChildren)
    .map((i) => ({ ...i, ano: Number((i.name.match(/\d{4}/) || [])[0]) }))
    .filter((i) => i.ano >= ini && i.ano <= fim)
    .sort((a, b) => a.ano - b.ano);
  const res = await getStats({
    subthemeKey: sub("ultimoFogo"), legendKey: legenda.key, pixelValue: pixelValues(folhas),
    year: [ini, fim], ...territoryParams(t),
  });

  // Degradê pelo tempo decorrido até `fim`: quente = recente, verde/azul = há mais tempo.
  const maxDecorrido = Math.max(1, fim - ini);
  const corDecorrido = (anos) => rampColor(DECORRIDO_RAMP, anos / maxDecorrido);
  const dados = totalsByClass(folhas, res).map((d) => ({
    ...d, ano: d.item.ano, decorrido: fim - d.item.ano, color: corDecorrido(fim - d.item.ano),
  }));
  const total = sum(dados.map((d) => d.value));
  const recentes = sum(dados.filter((d) => d.decorrido <= 4).map((d) => d.value));
  // Ano mediano (ponderado pela área) do último fogo.
  let acumulado = 0;
  const mediano = dados.find((d) => (acumulado += d.value) >= total / 2);

  const faixasDef = [[0, 1, "0 a 1 ano"], [2, 5, "2 a 5 anos"], [6, 10, "6 a 10 anos"], [11, 20, "11 a 20 anos"], [21, Infinity, "mais de 20 anos"]];
  const faixas = faixasDef
    .map(([a, b, label]) => {
      const membros = dados.filter((d) => d.decorrido >= a && d.decorrido <= b);
      // Cor do meio da faixa no mesmo degradê do gráfico por ano.
      const meio = (Math.min(b, maxDecorrido) + a) / 2;
      return membros.length ? { label, value: sum(membros.map((m) => m.value)), color: corDecorrido(meio) } : null;
    })
    .filter(Boolean);

  return {
    kpis: [
      { label: `Área queimada pela última vez nos últimos 5 anos (${Math.max(ini, fim - 4)}–${fim})`, value: pct(safeDiv(recentes, total)), note: ha(recentes) },
      { label: "Ano mediano do último fogo", value: mediano ? String(mediano.ano) : "—", note: "Metade da área queimou pela última vez até este ano" },
    ],
    cards: [
      {
        span: 12,
        title: "Área por ano do último fogo",
        subtitle: `Período ${ini}–${fim} · tempo decorrido até ${fim} = ${fim} − ano`,
        alterna: "linhas",
        chart: {
          kind: "bar",
          axisTitle: "Área (ha)",
          labels: dados.map((d) => String(d.ano)),
          tooltipTitles: dados.map((d) => `${d.ano} · ${d.decorrido} ano(s) antes de ${fim}`),
          values: dados.map((d) => d.value),
          colors: dados.map((d) => d.color),
        },
        legend: [
          { label: `Há mais tempo (${ini})`, color: DECORRIDO_RAMP[DECORRIDO_RAMP.length - 1] },
          { label: `Recente (${fim})`, color: DECORRIDO_RAMP[0] },
          { label: "Cor = tempo decorrido desde o último fogo", note: true },
        ],
      },
      {
        span: 6,
        title: "Tempo decorrido desde o último fogo",
        subtitle: `Em relação a ${fim}`,
        alterna: "rosca",
        chart: {
          kind: "hbar",
          labels: faixas.map((f) => `${f.label} · ${pct(safeDiv(f.value, total))}`),
          tooltipTitles: faixas.map((f) => f.label),
          values: faixas.map((f) => f.value),
          colors: faixas.map((f) => f.color),
        },
      },
      {
        span: 6,
        title: "Anos de último fogo com mais área",
        subtitle: "Top 5",
        ranking: {
          rows: [...dados].sort((a, b) => b.value - a.value).slice(0, 5)
            .map((d) => ({ label: `${d.ano} (${d.decorrido} ano(s) atrás)`, value: d.value, color: d.color })),
          total,
        },
      },
    ],
  };
}

async function loadSeveridade({ t, ini, fim }) {
  const legenda = await getLegend("severidade");
  const classes = levelItems(legenda, 0);
  const res = await getStats({
    subthemeKey: sub("severidade"), legendKey: legenda.key, pixelValue: pixelValues(classes),
    year: range(ini, fim), ...territoryParams(t),
  });

  const anos = byYear(res);
  const comDados = anos.filter((a) => a.area > 0);
  const porClasse = totalsByClass(classes, res);
  const total = sum(porClasse.map((c) => c.value));
  // Classes da legenda em ordem crescente de severidade: "Alta" ou mais = 3 últimas.
  const altas = classes.slice(-3);
  const somaAltas = (valores) => sum(altas.map((c) => valores.get(c.pixelValue) || 0));
  const predominante = maxBy(porClasse, (c) => c.value);
  const vermelho = altas[altas.length - 1]?.color;

  return {
    kpis: [
      {
        label: `Severidade ${altas[0]?.name || "alta"} ou maior`,
        value: pct(safeDiv(sum(comDados.map((a) => somaAltas(a.valores))), total)),
        note: altas.map((c) => c.name).join(" · "),
      },
      { label: "Classe predominante", value: predominante?.label || "—", note: predominante ? `${pct(safeDiv(predominante.value, total))} · ${ha(predominante.value)} em ${ini}–${fim}` : "" },
    ],
    cards: [
      {
        span: 12,
        title: `Severidade total entre ${ini} e ${fim}`,
        subtitle: "Participação de cada classe na área afetada do período",
        propbars: { rows: [{ segments: porClasse }] },
        alterna: "rosca",
        rosca: [{ slices: porClasse.filter((c) => c.value > 0), centerLabel: "afetados" }],
      },
      {
        span: 12,
        title: "Área afetada por ano e classe de severidade",
        alterna: "linhas",
        subtitle: comDados.length < anos.length
          ? `${anos.length - comDados.length} ano(s) do período sem dado de severidade na API`
          : `${ini}–${fim} · cores oficiais da legenda de severidade`,
        chart: {
          kind: "stacked",
          labels: anos.map((a) => String(a.ano)),
          series: classes.map((c) => ({ label: c.name, color: c.color, values: anos.map((a) => a.valores.get(c.pixelValue) || 0) })),
        },
        legend: classes.map((c) => ({ label: c.name, color: c.color })),
      },
      {
        span: 6,
        title: "Distribuição no período",
        subtitle: `${ini}–${fim} · participação de cada classe na área afetada`,
        alterna: "rosca",
        rosca: [{ slices: porClasse.filter((c) => c.value > 0), centerLabel: "afetados" }],
        chart: {
          kind: "hbar",
          percent: true,
          categoryTitle: "Severidade",
          axisTitle: "Participação na área afetada",
          labels: porClasse.map((c) => c.label),
          tooltipTitles: porClasse.map((c) => `${c.label} · ${ha(c.value)}`),
          values: porClasse.map((c) => safeDiv(c.value, total)),
          colors: porClasse.map((c) => c.color),
        },
      },
      {
        span: 6,
        title: `Proporção com severidade ${altas[0]?.name || "alta"} ou maior`,
        subtitle: "Por ano, sobre a área afetada no ano",
        alterna: "linhas",
        chart: {
          kind: "line",
          percent: true,
          axisTitle: "Proporção da área afetada no ano (%)",
          labels: comDados.map((a) => String(a.ano)),
          values: comDados.map((a) => safeDiv(somaAltas(a.valores), a.area)),
          color: vermelho,
        },
      },
    ],
  };
}

// Municípios no recorte atual: filtro por estado (UF no nome) e/ou bioma (parte do município
// dentro do bioma, ver getRecorteBioma).
async function municipiosNoEscopo(progress) {
  const c = state.conn;
  const estado = c.estados.find((e) => e.id === state.estadoId);
  const bioma = c.biomas.find((b) => b.id === state.biomaId);
  let recorte = null;
  if (bioma) {
    progress?.("Recortando os municípios pelo bioma…");
    recorte = await getRecorteBioma(bioma.id, (n, tot) => progress?.(`Lendo geometria dos municípios… ${n}/${tot} tiles`));
  }
  const doEstado = estado ? new Set((await getMunicipiosDoEstado(estado)).map((m) => m.id)) : null;
  const filtro = (l) => (!recorte || recorte.has(l.id)) && (!doEstado || doEstado.has(l.id));
  const partes = [estado && `de ${estado.name}`, bioma && `em ${bioma.name}`].filter(Boolean);
  return {
    filtro,
    recorte,
    biomaId: bioma?.id,
    biomaNome: bioma?.name,
    rotulo: partes.length ? `${rot("municipio", true)} ${partes.join(" ")}` : `${rot("municipio", true)} ${PAIS.de}`,
  };
}

// Nota sobre o recorte por bioma para rankings e mapas de municípios.
function avisoRecorte(escopo, r) {
  if (!escopo.recorte) return "";
  let txt = `Só a parte de cada município dentro do ${escopo.biomaNome}. ${r.divisa} município(s) na divisa usam a área exata da interseção calculada pela API`;
  const prontos = r.divisa - r.pendentes - r.falhas;
  if (r.pendentes) txt += `: ${prontos} prontos, ${r.pendentes} em cálculo no servidor (ficam de fora até terminar; a página se atualiza sozinha)`;
  if (r.falhas) txt += `; ${r.falhas} não puderam ser calculados agora`;
  return `${txt}.`;
}

async function loadTerritorios({ ini, fim }) {
  const c = state.conn;
  const total = await getLegend("anualTotal");
  const vermelho = levelItems(total, 0)[0].color;
  const estado = c.estados.find((e) => e.id === state.estadoId);
  const bioma = c.biomas.find((b) => b.id === state.biomaId);
  const catEstado = c.catEstado;
  const catBioma = c.catBioma;

  const progresso = (rotulo) => (n, tot) => setStatus(`${rotulo}: ${n}/${tot} anos`, true);
  const [estados, biomas] = await Promise.all([
    rankingPeriodo(catEstado, ini, fim, progresso(`Ranking de ${rot("estado", true).toLowerCase()}`)),
    catBioma ? rankingPeriodo(catBioma, ini, fim, progresso(`Ranking de ${rot("bioma", true).toLowerCase()}`)) : { anos: range(ini, fim), linhas: [] },
  ]);
  // Total do país: soma dos biomas (cobrem o país) ou, sem biomas, dos estados.
  const totalBrasil = sum((biomas.linhas.length ? biomas : estados).linhas.map((l) => l.value));
  const top3 = estados.linhas.slice(0, 3);
  const avisoBioma = bioma ? ` · ${rot("estado", true).toLowerCase()} inteiros (a API de ranking não recorta por ${rot("bioma").toLowerCase()})` : "";
  const ultimo = estados.anos.length - 1;
  const totalFimBrasil = sum((biomas.linhas.length ? biomas : estados).linhas.map((l) => l.porAno[ultimo] || 0));

  // Variação do ano final em relação à média dos anos anteriores do período (RAF 3.10.2).
  const variacao = estados.anos.length > 1
    ? estados.linhas
      .filter((l) => safeDiv(l.porAno[ultimo], totalFimBrasil) >= 0.01)
      .map((l) => {
        const media = mean(l.porAno.slice(0, ultimo));
        return { ...l, media, delta: safeDiv(l.porAno[ultimo], media) - 1 };
      })
      .sort((a, b) => b.delta - a.delta)
    : [];

  // Municípios: ranking do período e do ano final, no recorte de estado/bioma.
  const municipios = (async () => {
    const escopo = await municipiosNoEscopo((msg) => setStatus(msg, true));
    const mun = await rankingPeriodo(c.catMunicipio, ini, fim, progresso(`Ranking de ${rot("municipio", true).toLowerCase()}`));
    setStatus("");
    const r = aplicarRecorte(mun.linhas.filter(escopo.filtro), escopo, ini, fim);
    return { escopo, linhas: r.linhas, ultimo: mun.anos.length - 1, aviso: avisoRecorte(escopo, r) };
  })();
  municipios.catch(() => {});

  return {
    kpis: [
      {
        label: `${rot("estado")} com mais área queimada`,
        value: estados.linhas[0]?.name || "—",
        note: estados.linhas[0] ? `${ha(estados.linhas[0].value)} · ${pct(safeDiv(estados.linhas[0].value, totalBrasil))} ${PAIS.de}` : "",
      },
      {
        label: `Concentração nos 3 maiores (${rot("estado", true).toLowerCase()})`,
        value: pct(safeDiv(sum(top3.map((l) => l.value)), totalBrasil)),
        note: top3.map((l) => l.name).join(", "),
      },
      {
        label: `${rot("bioma")} com mais área queimada`,
        value: biomas.linhas[0]?.name || "—",
        note: biomas.linhas[0] ? `${pct(safeDiv(biomas.linhas[0].value, totalBrasil))} ${PAIS.de}` : "",
      },
    ],
    cards: [
      {
        span: 6,
        title: `${rot("estado", true)} que mais queimaram`,
        subtitle: `Soma das áreas anuais ${ini}–${fim} · top 10${avisoBioma}`,
        chart: hbarRanking(estados.linhas.slice(0, 10), vermelho, totalBrasil, state.estadoId),
      },
      {
        span: 6,
        title: `${rot("municipio", true)} que mais queimaram no período`,
        subtitle: `Soma das áreas anuais ${ini}–${fim} · top 10`,
        lazy: async (progress) => {
          progress(`Baixando ranking de ${rot("municipio", true).toLowerCase()}…`);
          const m = await municipios;
          const base = sum(m.linhas.map((l) => l.value));
          return {
            chart: hbarRanking(m.linhas.slice(0, 10), vermelho, base, state.municipioId),
            note: `${m.escopo.rotulo} · % sobre o total desses ${rot("municipio", true).toLowerCase()}. ${m.aviso}`,
          };
        },
      },
      {
        span: 6,
        title: `Variação em ${fim} em relação à média ${ini}–${fim - 1}`,
        subtitle: `${rot("estado", true)} com ao menos 1% da área queimada ${PAIS.de} no ano · vermelho = acima da média, azul = abaixo`,
        chart: variacao.length
          ? {
            kind: "hbar",
            percent: true,
            axisTitle: "Variação em relação à média (%)",
            labels: variacao.map((l) => l.name),
            tooltipTitles: variacao.map((l) => `${l.name}: ${ha(l.porAno[ultimo])} em ${fim} · média ${ha(l.media)}`),
            values: variacao.map((l) => l.delta),
            colors: variacao.map((l) => (l.delta >= 0 ? "#e34948" : "#2a78d6")),
          }
          : { kind: "hbar", labels: [], values: [] },
      },
      {
        span: 6,
        title: `${rot("municipio", true)} que mais queimaram em ${fim}`,
        subtitle: `Área queimada no ano final do período · top 10`,
        lazy: async (progress) => {
          progress(`Baixando ranking de ${rot("municipio", true).toLowerCase()}…`);
          const m = await municipios;
          const doAno = [...m.linhas].map((l) => ({ ...l, value: l.porAno[m.ultimo] || 0 }))
            .filter((l) => l.value > 0).sort((a, b) => b.value - a.value);
          return {
            chart: hbarRanking(doAno.slice(0, 10), vermelho, sum(doAno.map((l) => l.value)), state.municipioId),
            note: `${m.escopo.rotulo}. ${m.aviso}`,
          };
        },
      },
      {
        span: 12,
        title: `Evolução anual: os 3 maiores em área queimada (${rot("estado", true).toLowerCase()})`,
        subtitle: `${ini}–${fim} · cores categóricas (não há cor oficial por ${rot("estado").toLowerCase()})`,
        chart: {
          kind: "multiline",
          labels: estados.anos.map(String),
          series: top3.map((l, i) => ({ label: l.name, color: TERRITORY_COLORS[i], values: l.porAno })),
        },
        legend: top3.map((l, i) => ({ label: l.name, color: TERRITORY_COLORS[i], line: true })),
      },
      {
        span: 8,
        title: `Participação na área queimada por ${rot("bioma").toLowerCase()}, ano a ano`,
        subtitle: `${ini}–${fim} · cada ano soma 100% da área queimada ${PAIS.de}`,
        alterna: "linhas",
        chart: {
          kind: "stacked",
          percent: true,
          axisTitle: "Participação na área queimada (%)",
          labels: biomas.anos.map(String),
          series: biomas.linhas.map((l) => ({
            label: l.name,
            color: corBioma(l.name),
            values: l.porAno.map((v, i) => safeDiv(v, sum(biomas.linhas.map((b) => b.porAno[i] || 0)))),
          })),
        },
        legend: biomas.linhas.map((l) => ({ label: l.name, color: corBioma(l.name) })),
      },
      {
        span: 4,
        title: `${rot("bioma", true)}: área queimada no período`,
        subtitle: `Soma das áreas anuais e participação ${PAIS.em}`,
        ranking: {
          rows: biomas.linhas.map((l) => ({ label: l.name, value: l.value, color: corBioma(l.name) })),
          total: totalBrasil,
        },
      },
    ],
  };
}

function hbarRanking(linhas, cor, base, destaqueId) {
  return {
    kind: "hbar",
    labels: linhas.map((l) => `${l.name} · ${pct(safeDiv(l.value, base))}`),
    tooltipTitles: linhas.map((l) => l.name),
    values: linhas.map((l) => l.value),
    colors: linhas.map((l) => (destaqueId && l.id !== destaqueId ? withAlpha(cor, 0.45) : cor)),
  };
}

// Descobre o id de categoria de um território (os ids diferem entre prd e stg).
async function categoriaDe(territoryId) {
  const c = state.conn;
  return c.biomas.some((b) => b.id === territoryId) ? c.catBioma : c.catEstado;
}

// Categorias fundiárias disponíveis na API (a malha fundiária do RAF, com imóveis
// rurais privados, não está na API: aparece só como referência estática abaixo).
const CATEGORIAS_FUNDIARIAS = [
  { re: /^TERRAS_INDIGENAS/, label: "Terras Indígenas" },
  { re: /^UNIDADES_DE_CONSERVACAO/, label: "Unidades de Conservação" },
  { re: /^ASSENTAMENTOS/, label: "Assentamentos" },
  { re: /^QUILOMBOS/, label: "Territórios Quilombolas" },
  { re: /^FLORESTAS_PUBLICAS_NAO_DESTINADAS/, label: "Florestas Públicas Não Destinadas" },
];

async function loadFundiario({ t, ini, fim }) {
  const c = state.conn;
  const total = await getLegend("anualTotal");
  const vermelho = levelItems(total, 0)[0].color;
  const cats = (await cached(`${envId()}:cats`, () => fetchTerritoryCategories(c.env))).categories || [];
  const categorias = CATEGORIAS_FUNDIARIAS
    .map((def) => ({ ...def, cat: cats.find((x) => def.re.test(x.key)) }))
    .filter((d) => d.cat);
  const brasilInteiro = t.ids.length === 1 && t.ids[0] === c.brasil.id;
  const anos = range(ini, fim);
  const base = { subthemeKey: sub("anual"), legendKey: total.key, pixelValue: pixelValues(levelItems(total, 0)), year: anos };

  // Uma consulta por categoria (todas as áreas da categoria; recortada pelo filtro se houver).
  // O 1º cálculo de cada categoria pode levar vários minutos no servidor; depois fica em cache.
  const dados = (async () => {
    const totalRecorte = byYear(await getStats({ ...base, ...territoryParams(t) }));
    let prontas = 0;
    const resultados = await Promise.allSettled(categorias.map(async (d) => {
      const params = { ...base, territoryCategoryId: d.cat.id };
      if (!brasilInteiro) Object.assign(params, { territoryId: t.ids, spatialMethod: "intersection" });
      const res = await getStats(params);
      setStatus(`Categorias fundiárias: ${++prontas}/${categorias.length} prontas`, true);
      const porAno = byYear(res);
      return { ...d, porAno, value: sum(porAno.map((a) => a.area)) };
    }));
    setStatus("");
    return {
      totalRecorte: sum(totalRecorte.map((a) => a.area)),
      ok: resultados.filter((r) => r.status === "fulfilled").map((r) => r.value),
      falhas: resultados.map((r, i) => (r.status === "rejected" ? `${categorias[i].label}: ${r.reason.message}` : null)).filter(Boolean),
    };
  })();
  dados.catch(() => {}); // os cards tratam o erro

  const aviso = "Soma das áreas anuais (recorrências contam em cada ano); categorias podem se sobrepor (ex.: APAs contêm imóveis privados).";
  const notaFalhas = (d) => (d.falhas.length ? ` · Ainda em cálculo/indisponível: ${d.falhas.map((f) => f.split(":")[0]).join(", ")}` : "");

  return {
    kpis: [],
    cards: [
      {
        span: 12,
        title: `Fogo por categoria fundiária · ${ini}–${fim}`,
        subtitle: `${brasilInteiro ? "Brasil" : `Recorte: ${t.nome}`} · ${aviso}`,
        lazy: async (progress) => {
          progress("Calculando… o 1º cálculo de cada categoria pode levar alguns minutos no servidor");
          const d = await dados;
          return {
            topcards: [...d.ok].sort((a, b) => b.value - a.value).map((r) => ({
              title: r.label,
              value: ha(r.value),
              note: `${pct(safeDiv(r.value, d.totalRecorte))} da área queimada do recorte`,
            })),
          };
        },
      },
      {
        span: 6,
        title: "Ranking das categorias",
        subtitle: "Área queimada no período e participação no total do recorte",
        lazy: async (progress) => {
          progress("Aguardando as categorias…");
          const d = await dados;
          const linhas = [...d.ok].sort((a, b) => b.value - a.value);
          return {
            chart: {
              kind: "hbar",
              labels: linhas.map((l) => `${l.label} · ${pct(safeDiv(l.value, d.totalRecorte))}`),
              tooltipTitles: linhas.map((l) => l.label),
              values: linhas.map((l) => l.value),
              color: vermelho,
            },
            note: notaFalhas(d),
          };
        },
      },
      {
        span: 6,
        title: "Evolução anual nas principais categorias",
        subtitle: "Terras Indígenas, Unidades de Conservação e Assentamentos · cores categóricas",
        lazy: async (progress) => {
          progress("Aguardando as categorias…");
          const d = await dados;
          const principais = ["Terras Indígenas", "Unidades de Conservação", "Assentamentos"]
            .map((n) => d.ok.find((r) => r.label === n)).filter(Boolean);
          return {
            chart: {
              kind: "multiline",
              labels: anos.map(String),
              series: principais.map((r, i) => ({ label: r.label, color: TERRITORY_COLORS[i], values: r.porAno.map((a) => a.area) })),
            },
            legend: principais.map((r, i) => ({ label: r.label, color: TERRITORY_COLORS[i], line: true })),
          };
        },
      },
    ],
  };
}

function mapTypeControl() {
  return {
    type: "segmented",
    label: "Tipo de mapa",
    value: state.mapa.tipo,
    options: [{ value: "bruto", label: "Dados brutos" }, { value: "ranking", label: "Ranking por território" }],
    onChange: (v) => { state.mapa.tipo = v; loadCurrentTab(); },
  };
}

async function loadMapa(ctx) {
  return state.mapa.tipo === "ranking" ? loadMapaRanking(ctx) : loadMapaBruto(ctx);
}

async function loadMapaBruto({ t, ini, fim }) {
  // Subprodutos que o país tem, separados em anuais (um ano) e por período.
  const camadas = MAP_LAYERS.filter((c) => state.conn.subthemes[c.subtheme]);
  const camada = camadas.find((c) => c.id === state.mapa.camada) || camadas[0];
  const legenda = await getLegend(camada.legend);
  const m = state.mapa;
  const anoUnico = clamp(m.ano ?? fim, ini, fim); // ano mostrado nos subprodutos anuais
  const anos = camada.periodo ? [ini, fim] : anoUnico;
  // Cortina (só subprodutos anuais): o mesmo subproduto em dois anos escolhidos.
  const cortina = m.cortina === "on" && !camada.periodo;
  const anoA = clamp(m.anoA ?? Math.max(ini, fim - 10), ini, fim);
  const anoB = clamp(m.anoB ?? fim, ini, fim);
  const camada2 = !cortina && m.camada2 ? camadas.find((c) => c.id === m.camada2 && c.id !== camada.id) : null;
  const legenda2 = camada2 ? await getLegend(camada2.legend) : null;
  const pedirTiles = (cam, lg, ano) => getMapUrl({
    subthemeKey: sub(cam.subtheme), legendKey: lg.key, pixelValue: pixelValues(lg.items), year: ano, ...territoryParams(t),
  });

  const [tileUrl, tileUrl2, ativo, contornos] = await Promise.all([
    pedirTiles(camada, legenda, cortina ? anoB : anos),
    cortina ? pedirTiles(camada, legenda, anoA)
      : camada2 ? pedirTiles(camada2, legenda2, camada2.periodo ? [ini, fim] : anoUnico) : null,
    getGeometry(t.ids, t.metodo).catch(() => null),
    referenceOutlines(m.visualizar),
  ]);

  const itensLegenda = (cam, lg) => levelItems(lg, 0).map((i) => ({
    label: cam.id === "frequencia" ? frequencyLabel(i.name) : i.name,
    color: i.color,
  }));
  const nomeSub = (c) => c.label.replace(/\s*\((ano fim|período)\)/, "");
  const legendItems = [
    ...(camada2 ? [{ label: `Camada de baixo: ${nomeSub(camada)}`, head: true }] : []),
    ...itensLegenda(camada, legenda),
    ...(camada2 ? [{ label: `Camada de cima: ${nomeSub(camada2)}`, head: true }, ...itensLegenda(camada2, legenda2)] : []),
  ];
  const explicacao = cortina
    ? `Arraste a barra no mapa: à esquerda ${anoA}, à direita ${anoB}.`
    : camada2 ? "Use a barra no canto do mapa para deixar a camada de cima mais ou menos transparente." : "";
  const opcoesSub = (lista) => [...lista].sort((a, b) => Number(Boolean(a.periodo)) - Number(Boolean(b.periodo)))
    .map((c) => ({ value: c.id, label: nomeSub(c), grupo: c.periodo ? "Por período" : "Anuais" }));
  const anosOpc = range(ini, fim).map((a) => ({ value: String(a), label: String(a) }));

  return {
    toolbar: [
      mapTypeControl(),
      {
        type: "select",
        label: "Subproduto",
        value: camada.id,
        options: opcoesSub(camadas),
        onChange: (v) => { state.mapa.camada = v; loadCurrentTab(); },
      },
      ...(!camada.periodo && !cortina ? [{
        type: "select",
        label: "Ano",
        value: String(anoUnico),
        options: anosOpc,
        onChange: (v) => { state.mapa.ano = Number(v); loadCurrentTab(); },
      }] : []),
      ...(!camada.periodo ? [{
        type: "segmented",
        label: "Comparar dois anos (cortina)",
        value: cortina ? "on" : "off",
        options: [{ value: "off", label: "Não" }, { value: "on", label: "Sim" }],
        onChange: (v) => { state.mapa.cortina = v; if (v === "on") state.mapa.camada2 = ""; loadCurrentTab(); },
      }] : []),
      ...(cortina ? [
        { type: "select", label: "Ano da esquerda", value: String(anoA), options: anosOpc, onChange: (v) => { state.mapa.anoA = Number(v); loadCurrentTab(); } },
        { type: "select", label: "Ano da direita", value: String(anoB), options: anosOpc, onChange: (v) => { state.mapa.anoB = Number(v); loadCurrentTab(); } },
      ] : [{
        type: "select",
        label: "Sobrepor outro subproduto",
        value: camada2?.id || "",
        options: [{ value: "", label: "Nenhum" }, ...opcoesSub(camadas.filter((c) => c.id !== camada.id))],
        onChange: (v) => { state.mapa.camada2 = v; loadCurrentTab(); },
      }]),
      {
        type: "segmented",
        label: "Contornos",
        value: m.visualizar,
        options: MAP_VIEWS.filter((v) => v.id !== "biomas" || state.conn.biomas.length)
          .map((v) => ({ value: v.id, label: rot({ biomas: "bioma", estados: "estado", municipios: "municipio" }[v.id], true) })),
        onChange: (v) => { state.mapa.visualizar = v; loadCurrentTab(); },
      },
      {
        type: "segmented",
        label: "Destacar o recorte",
        value: m.destaque,
        options: [{ value: "off", label: "Não" }, { value: "on", label: "Sim" }],
        onChange: (v) => { state.mapa.destaque = v; loadCurrentTab(); },
      },
    ],
    kpis: [],
    cards: [{
      span: 12,
      title: cortina ? `${nomeSub(camada)} · ${anoA} × ${anoB}`
        : `${nomeSub(camada)} · ${camada.periodo ? `${ini}–${fim}` : anoUnico}`,
      subtitle: `Recorte: ${t.nome} · tiles oficiais do MapBiomas Fogo · clique num contorno para destacá-lo${explicacao ? ` · ${explicacao}` : ""}`,
      info: "Mapa oficial do MapBiomas Fogo, pixel a pixel. Escolha o subproduto e o ano; nos subprodutos anuais, a cortina compara dois anos lado a lado.",
      map: {
        tileUrl, tileUrl2, modo: cortina ? "cortina" : camada2 ? "sobrepor" : "",
        anoA, anoB, destacar: m.destaque === "on",
        ativo, contornos: contornos.feicoes, aviso: contornos.aviso,
      },      legend: legendItems,
    }],
  };
}

// Mapa coroplético: cada estado/município pintado pela área queimada no período.
async function loadMapaRanking({ t, ini, fim }) {
  const c = state.conn;
  const total = await getLegend("anualTotal");
  const vermelho = levelItems(total, 0)[0].color;
  const unidade = state.mapa.unidade;
  const progresso = (rotulo) => (n, tot) => setStatus(`${rotulo}: ${n}/${tot}`, true);

  let linhas, features, escopoRotulo, anosRk, aviso = "", emCalculo = false;
  if (unidade === "estados" || unidade === "biomas") {
    // Estados ou biomas inteiros: um ranking por ano da categoria + contorno de cada território.
    const ehBioma = unidade === "biomas";
    const lista = ehBioma ? c.biomas : c.estados;
    const nivel = ehBioma ? "bioma" : "estado";
    const [rk, geoms] = await Promise.all([
      rankingPeriodo(ehBioma ? c.catBioma : c.catEstado, ini, fim, progresso(`Ranking de ${rot(nivel, true).toLowerCase()} (anos)`)),
      mapLimit(lista, 6, (e) => getGeometry([e.id]).catch(() => null)),
    ]);
    linhas = rk.linhas;
    anosRk = rk.anos;
    const valor = new Map(linhas.map((l) => [l.id, l.value]));
    features = lista.map((e, i) => ({
      type: "Feature", geometry: geoms[i], properties: { id: e.id, name: e.name, value: valor.get(e.id) || 0 },
    })).filter((f) => f.geometry);
    escopoRotulo = `${rot(nivel, true)} ${PAIS.de}`;
    if (state.biomaId && !ehBioma) aviso = `${rot("estado", true)} inteiros: a API de ranking não recorta por ${rot("bioma").toLowerCase()}.`;
    if (state.estadoId && ehBioma) aviso = `${rot("bioma", true)} inteiros: a API de ranking não recorta por ${rot("estado").toLowerCase()}.`;
  } else {
    const [escopo, rk, shapes] = await Promise.all([
      municipiosNoEscopo((msg) => setStatus(msg, true)),
      rankingPeriodo(c.catMunicipio, ini, fim, progresso(`Ranking de ${rot("municipio", true).toLowerCase()} (anos)`)),
      getMunicipiosShapes(progresso(`Geometria dos ${rot("municipio", true).toLowerCase()} (tiles)`)),
    ]);
    const r = aplicarRecorte(rk.linhas.filter(escopo.filtro), escopo, ini, fim);
    linhas = r.linhas;
    anosRk = rk.anos;
    const valor = new Map(linhas.map((l) => [l.id, l]));
    // Com bioma: só a parte de cada município dentro dele (geometria recortada).
    const geometrias = escopo.recorte ? [...escopo.recorte.values()] : [...shapes.values()];
    // Um polígono por pedaço de tile: os pedaços de tiles vizinhos se sobrepõem na faixa de
    // buffer e, num mesmo MultiPolygon, o preenchimento even-odd do canvas abriria faixas vazias.
    features = geometrias
      .filter((m) => escopo.filtro({ id: m.id, name: m.name }))
      .map((m) => (ehBrasil ? m : { ...m, name: nomeMunicipio(m.name) }))
      .flatMap((m) => {
        const emCalculo = m.situacao === "divisa" && !r.valores?.has(m.id);
        return m.polygons.map((coords) => ({
          type: "Feature",
          geometry: { type: "Polygon", coordinates: coords },
          properties: {
            id: m.id, name: m.name, value: valor.get(m.id)?.value || 0, emCalculo,
            nota: m.situacao === "divisa" ? `Na divisa · ${pct(m.fracao)} do município no bioma${emCalculo ? " · área exata em cálculo" : " · área da interseção"}` : "",
          },
        }));
      });
    escopoRotulo = escopo.rotulo;
    aviso = avisoRecorte(escopo, r);
    emCalculo = r.pendentes > 0;
  }
  setStatus("");

  // Quantis com um valor por território (municípios aparecem em vários pedaços).
  const classes = quantileClasses([...new Map(features.filter((f) => !f.properties.emCalculo).map((f) => [f.properties.id, f.properties.value])).values()]);
  const corEmCalculo = "#9aa0a6";
  for (const f of features) f.properties.color = f.properties.emCalculo ? corEmCalculo : classes.colorOf(f.properties.value);
  const top = linhas.slice(0, 10);
  const recorte = state.biomaId || state.estadoId ? await getGeometry(t.ids, t.metodo).catch(() => null) : null;
  // Série anual de cada território (para linha do tempo, variação, detalhe e CSV).
  const porId = new Map(linhas.map((l) => [l.id, l]));
  const territorios = new Map();
  for (const f of features) {
    const { id, name } = f.properties;
    if (territorios.has(id)) continue;
    const serie = porId.get(id)?.porAno;
    territorios.set(id, { name, porAno: serie ? [...serie] : new Array(anosRk.length).fill(0), emCalculo: f.properties.emCalculo });
  }

  return {
    toolbar: [
      mapTypeControl(),
      {
        type: "segmented",
        label: "Pintar por",
        value: unidade,
        options: [
          ...(state.conn.biomas.length ? [{ value: "biomas", label: rot("bioma", true) }] : []),
          { value: "estados", label: rot("estado", true) },
          ...(state.conn.catMunicipio ? [{ value: "municipios", label: rot("municipio", true) }] : []),
        ],
        onChange: (v) => { state.mapa.unidade = v; loadCurrentTab(); },
      },
    ],
    kpis: [],
    cards: [
      {
        span: 12,
        title: `${escopoRotulo}: explore o fogo no mapa`,
        subtitle: `Ranking de ${ini}–${fim} · escolha o que mostrar nos passos abaixo e clique num território para ver o detalhe${unidade === "municipios" && !state.biomaId && !state.estadoId ? ` · escolha ${rot("bioma").toLowerCase()} ou ${rot("estado").toLowerCase()} no filtro para recortar` : ""}`,
        map: {
          dinamico: {
            features, territorios, anos: anosRk, unidade, contorno: recorte, bordas: unidade !== "municipios",
            emCalculo, corEmCalculo, recorteBioma: Boolean(state.biomaId),
          },
          aviso,
        },
      },
      {
        span: 12,
        title: `${rot({ biomas: "bioma", estados: "estado" }[unidade] || "municipio", true)} que mais queimaram`,
        subtitle: `Top 10 · soma ${ini}–${fim} · cor = classe da área queimada no período`,
        ranking: {
          rows: top.map((l) => ({ label: l.name, value: l.value, color: classes.colorOf(l.value) || vermelho })),
          total: sum(linhas.map((l) => l.value)),
        },
      },
    ],
  };
}

// 5 classes por quantis (só valores > 0) na rampa CHORO_RAMP.
function quantileClasses(valores) {
  const pos = valores.filter((v) => v > 0).sort((a, b) => a - b);
  if (!pos.length) return { colorOf: () => null, legend: [] };
  const q = (p) => pos[Math.min(pos.length - 1, Math.floor(p * pos.length))];
  const cortes = [q(0.2), q(0.4), q(0.6), q(0.8)];
  const colorOf = (v) => {
    if (!(v > 0)) return null;
    const i = cortes.findIndex((c) => v <= c);
    return CHORO_RAMP[i < 0 ? CHORO_RAMP.length - 1 : i];
  };
  const limites = [pos[0], ...cortes, pos[pos.length - 1]];
  const legend = CHORO_RAMP.map((color, i) => ({
    label: `${fmtCompact.format(i ? limites[i] : limites[0])}–${fmtCompact.format(limites[i + 1])} ha`,
    color,
  }));
  return { colorOf, legend };
}

// Contornos de referência do mapa de dados brutos, com nome (clicáveis: o clique destaca o contorno).
async function referenceOutlines(visualizar) {
  const c = state.conn;
  const feicao = (geometry, name) => (geometry ? { type: "Feature", geometry, properties: { name } } : null);
  if (visualizar === "biomas") {
    const gs = await Promise.all(c.biomas.map((b) => getGeometry([b.id]).catch(() => null)));
    return { feicoes: gs.map((g, i) => feicao(g, c.biomas[i].name)).filter(Boolean) };
  }
  if (visualizar === "estados") {
    const gs = await Promise.all(c.estados.map((e) => getGeometry([e.id]).catch(() => null)));
    return { feicoes: gs.map((g, i) => feicao(g, c.estados[i].name)).filter(Boolean) };
  }
  if (state.municipioId) {
    // Com bioma ativo, só a parte do município dentro do bioma (interseção feita pela API).
    const geom = state.biomaId
      ? await getGeometry([state.municipioId, state.biomaId], "intersection").catch(() => null)
      : await getGeometry([state.municipioId]).catch(() => null);
    const m = state.municipiosDoEstado.find((x) => x.id === state.municipioId);
    return { feicoes: [feicao(geom, m?.name || "")].filter(Boolean) };
  }
  // Sem município escolhido: todos os municípios (do estado filtrado, se houver), pelos tiles vetoriais.
  if (!c.catMunicipio) return { feicoes: [] };
  const estado = c.estados.find((e) => e.id === state.estadoId);
  const [shapes, doEstado] = await Promise.all([
    getMunicipiosShapes((n, tot) => setStatus(`Geometria dos ${rot("municipio", true).toLowerCase()} (tiles): ${n}/${tot}`, true)),
    estado ? getMunicipiosDoEstado(estado) : null,
  ]);
  setStatus("");
  const ids = doEstado ? new Set(doEstado.map((m) => m.id)) : null;
  const feicoes = [];
  for (const m of shapes.values()) {
    if (ids && !ids.has(m.id)) continue;
    for (const coords of m.polygons) feicoes.push({ type: "Feature", geometry: { type: "Polygon", coordinates: coords }, properties: { name: nomeMunicipio(m.name), id: m.id } });
  }
  return { feicoes };
}

// ---------------------------------------------------------------------------
// Renderização
// ---------------------------------------------------------------------------

// Copia o endereço com o hash atual (writeHash guarda país, idioma, aba, filtros e opções da aba).
async function copiarLink(btn) {
  writeHash();
  const url = location.href;
  let ok = false;
  try {
    await navigator.clipboard.writeText(url);
    ok = true;
  } catch {
    // Sem permissão de área de transferência (ex.: file://): seleciona o texto num campo e copia.
    const campo = document.createElement("textarea");
    campo.value = url;
    campo.style.position = "fixed";
    campo.style.opacity = "0";
    document.body.appendChild(campo);
    campo.select();
    try { ok = document.execCommand("copy"); } catch { ok = false; }
    campo.remove();
  }
  const original = btn.textContent;
  btn.textContent = ok ? "✓ Link copiado" : "Copie o endereço da barra do navegador";
  setTimeout(() => { if (btn.isConnected) btn.textContent = original; }, 2500);
}

function renderView(view) {
  destroyCharts();
  const tokens = themeTokens();
  aplicarPadroesChart(tokens);
  const t = activeTerritory();
  el.panel.innerHTML = `
    <div class="panel-head">
      <p class="context"><strong>${escapeHtml(t.nome)}</strong> · ${state.ini}–${state.fim}
        <button type="button" class="md-botao copiar-link" title="Copia um link que abre o painel nesta mesma análise (país, idioma, aba, filtros e opções)">🔗 Copiar link desta análise</button></p>
      ${headControls(view)}
    </div>
    ${view.kpis.length ? `<div class="kpis">${view.kpis.map(kpiHtml).join("")}</div>` : ""}
    <div class="grid">${view.cards.map((card, i) => cardHtml(card, i, view)).join("")}</div>`;

  view.toolbar?.forEach((ctl, i) => bindToolbar(ctl, i));
  el.panel.querySelector(".copiar-link")?.addEventListener("click", (e) => copiarLink(e.currentTarget));
  const seq = state.seq;
  view.cards.forEach((card, i) => {
    const root = el.panel.querySelector(`[data-card="${i}"]`);
    if (card.custom) return card.custom(root.querySelector(".custom-body"));
    if (card.map) return renderMap(root.querySelector(".map-box"), card.map);
    if (card.lazy) return fillLazyCard(card, root, seq, tokens);
    if (card.alterna) bindVista(card, i, root, tokens);
    montarGraficos(card, root, vistaDe(card, i), tokens);
  });
}

// ---------- Alternância do tipo de gráfico por card ----------
// alterna: "linhas" -> Barras | Linhas (séries por ano); "rosca" -> Barras | Rosca (distribuições);
// "anual" -> Barras | Linhas | Por bioma (o card fornece `porBioma`, carregado sob demanda).

const VISTAS = {
  linhas: [["barras", "Barras"], ["linhas", "Linhas"]],
  rosca: [["barras", "Barras"], ["rosca", "Rosca"]],
  anual: [["barras", "Barras"], ["linhas", "Linhas"], ["biomas", `Por ${rot("bioma").toLowerCase()}`]],
};

// Vistas que trocam o corpo do card inteiro (não dá para só mudar o tipo do Chart.js).
const VISTAS_DE_CORPO = ["rosca", "biomas"];

const ICONE_VISTA = {
  barras: `<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M2 14V8.5a1 1 0 0 1 1-1h1.5a1 1 0 0 1 1 1V14zm4.75 0V3a1 1 0 0 1 1-1h1.5a1 1 0 0 1 1 1v11zm4.75 0V6.5a1 1 0 0 1 1-1H14a1 1 0 0 1 1 1V14z"/></svg>`,
  linhas: `<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" d="M1.5 12.5c2-1 2.6-5 4.5-5s2.2 2.5 4 2.5 2.5-4.5 4.5-7"/></svg>`,
  biomas: `<svg viewBox="0 0 16 16" aria-hidden="true"><path fill="currentColor" d="M2 14v-3h3.5v3zm0-4V7.5h3.5V10zm4.75 4V9h3.5v5zm0-6V4h3.5v4zm4.75 6v-2.5H15V14zm0-3.5V6H15v4.5z"/></svg>`,
  rosca: `<svg viewBox="0 0 16 16" aria-hidden="true"><circle cx="8" cy="8" r="5.25" fill="none" stroke="currentColor" stroke-opacity="0.35" stroke-width="3"/><path fill="none" stroke="currentColor" stroke-width="3" d="M8 2.75A5.25 5.25 0 0 1 13.25 8"/></svg>`,
};

const chaveVista = (i) => `${state.tab}:${i}`;

function vistaDe(card, i) {
  if (!card.alterna) return null;
  return state.vistas[chaveVista(i)] || (card.chart?.kind === "line" ? "linhas" : "barras");
}

// Spec do gráfico na vista escolhida (o spec original descreve a vista padrão).
function specNaVista(spec, vista) {
  if (vista === "linhas") return spec.kind === "stacked" ? { ...spec, area: true } : { ...spec, kind: "line" };
  if (vista === "barras" && spec.kind === "line") return { ...spec, kind: "bar" };
  return spec;
}

// Roscas do card: as declaradas em `rosca` ou uma derivada das barras (categoria -> fatia).
function roscasDe(card) {
  const c = card.chart;
  const roscas = card.rosca || [{
    slices: c.values.map((v, k) => ({ label: c.tooltipTitles?.[k] || c.labels[k], value: v, color: c.colors?.[k] || c.color })),
  }];
  return roscas.map((r) => ({ ...r, slices: r.slices.filter((s) => s.value > 0) })).filter((r) => r.slices.length);
}

function vistaToggleHtml(card, i) {
  const atual = vistaDe(card, i);
  return `
        <div class="vista-toggle" role="group" aria-label="Tipo de gráfico">
          ${VISTAS[card.alterna].map(([v, rotulo]) => `
            <button type="button" data-vista="${v}" aria-pressed="${v === atual}" title="Ver como ${rotulo.toLowerCase()}">${ICONE_VISTA[v]}<span>${rotulo}</span></button>`).join("")}
        </div>`;
}

function bindVista(card, i, root, tokens) {
  const grupo = root.querySelector(".card-head .vista-toggle");
  grupo.addEventListener("click", (e) => {
    const btn = e.target.closest("button[data-vista]");
    const atual = vistaDe(card, i);
    if (!btn || btn.dataset.vista === atual) return;
    const nova = btn.dataset.vista;
    state.vistas[chaveVista(i)] = nova;
    grupo.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", String(b === btn)));
    const chart = root._charts?.[0];
    if (card.chart && chart && !VISTAS_DE_CORPO.includes(atual) && !VISTAS_DE_CORPO.includes(nova)) {
      trocarTipo(root, chart, chartConfig(specNaVista(card.chart, nova), tokens));
    } else {
      // Rosca não tem eixos: destrói a instância e recria o corpo do card com os mesmos dados.
      descartarGraficos(root);
      root.querySelector(".card-body").innerHTML = cardBodyHtml(card, nova);
      montarGraficos(card, root, nova, tokens);
    }
  });
}

// Barras <-> linhas na mesma instância: troca tipo, datasets e opções e anima com update().
function trocarTipo(root, chart, config) {
  try {
    chart.config.type = config.type;
    chart.data = config.data;
    chart.options = config.options;
    chart.update();
  } catch (err) {
    console.warn("Troca de tipo no lugar falhou; recriando o gráfico.", err);
    const canvas = chart.canvas;
    descartarGraficos(root);
    registrarGrafico(root, new Chart(canvas, config));
  }
}

function registrarGrafico(root, chart) {
  (root._charts ||= []).push(chart);
  state.charts.push(chart);
}

function descartarGraficos(root) {
  for (const c of root._charts || []) {
    c.destroy();
    state.charts = state.charts.filter((x) => x !== c);
  }
  root._charts = [];
}

function montarGraficos(card, root, vista, tokens) {
  root._vista = vista;
  if (vista === "biomas") {
    const body = root.querySelector(".card-body");
    const vivo = () => root.isConnected && root._vista === "biomas";
    card.porBioma((msg) => { if (vivo()) body.innerHTML = `<p class="empty busy-text">${escapeHtml(msg)}</p>`; })
      .then((res) => {
        const desenhar = () => {
          if (!vivo()) return;
          descartarGraficos(root);
          const spec = { ...res.chart, series: res.chart.series.map((s) => ({ ...s, color: corPaleta(s.label) })) };
          body.innerHTML = chartBoxHtml(spec, card.title) +
            legendRowHtml([...spec.series.map((s) => ({ label: s.label, color: s.color })), { label: res.nota, note: true }]) +
            (ehBrasil ? paletaToggleHtml() : ""); // as paletas terrosa/vermelhos são dos biomas do Brasil
          registrarGrafico(root, new Chart(body.querySelector("canvas"), chartConfig(spec, tokens)));
          body.querySelector(".paleta-toggle")?.addEventListener("click", (e) => {
            const btn = e.target.closest("button[data-paleta]");
            if (!btn || btn.dataset.paleta === state.paletaBioma) return;
            state.paletaBioma = btn.dataset.paleta;
            desenhar();
          });
        };
        desenhar();
      })
      .catch((err) => {
        console.error(err);
        if (vivo()) body.innerHTML = `<p class="empty">Não foi possível calcular por bioma: ${escapeHtml(err.message)}</p>`;
      });
    return;
  }
  if (vista === "rosca") {
    const canvases = root.querySelectorAll(".donut canvas");
    roscasDe(card).forEach((r, k) => {
      if (canvases[k]) registrarGrafico(root, new Chart(canvases[k], doughnutConfig(r, tokens)));
    });
    return;
  }
  const canvas = root.querySelector("canvas");
  if (!canvas) return;
  const config = card.chart ? chartConfig(specNaVista(card.chart, vista), tokens) : doughnutConfig(card.doughnut, tokens);
  registrarGrafico(root, new Chart(canvas, config));
}

// Padrões globais do Chart.js: fonte maior e cinza escuro do tema (rótulos, legendas, tooltips).
function aplicarPadroesChart(t) {
  Chart.defaults.color = t.text2;
  Chart.defaults.font.size = 13;
  Chart.defaults.plugins.legend.labels.color = t.text2;
  Chart.defaults.plugins.legend.labels.font = { size: 13 };
  Chart.defaults.plugins.tooltip.titleFont = { size: 14, weight: "600" };
  Chart.defaults.plugins.tooltip.bodyFont = { size: 13 };
  Chart.defaults.plugins.tooltip.footerFont = { size: 13, weight: "600" };
}

// Cards cujo conteúdo é caro (rankings): a aba aparece já, o card completa depois.
function fillLazyCard(card, root, seq, tokens) {
  const body = root.querySelector(".lazy-body");
  const vivo = () => seq === state.seq && root.isConnected;
  const progress = (msg) => { if (vivo()) body.innerHTML = `<p class="empty busy-text">${escapeHtml(msg)}</p>`; };
  card.lazy(progress)
    .then((res) => {
      if (!vivo()) return;
      root._res = res; // a tabela sob demanda usa o resultado do card
      const conteudo = res.chart ? chartBoxHtml(res.chart, card.title)
        : res.mini ? miniRoscasHtml(res.mini)
        : res.topcards ? topcardsHtml(res.topcards)
        : rankingHtml(res.ranking);
      body.innerHTML = conteudo +
        (res.legend?.length ? legendRowHtml(res.legend) : "") +
        (res.note ? `<p class="kpi-note">${escapeHtml(res.note)}</p>` : "");
      const canvas = body.querySelector("canvas");
      if (res.chart && canvas) state.charts.push(new Chart(canvas, chartConfig(res.chart, tokens)));
      if (res.mini) {
        body.querySelectorAll(".mini-donut canvas").forEach((cv, k) => state.charts.push(new Chart(cv, miniDonutConfig(res.mini[k], tokens))));
      }
    })
    .catch((err) => {
      console.error(err);
      if (vivo()) body.innerHTML = `<p class="empty">Não foi possível calcular: ${escapeHtml(err.message)}</p>`;
    });
}

// Opções de um <select>; com `grupo` nas opções, vira <optgroup> (ex.: Anuais / Por período).
function opcoesSelectHtml(ctl) {
  const opt = (o) => `<option value="${escapeHtml(o.value)}" ${o.value === ctl.value ? "selected" : ""}>${escapeHtml(o.label)}</option>`;
  if (!ctl.options.some((o) => o.grupo)) return ctl.options.map(opt).join("");
  const soltas = ctl.options.filter((o) => !o.grupo);
  const grupos = [...new Set(ctl.options.filter((o) => o.grupo).map((o) => o.grupo))];
  return soltas.map(opt).join("") + grupos.map((g) =>
    `<optgroup label="${escapeHtml(g)}">${ctl.options.filter((o) => o.grupo === g).map(opt).join("")}</optgroup>`).join("");
}

function toolbarHtml(ctl, i) {
  if (ctl.type === "chips") {
    return `
      <div class="field chips-field">
        <span>${escapeHtml(ctl.label)}</span>
        <div class="chips" role="group" aria-label="${escapeHtml(ctl.label)}" data-ctl="${i}">
          ${ctl.options.map((o) => `
            <button type="button" class="chip${o.dash ? " dash" : ""}" data-value="${escapeHtml(o.value)}"
              aria-pressed="${ctl.selected.includes(o.value)}"${o.color ? ` style="--c:${o.color}"` : ""}>${escapeHtml(o.label)}</button>`).join("")}
        </div>
      </div>`;
  }
  if (ctl.type === "segmented") {
    return `
      <div class="field">
        <span>${escapeHtml(ctl.label)}</span>
        <div class="segmented" role="group" aria-label="${escapeHtml(ctl.label)}" data-ctl="${i}">
          ${ctl.options.map((o) => `
            <button type="button" data-value="${escapeHtml(o.value)}" aria-pressed="${String(o.value) === String(ctl.value)}"
              ${o.title ? `title="${escapeHtml(o.title)}"` : ""}>${escapeHtml(o.label)}</button>`).join("")}
        </div>
      </div>`;
  }
  return `
    <label class="field">
      <span>${escapeHtml(ctl.label)}</span>
      <select data-ctl="${i}">
        ${opcoesSelectHtml(ctl)}
      </select>
    </label>`;
}

function bindToolbar(ctl, i) {
  const node = el.panel.querySelector(`[data-ctl="${i}"]`);
  if (ctl.type === "chips") {
    node.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (btn) ctl.onChange(btn.dataset.value);
    });
  } else if (ctl.type === "segmented") {
    node.addEventListener("click", (e) => {
      const btn = e.target.closest("button");
      if (btn && btn.dataset.value !== String(ctl.value)) ctl.onChange(btn.dataset.value);
    });
  } else {
    node.addEventListener("change", () => ctl.onChange(node.value));
  }
}

function kpiHtml(kpi) {
  return `
    <article class="card kpi">
      <p class="kpi-label">${escapeHtml(kpi.label)}</p>
      <p class="kpi-value">${escapeHtml(kpi.value)}</p>
      ${kpi.note ? `<p class="kpi-note">${escapeHtml(kpi.note)}</p>` : ""}
    </article>`;
}

// Controles da barra superior da aba (os marcados `inSection` vão para o card de seção).
function headControls(view) {
  const itens = (view.toolbar || []).map((ctl, i) => (ctl.inSection || ctl.card != null ? "" : toolbarHtml(ctl, i))).join("");
  return itens.trim() ? `<div class="toolbar">${itens}</div>` : "";
}

// Explicação curta de cada card (botão ⓘ), pelo título em português; `card.info` tem prioridade.
// Texto novo aqui precisa de linha em i18n.js.
const INFO_CARDS = [
  [/^Área queimada anual$/, "Soma da área que queimou em cada ano no território e no período escolhidos. A linha tracejada é a média do período."],
  [/^Anos que mais queimaram$/, "Os seis anos com mais área queimada no período e quanto cada um ficou acima ou abaixo da média anual."],
  [/^Vegetação natural x uso antrópico/, "Em que cobertura da terra o fogo ocorreu, cruzando as áreas queimadas com o mapa de uso e cobertura MapBiomas do mesmo ano. É a soma das áreas anuais: um lugar que queimou em dois anos conta duas vezes."],
  [/^Área queimada por ano e classe$/, "A área queimada de cada ano dividida pela classe de cobertura da terra onde o fogo ocorreu."],
  [/^Área queimada acumulada$/, "Área que queimou pelo menos uma vez entre o ano inicial e cada ano. Cada lugar conta uma vez só, mesmo que tenha queimado de novo; por isso a curva só sobe."],
  [/^Quanto d[oe] .* queimou ao menos uma vez/, "Parte do território que queimou pelo menos uma vez no período: área queimada acumulada dividida pela área total."],
  [/^Composição do acumulado por ano$/, "Em cada ano, a parte de baixo é o que já tinha queimado antes e a de cima é a área que queimou pela primeira vez no período. Partes de cima pequenas indicam que o fogo voltou a lugares já queimados."],
  [/^Anos que mais adicionaram área inédita$/, "Anos em que o fogo mais chegou a áreas que ainda não tinham queimado no período."],
  [/^Média mensal da área queimada/, "Área queimada média em cada mês, considerando todos os anos do período. Escolha anos de destaque para comparar com a média."],
  [/^Distribuição mensal/, "Soma de todos os anos do período, mês a mês: mostra em que meses se concentra a temporada de fogo."],
  [/^Soma do período e média por mês$/, "Resumo de cada mês: área total no período, parte do total e média por ano com desvio padrão."],
  [/^Área queimada por ano e mês/, "Cada barra é um ano (ou temporada), dividida pelos meses em que o fogo ocorreu."],
  [/^Proporção mensal da área queimada por/, "Para cada território, que parte da área queimada no período ocorreu em cada mês. Permite comparar o calendário do fogo entre territórios."],
  [/^Faixas de frequência$/, "Quantas vezes cada lugar queimou no período, agrupado em faixas. Áreas que queimam muitas vezes indicam uso repetido do fogo ou vegetação que convive com o fogo."],
  [/^Área queimada por classe de frequência$/, "Área em cada número de vezes que queimou no período. Em %, mostra a parte da área queimada do período em cada classe."],
  [/^Área queimada por ano e tamanho de cicatriz$/, "Área queimada de cada ano dividida pelo tamanho da cicatriz (mancha contínua de área queimada). Cicatrizes grandes costumam vir de incêndios que fugiram do controle."],
  [/^Distribuição por faixa de tamanho$/, "Parte da área queimada do período em cada faixa de tamanho de cicatriz."],
  [/^Proporção por ano em cicatrizes/, "Em cada ano, que parte da área queimada estava em cicatrizes da faixa de tamanho escolhida."],
  [/^Tempo desde o fogo anterior/, "Para as áreas que queimaram no ano final e já tinham queimado antes: quantos anos se passaram desde o fogo anterior."],
  [/^Distribuição por tempo de retorno$/, "Parte da área com fogo anterior em cada faixa de tempo de retorno."],
  [/^Intervalo médio de retorno/, "Média de anos entre queimadas sucessivas no mesmo lugar, para áreas que queimaram mais de uma vez. Intervalos curtos indicam fogo frequente."],
  [/^Distribuição por intervalo médio$/, "Parte da área com fogo recorrente em cada faixa de intervalo médio de retorno."],
  [/^Área por ano do último fogo$/, "Para cada lugar que queimou no período, o ano em que queimou pela última vez. Último fogo antigo = área há mais tempo sem queimar."],
  [/^Tempo decorrido desde o último fogo$/, "Há quantos anos, contados até o ano final, cada área queimou pela última vez."],
  [/^Anos de último fogo com mais área$/, "Os anos em que mais área queimou pela última vez."],
  [/^Severidade total/, "Severidade é o quanto o fogo alterou a vegetação, estimada com imagens de satélite de antes e depois do fogo. Mostra a parte da área afetada em cada classe."],
  [/^Área afetada por ano e classe de severidade$/, "Área afetada de cada ano dividida pela classe de severidade."],
  [/^Distribuição no período$/, "Parte da área afetada do período em cada classe de severidade."],
  [/^Proporção com severidade/, "Em cada ano, que parte da área afetada teve severidade igual ou maior que a classe indicada."],
  [/que mais queimaram( no período)?$/, "Ranking pela soma das áreas queimadas em cada ano do período."],
  [/que mais queimaram em \d{4}$/, "Ranking da área queimada no ano final do período."],
  [/^Variação em \d{4} em relação à média/, "Quanto a área queimada no ano final ficou acima (vermelho) ou abaixo (azul) da média dos anos anteriores do período, em cada território."],
  [/^Evolução anual: os 3 maiores/, "Série anual dos três territórios com mais área queimada no período."],
  [/^Participação na área queimada por .*, ano a ano$/, "Em cada ano, como a área queimada se divide entre os territórios (cada barra soma 100%)."],
  [/: área queimada no período$/, "Área queimada somada no período em cada território e sua parte no total."],
  [/^Fogo por categoria fundiária/, "Área queimada em Terras Indígenas, Unidades de Conservação, assentamentos e outras categorias fundiárias."],
  [/^Ranking das categorias$/, "Categorias fundiárias ordenadas pela área queimada no período."],
  [/^Evolução anual nas principais categorias$/, "Série anual da área queimada nas principais categorias fundiárias."],
  [/: explore o fogo no mapa$/, "Mapa por território: escolha o que mostrar, o ano ou a comparação, ou busque um território pelo nome. Clique num território para ver a série anual."],
  [/^Fábrica de GIFs$/, "Gera uma animação com um mapa por ano. Escolha país, território, produto e anos; o GIF é montado no seu navegador."],
];

function infoDoCard(card) {
  if (card.section) return null;
  return card.info || INFO_CARDS.find(([re]) => re.test(card.title || ""))?.[1] || null;
}

function cardHtml(card, i, view) {
  if (card.section) {
    const controles = (view.toolbar || []).map((ctl, j) => (ctl.inSection ? toolbarHtml(ctl, j) : "")).join("");
    return `
      <section class="section-head span-12" data-card="${i}">
        <div>
          <h2>${escapeHtml(card.title)}</h2>
          ${card.subtitle ? `<p>${escapeHtml(card.subtitle)}</p>` : ""}
        </div>
        ${controles.trim() ? `<div class="toolbar">${controles}</div>` : ""}
      </section>`;
  }
  const toggle = card.alterna ? vistaToggleHtml(card, i) : "";
  // Baixar o gráfico (PNG) e abrir a tabela (sob demanda) em todo card com dados.
  const temDados = !card.map && !card.custom && !card.tabela &&
    (card.chart || card.rosca || card.ranking || card.propbars || card.lazy || card.topcards || card.doughnut);
  const acoes = temDados ? `
        <div class="card-acoes">
          <button type="button" data-acao="png" title="Baixar gráfico (PNG)">⬇ PNG</button>
          <button type="button" data-acao="tabela" aria-pressed="false" title="Ver a tabela com os valores">▦ Tabela</button>
        </div>` : "";
  const info = infoDoCard(card);
  return `
    <article class="card span-${card.span}${card.chart?.big ? " card-big" : ""}" data-card="${i}">
      <header class="card-head${toggle || acoes || info ? " has-toggle" : ""}">
        <div>
          <h2>${escapeHtml(card.title)}${info ? ` <button type="button" class="card-info" data-acao="info" aria-expanded="false" title="O que este gráfico mostra?" aria-label="O que este gráfico mostra?">ⓘ</button>` : ""}</h2>
          ${card.subtitle ? `<p>${escapeHtml(card.subtitle)}</p>` : ""}
        </div>
        <div class="card-topo">${toggle}${acoes}</div>
      </header>
      ${info ? `<p class="card-info-texto" hidden>${escapeHtml(info)}</p>` : ""}
      ${(view.toolbar || []).some((c) => c.card === i) ? `<div class="toolbar card-controles">${(view.toolbar || []).map((c, j) => (c.card === i ? toolbarHtml(c, j) : "")).join("")}</div>` : ""}
      <div class="card-body">${cardBodyHtml(card, vistaDe(card, i))}</div>
    </article>`;
}

// Corpo do card na vista atual, com a legenda do card (a vista por bioma traz a sua ao carregar).
function cardBodyHtml(card, vista) {
  if (vista === "biomas") return `<p class="empty busy-text">Calculando ${rot("bioma", true).toLowerCase()}…</p>`;
  return conteudoHtml(card, vista) + (card.legend?.length ? legendRowHtml(card.legend) : "");
}

function conteudoHtml(card, vista) {
  if (vista === "rosca") return roscasHtml(roscasDe(card));
  if (card.topcards) return topcardsHtml(card.topcards);
  if (card.tabela) return tabelaHtml(card.tabela);
  if (card.custom) return `<div class="custom-body"></div>`;
  if (card.map?.dinamico) {
    return `${card.map.aviso ? `<p class="map-notice">${escapeHtml(card.map.aviso)}</p>` : ""}
      <div class="md-wrap">
        <div class="md-controles"></div>
        <p class="md-ajuda" aria-live="polite"></p>
        <div class="map-box"></div>
        <div class="md-legenda"></div>
        <div class="md-detalhe-box" hidden></div>
      </div>`;
  }
  if (card.map) return `${card.map.aviso ? `<p class="map-notice">${escapeHtml(card.map.aviso)}</p>` : ""}<div class="map-box"></div>`;
  if (card.lazy) return `<div class="lazy-body"><p class="empty busy-text">Calculando…</p></div>`;
  if (card.chart) return chartBoxHtml(card.chart, card.title);
  if (card.ranking) return rankingHtml(card.ranking);
  if (card.propbars) return propbarsHtml(card.propbars);
  return doughnutHtml(card.doughnut);
}

function roscasHtml(roscas) {
  if (roscas.length <= 1) return doughnutHtml(roscas[0] || { slices: [] });
  return `<div class="donut-grid">${roscas.map((r) => `
    <div>${r.titulo ? `<p class="propbar-label">${escapeHtml(r.titulo)}</p>` : ""}${doughnutHtml(r)}</div>`).join("")}</div>`;
}

// ---------- Ações dos cards: baixar o gráfico (PNG) e tabela sob demanda ----------

function acaoDoCard(e) {
  const btn = e.target.closest("[data-acao]");
  if (!btn) return;
  const root = btn.closest("[data-card]");
  const card = state.lastView?.cards?.[Number(root?.dataset.card)];
  if (!root || !card) return;
  if (btn.dataset.acao === "info") {
    const texto = root.querySelector(".card-info-texto");
    if (!texto) return;
    texto.hidden = !texto.hidden;
    btn.setAttribute("aria-expanded", String(!texto.hidden));
    return;
  }
  if (btn.dataset.acao === "png") return baixarGraficoPng(root, card);
  if (btn.dataset.acao === "csv") {
    const t = tabelaDoCard(root, card);
    const celula = (c) => (/[;"\n]/.test(String(c)) ? `"${String(c).replace(/"/g, '""')}"` : String(c));
    const csv = "﻿" + [t.colunas, ...t.linhas, ...(t.total ? [t.total] : [])].map((l) => l.map((c) => celula(tr(c))).join(";")).join("\r\n");
    return baixarArquivo(new Blob([csv], { type: "text/csv;charset=utf-8" }), `mapbiomas_fogo_${nomeSeguro(card.title).toLowerCase()}.csv`);
  }
  if (btn.dataset.acao === "tabela") {
    const aberta = root.querySelector(".card-tabela");
    btn.setAttribute("aria-pressed", String(!aberta));
    if (aberta) return aberta.remove();
    const t = tabelaDoCard(root, card);
    const div = document.createElement("div");
    div.className = "card-tabela";
    div.innerHTML = `${tabelaHtml(t)}
      <div class="card-tabela-acoes"><button type="button" class="md-botao" data-acao="csv">⬇ Baixar tabela (CSV)</button></div>`;
    root.appendChild(div);
  }
}

// Monta a tabela do card a partir do que está desenhado (gráficos Chart.js) ou dos dados do card.
function tabelaDoCard(root, card) {
  const fmtV = (v, pctc) => (v == null || Number.isNaN(v) ? "—" : pctc ? pct(v) : fmtFull.format(Math.round(v)));
  const res = root._res || {};
  const charts = [...root.querySelectorAll(".card-body canvas")].map((c) => Chart.getChart(c)).filter(Boolean);
  const titulo = (ch) => ch.canvas.closest(".donut-grid > div")?.querySelector(".propbar-label")?.textContent?.trim()
    || ch.canvas.closest("li")?.querySelector(".md-nome")?.textContent?.trim() || "";
  if (charts.length) {
    const roscas = charts.filter((ch) => ch.config.type === "doughnut");
    if (roscas.length) {
      const varias = roscas.length > 1;
      const linhas = roscas.flatMap((ch) => {
        const vals = ch.data.datasets[0]?.data || [];
        const tot = sum(vals.map(Number).filter(Number.isFinite));
        return (ch.data.labels || []).map((l, k) => [...(varias ? [titulo(ch)] : []), String(l), fmtV(vals[k]), pct(safeDiv(vals[k], tot))]);
      });
      return { colunas: [...(varias ? ["Grupo"] : []), "Classe", "Área (ha)", "%"], linhas };
    }
    const ch = charts[0];
    const pctc = !!ch.options.plugins?.tabela?.percent;
    const ds = ch.data.datasets.filter((d) => Array.isArray(d.data));
    const labels = ch.data.labels || [];
    const unidade = pctc ? "(%)" : "(ha)";
    const linhas = labels.map((l, k) => [String(l), ...ds.map((d) => {
      const v = d.data[k];
      return fmtV(typeof v === "object" && v !== null ? v.y ?? v.x : v, pctc);
    })]);
    const total = !pctc && ds.length === 1 && labels.length > 1
      ? ["Total do período", fmtV(sum(ds[0].data.map((v) => Number(v) || 0)), false)] : null;
    return { colunas: ["", ...ds.map((d) => `${d.label || "Valor"} ${unidade}`)], linhas, total };
  }
  const ranking = res.ranking || card.ranking;
  if (ranking) {
    const tot = ranking.total || sum(ranking.rows.map((r) => r.value));
    return { colunas: ["", "Área (ha)", "%"], linhas: ranking.rows.map((r) => [r.label, fmtV(r.value), pct(safeDiv(r.value, tot))]) };
  }
  if (card.propbars) {
    const linhas = card.propbars.rows.flatMap((r) => {
      const tot = sum(r.segments.map((s) => s.value));
      return r.segments.filter((s) => s.value > 0).map((s) => [r.label, s.label, fmtV(s.value), pct(safeDiv(s.value, tot))]);
    });
    return { colunas: ["Grupo", "Classe", "Área (ha)", "%"], linhas };
  }
  const tops = res.topcards || card.topcards;
  if (tops) return { colunas: ["", "Valor", "Nota"], linhas: tops.map((c) => [c.title, c.value, c.note || ""]) };
  return { colunas: [""], linhas: [] };
}

// PNG do card: os canvas do gráfico nas posições da tela + título, subtítulo e legenda.
async function baixarGraficoPng(root, card) {
  const body = root.querySelector(".card-body");
  const canvases = [...body.querySelectorAll("canvas")].filter((c) => c.offsetWidth);
  if (!canvases.length) return setStatus(tr("Este card não tem gráfico para baixar; use a tabela."));
  const br = body.getBoundingClientRect();
  const W = Math.ceil(br.width), pad = 16, cab = 52, esc = 2;
  const altura = Math.max(...canvases.map((c) => c.getBoundingClientRect().bottom)) - br.top;
  const itens = itensDaLegenda(root.querySelector(".legend-row"));
  const medir = document.createElement("canvas").getContext("2d");
  medir.font = "500 12px Montserrat, sans-serif";
  let x = pad, linha = 0;
  const pos = itens.map((it) => {
    const w = (it.color ? 18 : 0) + medir.measureText(it.label).width;
    if (x + w > W + pad && x > pad) { linha++; x = pad; }
    const p = { it, x, linha };
    x += w + 16;
    return p;
  });
  const rodape = (itens.length ? (linha + 1) * 20 + 8 : 0) + 24;
  const cv = Object.assign(document.createElement("canvas"), { width: (W + 2 * pad) * esc, height: (cab + altura + rodape) * esc });
  const ctx = cv.getContext("2d");
  ctx.scale(esc, esc);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W + 2 * pad, cab + altura + rodape);
  ctx.fillStyle = "#1f1f1f";
  ctx.font = "700 15px Montserrat, sans-serif";
  ctx.fillText(tr(card.title), pad, 24, W);
  ctx.fillStyle = "#555555";
  ctx.font = "500 11px Montserrat, sans-serif";
  ctx.fillText(tr(`${activeTerritory().nome} · ${state.ini}–${state.fim}${card.subtitle ? ` · ${card.subtitle}` : ""}`), pad, 42, W);
  for (const c of canvases) {
    const r = c.getBoundingClientRect();
    ctx.drawImage(c, pad + r.left - br.left, cab + r.top - br.top, r.width, r.height);
  }
  ctx.textBaseline = "middle";
  for (const { it, x: lx, linha: l } of pos) {
    const yy = cab + altura + 14 + l * 20;
    let tx = lx;
    if (it.color) {
      ctx.fillStyle = it.color;
      if (it.linha) ctx.fillRect(lx, yy - 1, 12, 3); else ctx.fillRect(lx, yy - 6, 12, 12);
      tx += 18;
    }
    ctx.fillStyle = "#333333";
    ctx.font = "500 12px Montserrat, sans-serif";
    ctx.fillText(it.label, tx, yy);
  }
  ctx.fillStyle = "#666666";
  ctx.font = "500 10px Montserrat, sans-serif";
  ctx.fillText(tr(`Fonte: ${PAIS.marca}${state.conn?.colecao ? ` · Coleção ${state.conn.colecao}` : ""} · plataforma.mapbiomas.org`), pad, cab + altura + rodape - 10, W);
  const blob = await new Promise((resolve) => cv.toBlob(resolve, "image/png"));
  if (blob) baixarArquivo(blob, `mapbiomas_fogo_${nomeSeguro(card.title).toLowerCase()}.png`);
}

function tabelaHtml(t) {
  if (!t.linhas.length) return `<p class="empty">Sem dados para o período selecionado.</p>`;
  const cel = (tag, v, i) => `<${tag}${i ? ` class="num"` : ""}>${escapeHtml(v)}</${tag}>`;
  return `<div class="tabela-wrap"><table class="tabela">
    <thead><tr>${t.colunas.map((c, i) => cel("th", c, i)).join("")}</tr></thead>
    <tbody>${t.linhas.map((l) => `<tr>${l.map((c, i) => cel("td", c, i)).join("")}</tr>`).join("")}</tbody>
    ${t.total ? `<tfoot><tr>${t.total.map((c, i) => cel("td", c, i)).join("")}</tr></tfoot>` : ""}
  </table></div>`;
}

function topcardsHtml(items) {
  if (!items.length) return `<p class="empty">Sem dados para o período selecionado.</p>`;
  return `<ol class="topcards">${items.map((c, k) => `
    <li>
      <span class="tc-rank">${k + 1}º</span>
      <strong class="tc-title">${escapeHtml(c.title)}</strong>
      <span class="tc-value">${escapeHtml(c.value)}</span>
      ${c.note ? `<span class="tc-note">${escapeHtml(c.note)}</span>` : ""}
    </li>`).join("")}</ol>`;
}

function chartBoxHtml(chart, title) {
  if (!(chart.values || chart.labels || []).length) return `<p class="empty">Sem dados para o período selecionado.</p>`;
  const altura = chart.height || (chart.kind === "hbar" ? Math.max(220, chart.labels.length * 34 + 40) : null);
  return `<div class="chart-box"${altura ? ` style="height:${altura}px"` : ""}><canvas role="img" aria-label="${escapeHtml(title)}"></canvas></div>`;
}

function legendRowHtml(items) {
  return `<ul class="legend-row">${items.map((it) => (it.head
    ? `<li class="legend-head">${escapeHtml(it.label)}</li>`
    : it.note
    ? `<li class="legend-note">${escapeHtml(it.label)}</li>`
    : `
    <li><i class="${it.line ? `ln${it.dash ? " dash" : ""}` : "sw"}" style="--c:${it.color}"></i>${escapeHtml(it.label)}</li>`)).join("")}</ul>`;
}

function rankingHtml({ rows, total, plain = false }) {
  if (!rows.length) return `<p class="empty">Sem dados para o período selecionado.</p>`;
  const max = Math.max(...rows.map((r) => r.value)) || 1;
  const items = rows.map((r, i) => `
    <li>
      ${plain ? "" : `<span class="rank">${i + 1}</span>`}
      <div>
        <div class="r-top">
          <span class="r-label"><i class="sw" style="--c:${r.color}"></i><span>${escapeHtml(r.label)}</span></span>
          <span class="r-value">${r.valueText ? escapeHtml(r.valueText) : ha(r.value)}${total ? `<small>${pct(safeDiv(r.value, total))}</small>` : ""}</span>
        </div>
        <div class="r-track"><span style="width:${(100 * r.value / max).toFixed(1)}%;--c:${r.color}"></span></div>
        ${r.detail ? `<div class="r-detail">${escapeHtml(r.detail)}</div>` : ""}
      </div>
    </li>`).join("");
  return `<ol class="ranking${plain ? " plain" : ""}">${items}</ol>`;
}

// Barras empilhadas 100% em HTML/CSS (uma por linha), com % dentro dos segmentos que cabem
// e uma legenda com área e participação embaixo.
function propbarsHtml({ rows, legend }) {
  const comDados = rows.filter((r) => sum(r.segments.map((s) => s.value)) > 0);
  if (!comDados.length) return `<p class="empty">Sem área queimada no período selecionado.</p>`;
  const barras = comDados.map((r) => {
    const total = sum(r.segments.map((s) => s.value));
    const segs = r.segments.filter((s) => s.value > 0).map((s) => {
      const p = safeDiv(s.value, total);
      const texto = r.nomes && p >= 0.18 ? `${s.label} · ${pct(p)}` : p >= 0.06 ? pct(p) : "";
      return `<span style="flex:${p} 1 0;--c:${s.color};--fg:${textOn(s.color)}" title="${escapeHtml(`${s.label}: ${pct(p)} · ${ha(s.value)}`)}">${escapeHtml(texto)}</span>`;
    }).join("");
    return `${r.label ? `<p class="propbar-label">${escapeHtml(r.label)}</p>` : ""}<div class="propbar" role="img" aria-label="${escapeHtml(r.label || "Proporção por classe")}">${segs}</div>`;
  }).join("");
  const itens = legend || comDados[comDados.length - 1].segments;
  const totalLegenda = sum(itens.map((s) => s.value));
  const lista = itens.filter((s) => s.value > 0).map((s) => `
    <li>
      <i class="sw" style="--c:${s.color}"></i>
      <span>${escapeHtml(s.label)}</span>
      <span class="l-value">${ha(s.value)}</span>
      <span class="l-share">${pct(safeDiv(s.value, totalLegenda))}</span>
    </li>`).join("");
  return `<div class="propbars">${barras}<ul class="legend-list">${lista}</ul></div>`;
}

function doughnutHtml({ slices, centerLabel = "total" }) {
  if (!slices.length) return `<p class="empty">Sem área queimada no período selecionado.</p>`;
  const total = sum(slices.map((s) => s.value));
  const legend = slices.map((s) => `
    <li>
      <i class="sw" style="--c:${s.color}"></i>
      <span>${escapeHtml(s.label)}</span>
      <span class="l-value">${ha(s.value)}</span>
      <span class="l-share">${pct(safeDiv(s.value, total))}</span>
    </li>`).join("");
  return `
    <div class="donut">
      <div class="donut-chart">
        <canvas role="img" aria-label="Distribuição por classe"></canvas>
        <div class="donut-center"><strong>${total >= 1e6 ? mha(total) : ha(total)}</strong><span>${escapeHtml(centerLabel)}</span></div>
      </div>
      <ul class="legend-list">${legend}</ul>
    </div>`;
}

// ---------- Mapa (Leaflet) ----------

function renderMap(box, spec) {
  const dark = document.documentElement.dataset.theme === "dark" ||
    (document.documentElement.dataset.theme !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
  const tokens = themeTokens();
  // Zoom inteiro: em zoom fracionário os tiles do basemap são escalados e deixam costuras.
  // preferCanvas: vetores em canvas (podem ir para o PNG baixado); tiles com crossOrigin pelo mesmo motivo.
  const map = L.map(box, { zoomSnap: 1, attributionControl: true, preferCanvas: true });
  state.map = map;
  L.control.scale({ imperial: false, position: "bottomright" }).addTo(map);

  // Basemap sem chave de API (o da CARTO passou a exigir chave).
  const canvas = dark ? "World_Dark_Gray_Base" : "World_Light_Gray_Base";
  L.tileLayer(`https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/${canvas}/MapServer/tile/{z}/{y}/{x}`, {
    maxZoom: 16,
    crossOrigin: "anonymous",
    attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap",
  }).addTo(map);

  if (spec.dinamico) return renderMapaDinamico(map, spec.dinamico, tokens, box);
  if (spec.choropleth) return renderChoropleth(map, spec.choropleth, tokens);

  const atribuicao = `MapBiomas Fogo · Coleção ${escapeHtml(state.conn.colecao || "?")}`;
  const principal = L.tileLayer(spec.tileUrl, { maxZoom: 18, opacity: 0.95, crossOrigin: "anonymous", attribution: atribuicao }).addTo(map);
  controleNoMapa(map, "topright", `<button type="button" class="md-botao">⬇ Baixar mapa (PNG)</button>`, (div) => {
    div.querySelector("button").addEventListener("click", () => {
      const card = box.closest(".card");
      baixarMapaPng(map, box, card?.querySelector("h2")?.textContent?.replace("ⓘ", "").trim() || "Mapa", card?.querySelector(".legend-row"));
    });
  });
  if (spec.modo === "sobrepor" && spec.tileUrl2) {
    const deCima = L.tileLayer(spec.tileUrl2, { maxZoom: 18, opacity: 0.8, crossOrigin: "anonymous" }).addTo(map);
    controleNoMapa(map, "bottomleft", `
      <label class="map-ctl-label">Camada de cima: transparência
        <input type="range" min="0" max="100" value="80" aria-label="Opacidade da camada de cima">
      </label>`, (div) => {
      div.querySelector("input").addEventListener("input", (e) => deCima.setOpacity(Number(e.target.value) / 100));
    });
  }
  if (spec.modo === "cortina" && spec.tileUrl2) {
    // Esquerda = ano A (tileUrl2), direita = ano final (principal), como o plugin leaflet-side-by-side.
    const esquerda = L.tileLayer(spec.tileUrl2, { maxZoom: 18, opacity: 0.95, crossOrigin: "anonymous" }).addTo(map);
    montarCortina(map, box, esquerda, principal, spec.anoA, spec.anoB);
  }

  // Contornos clicáveis (preenchimento invisível): o clique destaca o contorno e mostra o nome.
  const contornos = (spec.contornos || []).filter(Boolean);
  if (contornos.length) {
    let destaque = null;
    const camadaContornos = L.geoJSON({ type: "FeatureCollection", features: contornos }, {
      style: { color: tokens.muted, weight: 0.8, fill: true, fillOpacity: 0 },
      onEachFeature: (f, layer) => {
        if (f.properties?.name) layer.bindTooltip(escapeHtml(f.properties.name), { sticky: true });
        layer.on("click", () => {
          if (destaque) map.removeLayer(destaque);
          const mesmo = f.properties?.id ? contornos.filter((g) => g.properties?.id === f.properties.id) : [f];
          const fc = { type: "FeatureCollection", features: mesmo };
          destaque = L.layerGroup([
            L.geoJSON(fc, { interactive: false, style: { color: "#ffffff", weight: 6, opacity: 0.9, fill: false } }),
            L.geoJSON(fc, { interactive: false, style: { color: tokens.text, weight: 2.5, fill: false } }),
          ]).addTo(map);
          const b = L.geoJSON(fc).getBounds();
          if (b.isValid()) map.fitBounds(b, { padding: [40, 40], maxZoom: 10 });
        });
      },
    }).addTo(map);
    camadaContornos.bringToBack();
  }

  let bounds = null;
  if (spec.ativo && spec.destacar) {
    // Escurece tudo fora do recorte (bioma, estado ou município filtrado).
    try {
      const alvo = spec.ativo.type === "Feature" ? spec.ativo : { type: "Feature", properties: {}, geometry: spec.ativo };
      const mascara = turf.mask(alvo);
      L.geoJSON(mascara, {
        interactive: false,
        style: { stroke: false, fillColor: tokens.surface, fillOpacity: 0.75 },
      }).addTo(map);
    } catch (err) {
      console.warn("[mapa] não foi possível destacar o recorte:", err);
    }
  }
  if (spec.ativo) {
    const ativo = L.geoJSON(spec.ativo, {
      interactive: false,
      style: { color: tokens.text, weight: 2, fill: false },
    }).addTo(map);
    bounds = ativo.getBounds();
  }
  if (bounds && bounds.isValid()) map.fitBounds(bounds, { padding: [16, 16] });
  else map.setView([-14.5, -52], 4);
}

// ---------- Mapa dinâmico (estados/municípios) ----------
// Tudo roda no navegador com o ranking anual já carregado: trocar indicador, ano, estilo
// ou território não faz novas consultas à API.
//   1. O que mostrar: área queimada (ha) · % do território · mudança entre dois períodos
//   2. Quando: período todo ou um ano (com ▶ para animar) / ano que divide os dois períodos
//   3. Desenho: cores no território (coroplético) ou bolhas proporcionais · baixar CSV
//   Clique num território: quadro com série anual, números-resumo e (estados) atalho para filtrar.

const DIV_RAMP = ["#2a78d6", "#9ec5f0", "#d9dbe0", "#f5a99c", "#d7301f"]; // azul = menos fogo, vermelho = mais
const haCurto = (v) => (Math.abs(v) >= 1e6 ? mha(v) : ha(v));

function classesPorQuantis(valores, fmt) {
  const pos = valores.filter((v) => v > 0).sort((a, b) => a - b);
  if (!pos.length) return { colorOf: () => null, legend: [] };
  const q = (p) => pos[Math.min(pos.length - 1, Math.floor(p * pos.length))];
  const cortes = [q(0.2), q(0.4), q(0.6), q(0.8)];
  const colorOf = (v) => {
    if (!(v > 0)) return null;
    const i = cortes.findIndex((c) => v <= c);
    return CHORO_RAMP[i < 0 ? CHORO_RAMP.length - 1 : i];
  };
  const lim = [pos[0], ...cortes, pos[pos.length - 1]];
  return { colorOf, legend: CHORO_RAMP.map((color, i) => ({ label: `${fmt(lim[i])} a ${fmt(lim[i + 1])}`, color })) };
}

function classesDivergentes(valores, fmt) {
  const abs = valores.map(Math.abs).filter((v) => v > 0).sort((a, b) => a - b);
  if (!abs.length) return { colorOf: () => DIV_RAMP[2], legend: [{ label: "Sem mudança", color: DIV_RAMP[2] }] };
  const q = (p) => abs[Math.min(abs.length - 1, Math.floor(p * abs.length))];
  const t1 = q(0.3), t2 = q(0.75);
  const colorOf = (v) => (v >= t2 ? DIV_RAMP[4] : v > t1 ? DIV_RAMP[3] : v <= -t2 ? DIV_RAMP[0] : v < -t1 ? DIV_RAMP[1] : DIV_RAMP[2]);
  return {
    colorOf,
    legend: [
      { label: `Queimou bem menos (${fmt(-t2)} ou menos)`, color: DIV_RAMP[0] },
      { label: "Queimou menos", color: DIV_RAMP[1] },
      { label: `Quase igual (${fmt(-t1)} a ${fmt(t1)})`, color: DIV_RAMP[2] },
      { label: "Queimou mais", color: DIV_RAMP[3] },
      { label: `Queimou bem mais (${fmt(t2)} ou mais)`, color: DIV_RAMP[4] },
    ],
  };
}

function renderMapaDinamico(map, spec, tokens, box) {
  const wrap = box.closest(".md-wrap");
  const root = box.closest("[data-card]");
  const ctlBox = wrap.querySelector(".md-controles");
  const ajuda = wrap.querySelector(".md-ajuda");
  const legBox = wrap.querySelector(".md-legenda");
  const detBox = wrap.querySelector(".md-detalhe-box");
  const { anos, territorios, unidade } = spec;
  const n = anos.length;
  const ano0 = anos[0], anoN = anos[n - 1];
  const nivelUnid = { biomas: "bioma", estados: "estado" }[unidade] || "municipio";
  const nomeUnid = rot(nivelUnid).toLowerCase();
  const unidPlural = rot(nivelUnid, true).toLowerCase();
  const vazio = resolveColor("var(--surface-muted)");
  // comp: "anos" = um ano em relação a outro; "media" = um ano em relação à média do período.
  const st = {
    indicador: "area", periodo: "todo", comp: "anos",
    ano: anoN, anoA: anos[Math.max(0, n - 2)], anoB: anoN, anoM: anoN, sel: null, timer: null,
  };
  const idx = (a) => a - ano0;
  const mediaDe = (t) => somaFaixa(t.porAno, 0, n - 1) / n;

  const somaFaixa = (a, de, ate) => { let s = 0; for (let i = de; i <= ate; i++) s += a[i] || 0; return s; };

  // Área de cada território (ha) pelo contorno do mapa: calculada só quando pedida.
  let areas = null;
  function areaDe(id) {
    if (!areas) {
      areas = new Map();
      for (const f of spec.features) {
        try { areas.set(f.properties.id, (areas.get(f.properties.id) || 0) + turf.area(f) / 1e4); } catch { /* geometria inválida */ }
      }
    }
    return areas.get(id) || 0;
  }

  // Valor do território no indicador atual (anoIdx: índice do ano; null = período todo).
  function valor(id, anoIdx = st.periodo === "ano" ? st.ano - ano0 : null) {
    const t = territorios.get(id);
    if (!t) return 0;
    if (st.indicador === "mudanca") {
      return st.comp === "media"
        ? (t.porAno[idx(st.anoM)] || 0) - mediaDe(t)
        : (t.porAno[idx(st.anoB)] || 0) - (t.porAno[idx(st.anoA)] || 0);
    }
    const v = anoIdx == null ? somaFaixa(t.porAno, 0, n - 1) : t.porAno[anoIdx] || 0;
    if (st.indicador === "pct") {
      const a = areaDe(id);
      return a ? v / a : 0;
    }
    return v;
  }
  const sinal = (v) => (v >= 0 ? "+" : "−");
  const fmtCurto = (v) => (st.indicador === "pct" ? pct(v)
    : st.indicador === "mudanca" ? `${sinal(v)}${haCurto(Math.abs(v))}` : haCurto(v));
  const fmtLongo = (v) => (st.indicador === "pct" ? pct(v)
    : st.indicador === "mudanca" ? `${sinal(v)}${ha(Math.abs(v))}` : ha(v));
  const compRotulo = () => (st.comp === "media" ? `${st.anoM} × média ${ano0}–${anoN}` : `${st.anoB} × ${st.anoA}`);

  let classes = { colorOf: () => null, legend: [] };
  let maxAbs = 0;
  let cores = new Map();
  function recalcular() {
    const ids = [...territorios.keys()].filter((id) => !territorios.get(id).emCalculo);
    let pool;
    if (st.indicador !== "mudanca" && st.periodo === "ano") {
      // Mesma escala para todos os anos: a animação compara anos de verdade.
      pool = [];
      for (const id of ids) for (let i = 0; i < n; i++) pool.push(valor(id, i));
    } else {
      pool = ids.map((id) => valor(id));
    }
    classes = st.indicador === "mudanca" ? classesDivergentes(pool, fmtCurto) : classesPorQuantis(pool, fmtCurto);
    maxAbs = pool.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    cores = new Map();
    for (const [id, t] of territorios) cores.set(id, t.emCalculo ? spec.corEmCalculo : classes.colorOf(valor(id)));
  }

  function estilo(f) {
    const cor = cores.get(f.properties.id) || vazio;
    const borda = spec.bordas ? { color: tokens.surface, weight: 1 } : { color: cor, weight: 0.4 };
    return { fillColor: cor, fillOpacity: 0.9, ...borda };
  }

  function dica(id, nota) {
    const t = territorios.get(id);
    const quando = st.indicador === "mudanca" ? compRotulo()
      : st.periodo === "ano" ? `em ${st.ano}` : `soma ${ano0}–${anoN}`;
    const v = t?.emCalculo ? "Calculando a área da interseção…" : `${fmtLongo(valor(id))} <small>(${quando})</small>`;
    return `<strong>${escapeHtml(t?.name || "")}</strong><br>${v}${nota ? `<br><small>${escapeHtml(nota)}</small>` : ""}<br><small>Clique para ver o detalhe</small>`;
  }

  const poligonos = L.geoJSON({ type: "FeatureCollection", features: spec.features }, {
    renderer: L.canvas(),
    style: estilo,
    onEachFeature: (f, layer) => {
      const id = f.properties.id;
      layer.bindTooltip(() => dica(id, f.properties.nota), { sticky: true });
      layer.on("mouseover", () => layer.setStyle({ weight: 2, color: tokens.text }));
      layer.on("mouseout", () => poligonos.resetStyle(layer));
      layer.on("click", () => selecionar(id));
    },
    attribution: `MapBiomas Fogo · Coleção ${escapeHtml(state.conn.colecao || "?")} · geometrias da API MapBiomas`,
  }).addTo(map);

  if (spec.contorno) {
    L.geoJSON(spec.contorno, { interactive: false, style: { color: tokens.text, weight: 1.5, fill: false } }).addTo(map);
  }

  // Contorno do território clicado: linha escura com halo claro, por cima de tudo.
  let destaque = null;
  function destacar(id) {
    if (destaque) map.removeLayer(destaque);
    destaque = null;
    if (!id) return;
    const pedacos = spec.features.filter((f) => f.properties.id === id);
    if (!pedacos.length) return;
    const fc = { type: "FeatureCollection", features: pedacos };
    destaque = L.layerGroup([
      L.geoJSON(fc, { interactive: false, style: { color: "#ffffff", weight: 6, opacity: 0.9, fill: false } }),
      L.geoJSON(fc, { interactive: false, style: { color: tokens.text, weight: 2.5, fill: false } }),
    ]).addTo(map);
    const b = L.geoJSON(fc).getBounds();
    if (b.isValid()) map.fitBounds(b, { padding: [40, 40], maxZoom: 9 });
  }

  function desenhar() {
    recalcular();
    poligonos.setStyle(estilo);
    atualizarTextos();
    atualizarControles();
  }

  function tituloIndicador() {
    if (st.indicador === "mudanca") return `Diferença na área queimada: ${compRotulo()}`;
    const quando = st.periodo === "ano" ? `em ${st.ano}` : `${ano0}–${anoN} (soma dos anos)`;
    return st.indicador === "pct" ? `% do território que queimou ${quando}` : `Área queimada ${quando}`;
  }

  function atualizarTextos() {
    const periodoTxt = st.periodo === "ano" ? `em ${st.ano}` : `de ${ano0} a ${anoN} (soma dos anos)`;
    let s;
    if (st.indicador === "area") {
      s = `Mostra quantos hectares queimaram em cada ${nomeUnid} ${periodoTxt}. Quanto mais escuro, mais queimou.`;
    } else if (st.indicador === "pct") {
      s = `Mostra que parte de cada ${nomeUnid} queimou ${periodoTxt} (área queimada ÷ área do ${nomeUnid}), ` +
        `para comparar ${unidPlural} pequenos e grandes.` +
        (st.periodo === "todo" ? " Na soma dos anos pode passar de 100%, porque o mesmo lugar pode queimar mais de uma vez." : "") +
        " A área do território vem do contorno do mapa (valor aproximado).";
    } else if (st.comp === "media") {
      s = `Compara a área queimada em ${st.anoM} com a média anual de ${ano0}–${anoN}. ` +
        `Vermelho = ${st.anoM} queimou mais que a média; azul = queimou menos.`;
    } else {
      s = `Compara a área queimada em ${st.anoB} com a de ${st.anoA} (${st.anoB} menos ${st.anoA}). ` +
        `Vermelho = ${st.anoB} queimou mais; azul = queimou menos.`;
    }
    if (st.periodo === "ano" && st.indicador !== "mudanca") s += " Aperte ▶ para ver ano a ano: as cores usam a mesma escala em todos os anos.";
    ajuda.textContent = `${s} Clique num ${nomeUnid} para ver o contorno e o detalhe.`;

    const itens = [{ label: tituloIndicador(), head: true }, ...classes.legend];
    if (st.indicador !== "mudanca") itens.push({ label: "Sem área queimada", color: "var(--surface-muted)" });
    if (spec.emCalculo) itens.push({ label: "Na divisa · área exata em cálculo", color: spec.corEmCalculo });
    legBox.innerHTML = legendRowHtml(itens);
  }

  // ----- Controles (passos 1, 2 e 3) -----
  const seg = (grupo, opcoes) => `
    <div class="segmented" role="group" data-g="${grupo}">
      ${opcoes.map(([v, r]) => `<button type="button" data-v="${v}" aria-pressed="false">${escapeHtml(r)}</button>`).join("")}
    </div>`;
  // Busca por nome (só com mais de 20 territórios): nome exibido -> id (nomes repetidos ganham o id entre parênteses).
  const comBusca = territorios.size > 20;
  const semAcento = (s) => String(s).normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().trim();
  const nomesBusca = new Map();
  const contagemNomes = new Map();
  for (const t of territorios.values()) contagemNomes.set(t.name, (contagemNomes.get(t.name) || 0) + 1);
  for (const [id, t] of territorios) nomesBusca.set(contagemNomes.get(t.name) > 1 ? `${t.name} (${id})` : t.name, id);
  const indicadores = [["area", "Área queimada (ha)"], ["pct", "% do território"], ...(n >= 2 ? [["mudanca", "Comparar anos"]] : [])];
  const selAno = (r, v, rotulo) => `<select class="md-select" data-r="${r}" aria-label="${rotulo}">${anos.map((a) => `<option value="${a}"${a === v ? " selected" : ""}>${a}</option>`).join("")}</select>`;
  ctlBox.innerHTML = `
    <div class="md-passo"><span class="md-num">1</span><span class="md-rotulo">O que mostrar</span>${seg("indicador", indicadores)}</div>
    <div class="md-passo" data-quando="area pct"><span class="md-num">2</span><span class="md-rotulo">Quando</span>
      ${seg("periodo", [["todo", `Período todo (${ano0}–${anoN})`], ["ano", "Um ano"]])}
      <span class="md-ano" data-quando-periodo>
        <button type="button" class="md-play" aria-label="Animar ano a ano">▶</button>
        <input type="range" min="${ano0}" max="${anoN}" step="1" value="${st.ano}" data-r="ano" aria-label="Ano">
        <output class="md-valor" data-o="ano"></output>
      </span>
    </div>
    <div class="md-passo" data-quando="mudanca"><span class="md-num">2</span><span class="md-rotulo">Comparar</span>
      ${seg("comp", [["anos", "Um ano × outro ano"], ["media", "Um ano × média histórica"]])}
      <span class="md-ano" data-quando-comp="anos">${selAno("anoB", st.anoB, "Ano")}<span>em relação a</span>${selAno("anoA", st.anoA, "Ano de referência")}</span>
      <span class="md-ano" data-quando-comp="media">${selAno("anoM", st.anoM, "Ano")}<span>em relação à média ${ano0}–${anoN}</span></span>
    </div>
    ${comBusca ? `<div class="md-passo"><span class="md-num">3</span><span class="md-rotulo">Buscar</span>
      <input type="search" class="md-busca" list="md-busca-lista-${root.dataset.card}" placeholder="Digite o nome do ${nomeUnid}…" aria-label="Buscar ${nomeUnid} pelo nome" autocomplete="off">
      <datalist id="md-busca-lista-${root.dataset.card}">${[...nomesBusca.keys()].sort((a, b) => a.localeCompare(b, IDIOMA.locale)).map((nm) => `<option value="${escapeHtml(nm)}"></option>`).join("")}</datalist>
      <span class="md-busca-msg" aria-live="polite"></span>
    </div>` : ""}
    <div class="md-passo"><span class="md-num">${comBusca ? 4 : 3}</span><span class="md-rotulo">Baixar</span>
      <button type="button" class="md-botao md-png">⬇ Baixar mapa (PNG)</button>
      <button type="button" class="md-botao md-csv" title="Planilha com o valor de cada ${nomeUnid} e a série anual">⬇ Baixar dados (CSV)</button>
    </div>`;

  function atualizarControles() {
    ctlBox.querySelectorAll("[data-g]").forEach((g) => g.querySelectorAll("button").forEach((b) => {
      b.setAttribute("aria-pressed", String(st[g.dataset.g] === b.dataset.v));
    }));
    ctlBox.querySelectorAll("[data-quando]").forEach((d) => { d.hidden = !d.dataset.quando.split(" ").includes(st.indicador); });
    ctlBox.querySelector("[data-quando-periodo]").hidden = st.periodo !== "ano";
    ctlBox.querySelector('[data-r="ano"]').value = String(st.ano);
    ctlBox.querySelector('[data-o="ano"]').textContent = String(st.ano);
    ctlBox.querySelectorAll("[data-quando-comp]").forEach((d) => { d.hidden = d.dataset.quandoComp !== st.comp; });
    const play = ctlBox.querySelector(".md-play");
    play.textContent = st.timer ? "⏸" : "▶";
    play.setAttribute("aria-label", st.timer ? "Pausar animação" : "Animar ano a ano");
  }

  function pararPlay() {
    if (!st.timer) return;
    clearInterval(st.timer);
    st.timer = null;
    if (box.isConnected) atualizarControles();
  }
  function alternarPlay() {
    if (st.timer) return pararPlay();
    if (st.ano >= anoN) st.ano = ano0;
    st.timer = setInterval(() => {
      if (!box.isConnected || st.ano >= anoN) return pararPlay();
      st.ano += 1;
      desenhar();
    }, 900);
    desenhar();
  }

  ctlBox.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    if (b.classList.contains("md-play")) return alternarPlay();
    if (b.classList.contains("md-csv")) return baixarCsv();
    if (b.classList.contains("md-png")) return baixarMapaPng(map, box, tituloIndicador(), legBox);
    const g = b.closest("[data-g]")?.dataset.g;
    if (!g || st[g] === b.dataset.v) return;
    pararPlay();
    st[g] = b.dataset.v;
    desenhar();
  });
  ctlBox.querySelector('[data-r="ano"]').addEventListener("input", (e) => { pararPlay(); st.ano = Number(e.target.value); desenhar(); });
  // Busca: nome exato da lista; senão, sem acento, o primeiro que começa com (ou contém) o texto.
  const busca = ctlBox.querySelector(".md-busca");
  const buscaMsg = ctlBox.querySelector(".md-busca-msg");
  function buscar(final) {
    if (!busca) return;
    const txt = busca.value.trim();
    buscaMsg.textContent = "";
    if (!txt) return;
    let id = nomesBusca.get(txt);
    if (!id && final) {
      const alvo = semAcento(txt);
      const nomes = [...nomesBusca.keys()];
      const achado = nomes.find((nm) => semAcento(nm) === alvo) || nomes.find((nm) => semAcento(nm).startsWith(alvo))
        || nomes.find((nm) => semAcento(nm).includes(alvo));
      if (achado) { id = nomesBusca.get(achado); busca.value = achado; }
      else buscaMsg.textContent = "Nenhum território com esse nome neste mapa.";
    }
    if (id && id !== st.sel) selecionar(id);
  }
  busca?.addEventListener("input", () => buscar(false));
  busca?.addEventListener("change", () => buscar(true));
  busca?.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); buscar(true); } });
  ctlBox.querySelectorAll("select[data-r]").forEach((s) => s.addEventListener("change", () => {
    st[s.dataset.r] = Number(s.value);
    desenhar();
  }));

  // ----- Detalhe do território clicado (contorno em destaque + quadro abaixo) -----
  function selecionar(id) {
    st.sel = id;
    destacar(id);
    mostrarDetalhe(id);
  }

  function mostrarDetalhe(id) {
    const t = territorios.get(id);
    if (!t) return;
    const total = somaFaixa(t.porAno, 0, n - 1);
    const pico = t.porAno.reduce((best, v, i) => (v > t.porAno[best] ? i : best), 0);
    const comFogo = t.porAno.filter((v) => v > 0).length;
    const areaT = unidade !== "municipios" ? areaDe(id) : 0;
    const filtroEl = unidade === "biomas" ? el.bioma : el.estado;
    const podeFiltrar = unidade !== "municipios" && filtroEl.value !== id && [...filtroEl.options].some((o) => o.value === id);
    detBox.hidden = false;
    detBox.innerHTML = `
      <div class="md-det-cab">
        <h3>${escapeHtml(t.name)}</h3>
        <div class="md-det-acoes">
          ${podeFiltrar ? `<button type="button" class="md-botao" data-a="filtrar">Filtrar o painel por ${escapeHtml(nomeUnid)}</button>` : ""}
          <button type="button" class="md-botao" data-a="fechar" aria-label="Fechar detalhe">✕ Fechar</button>
        </div>
      </div>
      <ul class="md-det-kpis">
        <li><span>Soma ${ano0}–${anoN}</span><strong>${ha(total)}</strong></li>
        <li><span>Média por ano</span><strong>${ha(total / n)}</strong></li>
        <li><span>Ano que mais queimou</span><strong>${total ? `${anos[pico]} · ${ha(t.porAno[pico])}` : "—"}</strong></li>
        <li><span>Anos com fogo</span><strong>${comFogo} de ${n}</strong></li>
        ${areaT ? `<li><span>Área (aprox.)</span><strong>${ha(areaT)}</strong></li>` : ""}
      </ul>
      <div class="chart-box" style="height:240px"><canvas role="img" aria-label="Área queimada por ano em ${escapeHtml(t.name)}"></canvas></div>
      <ul class="legend-row">
        <li><i class="sw" style="--c:${CHORO_RAMP[3]}"></i>Área queimada no ano</li>
        <li><i class="ln dash" style="--c:var(--text-2)"></i>Média ${ano0}–${anoN}</li>
      </ul>`;
    descartarGraficos(root);
    registrarGrafico(root, new Chart(detBox.querySelector("canvas"), chartConfig({
      kind: "bar", labels: anos.map(String), values: t.porAno, color: CHORO_RAMP[3], meanLine: total / n, axisTitle: "Área queimada (ha)",
    }, tokens)));
    detBox.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "nearest" });
  }

  detBox.addEventListener("click", (e) => {
    const acao = e.target.closest("[data-a]")?.dataset.a;
    if (acao === "fechar") {
      detBox.hidden = true;
      descartarGraficos(root);
      st.sel = null;
      destacar(null);
    } else if (acao === "filtrar" && st.sel) {
      pararPlay();
      const alvo = unidade === "biomas" ? el.bioma : el.estado;
      alvo.value = st.sel;
      alvo.dispatchEvent(new Event("change"));
    }
  });

  // ----- CSV (separador ";" e vírgula decimal, abre direto no Excel em português) -----
  function baixarCsv() {
    const num = (v) => String(Math.round(v * 100) / 100).replace(".", ",");
    const sufixo = st.indicador === "mudanca" ? (st.comp === "media" ? `${st.anoM}_menos_media_${ano0}-${anoN}` : `${st.anoB}_menos_${st.anoA}`)
      : st.periodo === "ano" ? String(st.ano) : `${ano0}-${anoN}`;
    const nomeInd = { area: "area_queimada_ha", pct: "pct_do_territorio", mudanca: "diferenca_area_queimada_ha" }[st.indicador];
    const cab = [nomeUnid, `${nomeInd}_${sufixo}`, `soma_${ano0}-${anoN}_ha`, ...anos.map((a) => `${a}_ha`)];
    const linhasCsv = [...territorios.entries()]
      .sort((a, b) => a[1].name.localeCompare(b[1].name, "pt-BR"))
      .map(([id, t]) => [
        t.name,
        num(st.indicador === "pct" ? valor(id) * 100 : valor(id)),
        num(somaFaixa(t.porAno, 0, n - 1)),
        ...t.porAno.map(num),
      ]);
    const celula = (c) => (/[;"\n]/.test(c) ? `"${c.replace(/"/g, '""')}"` : c);
    const csv = "﻿" + [cab, ...linhasCsv].map((l) => l.map(celula).join(";")).join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `mapbiomas_fogo_${unidade}_${nomeInd}_${sufixo}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  const bounds = poligonos.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds, { padding: [12, 12] });
  else map.setView([-14.5, -52], 4);
  desenhar();
}

// ---------- Baixar o mapa (PNG) ----------
// Copia o que está na tela: tiles (basemap e camadas, com transparência e cortina), vetores
// (canvas do Leaflet), e acrescenta título, legenda, barra de escala e fonte.

// Barra de escala: comprimento "redondo" (1, 2 ou 5 × 10^n) que caiba em ~120 px no centro do mapa.
function escalaDoMapa(map, largura = 120) {
  const s = map.getSize();
  const a = map.containerPointToLatLng([s.x / 2 - 50, s.y / 2]);
  const b = map.containerPointToLatLng([s.x / 2 + 50, s.y / 2]);
  const mpp = map.distance(a, b) / 100;
  const alvo = mpp * largura;
  const pot = 10 ** Math.floor(Math.log10(alvo));
  const metros = [1, 2, 5, 10].map((k) => k * pot).filter((v) => v <= alvo).pop() || pot;
  return { px: metros / mpp, rotulo: metros >= 1000 ? `${fmtFull.format(metros / 1000)} km` : `${fmtFull.format(metros)} m` };
}

function itensDaLegenda(el) {
  return [...(el?.querySelectorAll("li") || [])].map((li) => {
    const i = li.querySelector("i");
    const cor = i ? (i.style.getPropertyValue("--c") || i.style.background) : null;
    return { label: li.textContent.trim(), head: li.classList.contains("legend-head"), note: li.classList.contains("legend-note"), color: cor ? resolveColor(cor.trim()) : null, linha: i?.classList.contains("ln") };
  }).filter((x) => x.label);
}

async function baixarMapaPng(map, box, titulo, legendaEl) {
  const W = box.clientWidth, H = box.clientHeight;
  const pad = 16, cab = 58, esc = 2;
  const medir = document.createElement("canvas").getContext("2d");
  medir.font = "500 12px Montserrat, sans-serif";
  const itens = itensDaLegenda(legendaEl);
  // Legenda em linhas que cabem na largura.
  let x = pad, linha = 0;
  const pos = itens.map((it) => {
    const w = (it.color ? 18 : 0) + medir.measureText(it.label).width;
    if (it.head || x + w > W + pad && x > pad) { if (x > pad) linha++; x = pad; }
    const p = { it, x, linha };
    x = it.head ? W + pad : x + w + 16; // título da legenda ocupa a linha toda
    return p;
  });
  const linhas = itens.length ? linha + 1 : 0;
  const rodape = 14 + linhas * 20 + 26;
  const cv = Object.assign(document.createElement("canvas"), { width: (W + 2 * pad) * esc, height: (cab + H + rodape) * esc });
  const ctx = cv.getContext("2d");
  ctx.scale(esc, esc);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, W + 2 * pad, cab + H + rodape);
  ctx.fillStyle = "#1f1f1f";
  ctx.font = "700 16px Montserrat, sans-serif";
  ctx.fillText(tr(titulo), pad, 26, W);
  ctx.fillStyle = "#555555";
  ctx.font = "500 12px Montserrat, sans-serif";
  ctx.fillText(tr(`${PAIS.marca}${state.conn?.colecao ? ` · Coleção ${state.conn.colecao}` : ""} · ${activeTerritory().nome} · ${state.ini}–${state.fim}`), pad, 46, W);

  // Mapa: tiles e canvas do Leaflet nas posições da tela.
  const mr = box.getBoundingClientRect();
  ctx.save();
  ctx.beginPath();
  ctx.rect(pad, cab, W, H);
  ctx.clip();
  ctx.fillStyle = resolveColor("var(--surface-muted)") || "#eeeeee";
  ctx.fillRect(pad, cab, W, H);
  for (const img of box.querySelectorAll(".leaflet-tile-pane img")) {
    if (!img.complete || !img.naturalWidth) continue;
    const r = img.getBoundingClientRect();
    const camada = img.closest(".leaflet-layer");
    ctx.save();
    if (camada?.style.clip) {
      const m = camada.style.clip.match(/rect\(([-\d.]+)px,?\s*([-\d.]+)px,?\s*([-\d.]+)px,?\s*([-\d.]+)px\)/);
      if (m) {
        const o = camada.getBoundingClientRect();
        const [t, d, b, e] = m.slice(1).map(Number);
        ctx.beginPath();
        ctx.rect(pad + o.left - mr.left + e, cab + o.top - mr.top + t, d - e, b - t);
        ctx.clip();
      }
    }
    ctx.globalAlpha = Number(getComputedStyle(camada || img).opacity) || 1;
    ctx.drawImage(img, pad + r.left - mr.left, cab + r.top - mr.top, r.width, r.height);
    ctx.restore();
  }
  for (const c of box.querySelectorAll(".leaflet-pane canvas")) {
    const r = c.getBoundingClientRect();
    ctx.drawImage(c, pad + r.left - mr.left, cab + r.top - mr.top, r.width, r.height);
  }
  const linhaCortina = box.querySelector(".cortina-linha");
  if (linhaCortina) {
    const r = linhaCortina.getBoundingClientRect();
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(pad + r.left - mr.left, cab, 3, H);
  }
  ctx.restore();
  ctx.strokeStyle = "#d0d0d0";
  ctx.strokeRect(pad + 0.5, cab + 0.5, W - 1, H - 1);

  // Barra de escala (canto inferior esquerdo do mapa).
  const e = escalaDoMapa(map);
  const ex = pad + 12, ey = cab + H - 16;
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  ctx.fillRect(ex - 6, ey - 20, e.px + 12 + 60, 28);
  ctx.fillStyle = "#1f1f1f";
  ctx.fillRect(ex, ey, e.px, 3);
  ctx.fillRect(ex, ey - 6, 2, 9);
  ctx.fillRect(ex + e.px - 2, ey - 6, 2, 9);
  ctx.font = "600 11px Montserrat, sans-serif";
  ctx.fillText(e.rotulo, ex + e.px + 6, ey + 4);

  // Legenda e fonte.
  ctx.textBaseline = "middle";
  const y0 = cab + H + 14 + 10;
  for (const { it, x: lx, linha: l } of pos) {
    const yy = y0 + l * 20;
    let tx = lx;
    if (it.color) {
      ctx.fillStyle = it.color;
      if (it.linha) ctx.fillRect(lx, yy - 1, 12, 3); else ctx.fillRect(lx, yy - 6, 12, 12);
      tx += 18;
    }
    ctx.fillStyle = it.head ? "#1f1f1f" : "#333333";
    ctx.font = `${it.head ? 700 : 500} 12px Montserrat, sans-serif`;
    ctx.fillText(it.label, tx, yy);
  }
  ctx.fillStyle = "#666666";
  ctx.font = "500 10px Montserrat, sans-serif";
  ctx.fillText(tr(`Fonte: ${PAIS.marca} · plataforma.mapbiomas.org · mapa base Esri`), pad, cab + H + rodape - 12, W);

  try {
    const blob = await new Promise((resolve, reject) => cv.toBlob((b) => (b ? resolve(b) : reject(new Error("vazio"))), "image/png"));
    baixarArquivo(blob, `mapbiomas_fogo_mapa_${nomeSeguro(titulo).toLowerCase()}.png`);
  } catch (err) {
    console.error(err);
    setStatus(tr("O navegador bloqueou a leitura das imagens do mapa. Abra o painel pelo site publicado (mapbiomas.github.io) em vez do arquivo local."));
  }
}

// Controle HTML dentro do mapa (cliques e arrastes não movem o mapa).
function controleNoMapa(map, posicao, html, aoMontar) {
  const ctl = L.control({ position: posicao });
  ctl.onAdd = () => {
    const div = L.DomUtil.create("div", "map-ctl");
    div.innerHTML = html;
    L.DomEvent.disableClickPropagation(div);
    L.DomEvent.disableScrollPropagation(div);
    aoMontar?.(div);
    return div;
  };
  ctl.addTo(map);
  return ctl;
}

// Cortina: recorta a camada da esquerda e a da direita numa linha vertical que o usuário arrasta.
function montarCortina(map, box, esquerda, direita, anoA, anoB) {
  const linha = L.DomUtil.create("div", "cortina-linha", box);
  L.DomEvent.disableClickPropagation(linha);
  let frac = 0.5;
  controleNoMapa(map, "bottomleft", `
    <div class="cortina-ctl">
      <span>◀ ${anoA}</span>
      <input type="range" min="0" max="100" value="50" aria-label="Posição da cortina">
      <span>${anoB} ▶</span>
    </div>`, (div) => {
    div.querySelector("input").addEventListener("input", (e) => { frac = Number(e.target.value) / 100; aplicar(); });
  });
  function aplicar() {
    if (!map._loaded) return; // o mapa ainda não tem centro/zoom (o evento "load" chama de novo)
    const nw = map.containerPointToLayerPoint([0, 0]);
    const se = map.containerPointToLayerPoint(map.getSize());
    const x = nw.x + (se.x - nw.x) * frac;
    const ce = esquerda.getContainer(), cd = direita.getContainer();
    if (ce) ce.style.clip = `rect(${nw.y}px, ${x}px, ${se.y}px, ${nw.x}px)`;
    if (cd) cd.style.clip = `rect(${nw.y}px, ${se.x}px, ${se.y}px, ${x}px)`;
    linha.style.left = `${frac * 100}%`;
  }
  map.on("move zoomend resize load", aplicar);
  aplicar();
}

function renderChoropleth(map, spec, tokens) {
  const vazio = resolveColor("var(--surface-muted)");
  const camada = L.geoJSON({ type: "FeatureCollection", features: spec.features }, {
    renderer: L.canvas(),
    style: (f) => {
      const cor = f.properties.color || vazio;
      // Estados com borda clara; municípios sem borda (os tiles cortam os polígonos em pedaços).
      return spec.bordas
        ? { fillColor: cor, fillOpacity: 0.9, color: tokens.surface, weight: 1 }
        : { fillColor: cor, fillOpacity: 0.9, color: cor, weight: 0.4 };
    },
    onEachFeature: (f, layer) => {
      const valor = f.properties.emCalculo ? "Calculando a área da interseção…" : ha(f.properties.value);
      const nota = f.properties.nota ? `<br><small>${escapeHtml(f.properties.nota)}</small>` : "";
      layer.bindTooltip(`<strong>${escapeHtml(f.properties.name)}</strong><br>${valor}${nota}`, { sticky: true });
      layer.on("mouseover", () => layer.setStyle({ weight: 2, color: tokens.text }));
      layer.on("mouseout", () => camada.resetStyle(layer));
    },
    attribution: `MapBiomas Fogo · Coleção ${escapeHtml(state.conn.colecao || "?")} · geometrias da API MapBiomas`,
  }).addTo(map);

  if (spec.contorno) {
    L.geoJSON(spec.contorno, { interactive: false, style: { color: tokens.text, weight: 1.5, fill: false } }).addTo(map);
  }
  const bounds = camada.getBounds();
  if (bounds.isValid()) map.fitBounds(bounds, { padding: [12, 12] });
  else map.setView([-14.5, -52], 4);
}

// ---------- Configurações do Chart.js ----------

function themeTokens() {
  const css = getComputedStyle(document.documentElement);
  const v = (name) => css.getPropertyValue(name).trim();
  return {
    surface: v("--surface"), text: v("--text"), text2: v("--text-2"), muted: v("--text-muted"),
    grid: v("--grid"), axis: v("--axis"), brand: v("--brand"),
  };
}

// Cores dadas como token CSS ("var(--axis)") viram o valor atual do tema para o canvas.
const resolveColor = (color, _t) => {
  const m = /^var\((--[\w-]+)\)$/.exec(color || "");
  return m ? getComputedStyle(document.documentElement).getPropertyValue(m[1]).trim() : color;
};

function baseOptions(t) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: reducedMotion() ? 0 : 300 },
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: t.surface,
        titleColor: t.text,
        bodyColor: t.text2,
        footerColor: t.text,
        borderColor: t.axis,
        borderWidth: 1,
        padding: 10,
        cornerRadius: 8,
        boxWidth: 8,
        boxHeight: 8,
        boxPadding: 4,
        usePointStyle: true,
        titleFont: { size: 14, weight: "600" },
        bodyFont: { size: 13 },
      },
    },
  };
}

// Maior valor do gráfico (nos empilhados, o maior total de uma barra).
function maiorValorDoGrafico(spec) {
  const num = (v) => (Number.isFinite(+v) ? +v : 0);
  if (spec.values) return Math.max(0, ...spec.values.map(num));
  const series = spec.series || [spec.base, spec.top].filter(Boolean);
  const empilha = spec.kind === "stacked" || spec.kind === "combo";
  let m = 0;
  const n = Math.max(0, ...series.map((x) => x.values?.length || 0));
  for (let i = 0; i < n; i++) {
    const vs = series.map((x) => num(x.values?.[i]));
    m = Math.max(m, empilha ? sum(vs) : Math.max(0, ...vs));
  }
  return m;
}

function chartConfig(spec, t) {
  // Mha com valores abaixo de 1 milhão de ha fica com números minúsculos no eixo: passa para mil ha.
  if (spec.unit === "Mha" && !spec.percent && maiorValorDoGrafico(spec) < 1e6) {
    spec = { ...spec, unit: "mil", axisTitle: spec.axisTitle?.replace("(Mha)", "(mil ha)") };
  }
  const options = baseOptions(t);
  options.plugins.tabela = { percent: !!spec.percent }; // lido pela tabela sob demanda
  options.plugins.rotulosBarras = { formato: spec.rotulos || "" }; // "pct": % de cada barra escrita sobre ela
  if (spec.rotulos) options.layout = { padding: { top: 18, right: spec.kind === "hbar" ? 44 : 8 } }; // espaço para os rótulos
  const horizontal = spec.kind === "hbar";
  const stacked = spec.kind === "stacked" || spec.kind === "combo";
  const multi = stacked || spec.kind === "grouped";
  const valueOf = (ctx) => (horizontal ? ctx.parsed.x : ctx.parsed.y);
  const fmtValor = spec.percent ? (v) => pct(v)
    : spec.unit === "Mha" ? (v) => mha(v, 2)
    : spec.unit === "mil" ? (v) => ha(v)
    : (v) => `${fmtFull.format(v)} ha`;
  const fonte = spec.big ? 14 : 13;
  if (spec.big) {
    options.plugins.tooltip.titleFont = { size: 15, weight: "600" };
    options.plugins.tooltip.bodyFont = { size: 14 };
  }
  // Sem linhas de grade internas: só as linhas dos eixos X e Y.
  const categoryAxis = {
    stacked,
    grid: { display: false },
    border: { display: true, color: t.axis },
    ticks: { color: t.text2, maxRotation: 0, autoSkip: !horizontal, autoSkipPadding: 12, font: { size: fonte } },
  };
  const valueAxis = {
    stacked,
    beginAtZero: true,
    grid: { display: false },
    border: { display: true, color: t.axis },
    ticks: { color: t.text2, maxTicksLimit: 6, font: { size: fonte }, callback: (v) => (spec.percent ? pct(v) : fmtCompact.format(v)) },
  };
  if (spec.percent && stacked) valueAxis.max = 1;
  if (spec.unit === "Mha") {
    valueAxis.ticks.callback = (v) => fmtMhaEixo.format(v / 1e6);
  } else if (spec.unit === "mil") {
    valueAxis.ticks.callback = (v) => fmtMhaEixo.format(v / 1e3);
  }
  // Título padrão do eixo de valores quando o gráfico não define um.
  const tituloEixo = spec.axisTitle || (spec.percent ? "Participação (%)" : "Área queimada (ha)");
  if (tituloEixo) {
    valueAxis.title = { display: true, text: tituloEixo, color: t.text2, font: { size: fonte + 1, weight: "600" } };
  }
  if (spec.categoryTitle) {
    categoryAxis.title = { display: true, text: spec.categoryTitle, color: t.text2, font: { size: fonte + 1, weight: "600" } };
  }
  options.scales = horizontal ? { y: categoryAxis, x: valueAxis } : { x: categoryAxis, y: valueAxis };
  if (horizontal) {
    options.indexAxis = "y";
    options.interaction.axis = "y";
  }

  options.plugins.tooltip.callbacks = {
    title: (items) => (spec.tooltipTitles ? spec.tooltipTitles[items[0].dataIndex] : items[0].label),
    label: (ctx) => ` ${multi ? `${ctx.dataset.label}: ` : ""}${fmtValor(valueOf(ctx))}`,
  };
  if (spec.tooltipExtra) options.plugins.tooltip.callbacks.afterLabel = (ctx) => spec.tooltipExtra[ctx.dataIndex];
  if (stacked) {
    options.plugins.tooltip.filter = (item) => valueOf(item) > 0;
    options.plugins.tooltip.itemSort = (a, b) => valueOf(b) - valueOf(a);
  }
  if (spec.kind === "stacked" && !spec.percent) {
    options.plugins.tooltip.callbacks.footer = (items) => `Total: ${fmtValor(sum(items.map(valueOf)))}`;
  }
  if (spec.meanLine != null) {
    options.plugins.tooltip.callbacks.label = (ctx) => ` ${ctx.dataset.label}: ${fmtValor(ctx.parsed.y)}`;
  }

  if (spec.kind === "line") return lineConfig(spec, t, options);
  if (spec.kind === "band" || spec.kind === "multiline") return multiLineConfig(spec, t, options);

  const barStyle = { maxBarThickness: 24, barPercentage: 0.85, categoryPercentage: 0.9 };
  let datasets;
  if (spec.kind === "grouped") {
    // Barras lado a lado por categoria (ex.: % mensal de cada bioma, como a Figura 14 do RAF).
    datasets = spec.series.map((s) => ({
      label: s.label,
      data: s.values,
      backgroundColor: resolveColor(s.color, t),
      maxBarThickness: 14,
      barPercentage: 0.9,
      categoryPercentage: 0.85,
      borderRadius: 3,
      borderSkipped: "start",
    }));
  } else if (spec.kind === "stacked" && spec.area) {
    // Vista "Linhas" do empilhado: áreas empilhadas com linhas suaves.
    datasets = spec.series.map((s, k) => {
      const cor = resolveColor(s.color, t);
      return {
        label: s.label,
        data: s.values,
        borderColor: cor,
        backgroundColor: withAlpha(cor, 0.7),
        borderWidth: 1.5,
        pointRadius: 0,
        pointHoverRadius: 4,
        pointHoverBorderColor: t.surface,
        pointHoverBorderWidth: 2,
        fill: k ? "-1" : "origin",
        cubicInterpolationMode: "monotone",
      };
    });
    options.plugins.crosshair = { color: t.axis };
    return { type: "line", data: { labels: spec.labels, datasets }, options };
  } else if (spec.kind === "stacked") {
    datasets = spec.series.map((s) => ({
      ...barStyle,
      label: s.label,
      data: s.values,
      backgroundColor: resolveColor(s.color, t),
      borderColor: t.surface,
      borderWidth: { top: 1, right: 0, bottom: 0, left: 0 },
      borderSkipped: false,
      stack: "classes",
    }));
  } else if (spec.kind === "combo") {
    // Um eixo só: a barra de cada ano = acumulado anterior (neutro) + área inédita (cor oficial);
    // a linha acompanha o topo das barras (= área acumulada).
    datasets = [
      {
        type: "line",
        label: spec.line.label,
        data: spec.line.values,
        borderColor: spec.line.color,
        backgroundColor: spec.line.color,
        borderWidth: 2,
        pointRadius: 0,
        pointHoverRadius: 5,
        pointHoverBorderWidth: 2,
        pointHoverBorderColor: t.surface,
        tension: 0,
        stack: "linha",
        order: 0,
      },
      { ...barStyle, label: spec.base.label, data: spec.base.values, backgroundColor: t.axis, stack: "barra", order: 1 },
      {
        ...barStyle,
        label: spec.top.label,
        data: spec.top.values,
        backgroundColor: spec.top.color,
        borderColor: t.surface,
        borderWidth: { top: 0, right: 0, bottom: 1, left: 0 },
        borderRadius: 4,
        borderSkipped: "bottom",
        stack: "barra",
        order: 1,
      },
    ];
    options.plugins.tooltip.itemSort = undefined;
    options.plugins.tooltip.filter = undefined;
  } else {
    const colors = (spec.colors || spec.values.map(() => spec.color)).map((c) => resolveColor(c, t));
    datasets = [{
      ...barStyle,
      label: "Área queimada",
      data: spec.values,
      backgroundColor: colors,
      borderRadius: 4,
      borderSkipped: "start",
      order: 1,
    }];
    if (spec.meanLine != null) datasets.push(linhaMedia(spec, t));
  }
  return { type: "bar", data: { labels: spec.labels, datasets }, options };
}

// Linha de referência da média do período (tracejada = referência, não dado).
function linhaMedia(spec, t) {
  return {
    type: "line",
    label: "Média do período",
    data: spec.values.map(() => spec.meanLine),
    borderColor: t.text2,
    borderWidth: 1.5,
    borderDash: [6, 4],
    pointRadius: 0,
    pointHoverRadius: 0,
    fill: false,
    order: 0,
  };
}

// Série única como linha suave (monotone: curva sem passar do valor real entre os anos).
// Com uma cor por ano (`colors`), cada trecho e ponto usa a cor do ano e não há preenchimento.
function lineConfig(spec, t, options) {
  const last = spec.values.length - 1;
  const cores = spec.colors?.map((c) => resolveColor(c, t));
  const cor = resolveColor(spec.color, t) || cores?.[last] || t.text2;
  options.plugins.crosshair = { color: t.axis };
  const datasets = [{
    label: "Área queimada",
    data: spec.values,
    borderColor: cor,
    backgroundColor: cores ? cor : withAlpha(cor, 0.12),
    borderWidth: 2.25,
    fill: cores ? false : "origin",
    cubicInterpolationMode: "monotone",
    segment: cores ? { borderColor: (ctx) => cores[ctx.p1DataIndex] } : undefined,
    pointRadius: cores ? 2.5 : spec.values.map((_, i) => (i === last ? 4 : 0)),
    pointBackgroundColor: cores || cor,
    pointBorderColor: cores ? cores : t.surface,
    pointBorderWidth: cores ? 0 : 2,
    pointHoverRadius: 5,
    order: 1,
  }];
  if (spec.meanLine != null) datasets.push(linhaMedia(spec, t));
  return { type: "line", data: { labels: spec.labels, datasets }, options };
}

// Linhas múltiplas; "band" = faixa média ± DP (preenchida entre os limites) + média + anos.
function multiLineConfig(spec, t, options) {
  const line = (s, extra = {}) => ({
    type: "line",
    label: s.label,
    data: s.values,
    borderColor: s.color,
    backgroundColor: s.color,
    borderWidth: 2,
    borderDash: s.dash || [],
    pointRadius: 0,
    pointHoverRadius: 4,
    pointHoverBorderColor: t.surface,
    pointHoverBorderWidth: 2,
    tension: 0,
    fill: false,
    ...extra,
  });
  let datasets;
  if (spec.kind === "band") {
    // Como a Figura 13 do RAF: média histórica como área cinza; faixa ± 1 DP mais clara;
    // anos selecionados como linhas com marcadores.
    const cinza = resolveColor(spec.mean.color, t);
    const limite = { borderWidth: 0, pointRadius: 0, pointHoverRadius: 0, backgroundColor: withAlpha(cinza, 0.12), tension: 0 };
    datasets = [
      { ...limite, label: "Média + 1 DP", data: spec.upper, fill: false },
      { ...limite, label: "Média − 1 DP", data: spec.lower, fill: "-1" },
      line({ ...spec.mean, color: cinza }, { borderWidth: 1.5, fill: "origin", backgroundColor: withAlpha(cinza, 0.35) }),
      ...spec.years.map((s) => line(s, {
        borderWidth: 2.25, pointRadius: 3.5, pointBackgroundColor: s.color, pointBorderColor: t.surface, pointBorderWidth: 1.5,
      })),
    ];
  } else {
    datasets = spec.series.map((s) => line(s));
  }
  options.plugins.crosshair = { color: t.axis };
  options.plugins.tooltip.callbacks.label = (ctx) => ` ${ctx.dataset.label}: ${fmtFull.format(ctx.parsed.y)} ha`;
  return { type: "line", data: { labels: spec.labels, datasets }, options };
}

function doughnutConfig({ slices }, t) {
  const options = baseOptions(t);
  const total = sum(slices.map((s) => s.value));
  options.interaction = { mode: "nearest", intersect: true };
  options.cutout = "56%"; // anel mais largo, com espaço para a % dentro de cada fatia
  options.plugins.rotulosRosca = { total };
  options.plugins.tooltip.callbacks = {
    label: (ctx) => ` ${fmtFull.format(ctx.parsed)} ha (${pct(safeDiv(ctx.parsed, total))})`,
  };
  return {
    type: "doughnut",
    data: {
      labels: slices.map((s) => s.label),
      datasets: [{
        data: slices.map((s) => s.value),
        backgroundColor: slices.map((s) => resolveColor(s.color, t)),
        borderColor: t.surface,
        borderWidth: 2,
        hoverOffset: 4,
      }],
    },
    options,
  };
}

// % de cada barra sobre ela (gráficos com `rotulos: "pct"`; barras abaixo de 0,5% ficam sem rótulo).
const rotulosBarrasPlugin = {
  id: "rotulosBarras",
  afterDatasetsDraw(chart, _args, opts) {
    if (opts?.formato !== "pct" || chart.config.type !== "bar") return;
    const ds = chart.data.datasets[0];
    if (!ds) return;
    const vals = ds.data.map((v) => Number(v) || 0);
    const tot = sum(vals);
    const horizontal = chart.options.indexAxis === "y";
    const { ctx } = chart;
    ctx.save();
    ctx.font = "600 11px Montserrat, sans-serif";
    ctx.fillStyle = resolveColor("var(--text-2)") || "#333";
    ctx.textAlign = horizontal ? "left" : "center";
    ctx.textBaseline = horizontal ? "middle" : "bottom";
    chart.getDatasetMeta(0).data.forEach((barra, i) => {
      const p = safeDiv(vals[i], tot);
      if (p < 0.005) return;
      const txt = pct(p);
      if (horizontal) ctx.fillText(txt, barra.x + 4, barra.y);
      else ctx.fillText(txt, barra.x, barra.y - 3);
    });
    ctx.restore();
  },
};

// % de cada fatia escrita dentro da rosca (fatias com 4% ou mais, para não sobrepor).
const rotulosRoscaPlugin = {
  id: "rotulosRosca",
  afterDatasetsDraw(chart, _args, opts) {
    if (chart.config.type !== "doughnut" || !opts?.total) return;
    const { ctx } = chart;
    const meta = chart.getDatasetMeta(0);
    ctx.save();
    ctx.font = "700 12px Montserrat, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.lineJoin = "round";
    meta.data.forEach((arco, i) => {
      const v = chart.data.datasets[0].data[i];
      const p = safeDiv(v, opts.total);
      if (p < 0.04 || arco.hidden) return;
      const { x, y } = arco.tooltipPosition();
      const txt = pct(p);
      ctx.lineWidth = 3;
      ctx.strokeStyle = "rgba(0,0,0,0.45)";
      ctx.strokeText(txt, x, y);
      ctx.fillStyle = "#ffffff";
      ctx.fillText(txt, x, y);
    });
    ctx.restore();
  },
};

// Linha vertical acompanhando o tooltip nos gráficos de linha. Registrado globalmente
// (a troca barras -> linhas reaproveita a instância); só desenha com `plugins.crosshair.color`.
const crosshairPlugin = {
  id: "crosshair",
  afterDatasetsDraw(chart, _args, opts) {
    const active = chart.tooltip?.getActiveElements?.() || [];
    if (!opts?.color || !active.length) return;
    const { ctx, chartArea } = chart;
    const x = active[0].element.x;
    ctx.save();
    ctx.strokeStyle = opts.color;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    ctx.restore();
  },
};

function destroyCharts() {
  state.charts.forEach((c) => c.destroy());
  state.charts = [];
  if (state.map) {
    state.map.remove();
    state.map = null;
  }
}

function renderError(err, retry) {
  destroyCharts();
  state.lastView = null;
  el.panel.innerHTML = `
    <div class="card error-card">
      <h2>Não foi possível carregar os dados</h2>
      <p>${escapeHtml(err.message)}</p>
      <button type="button" class="btn">Tentar novamente</button>
    </div>`;
  el.panel.querySelector("button").addEventListener("click", retry);
}

// ---------------------------------------------------------------------------
// Interface: filtros, abas, status
// ---------------------------------------------------------------------------

// Seletor de país no topo, logo, marca e textos do país. Trocar de país recarrega a página
// (cada país tem sua própria árvore de temas, categorias e territórios na API).
// Coleção de cada país (nota informativa do tema de fogo de cada região), mostrada nos seletores de país.
const COLECOES = new Map();
const rotuloPais = (p) => {
  const c = COLECOES.get(p.region);
  return c === undefined ? p.nome : `${p.nome} · ${c ? `Col. ${c}` : "coleção não informada"}`;
};
function rotularColecoes() {
  for (const p of PAISES) {
    gifRegiao(p.region).then((r) => {
      COLECOES.set(p.region, r.colecao || null);
      document.querySelectorAll(`option[data-pais="${p.region}"]`).forEach((o) => { o.textContent = rotuloPais(p); });
    }).catch(() => {});
  }
}

function montarPais() {
  const sel = document.getElementById("f-pais");
  sel.innerHTML = PAISES.map((p) => `<option value="${p.region}" data-pais="${p.region}"${p === PAIS ? " selected" : ""}>${escapeHtml(rotuloPais(p))}</option>`).join("");
  rotularColecoes();
  sel.addEventListener("change", () => {
    const p = new URLSearchParams(sel.value === "brazil" ? "" : `pais=${sel.value}`);
    location.hash = p.toString();
    location.reload();
  });
  document.getElementById("logo").src = PAIS.logo;
  document.getElementById("logo").alt = PAIS.marca;
  document.getElementById("marca").textContent = PAIS.marca;
  document.getElementById("pais-titulo").textContent = PAIS.nome;
  document.getElementById("footer-fonte").textContent = `Dados ${PAIS.de}: ${PAIS.marca}, via API pública da Plataforma MapBiomas.`;
  document.title = `${PAIS.marca} · ${PAIS.nome}`;
  document.documentElement.lang = IDIOMA.locale;
  if (!ehBrasil) gifState.region = PAIS.region;
  // Idioma: pt, es, en (e bahasa indonésio para a Indonésia). Trocar recarrega a página.
  const selIdioma = document.getElementById("f-idioma");
  selIdioma.innerHTML = IDIOMAS_DO_PAIS.map((i) => `<option value="${i.id}"${i === IDIOMA ? " selected" : ""}>${escapeHtml(i.nome)}</option>`).join("");
  selIdioma.addEventListener("change", () => {
    const p = new URLSearchParams(location.hash.slice(1));
    if (selIdioma.value === IDIOMA_PADRAO) p.delete("lang"); else p.set("lang", selIdioma.value);
    location.hash = p.toString();
    location.reload();
  });
}

function renderTabs() {
  if (!abasDisponiveis().some((t) => t.id === state.tab)) state.tab = "anual";
  el.tabs.innerHTML = abasDisponiveis().map((t) => `
    <button type="button" role="tab" id="tab-${t.id}" data-tab="${t.id}" aria-controls="panel"
      aria-selected="${t.id === state.tab}" tabindex="${t.id === state.tab ? 0 : -1}">${t.label}</button>`).join("");
  el.panel.setAttribute("aria-labelledby", `tab-${state.tab}`);
}

function selectTab(id, focus = false) {
  if (id === state.tab) return;
  state.tab = id;
  for (const btn of el.tabs.querySelectorAll("[role=tab]")) {
    const selected = btn.dataset.tab === id;
    btn.setAttribute("aria-selected", selected);
    btn.tabIndex = selected ? 0 : -1;
    if (selected && focus) btn.focus();
  }
  el.panel.setAttribute("aria-labelledby", `tab-${id}`);
  destroyCharts();
  state.lastView = null;
  el.panel.innerHTML = `<p class="placeholder">Carregando ${escapeHtml(TABS.find((t) => t.id === id).label)}…</p>`;
  loadCurrentTab();
}

const optionsHtml = (items, emptyLabel) =>
  `<option value="">${escapeHtml(emptyLabel)}</option>` +
  items.map((t) => `<option value="${escapeHtml(t.id)}">${escapeHtml(t.name)}</option>`).join("");

function fillFilters() {
  const c = state.conn;
  document.getElementById("rot-bioma").textContent = rot("bioma");
  document.getElementById("rot-estado").textContent = rot("estado");
  document.getElementById("rot-municipio").textContent = rot("municipio");
  el.bioma.closest(".field").hidden = !c.biomas.length;
  el.municipio.closest(".field").hidden = !c.catMunicipio;
  el.bioma.innerHTML = optionsHtml(c.biomas, `Todos (${PAIS.nome})`);
  el.estado.innerHTML = optionsHtml(c.estados, `Todos (${rot("estado", true).toLowerCase()})`);
  el.municipio.innerHTML = optionsHtml([], `Todos (${rot("municipio", true).toLowerCase()})`);

  for (const input of [el.ini, el.fim]) {
    input.min = String(c.anoMin);
    input.max = String(c.anoMax);
    input.step = "1";
  }
  document.getElementById("f-min").textContent = c.anoMin;
  document.getElementById("f-max").textContent = c.anoMax;
  state.ini = clamp(state.ini, c.anoMin, c.anoMax);
  state.fim = clamp(state.fim, state.ini, c.anoMax);
  el.ini.value = String(state.ini);
  el.fim.value = String(state.fim);
  updatePeriodSlider();

  if (c.biomas.some((b) => b.id === state.biomaId)) el.bioma.value = state.biomaId;
  else state.biomaId = "";
  if (c.estados.some((e) => e.id === state.estadoId)) el.estado.value = state.estadoId;
  else state.estadoId = "";
}

function updatePeriodSlider() {
  const min = Number(el.ini.min), max = Number(el.ini.max);
  const a = Number(el.ini.value), b = Number(el.fim.value);
  const pos = (v) => (max > min ? (100 * (v - min)) / (max - min) : 0);
  const fill = document.getElementById("f-fill");
  fill.style.left = `${pos(a)}%`;
  fill.style.width = `${pos(b) - pos(a)}%`;
  document.getElementById("f-periodo").textContent = a === b ? String(a) : `${a}–${b}`;
  // O polegar do início fica por cima quando os dois se encontram no fim da escala.
  el.ini.style.zIndex = a === max ? "4" : "3";
}

async function fillMunicipios() {
  state.municipiosDoEstado = [];
  const estado = state.conn.estados.find((e) => e.id === state.estadoId);
  if (!estado) {
    state.municipioId = "";
    el.municipio.innerHTML = optionsHtml([], `Todos (${rot("municipio", true).toLowerCase()})`);
    el.municipio.disabled = true;
    return;
  }
  el.municipio.disabled = true;
  el.municipio.innerHTML = `<option value="">Carregando ${rot("municipio", true).toLowerCase()}…</option>`;
  try {
    state.municipiosDoEstado = await getMunicipiosDoEstado(estado);
    el.municipio.innerHTML = optionsHtml(state.municipiosDoEstado, `Todos (${state.municipiosDoEstado.length} ${rot("municipio", true).toLowerCase()})`);
    if (!state.municipiosDoEstado.some((m) => m.id === state.municipioId)) state.municipioId = "";
    el.municipio.value = state.municipioId;
    el.municipio.disabled = false;
  } catch (err) {
    console.error(err);
    state.municipioId = "";
    el.municipio.innerHTML = `<option value="">Falha ao carregar ${rot("municipio", true).toLowerCase()}</option>`;
  }
}

function setStatus(text, busy = false) {
  el.status.textContent = text;
  el.status.classList.toggle("busy", busy);
}

function updateSourceInfo() {
  const conn = state.conn;
  if (!conn) {
    el.badge.textContent = "API indisponível";
    el.badge.dataset.state = "error";
    return;
  }
  const fallback = conn.env.id === "stg";
  el.badge.textContent = `API: ${conn.env.label}${fallback ? " (fallback)" : ""}`;
  el.badge.dataset.state = conn.colecaoOk ? "ok" : "warn";
  el.badge.title = conn.failures.length ? conn.failures.join("\n") : conn.env.baseUrl;
  document.getElementById("colecao-titulo").textContent = conn.colecao ? ` Col. ${conn.colecao}` : "";
  document.title = `${PAIS.marca}${conn.colecao ? ` Col. ${conn.colecao}` : ""} · ${PAIS.nome}`;
  el.subtitle.textContent = `Estatísticas de área queimada · ${conn.anoMin}–${conn.anoMax}`;
  el.footerEnv.textContent = `Ambiente: ${conn.env.baseUrl}.`;

  el.banner.hidden = conn.colecaoOk;
  if (!conn.colecaoOk) {
    el.banner.textContent =
      `A produção (Coleção ${CONFIG.colecaoExigida}) não respondeu. Os dados exibidos vêm do ${conn.env.label}, ` +
      `que serve a Coleção ${conn.colecao || "desconhecida"}, e não a ${CONFIG.colecaoExigida}. ` +
      `Recarregue a página mais tarde para voltar à produção.`;
  }
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  if (TABS.some((t) => t.id === p.get("aba"))) state.tab = p.get("aba");
  state.biomaId = p.get("bioma") || "";
  state.estadoId = p.get("estado") || "";
  state.municipioId = p.get("municipio") || "";
  if (Number(p.get("ini"))) state.ini = Number(p.get("ini"));
  if (Number(p.get("fim"))) state.fim = Number(p.get("fim"));
  if (p.has("nivel")) state.nivel = Number(p.get("nivel"));
  if (p.get("anos")) {
    state.mensalAnos = p.get("anos").split(",").map(Number).filter(Boolean).slice(0, SERIES_COLORS.length);
    state.mensalAnosTocados = true;
  }
  if (["bruto", "ranking"].includes(p.get("mapa"))) state.mapa.tipo = p.get("mapa");
  if (["biomas", "estados", "municipios"].includes(p.get("unidade"))) state.mapa.unidade = p.get("unidade");
  if (MAP_LAYERS.some((c) => c.id === p.get("camada"))) state.mapa.camada = p.get("camada");
  if (MAP_VIEWS.some((v) => v.id === p.get("vis"))) state.mapa.visualizar = p.get("vis");
  if (["todas", "15", "9"].includes(p.get("freq"))) state.freqGrupo = p.get("freq");
  const temp = (p.get("temporada") || "").split("-").map(Number);
  if (temp.length === 2 && temp.every((m) => m >= 1 && m <= 12)) [state.mesIni, state.mesFim] = temp;
  if (["oficial", "fogo"].includes(p.get("paleta"))) state.paletaMes = p.get("paleta");
  if (["bioma", "estado"].includes(p.get("por"))) state.mensalNivel = p.get("por");
  if (["ha", "pct"].includes(p.get("unid"))) state.freqUnid = p.get("unid");
  if (p.get("cicatriz")) state.cicatrizClasse = p.get("cicatriz");
}

function writeHash() {
  const p = new URLSearchParams({ aba: state.tab, ini: state.ini, fim: state.fim });
  if (!ehBrasil) p.set("pais", PAIS.region);
  if (IDIOMA.id !== IDIOMA_PADRAO) p.set("lang", IDIOMA.id);
  if (state.biomaId) p.set("bioma", state.biomaId);
  if (state.estadoId) p.set("estado", state.estadoId);
  if (state.municipioId) p.set("municipio", state.municipioId);
  if (state.tab === "anual") p.set("nivel", state.nivel);
  if (state.tab === "mensal" && state.mensalAnos.length) p.set("anos", state.mensalAnos.join(","));
  if (state.tab === "mensal") {
    if (state.mesIni !== 1 || state.mesFim !== 12) p.set("temporada", `${state.mesIni}-${state.mesFim}`);
    if (state.paletaMes !== "oficial") p.set("paleta", state.paletaMes);
    if (state.mensalNivel !== "bioma") p.set("por", state.mensalNivel);
  }
  if (state.tab === "frequencia") {
    p.set("freq", state.freqGrupo);
    if (state.freqUnid !== "ha") p.set("unid", state.freqUnid);
  }
  if (state.tab === "cicatriz" && state.cicatrizClasse !== "grandes") p.set("cicatriz", state.cicatrizClasse);
  if (state.tab === "mapa") {
    p.set("mapa", state.mapa.tipo);
    if (state.mapa.tipo === "ranking") p.set("unidade", state.mapa.unidade);
    else { p.set("camada", state.mapa.camada); p.set("vis", state.mapa.visualizar); }
  }
  try { history.replaceState(null, "", `#${p}`); } catch { /* file:// em alguns navegadores */ }
}

async function loadCurrentTab() {
  const seq = ++state.seq;
  const tab = TABS.find((t) => t.id === state.tab);
  writeHash();
  el.panel.classList.add("is-loading");
  setStatus("Carregando dados…", true);
  try {
    const view = await tab.load({ t: activeTerritory(), ini: state.ini, fim: state.fim });
    if (seq !== state.seq) return; // o usuário já mudou de aba/filtro
    state.lastView = view;
    renderView(view);
    setStatus("");
  } catch (err) {
    if (seq !== state.seq) return;
    console.error(err);
    renderError(err, loadCurrentTab);
    setStatus("");
  } finally {
    if (seq === state.seq) el.panel.classList.remove("is-loading");
  }
}

async function start() {
  el.panel.innerHTML = `<p class="placeholder">Conectando à API do MapBiomas…</p>`;
  setStatus("Descobrindo ambiente da API e validando a coleção…", true);
  try {
    state.conn = await connect();
  } catch (err) {
    console.error(err);
    updateSourceInfo();
    setStatus("");
    renderError(err, start);
    return;
  }
  updateSourceInfo();
  renderTabs(); // esconde as abas de produtos que o país não tem
  fillFilters();
  await fillMunicipios();
  for (const s of [el.bioma, el.estado, el.ini, el.fim]) s.disabled = false;
  await document.fonts?.ready;
  loadCurrentTab();
}

function init() {
  Chart.defaults.font.family = "'Montserrat', system-ui, -apple-system, 'Segoe UI', sans-serif";
  Chart.register(crosshairPlugin, traducaoChartPlugin, rotulosRoscaPlugin, rotulosBarrasPlugin);
  el.panel.addEventListener("click", acaoDoCard);
  readHash();
  montarPais();
  iniciarTraducao();
  renderTabs();

  el.tabs.addEventListener("click", (e) => {
    const btn = e.target.closest("[role=tab]");
    if (btn) selectTab(btn.dataset.tab);
  });
  el.tabs.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const abas = abasDisponiveis();
    const i = abas.findIndex((t) => t.id === state.tab);
    const next = (i + (e.key === "ArrowRight" ? 1 : -1) + abas.length) % abas.length;
    selectTab(abas[next].id, true);
  });

  el.bioma.addEventListener("change", () => {
    state.biomaId = el.bioma.value;
    loadCurrentTab();
  });
  el.estado.addEventListener("change", async () => {
    state.estadoId = el.estado.value;
    state.municipioId = "";
    await fillMunicipios();
    loadCurrentTab();
  });
  el.municipio.addEventListener("change", () => {
    state.municipioId = el.municipio.value;
    loadCurrentTab();
  });
  // Slider de período: "input" atualiza o rótulo enquanto arrasta; "change" (ao soltar) recarrega.
  el.ini.addEventListener("input", () => {
    if (Number(el.ini.value) > Number(el.fim.value)) el.ini.value = el.fim.value;
    updatePeriodSlider();
  });
  el.fim.addEventListener("input", () => {
    if (Number(el.fim.value) < Number(el.ini.value)) el.fim.value = el.ini.value;
    updatePeriodSlider();
  });
  for (const input of [el.ini, el.fim]) {
    input.addEventListener("change", () => {
      const ini = Number(el.ini.value), fim = Number(el.fim.value);
      if (ini === state.ini && fim === state.fim) return;
      state.ini = ini;
      state.fim = fim;
      loadCurrentTab();
    });
  }

  // Redesenha com os tokens do tema quando o sistema alterna claro/escuro.
  matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => {
    if (state.lastView) renderView(state.lastView);
  });

  start();
}

// ---------------------------------------------------------------------------
// Fábrica de GIFs (padrão de utils/mapbiomas_gif_factory, feita no navegador)
// ---------------------------------------------------------------------------
// Um quadro por ano com os tiles oficiais da Plataforma MapBiomas (a mesma API serve
// outros países pela "região"), com título, território, ano, legenda, fonte e logo.
// O GIF é codificado no navegador com gifenc. Esta aba não usa os filtros do topo.

const GIF_PAISES = [
  { region: "brazil", nome: "Brasil", lang: "pt-BR", logo: "logos/MapBiomas_Fogo_PT.png" },
  { region: "argentina", nome: "Argentina", de: "da Argentina", em: "na Argentina", lang: "es-ES", logo: "logos/MapBiomas_Fuego_ES.png" },
  { region: "bolivia", nome: "Bolívia", de: "da Bolívia", em: "na Bolívia", lang: "es-ES", logo: "logos/MapBiomas_Fuego_ES.png" },
  { region: "chile", nome: "Chile", de: "do Chile", em: "no Chile", lang: "es-ES", logo: "logos/MapBiomas_Fuego_ES.png" },
  { region: "colombia", nome: "Colômbia", de: "da Colômbia", em: "na Colômbia", lang: "es-ES", logo: "logos/MapBiomas_Fuego_ES.png" },
  { region: "paraguay", nome: "Paraguai", colecaoPadrao: "1", de: "do Paraguai", em: "no Paraguai", lang: "es-ES", logo: "logos/MapBiomas_Fuego_ES.png" },
  { region: "peru", nome: "Peru", de: "do Peru", em: "no Peru", lang: "es-ES", logo: "logos/MapBiomas_Fuego_ES.png" },
  { region: "indonesia", nome: "Indonésia", colecaoPadrao: "2", de: "da Indonésia", em: "na Indonésia", lang: "en-US", logo: "logos/MapBiomas_Fire_EN.png" },
];

// Textos gravados no GIF, no idioma do país.
const GIF_TEXTOS = {
  "pt-BR": { marca: "MapBiomas Fogo", fonte: "Fonte: MapBiomas Fogo", colecao: "Coleção", anual: "Área queimada anual" },
  "es-ES": { marca: "MapBiomas Fuego", fonte: "Fuente: MapBiomas Fuego", colecao: "Colección", anual: "Área quemada anual" },
  "en-US": { marca: "MapBiomas Fire", fonte: "Source: MapBiomas Fire", colecao: "Collection", anual: "Annual burned area" },
  "id-ID": { marca: "MapBiomas Fire", fonte: "Sumber: MapBiomas Fire", colecao: "Koleksi", anual: "Area terbakar tahunan" },
};

// ms por quadro: lento = antigo normal, normal = antigo rápido, rápido ainda mais rápido.
const GIF_VELOCIDADES = [["lento", "Lento", 700], ["normal", "Normal", 350], ["rapido", "Rápido", 180]];
const GIF_TAMANHOS = [["p", "Pequeno", 640], ["m", "Médio", 900], ["g", "Grande", 1200]];
const GIF_FUNDOS = [["mapa", "Mapa base cinza"], ["branco", "Fundo branco"]];
const GIFENC_URL = "https://cdn.jsdelivr.net/npm/gifenc@1.0.3/dist/gifenc.esm.js";
const GIF_BASEMAP = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}";

const gifState = {
  region: "brazil", produto: "", legenda: "", categoria: "", territorio: "", ini: null, fim: null,
  velocidade: "normal", tamanho: "m", fundo: "mapa", limites: "nenhum", resultado: null, gerando: false,
};

const gifEnv = () => ENVIRONMENTS[0]; // produção
const gifApi = (region, path, params) => apiGet(gifEnv(), `/${region}${path}`, params);
const nomeEm = (n, lang) => (typeof n === "string" ? n
  : n?.[lang] || (lang === "id-ID" && n?.["en-US"]) || n?.["pt-BR"] || n?.["es-ES"] || n?.["en-US"] || Object.values(n || {})[0] || "");
// Nome de arquivo seguro, como safe_name() do gif_factory.
const nomeSeguro = (v) => String(v || "").normalize("NFKD").replace(/[̀-ͯ]/g, "")
  .replace(/[^A-Za-z0-9_-]+/g, "_").replace(/^[_-]+|[_-]+$/g, "").slice(0, 80) || "sem_nome";

// Tema de fogo do país: produtos com série anual, categorias de território, anos e coleção.
function gifRegiao(region) {
  return cached(`gif:${region}`, async () => {
    const [themes, cats] = await Promise.all([
      gifApi(region, "/themes", { page: 1, pageSize: 100 }),
      gifApi(region, "/territories/categories", { page: 1, pageSize: 500 }),
    ]);
    const fire = (themes.themes || []).find((t) => /fire|fogo|fuego/i.test(`${t.key} ${JSON.stringify(t.name)}`));
    if (!fire) throw new Error("Este país não tem o tema de fogo na Plataforma MapBiomas.");
    // prd aninha subtemas em `children` (às vezes em dois níveis); alguns países os deixam soltos.
    const folhas = (arr) => (arr || []).flatMap((s) => (s.children?.length ? folhas(s.children)
      : s.subthemes?.length ? folhas(s.subthemes) : [s]));
    const produtos = folhas(fire.subthemes).filter((s) => !["last_year", "mean_interval"].includes(s.analysis)
      && !/mensal|monthly|mensual|bulanan|mosaico|monitor/i.test(JSON.stringify(s.name)));
    const nota = JSON.stringify([fire.description, fire.informativeNote]).replace(/<[^>]+>/g, " ");
    const m = nota.match(/(?:entre|between|from|de|period(?:o)?)\s+(\d{4})\s*(?:e|and|to|a|y|al|–|-)\s*(\d{4})/i)
      || nota.match(/\b((?:19|20)\d{2})\s*[–-]\s*((?:19|20)\d{2})\b/);
    const anoOk = (a) => a >= 1980 && a <= new Date().getFullYear();
    const anoMin = m && anoOk(Number(m[1])) ? Number(m[1]) : null;
    const anoMax = m && anoOk(Number(m[2])) && Number(m[2]) >= anoMin ? Number(m[2]) : null;
    const colecao = colecaoDaNota(nota);
    const colecaoFinal = colecao || PAISES.find((p) => p.region === region)?.colecaoPadrao || null;
    return { produtos, categorias: cats.categories || [], colecao: colecaoFinal, anoMin: anoMax ? anoMin : null, anoMax };
  });
}

function gifLegendas(region, subthemeKey) {
  return cached(`gif:${region}:legs:${subthemeKey}`, async () => {
    const l = await gifApi(region, "/legends", { page: 1, pageSize: 100, subthemeKey });
    return [...(l.legends || [])].sort((a, b) => (a.order ?? 0) - (b.order ?? 0));
  });
}

function gifLegenda(region, id, lang) {
  return cached(`gif:${region}:leg:${id}:${lang}`, async () => {
    const d = await gifApi(region, `/legends/${id}`);
    const lg = d.legend || d;
    const itens = [];
    const andar = (arr, depth) => (arr || []).forEach((i) => {
      if (!/modis/i.test(nomeEm(i.name, lang))) itens.push({ pixelValue: i.pixelValue, name: nomeEm(i.name, lang) || i.key, color: safeColor(i.color), depth });
      andar(i.children, depth + 1);
    });
    andar(lg.items, 0);
    return { key: lg.key, nome: nomeEm(lg.name, lang), items: itens };
  });
}

function gifTerritorios(region, categoryId) {
  return cached(`gif:${region}:ter:${categoryId}`, async () => {
    const pagina = (page) => gifApi(region, "/territories", { page, pageSize: 1000, categoryId });
    const first = await pagina(1);
    const rest = await Promise.all(range(2, first.numberOfPages || 1).map(pagina));
    return [first, ...rest].flatMap((r) => r.territories || [])
      // nameFormatted traz a UF nos municípios do Brasil ("X - MT"); tira só sufixos numéricos ("Chile - 1").
      .map((t) => ({ id: t.id, name: (t.nameFormatted || t.name).replace(/\s-\s\d+$/, "") }))
      .sort((a, b) => a.name.localeCompare(b.name, "pt-BR"));
  });
}

async function loadGif() {
  return {
    kpis: [],
    cards: [{
      span: 12,
      title: "Fábrica de GIFs",
      subtitle: "Animação ano a ano com os mapas oficiais da Plataforma MapBiomas, no padrão do MapBiomas GIF Factory · esta aba tem filtros próprios (país, território e anos) e não usa os filtros do topo",
      custom: montarGif,
    }],
  };
}

function montarGif(body) {
  const seg = (grupo, opcoes) => `
    <div class="segmented" role="group" data-g="${grupo}">
      ${opcoes.map(([v, r]) => `<button type="button" data-v="${v}" aria-pressed="${gifState[grupo] === v}">${escapeHtml(r)}</button>`).join("")}
    </div>`;
  body.innerHTML = `
    <div class="gif-form">
      <div class="md-passo"><span class="md-num">1</span><span class="md-rotulo">País</span>
        <select data-f="region" aria-label="País">${GIF_PAISES.map((p) => `<option value="${p.region}" data-pais="${p.region}"${p.region === gifState.region ? " selected" : ""}>${escapeHtml(rotuloPais(p))}</option>`).join("")}</select>
      </div>
      <div class="md-passo"><span class="md-num">2</span><span class="md-rotulo">Território</span>
        <select data-f="categoria" aria-label="Tipo de território"></select>
        <select data-f="territorio" aria-label="Território"></select>
      </div>
      <div class="md-passo"><span class="md-num">3</span><span class="md-rotulo">Produto</span>
        <select data-f="produto" aria-label="Produto"></select>
        <select data-f="legenda" aria-label="Legenda" hidden></select>
      </div>
      <div class="md-passo"><span class="md-num">4</span><span class="md-rotulo">Anos</span>
        <select data-f="ini" aria-label="Ano inicial"></select><span>até</span>
        <select data-f="fim" aria-label="Ano final"></select>
      </div>
      <div class="md-passo"><span class="md-num">5</span><span class="md-rotulo">Aparência</span>
        ${seg("velocidade", GIF_VELOCIDADES)}
        ${seg("tamanho", GIF_TAMANHOS.map(([v, r, px]) => [v, `${r} (${px} px)`]))}
        ${seg("fundo", GIF_FUNDOS)}
      </div>
      <div class="md-passo"><span class="md-num">6</span><span class="md-rotulo">Limites no mapa</span>
        <div class="segmented" role="group" data-g="limites"></div>
      </div>
      <div class="gif-acoes">
        <button type="button" class="md-botao gif-gerar">▶ Gerar GIF</button>
        <button type="button" class="md-botao gif-gerar gif-baixar">⬇ Gerar e baixar GIF</button>
        <span class="gif-status" aria-live="polite"></span>
      </div>
      <progress class="gif-prog" max="1" value="0" hidden></progress>
    </div>
    <p class="md-ajuda gif-ajuda"></p>
    <div class="gif-resultado"></div>`;

  const q = (s) => body.querySelector(s);
  const sel = (f) => q(`[data-f="${f}"]`);
  const status = (msg) => { q(".gif-status").textContent = msg; };
  const pais = () => GIF_PAISES.find((p) => p.region === gifState.region) || GIF_PAISES[0];
  const opcoes = (el, itens, atual) => {
    el.innerHTML = itens.map((o) => `<option value="${escapeHtml(o.value)}"${String(o.value) === String(atual) ? " selected" : ""}>${escapeHtml(o.label)}</option>`).join("");
    if (itens.length && !itens.some((o) => String(o.value) === String(atual))) el.value = String(itens[0].value);
    return el.value;
  };
  let reg = null;

  function ajuda() {
    const p = pais();
    const prod = reg?.produtos.find((s) => s.key === gifState.produto);
    const acumula = prod && (prod.dataType === "year_interval" || ["accumulated", "frequency"].includes(prod.analysis));
    const anos = gifState.ini && gifState.fim ? gifState.fim - gifState.ini + 1 : 0;
    q(".gif-ajuda").textContent =
      `O GIF terá ${anos} quadro(s), um por ano${acumula ? `, somando desde ${gifState.ini} (cada quadro mostra o acumulado até aquele ano)` : ""}. ` +
      `Cada quadro traz título, território, ano, legenda e fonte em ${{ pt: "português", es: "espanhol", en: "inglês", id: "bahasa indonésio" }[IDIOMA.id]}. ` +
      "Os mapas vêm da Plataforma MapBiomas; territórios pequenos ficam mais nítidos. Gerar leva de alguns segundos a poucos minutos.";
  }

  async function carregarPais() {
    status("Carregando o país…");
    body.querySelectorAll(".gif-gerar").forEach((x) => { x.disabled = true; });
    try {
      reg = await gifRegiao(gifState.region);
      const p = pais();
      // Limites que o país tem (biomas/ecorregiões e o 1º nível administrativo).
      const cfg = PAISES.find((x) => x.region === p.region);
      const limOpc = [["nenhum", "Nenhum"],
        ...["bioma", "estado"].filter((k) => cfg?.cats?.[k] && reg.categorias.some((c) => cfg.cats[k].test(c.key))).map((k) => [k, cfg.rotulos[k][1]])];
      if (!limOpc.some(([v]) => v === gifState.limites)) gifState.limites = "nenhum";
      q('[data-g="limites"]').innerHTML = limOpc.map(([v, r]) => `<button type="button" data-v="${v}" aria-pressed="${gifState.limites === v}">${escapeHtml(r)}</button>`).join("");
      gifState.produto = opcoes(sel("produto"), reg.produtos.map((s) => ({
        value: s.key,
        label: nomeEm(s.name, IDIOMA.gif) + (s.dataType === "year_interval" || s.analysis ? " (acumulado ano a ano)" : ""),
      })), gifState.produto);
      const preferida = reg.categorias.find((c) => /^POLITICAL_LEVEL_1$|^NIVEL_POLITICO_1|^BRASIL_|^PAIS_/i.test(c.key)) || reg.categorias[0];
      gifState.categoria = opcoes(sel("categoria"), reg.categorias.map((c) => ({ value: c.id, label: nomeEm(c.name, IDIOMA.gif) || c.key })),
        reg.categorias.some((c) => String(c.id) === String(gifState.categoria)) ? gifState.categoria : preferida?.id);
      const min = reg.anoMin || 1985, max = reg.anoMax || new Date().getFullYear() - 1;
      const anos = range(min, max).map((a) => ({ value: a, label: String(a) }));
      gifState.ini = Number(opcoes(sel("ini"), anos, clamp(gifState.ini ?? min, min, max)));
      gifState.fim = Number(opcoes(sel("fim"), anos, clamp(gifState.fim ?? max, gifState.ini, max)));
      await Promise.all([carregarLegendas(), carregarTerritorios()]);
      status(reg.anoMax ? `${reg.anoMin}–${reg.anoMax}${reg.colecao ? ` · Coleção ${reg.colecao}` : ""}`
        : "A Plataforma não informa o período deste país: confira se os anos escolhidos têm dados.");
      body.querySelectorAll(".gif-gerar").forEach((x) => { x.disabled = false; });
    } catch (err) {
      console.error(err);
      status(`Não foi possível carregar: ${err.message}`);
    }
    ajuda();
  }

  async function carregarLegendas() {
    const lista = await gifLegendas(gifState.region, gifState.produto);
    const el = sel("legenda");
    gifState.legenda = opcoes(el, lista.map((l) => ({ value: l.id, label: `Legenda: ${nomeEm(l.name, IDIOMA.gif) || l.key}` })), gifState.legenda);
    el.hidden = lista.length < 2;
  }

  async function carregarTerritorios() {
    const el = sel("territorio");
    el.innerHTML = `<option>Carregando…</option>`;
    const lista = await gifTerritorios(gifState.region, gifState.categoria);
    gifState.territorio = opcoes(el, lista.map((t) => ({ value: t.id, label: t.name })), gifState.territorio);
  }

  body.addEventListener("change", async (e) => {
    const f = e.target.dataset.f;
    if (!f) return;
    gifState[f] = ["ini", "fim"].includes(f) ? Number(e.target.value) : e.target.value;
    try {
      if (f === "region") {
        Object.assign(gifState, { produto: "", legenda: "", categoria: "", territorio: "", ini: null, fim: null });
        await carregarPais();
        return;
      }
      if (f === "produto") { gifState.legenda = ""; await carregarLegendas(); }
      if (f === "categoria") { gifState.territorio = ""; await carregarTerritorios(); }
      if (f === "ini" && gifState.fim < gifState.ini) { gifState.fim = gifState.ini; sel("fim").value = String(gifState.fim); }
      if (f === "fim" && gifState.ini > gifState.fim) { gifState.ini = gifState.fim; sel("ini").value = String(gifState.ini); }
    } catch (err) {
      status(`Não foi possível carregar: ${err.message}`);
    }
    ajuda();
  });

  body.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    const g = b.closest("[data-g]")?.dataset.g;
    if (g) {
      gifState[g] = b.dataset.v;
      b.parentElement.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
      return;
    }
    if (b.classList.contains("gif-gerar")) {
      // "Gerar e baixar": ao terminar, o download começa sozinho (a prévia continua abaixo).
      const baixar = b.classList.contains("gif-baixar");
      const antes = gifState.resultado?.url;
      gerarGif(body, reg, status).then(() => {
        const r = gifState.resultado;
        if (baixar && r?.url && r.url !== antes) {
          baixarArquivo(r.url, r.nome);
          status("Pronto! O download do GIF começou.");
        }
      }).finally(() => { body.querySelectorAll(".gif-gerar").forEach((x) => { x.disabled = false; }); });
    }
    if (b.dataset.a === "png" && gifState.resultado?.png) baixarArquivo(gifState.resultado.png, gifState.resultado.nome.replace(/\.gif$/, "_ultimo_quadro.png"));
  });

  if (gifState.resultado) mostrarResultadoGif(body);
  carregarPais();
}

function baixarArquivo(blobOuUrl, nome) {
  const url = typeof blobOuUrl === "string" ? blobOuUrl : URL.createObjectURL(blobOuUrl);
  const a = document.createElement("a");
  a.href = url;
  a.download = nome;
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (typeof blobOuUrl !== "string") setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function mostrarResultadoGif(body) {
  const r = gifState.resultado;
  body.querySelector(".gif-resultado").innerHTML = `
    <figure class="gif-preview">
      <img src="${r.url}" alt="${escapeHtml(r.titulo)}">
      <figcaption>${escapeHtml(r.titulo)} · ${r.quadros} quadros · ${(r.bytes / 1e6).toLocaleString(LOCALE, { maximumFractionDigits: 1 })} MB</figcaption>
    </figure>
    <div class="gif-acoes">
      <a class="md-botao" href="${r.url}" download="${escapeHtml(r.nome)}">⬇ Baixar GIF</a>
      <button type="button" class="md-botao" data-a="png">⬇ Baixar último quadro (PNG)</button>
    </div>`;
}

// ----- Geração -----

const lonParaX = (lon) => (lon + 180) / 360;
const latParaY = (lat) => {
  const r = (Math.max(-85, Math.min(85, lat)) * Math.PI) / 180;
  return (1 - Math.log(Math.tan(r) + 1 / Math.cos(r)) / Math.PI) / 2;
};

// Anéis [lon, lat] de Polygon / MultiPolygon / GeometryCollection / Feature(Collection).
function aneisDe(g) {
  if (!g) return [];
  if (g.type === "Feature") return aneisDe(g.geometry);
  if (g.type === "FeatureCollection") return g.features.flatMap((f) => aneisDe(f.geometry));
  if (g.type === "GeometryCollection") return g.geometries.flatMap(aneisDe);
  if (g.type === "Polygon") return g.coordinates;
  if (g.type === "MultiPolygon") return g.coordinates.flat();
  return [];
}

async function carregarImagem(url) {
  try {
    const resp = await fetch(url, { mode: "cors" });
    if (!resp.ok) return null;
    return await createImageBitmap(await resp.blob());
  } catch {
    return null;
  }
}

// Logo só entra se não "sujar" o canvas (aberto como arquivo local, o navegador bloqueia).
async function carregarLogo(src) {
  const img = await new Promise((resolve) => {
    const i = new Image();
    i.onload = () => resolve(i);
    i.onerror = () => resolve(null);
    i.src = src;
  });
  if (!img) return null;
  try {
    const c = document.createElement("canvas");
    c.width = c.height = 2;
    c.getContext("2d").drawImage(img, 0, 0, 2, 2);
    c.getContext("2d").getImageData(0, 0, 1, 1);
    return img;
  } catch {
    return null;
  }
}

async function gerarGif(body, reg, status) {
  if (gifState.gerando || !reg) return;
  const q = (s) => body.querySelector(s);
  const prog = q(".gif-prog");
  const vivo = () => body.isConnected;
  gifState.gerando = true;
  body.querySelectorAll(".gif-gerar").forEach((x) => { x.disabled = true; });
  prog.hidden = false;
  prog.value = 0;
  try {
    const p = GIF_PAISES.find((x) => x.region === gifState.region);
    const lang = IDIOMA.gif; // textos do quadro no idioma escolhido na página
    const tx = GIF_TEXTOS[lang];
    const produto = reg.produtos.find((s) => s.key === gifState.produto);
    const lista = await gifLegendas(p.region, produto.key);
    const ref = lista.find((l) => String(l.id) === String(gifState.legenda)) || lista[0];
    const legenda = await gifLegenda(p.region, ref.id, lang);
    const territorio = (await gifTerritorios(p.region, gifState.categoria)).find((t) => String(t.id) === String(gifState.territorio));
    if (!territorio) throw new Error("Escolha um território.");
    const anos = range(gifState.ini, gifState.fim);
    const acumula = produto.dataType === "year_interval" || ["accumulated", "frequency"].includes(produto.analysis);
    const delay = GIF_VELOCIDADES.find((v) => v[0] === gifState.velocidade)[2];
    const W = GIF_TAMANHOS.find((v) => v[0] === gifState.tamanho)[2];
    const s = W / 900;

    status("Buscando o contorno do território…");
    const geo = await gifApi(p.region, "/territories/geometry", { territoryId: [territorio.id], spatialMethod: "union", simplify: "true" });
    const aneis = aneisDe(geo.geometry || geo);
    if (!aneis.length) throw new Error("A Plataforma não devolveu o contorno deste território.");

    // Endereço dos tiles de cada ano (cumulativos: do ano inicial até o ano do quadro).
    let feitos = 0;
    const urls = await mapLimit(anos, 4, async (a) => {
      const r = await gifApi(p.region, "/maps/map", {
        region: p.region, subthemeKey: produto.key, legendKey: legenda.key, pixelValue: legenda.items.map((i) => i.pixelValue),
        year: acumula ? [gifState.ini, a] : a, territoryId: territorio.id, spatialMethod: "union",
      });
      status(`Preparando os mapas de cada ano: ${++feitos}/${anos.length}`);
      prog.value = 0.15 * (feitos / anos.length);
      return r.url;
    });
    if (!vivo()) return;

    // ---- Layout do quadro ----
    const pad = Math.round(24 * s);
    const header = Math.round(78 * s);
    const meio = document.createElement("canvas").getContext("2d");
    const itensLeg = legenda.items.filter((i) => i.depth === 0);
    meio.font = `500 ${Math.round(14 * s)}px Montserrat, sans-serif`;
    const sw = Math.round(14 * s), gapLeg = Math.round(18 * s), alturaLinha = Math.round(24 * s);
    let lx = pad, linha = 0;
    const posLeg = itensLeg.map((it) => {
      const w = sw + 6 * s + meio.measureText(it.name).width;
      if (lx + w > W - pad && lx > pad) { lx = pad; linha++; }
      const pos = { it, x: lx, linha };
      lx += w + gapLeg;
      return pos;
    });
    const footer = Math.round(18 * s) + (linha + 1) * alturaLinha + Math.round(30 * s);

    // Caixa do território em Web Mercator (unidades de 0 a 1) e zoom dos tiles.
    let x1 = 1, x2 = 0, y1 = 1, y2 = 0;
    for (const anel of aneis) for (const [lon, lat] of anel) {
      const x = lonParaX(lon), y = latParaY(lat);
      if (x < x1) x1 = x; if (x > x2) x2 = x; if (y < y1) y1 = y; if (y > y2) y2 = y;
    }
    const margem = 0.05;
    const dx = (x2 - x1) * (1 + 2 * margem) || 1e-4, dy = (y2 - y1) * (1 + 2 * margem) || 1e-4;
    const MW = W - 2 * pad;
    const MH = Math.round(clamp(MW * (dy / dx), MW * 0.55, MW * 1.5));
    const H = header + MH + footer;
    const z = clamp(Math.floor(Math.log2(Math.min(MW / (dx * 256), MH / (dy * 256)))), 1, 13);
    const k = Math.min(MW / (dx * 256 * 2 ** z), MH / (dy * 256 * 2 ** z)); // escala dos tiles (1 a 2)
    const mundo = 256 * 2 ** z;
    const cx = ((x1 + x2) / 2) * mundo, cy = ((y1 + y2) / 2) * mundo;
    const tile = 256 * k;
    const txIni = Math.floor((cx - MW / (2 * k)) / 256), txFim = Math.floor((cx + MW / (2 * k)) / 256);
    const tyIni = Math.max(0, Math.floor((cy - MH / (2 * k)) / 256)), tyFim = Math.min(2 ** z - 1, Math.floor((cy + MH / (2 * k)) / 256));
    const tiles = [];
    for (let tx = txIni; tx <= txFim; tx++) for (let ty = tyIni; ty <= tyFim; ty++) tiles.push({ tx, ty });
    const telaX = (tx) => MW / 2 + (tx * 256 - cx) * k;
    const telaY = (ty) => MH / 2 + (ty * 256 - cy) * k;
    const novoCanvas = (w, h) => Object.assign(document.createElement("canvas"), { width: w, height: h });
    const desenharTiles = async (ctx, molde) => {
      const imgs = await mapLimit(tiles, 8, ({ tx, ty }) => {
        const n = 2 ** z, xx = ((tx % n) + n) % n;
        return carregarImagem(molde.replace("{z}", z).replace("{x}", xx).replace("{y}", ty));
      });
      imgs.forEach((img, i) => { if (img) ctx.drawImage(img, telaX(tiles[i].tx), telaY(tiles[i].ty), tile + 0.5, tile + 0.5); });
    };
    const caminho = (ctx, lista = aneis) => {
      ctx.beginPath();
      for (const anel of lista) anel.forEach(([lon, lat], i) => {
        const px = MW / 2 + (lonParaX(lon) * mundo - cx) * k, py = MH / 2 + (latParaY(lat) * mundo - cy) * k;
        if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      });
    };

    // Fundo (uma vez) e contorno.
    status("Desenhando o mapa base…");
    const fundo = novoCanvas(MW, MH);
    const fctx = fundo.getContext("2d");
    fctx.fillStyle = "#ffffff";
    fctx.fillRect(0, 0, MW, MH);
    if (gifState.fundo === "mapa") await desenharTiles(fctx, GIF_BASEMAP);
    else { caminho(fctx); fctx.fillStyle = "#f1f1f1"; fctx.fill("evenodd"); }
    const contorno = novoCanvas(MW, MH);
    const cctx = contorno.getContext("2d");
    // Limites de referência (ex.: biomas do Brasil) em cinza, por baixo do contorno do território.
    const cfgPais = PAISES.find((x) => x.region === p.region);
    const reLim = gifState.limites !== "nenhum" ? cfgPais?.cats?.[gifState.limites] : null;
    const catLim = reLim ? reg.categorias.find((c) => reLim.test(c.key)) : null;
    if (catLim) {
      status("Desenhando os limites…");
      const terrs = (await gifTerritorios(p.region, catLim.id)).filter((x) => !/\s-\s(costa|mar)$/i.test(x.name));
      const geoms = await mapLimit(terrs, 6, (x) => gifApi(p.region, "/territories/geometry", { territoryId: [x.id], spatialMethod: "union", simplify: "true" })
        .then((g) => g.geometry || g).catch(() => null));
      cctx.strokeStyle = "rgba(85,85,85,0.95)";
      cctx.lineWidth = Math.max(1.2, 1.2 * s);
      cctx.lineJoin = "round";
      for (const g of geoms) { const a = aneisDe(g); if (a.length) { caminho(cctx, a); cctx.stroke(); } }
    }
    caminho(cctx);
    cctx.strokeStyle = "#2b2b2b";
    cctx.lineWidth = Math.max(1, 1.3 * s);
    cctx.lineJoin = "round";
    cctx.stroke();
    const logo = await carregarLogo(MARCA[1]); // logo no idioma do quadro

    const { GIFEncoder, quantize, applyPalette } = await import(GIFENC_URL);
    const gif = GIFEncoder();
    const quadro = novoCanvas(W, H);
    const ctx = quadro.getContext("2d");
    const dados = novoCanvas(MW, MH);
    const dctx = dados.getContext("2d");
    // Subtemas chamados só "Anual" viram "Área queimada anual" no título do GIF.
    const nomeProduto = /^(anual|annual|tahunan)$/i.test(nomeEm(produto.name, lang).trim()) ? tx.anual : nomeEm(produto.name, lang);
    const titulo = `${tx.marca} · ${nomeProduto}`;
    const subtitulo = `${territorio.name}${legenda.nome ? ` · ${legenda.nome}` : ""}`;
    const fonte = `${tx.fonte}${reg.colecao ? ` · ${tx.colecao} ${reg.colecao}` : ""} · plataforma.mapbiomas.org`;

    for (let i = 0; i < anos.length; i++) {
      if (!vivo()) return;
      status(`Desenhando o quadro ${i + 1} de ${anos.length} (${anos[i]})…`);
      dctx.clearRect(0, 0, MW, MH);
      if (urls[i]) await desenharTiles(dctx, urls[i]);

      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, W, H);
      // Cabeçalho: título, território e logo.
      ctx.fillStyle = "#1f1f1f";
      ctx.textBaseline = "alphabetic";
      ctx.font = `700 ${Math.round(21 * s)}px Montserrat, sans-serif`;
      ctx.fillText(titulo, pad, Math.round(38 * s), W - 2 * pad - (logo ? 170 * s : 0));
      ctx.fillStyle = "#444444";
      ctx.font = `500 ${Math.round(15 * s)}px Montserrat, sans-serif`;
      ctx.fillText(subtitulo, pad, Math.round(62 * s), W - 2 * pad - (logo ? 170 * s : 0));
      if (logo) {
        const lh = Math.round(46 * s), lw = (logo.width / logo.height) * lh;
        ctx.drawImage(logo, W - pad - lw, Math.round(16 * s), lw, lh);
      }
      // Mapa.
      ctx.save();
      ctx.beginPath();
      ctx.rect(pad, header, MW, MH);
      ctx.clip();
      ctx.drawImage(fundo, pad, header);
      ctx.drawImage(dados, pad, header);
      ctx.drawImage(contorno, pad, header);
      ctx.restore();
      ctx.strokeStyle = "#d0d0d0";
      ctx.lineWidth = 1;
      ctx.strokeRect(pad + 0.5, header + 0.5, MW - 1, MH - 1);
      // Ano em destaque, com contorno branco para ler sobre o mapa.
      const rotulo = acumula ? `${gifState.ini}–${anos[i]}` : String(anos[i]);
      ctx.font = `800 ${Math.round(44 * s)}px Montserrat, sans-serif`;
      ctx.lineWidth = Math.round(6 * s);
      ctx.strokeStyle = "rgba(255,255,255,0.92)";
      ctx.fillStyle = "#1f1f1f";
      const ax = pad + Math.round(16 * s), ay = header + MH - Math.round(24 * s);
      ctx.strokeText(rotulo, ax, ay);
      ctx.fillText(rotulo, ax, ay);
      // Linha do tempo (quanto da série já passou).
      const barY = header + MH - Math.round(10 * s), barW = MW - Math.round(32 * s);
      ctx.fillStyle = "rgba(0,0,0,0.12)";
      ctx.fillRect(ax, barY, barW, Math.max(3, 4 * s));
      ctx.fillStyle = "#ff5340";
      ctx.fillRect(ax, barY, barW * ((i + 1) / anos.length), Math.max(3, 4 * s));
      // Legenda e fonte.
      ctx.font = `500 ${Math.round(14 * s)}px Montserrat, sans-serif`;
      ctx.textBaseline = "middle";
      const y0 = header + MH + Math.round(18 * s) + alturaLinha / 2;
      for (const { it, x, linha: l } of posLeg) {
        const yy = y0 + l * alturaLinha;
        ctx.fillStyle = it.color;
        ctx.fillRect(x, yy - sw / 2, sw, sw);
        ctx.strokeStyle = "rgba(0,0,0,0.25)";
        ctx.strokeRect(x + 0.5, yy - sw / 2 + 0.5, sw - 1, sw - 1);
        ctx.fillStyle = "#333333";
        ctx.fillText(it.name, x + sw + 6 * s, yy);
      }
      ctx.fillStyle = "#666666";
      ctx.font = `500 ${Math.round(12 * s)}px Montserrat, sans-serif`;
      ctx.fillText(fonte, pad, H - Math.round(16 * s), W - 2 * pad);

      let img;
      try {
        img = ctx.getImageData(0, 0, W, H).data;
      } catch {
        throw new Error("O navegador bloqueou a leitura das imagens do mapa. Abra o painel pelo site publicado (mapbiomas.github.io) em vez do arquivo local.");
      }
      const paleta = quantize(img, 256);
      gif.writeFrame(applyPalette(img, paleta), W, H, { palette: paleta, delay: i === anos.length - 1 ? delay * 3 : delay });
      prog.value = 0.15 + 0.85 * ((i + 1) / anos.length);
      await sleep(0); // deixa a página respirar entre quadros
    }
    gif.finish();
    const blob = new Blob([gif.bytes()], { type: "image/gif" });
    const png = await new Promise((resolve) => quadro.toBlob(resolve, "image/png"));
    if (gifState.resultado?.url) URL.revokeObjectURL(gifState.resultado.url);
    const sufixo = anos.length > 1 ? `${anos[0]}-${anos[anos.length - 1]}` : String(anos[0]);
    gifState.resultado = {
      url: URL.createObjectURL(blob), png, bytes: blob.size, quadros: anos.length,
      titulo: `${titulo} · ${territorio.name} · ${sufixo}`,
      nome: `mapbiomas_fogo_${nomeSeguro(p.nome)}_${nomeSeguro(nomeProduto)}_${nomeSeguro(territorio.name)}_${sufixo}.gif`.toLowerCase(),
    };
    if (vivo()) {
      mostrarResultadoGif(body);
      status("Pronto! Veja a prévia abaixo e baixe o GIF.");
      body.querySelector(".gif-resultado").scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "nearest" });
    }
  } catch (err) {
    console.error(err);
    if (vivo()) status(`Não foi possível gerar o GIF: ${err.message}`);
  } finally {
    gifState.gerando = false;
    if (vivo()) prog.hidden = true;
  }
}

// ---------------------------------------------------------------------------
// Utilitários
// ---------------------------------------------------------------------------

const fmtFull = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 0 });
const fmtCompact = new Intl.NumberFormat(LOCALE, { notation: "compact", maximumFractionDigits: 1 });
const fmtPct = new Intl.NumberFormat(LOCALE, { style: "percent", maximumFractionDigits: 1 });

// Hectares por extenso no idioma da página: "8,7 milhões de ha", "8,7 millones de ha", "8.7 million ha", "8,7 juta ha".
const PALAVRAS_HA = {
  pt: { um: "milhão", varios: "milhões", mil: "mil", de: " de" },
  es: { um: "millón", varios: "millones", mil: "mil", de: " de" },
  en: { um: "million", varios: "million", mil: "thousand", de: "" },
  id: { um: "juta", varios: "juta", mil: "ribu", de: "" },
};
const ha = (v) => {
  const n = v || 0;
  const p = PALAVRAS_HA[IDIOMA.id];
  const f = (x) => new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 }).format(x);
  if (Math.abs(n) >= 1e6) return `${f(n / 1e6)} ${Math.abs(n) < 2e6 ? p.um : p.varios}${p.de} ha`;
  if (Math.abs(n) >= 1e3) return `${f(n / 1e3)} ${p.mil} ha`;
  return `${f(n)} ha`;
};
const fmtMhaEixo = new Intl.NumberFormat(LOCALE, { maximumFractionDigits: 1 });
// Milhões de hectares: 18.410.000 -> "18,4 Mha".
const mha = (v, casas = 1) =>
  `${new Intl.NumberFormat(LOCALE, { maximumFractionDigits: casas }).format((v || 0) / 1e6)} Mha`;
const pct = (v) => fmtPct.format(Number.isFinite(v) ? v : 0);
const sum = (arr) => arr.reduce((s, v) => s + v, 0);
const mean = (arr) => (arr.length ? sum(arr) / arr.length : 0);
const safeDiv = (a, b) => (b ? a / b : 0);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const range = (a, b) => Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

// Texto no idioma da página, para frases com números no meio (o i18n.js troca só trechos fixos).
const porIdioma = (o) => o[IDIOMA.id] ?? o.pt;

function maxBy(arr, fn) {
  return arr.reduce((best, x) => (best === undefined || fn(x) > fn(best) ? x : best), undefined);
}

function findLastIndex(arr, fn) {
  for (let i = arr.length - 1; i >= 0; i--) if (fn(arr[i])) return i;
  return -1;
}

function norm(text) {
  return String(text || "").normalize("NFD").replace(/[̀-ͯ]/g, "").trim().toLowerCase();
}

function safeColor(color) {
  return /^#[0-9a-f]{3}([0-9a-f]{3})?$/i.test(color || "") ? color.toLowerCase() : "#9aa0a6";
}

// Escurece uma cor hex misturando-a com preto (amount 0–1).
function shade(hex, amount) {
  let h = hex.replace("#", "");
  if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h, 16);
  const canal = (v) => Math.round(v * (1 - amount)).toString(16).padStart(2, "0");
  return `#${canal((n >> 16) & 255)}${canal((n >> 8) & 255)}${canal(n & 255)}`;
}

// Cor na posição t ∈ [0, 1] de um degradê com paradas igualmente espaçadas.
function rampColor(stops, t) {
  const rgb = (hex) => { const n = parseInt(hex.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255]; };
  const x = Math.min(1, Math.max(0, Number.isFinite(t) ? t : 0)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const [a, b] = [rgb(stops[i]), rgb(stops[i + 1])];
  const f = x - i;
  return `#${a.map((v, k) => Math.round(v + (b[k] - v) * f).toString(16).padStart(2, "0")).join("")}`;
}

// Texto legível sobre um fundo (claro sobre cores escuras e vice-versa).
function textOn(hex) {
  const n = parseInt(String(hex).replace("#", "").padEnd(6, "0").slice(0, 6), 16);
  const lum = (0.299 * ((n >> 16) & 255) + 0.587 * ((n >> 8) & 255) + 0.114 * (n & 255)) / 255;
  return lum > 0.6 ? "#1f2328" : "#ffffff";
}

function withAlpha(hex, alpha) {
  let h = hex.replace("#", "");
  if (h.length === 3) h = [...h].map((c) => c + c).join("");
  const n = parseInt(h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

init();
