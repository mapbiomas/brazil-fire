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

const CONFIG = {
  colecaoExigida: "5.1",
  region: "brazil",
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

// Como achar cada subtema na árvore do tema de fogo (válido em prd e stg).
const SUBTHEMES = {
  anual: { name: "Anual" },
  mensal: { name: "Mensal" },
  cicatriz: { name: "Tamanho da cicatriz" },
  retorno: { name: "Retorno do último fogo" },
  acumulado: { analysis: "accumulated" },
  frequencia: { analysis: "frequency" },
  mediaRetorno: { analysis: "mean_interval" },
  ultimoFogo: { analysis: "last_year" },
  severidade: { name: "Severidade" },
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
const corBioma = (nome) => BIOMA_CORES[norm(nome)] || "#9aa0a6";

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
const corPaleta = (nome) => PALETAS_BIOMA[state.paletaBioma]?.cores[ordemBioma(nome)] || corBioma(nome);

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

const TABS = [
  { id: "anual", label: "Fogo Anual", load: loadAnual },
  { id: "acumulado", label: "Fogo Acumulado", load: loadAcumulado },
  { id: "mensal", label: "Fogo Mensal", load: loadMensal },
  { id: "frequencia", label: "Frequência do Fogo", load: loadFrequencia },
  { id: "ultimoFogo", label: "Ano do Último Fogo", load: loadUltimoFogo },
  { id: "cicatriz", label: "Tamanho de Cicatriz", load: loadCicatriz },
  { id: "severidade", label: "Severidade", load: loadSeveridade },
  { id: "retorno", label: "Intervalo de Retorno", load: loadRetorno },
  { id: "fundiario", label: "Categorias Fundiárias", load: loadFundiario },
  { id: "territorios", label: "Territórios", load: loadTerritorios },
  { id: "mapa", label: "Mapas Interativos", load: loadMapa },
];

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
  mapa: { tipo: "bruto", unidade: "estados", camada: "acumulado", visualizar: "biomas" },
  freqGrupo: "todas", // Frequência: "todas" (1 a N vezes), "15" (1 a 15 e 16+), "9" (1 a 9 e 10+)
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
        "ou reduza o período/recorte. Séries longas para o Brasil inteiro podem levar mais de 10 min."
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

function readInformativeNote(fire) {
  const text = JSON.stringify([fire.description, fire.informativeNote]);
  const colecao =
    (text.match(/Cole[cç][aã]o\s+(\d+(?:\.\d+)?)\s+do\s+MapBiomas\s+Fogo/i) ||
      text.match(/(?:Fire|Fogo)\s+Collection\s+(\d+(?:\.\d+)?)/i) || [])[1] || null;
  const anos = text.match(/(?:entre|between|from)\s+(\d{4})\s+(?:e|and|to)\s+(\d{4})/i);
  return {
    colecao,
    anoMin: anos ? Number(anos[1]) : CONFIG.anoMinPadrao,
    anoMax: anos ? Number(anos[2]) : CONFIG.anoMaxPadrao,
  };
}

async function discoverEnvironment(env) {
  const [themes, categories] = await Promise.all([fetchThemes(env), fetchTerritoryCategories(env)]);

  const fire = (themes.themes || []).find(
    (t) => /^(fire|fogo)/i.test(t.key) || norm(t.name?.["pt-BR"]) === "fogo"
  );
  if (!fire) throw new Error("tema de fogo não encontrado");

  // prd aninha subtemas em `children`; stg em `subthemes`.
  const leaves = (fire.subthemes || []).flatMap((group) => group.children || group.subthemes || []);
  const subthemes = {};
  for (const [id, rule] of Object.entries(SUBTHEMES)) {
    const found = leaves.find((s) =>
      rule.analysis ? s.analysis === rule.analysis : norm(s.name?.["pt-BR"]) === norm(rule.name)
    );
    if (!found) throw new Error(`subtema "${rule.name || rule.analysis}" não encontrado`);
    subthemes[id] = found.key;
  }

  const cats = categories.categories || [];
  const findCat = (re) => cats.find((c) => re.test(c.key));
  const catBrasil = findCat(/^BRASIL|^POLITICAL_LEVEL_1$/);
  const catBioma = findCat(/^BIOMES$/);
  const catEstado = findCat(/^ESTADOS|^POLITICAL_LEVEL_3$/);
  const catMunicipio = findCat(/^MUNICIPIOS_IBGE|^POLITICAL_LEVEL_4$/);
  if (!catBrasil || !catBioma || !catEstado || !catMunicipio) throw new Error("categorias de território não encontradas");

  const [brasil, biomas, estados] = await Promise.all([
    fetchTerritories(env, catBrasil.id),
    fetchTerritories(env, catBioma.id),
    fetchTerritories(env, catEstado.id),
  ]);
  const territoryBrasil = (brasil.territories || [])[0];
  if (!territoryBrasil) throw new Error("território Brasil não encontrado");

  const byName = (a, b) => a.name.localeCompare(b.name, "pt-BR");
  return {
    env,
    subthemes,
    ...readInformativeNote(fire),
    brasil: { id: territoryBrasil.id, name: "Brasil" },
    biomas: (biomas.territories || []).map((t) => ({ id: t.id, name: t.name })).sort(byName),
    estados: (estados.territories || [])
      .map((t) => ({ id: t.id, name: t.name, uf: UF_POR_ESTADO[norm(t.name)] }))
      .sort(byName),
    catMunicipio: catMunicipio.id,
  };
}

async function connect() {
  const failures = [];
  let alternativa = null;
  for (const env of ENVIRONMENTS) {
    try {
      const conn = await discoverEnvironment(env);
      if (conn.colecao === CONFIG.colecaoExigida) return { ...conn, failures, colecaoOk: true };
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
    const chosen = def.match ? ordered.find(def.match) : ordered[0];
    if (!chosen) throw new Error(`Legenda "${id}" não encontrada no subtema ${subthemeKey}.`);
    const detail = await fetchLegendDetail(env, chosen.id);
    const legend = detail.legend || detail;
    return { key: legend.key, items: flattenLegend(legend.items) };
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
    const resp = await fetchRanking(state.conn.env, {
      subthemeKey: sub("anual"), legendKey: total.key, pixelValue: pixelValues(levelItems(total, 0)),
      year: ano, territoryCategoryId: categoryId, pageSize: 6000,
    });
    return new Map((resp.ranking || []).map((r) => [r.territory.id, { name: r.territory.name, value: r.value || 0 }]));
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
    const [w, s, e, n] = BRASIL_BBOX;
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
      name: item.name?.["pt-BR"] || item.name?.["en-US"] || item.key,
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
  if (!partes.length) return { ids: [c.brasil.id], nome: "Brasil", metodo: "union" };
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
  return {
    chart: {
      kind: "stacked", height: 300, big: true, unit: "Mha", axisTitle,
      labels: anos.map(String), series: series.filter((s) => sum(s.values) > 0),
    },
    nota: base.length ? "Parte do recorte dentro de cada bioma · Mha = milhões de hectares" : "Mha = milhões de hectares",
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
    subtitle = "Área queimada acumulada no período ÷ área total do bioma · primeira rosca: o Brasil todo";
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
  const [total, cobertura, natural] = await Promise.all([
    getLegend("anualTotal"), getLegend("anualCobertura"), getLegend("anualNatural"),
  ]);
  const niveis = range(0, maxDepth(cobertura) + 1);
  if (!niveis.includes(state.nivel)) state.nivel = 1;
  const legenda = state.nivel === 0 ? natural : cobertura;
  const classes = state.nivel === 0 ? levelItems(natural, 0) : cutLevel(cobertura, state.nivel - 1);
  const totalItems = levelItems(total, 0);
  const naturalItems = levelItems(natural, 0);
  const anosPeriodo = range(ini, fim);
  const pedir = (lg, itens) => getStats({
    subthemeKey: sub("anual"), legendKey: lg.key, pixelValue: pixelValues(itens), year: anosPeriodo, ...territoryParams(t),
  });

  const [resTotal, resClasses, resNatural] = await Promise.all([
    pedir(total, totalItems), pedir(legenda, classes), pedir(natural, naturalItems),
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

  return {
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
        note: areaTerritorio ? `de ${ha(areaTerritorio)} do recorte` : "Área do território não informada pela API",
      },
    ],
    cards: [
      {
        span: 12,
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
          { label: `Média ${ini}–${fim}: ${mha(media)} por ano`, color: "var(--text-2)", line: true, dash: true },
          { label: "Mha = milhões de hectares", note: true },
        ],
      },
      {
        span: 12,
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
            { label: "Natural x antrópico", segments: naturalAntropico, nomes: true },
            ...(state.nivel === 0 ? [] : [{ label: "Por classe", segments: ordenarPorGrupo(porClasse) }]),
          ],
          legend: state.nivel === 0 ? naturalAntropico : ordenarPorGrupo(porClasse),
        },
        alterna: "rosca",
        rosca: [
          { titulo: "Natural x antrópico", slices: naturalAntropico, centerLabel: "queimados" },
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
          { label: "Mha = milhões de hectares", note: true },
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

async function loadMensal({ t, ini, fim }) {
  const [legenda, total] = await Promise.all([getLegend("mensal"), getLegend("anualTotal")]);
  const meses = levelItems(legenda, 0);
  const res = await getStats({
    subthemeKey: sub("mensal"), legendKey: legenda.key, pixelValue: pixelValues(meses),
    year: range(ini, fim), ...territoryParams(t),
  });

  const anos = byYear(res);
  const n = anos.length;
  // Matriz ano × mês -> soma, média e desvio padrão amostral (n − 1), como a Tabela 9 do RAF.
  const porMes = meses.map((m) => anos.map((a) => a.valores.get(m.pixelValue) || 0));
  const somas = porMes.map(sum);
  const totalPeriodo = sum(somas);
  const medias = porMes.map(mean);
  const dps = porMes.map((vals, i) => (n > 1 ? Math.sqrt(sum(vals.map((v) => (v - medias[i]) ** 2)) / (n - 1)) : 0));
  const picoIdx = medias.indexOf(Math.max(...medias));
  const curtos = meses.map((m) => m.name.slice(0, 3));
  // Trimestre consecutivo de maior soma (o RAF destaca agosto–outubro).
  const trimestres = meses.map((_, i) => ({ i, soma: somas[i] + somas[(i + 1) % 12] + somas[(i + 2) % 12] }));
  const trimestre = maxBy(trimestres, (x) => x.soma);
  const nomeTri = trimestre ? `${curtos[trimestre.i]}–${curtos[(trimestre.i + 2) % 12]}` : "—";

  // Por padrão destaca o ano final em vermelho, como a Figura 13 do RAF.
  const disponiveis = anos.map((a) => a.ano);
  if (!state.mensalAnosTocados) state.mensalAnos = [fim];
  state.mensalAnos = state.mensalAnos.filter((a) => disponiveis.includes(a));
  const selecionados = state.mensalAnos.map((ano, i) => {
    const linha = anos.find((a) => a.ano === ano);
    return { label: String(ano), ...SERIES_COLORS[i], values: meses.map((m) => linha.valores.get(m.pixelValue) || 0) };
  });
  const anoFim = anos.find((a) => a.ano === fim);
  const picoFim = anoFim ? maxBy(meses, (m) => anoFim.valores.get(m.pixelValue) || 0) : null;
  const semFiltroLocal = !state.estadoId && !state.municipioId;
  const vermelho = levelItems(total, 0)[0].color;

  return {
    toolbar: [{
      type: "chips",
      label: `Anos em destaque (até ${SERIES_COLORS.length})`,
      options: disponiveis.map((a) => {
        const slot = SERIES_COLORS[state.mensalAnos.indexOf(a)];
        return { value: a, label: String(a), color: slot?.color, dash: !!slot?.dash };
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
    }],
    kpis: [
      { label: "Mês de pico (média histórica)", value: meses[picoIdx]?.name || "—", note: `${ha(medias[picoIdx])} ± ${ha(dps[picoIdx])} · ${pct(safeDiv(somas[picoIdx], totalPeriodo))} da área do período` },
      { label: "Trimestre de maior área queimada", value: nomeTri, note: `${pct(safeDiv(trimestre?.soma || 0, totalPeriodo))} da área queimada em ${ini}–${fim}` },
      {
        label: `Mês de pico em ${fim}`,
        value: picoFim?.name || "—",
        note: picoFim && meses[picoIdx] && picoFim.pixelValue !== meses[picoIdx].pixelValue
          ? `Deslocado em relação à média (${meses[picoIdx].name})`
          : "Igual ao pico da média histórica",
      },
    ],
    cards: [
      {
        span: 8,
        title: `Média mensal da área queimada ${ini}–${fim}`,
        subtitle: n > 1
          ? `Área cinza = média (n = ${n} anos) · faixa clara = ± 1 desvio padrão · linhas = anos em destaque`
          : "Escolha um período com mais de um ano para a média e o desvio padrão",
        chart: {
          kind: "band",
          labels: curtos,
          tooltipTitles: meses.map((m) => m.name),
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
      {
        span: 4,
        title: "Distribuição mensal",
        subtitle: "Soma no período · % do total · média ± DP",
        ranking: {
          plain: true,
          total: totalPeriodo,
          rows: meses.map((m, i) => ({
            label: m.name,
            value: somas[i],
            color: m.color,
            detail: `média ${fmtCompact.format(medias[i])} ± ${fmtCompact.format(dps[i])} ha por ano`,
          })),
        },
      },
      {
        span: 12,
        title: `Proporção mensal da área queimada por bioma · ${ini}–${fim}`,
        subtitle: semFiltroLocal
          ? "Para cada bioma, % da sua área queimada no período que ocorreu em cada mês (como a Figura 14 do RAF)"
          : "Biomas inteiros (este gráfico não aplica o filtro de estado/município)",
        lazy: async (progress) => {
          let feitos = 0;
          const biomas = state.conn.biomas;
          const series = await mapLimit(biomas, 3, async (b) => {
            const r = await getStats({
              subthemeKey: sub("mensal"), legendKey: legenda.key, pixelValue: pixelValues(meses),
              year: range(ini, fim), territoryId: [b.id], spatialMethod: "union",
            });
            progress(`Calculando biomas… ${++feitos}/${biomas.length}`);
            const somaMes = meses.map((m) => sum(byYear(r).map((a) => a.valores.get(m.pixelValue) || 0)));
            const tot = sum(somaMes);
            return { label: b.name, color: corBioma(b.name), values: somaMes.map((v) => safeDiv(v, tot)) };
          });
          return {
            chart: { kind: "grouped", percent: true, labels: meses.map((m) => m.name.toLowerCase()), series },
            legend: series.map((s) => ({ label: s.label, color: s.color })),
          };
        },
      },
    ],
  };
}

async function loadFrequencia({ t, ini, fim }) {
  const legenda = await getLegend("frequencia");
  const classes = levelItems(legenda, 1); // "1 vez", "2 vezes", …
  const faixas = levelItems(legenda, 0);
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
  const porFaixa = faixasComDados.map((f, i) => ({
    label: frequencyLabel(f.name),
    color: rampColor(FREQ_RAMP, safeDiv(i, faixasComDados.length - 1)),
    value: sum(porClasse.filter((c) => c.item.parent === f.pixelValue).map((c) => c.value)),
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

  return {
    toolbar: [{
      type: "segmented",
      label: "Classes de frequência",
      value: state.freqGrupo,
      options: ["todas", "15", "9"].map((v) => ({ value: v, label: rotuloGrupo[v] })),
      onChange: (v) => { state.freqGrupo = v; loadCurrentTab(); },
    }],
    kpis: [
      { label: `Área queimada ao menos 1 vez (${ini}–${fim})`, value: ha(total), note: "Soma de todas as classes de frequência" },
      { label: "Queimou 2 vezes ou mais", value: ha(total - umaVez), note: `${pct(safeDiv(total - umaVez, total))} da área queimada` },
      {
        label: "Maior frequência observada",
        value: visiveis.length ? `${visiveis[visiveis.length - 1].short} vez(es)` : "—",
        note: `em ${fim - ini + 1} ano(s) de período`,
      },
    ],
    cards: [
      {
        span: 5,
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
        span: 7,
        title: "Área por número de vezes que queimou",
        subtitle: `${ini}–${fim} · classes ${rotuloGrupo[state.freqGrupo]}`,
        alterna: "rosca",
        chart: {
          kind: "bar",
          axisTitle: "Área queimada (ha)",
          categoryTitle: "Número de vezes que queimou",
          labels: barras.map((c) => c.short),
          tooltipTitles: barras.map((c) => c.label),
          values: barras.map((c) => c.value),
          colors: barras.map((c) => c.color),
        },
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
  const somaGrandes = (valores) => sum(grandes.map((g) => somaGrupo(g, valores)));
  const anosComFogo = anos.filter((a) => a.area > 0);

  return {
    kpis: [
      { label: `Área queimada ${ini}–${fim}`, value: ha(total), note: "Soma das áreas anuais em todas as faixas de tamanho" },
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
        title: "Proporção em cicatrizes grandes por ano",
        subtitle: `Faixas ${grandes.map((g) => g.label).join(" e ")}, sobre a área queimada no ano`,
        alterna: "linhas",
        chart: {
          kind: "line",
          percent: true,
          labels: anosComFogo.map((a) => String(a.ano)),
          values: anosComFogo.map((a) => safeDiv(somaGrandes(a.valores), a.area)),
          color: grandes[grandes.length - 1]?.color,
        },
      },
    ],
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

async function loadRetorno({ t, ini, fim }) {
  const [lgRetorno, lgMedia] = await Promise.all([getLegend("retorno"), getLegend("mediaRetorno")]);
  const faixasRetorno = levelItems(lgRetorno, 0);
  const faixasMedia = levelItems(lgMedia, 0);

  const [retorno, media] = await Promise.all([
    getStats({
      subthemeKey: sub("retorno"), legendKey: lgRetorno.key, pixelValue: pixelValues(faixasRetorno),
      year: fim, ...territoryParams(t),
    }),
    getStats({
      subthemeKey: sub("mediaRetorno"), legendKey: lgMedia.key, pixelValue: pixelValues(faixasMedia),
      year: [ini, fim], ...territoryParams(t),
    }),
  ]);

  const dRetorno = totalsByClass(faixasRetorno, retorno);
  const dMedia = totalsByClass(faixasMedia, media);
  const totRetorno = sum(dRetorno.map((d) => d.value));
  const totMedia = sum(dMedia.map((d) => d.value));
  const curto = dRetorno[0];
  const predominante = maxBy(dMedia, (d) => d.value);

  return {
    kpis: [
      {
        label: `Retorno em ${curto ? curto.label : "—"} (${fim})`,
        value: pct(safeDiv(curto?.value || 0, totRetorno)),
        note: `${ha(curto?.value || 0)} de ${ha(totRetorno)} queimados com fogo anterior`,
      },
      {
        label: `Intervalo médio predominante (${ini}–${fim})`,
        value: predominante?.label || "—",
        note: predominante ? `${pct(safeDiv(predominante.value, totMedia))} da área com recorrência` : "",
      },
    ],
    cards: [
      {
        span: 7,
        title: `Tempo desde o fogo anterior · ${fim}`,
        subtitle: "Áreas queimadas no ano final que já tinham queimado antes",
        alterna: "rosca",
        chart: {
          kind: "bar",
          labels: dRetorno.map((d) => d.label),
          values: dRetorno.map((d) => d.value),
          colors: dRetorno.map((d) => d.color),
        },
      },
      {
        span: 5,
        title: `Intervalo médio de retorno · ${ini}–${fim}`,
        subtitle: "Média de anos entre queimadas sucessivas",
        ranking: { rows: dMedia, total: totMedia, plain: true },
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
      { label: `Área com fogo entre ${ini} e ${fim}`, value: ha(total), note: "Cada área conta uma vez, no ano do último fogo" },
      { label: `Último fogo nos últimos 5 anos (${Math.max(ini, fim - 4)}–${fim})`, value: pct(safeDiv(recentes, total)), note: ha(recentes) },
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
      { label: `Área afetada ${ini}–${fim}`, value: ha(total), note: "Soma anual das classes de severidade potencial (beta)" },
      {
        label: `Severidade ${altas[0]?.name || "alta"} ou maior`,
        value: pct(safeDiv(sum(comDados.map((a) => somaAltas(a.valores))), total)),
        note: altas.map((c) => c.name).join(" · "),
      },
      { label: "Classe predominante", value: predominante?.label || "—", note: predominante ? pct(safeDiv(predominante.value, total)) : "" },
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
  const filtro = (l) => (!recorte || recorte.has(l.id)) && (!estado || l.name.endsWith(` - ${estado.uf}`));
  const partes = [estado && `de ${estado.name}`, bioma && `no ${bioma.name}`].filter(Boolean);
  return {
    filtro,
    recorte,
    biomaId: bioma?.id,
    biomaNome: bioma?.name,
    rotulo: partes.length ? `Municípios ${partes.join(" ")}` : "Municípios do Brasil",
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
  const catEstado = await categoriaDe(c.estados[0].id);
  const catBioma = await categoriaDe(c.biomas[0].id);

  const progresso = (rotulo) => (n, tot) => setStatus(`${rotulo}: ${n}/${tot} anos`, true);
  const [estados, biomas] = await Promise.all([
    rankingPeriodo(catEstado, ini, fim, progresso("Ranking de estados")),
    rankingPeriodo(catBioma, ini, fim, progresso("Ranking de biomas")),
  ]);
  const totalBrasil = sum(biomas.linhas.map((l) => l.value));
  const top3 = estados.linhas.slice(0, 3);
  const avisoBioma = bioma ? " · estados inteiros (a API de ranking não recorta por bioma)" : "";
  const ultimo = estados.anos.length - 1;
  const totalFimBrasil = sum(biomas.linhas.map((l) => l.porAno[ultimo] || 0));

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
    const mun = await rankingPeriodo(c.catMunicipio, ini, fim, progresso("Ranking de municípios"));
    setStatus("");
    const r = aplicarRecorte(mun.linhas.filter(escopo.filtro), escopo, ini, fim);
    return { escopo, linhas: r.linhas, ultimo: mun.anos.length - 1, aviso: avisoRecorte(escopo, r) };
  })();
  municipios.catch(() => {});

  return {
    kpis: [
      {
        label: "Estado com mais área queimada",
        value: estados.linhas[0]?.name || "—",
        note: estados.linhas[0] ? `${ha(estados.linhas[0].value)} · ${pct(safeDiv(estados.linhas[0].value, totalBrasil))} do Brasil` : "",
      },
      {
        label: "Concentração nos 3 maiores estados",
        value: pct(safeDiv(sum(top3.map((l) => l.value)), totalBrasil)),
        note: top3.map((l) => l.name).join(", "),
      },
      {
        label: "Bioma com mais área queimada",
        value: biomas.linhas[0]?.name || "—",
        note: biomas.linhas[0] ? `${pct(safeDiv(biomas.linhas[0].value, totalBrasil))} do Brasil` : "",
      },
    ],
    cards: [
      {
        span: 6,
        title: "Estados que mais queimaram",
        subtitle: `Soma das áreas anuais ${ini}–${fim} · top 10${avisoBioma}`,
        chart: hbarRanking(estados.linhas.slice(0, 10), vermelho, totalBrasil, state.estadoId),
      },
      {
        span: 6,
        title: "Municípios que mais queimaram no período",
        subtitle: `Soma das áreas anuais ${ini}–${fim} · top 10`,
        lazy: async (progress) => {
          progress("Baixando ranking de municípios…");
          const m = await municipios;
          const base = sum(m.linhas.map((l) => l.value));
          return {
            chart: hbarRanking(m.linhas.slice(0, 10), vermelho, base, state.municipioId),
            note: `${m.escopo.rotulo} · % sobre o total desses municípios. ${m.aviso}`,
          };
        },
      },
      {
        span: 6,
        title: `Variação em ${fim} em relação à média ${ini}–${fim - 1}`,
        subtitle: "Estados com ao menos 1% da área queimada do Brasil no ano · vermelho = acima da média, azul = abaixo",
        chart: variacao.length
          ? {
            kind: "hbar",
            percent: true,
            labels: variacao.map((l) => l.name),
            tooltipTitles: variacao.map((l) => `${l.name}: ${ha(l.porAno[ultimo])} em ${fim} · média ${ha(l.media)}`),
            values: variacao.map((l) => l.delta),
            colors: variacao.map((l) => (l.delta >= 0 ? "#e34948" : "#2a78d6")),
          }
          : { kind: "hbar", labels: [], values: [] },
      },
      {
        span: 6,
        title: `Municípios que mais queimaram em ${fim}`,
        subtitle: `Área queimada no ano final do período · top 10`,
        lazy: async (progress) => {
          progress("Baixando ranking de municípios…");
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
        title: "Evolução anual dos 3 estados com mais área queimada",
        subtitle: `${ini}–${fim} · cores categóricas (não há cor oficial por estado)`,
        chart: {
          kind: "multiline",
          labels: estados.anos.map(String),
          series: top3.map((l, i) => ({ label: l.name, color: TERRITORY_COLORS[i], values: l.porAno })),
        },
        legend: top3.map((l, i) => ({ label: l.name, color: TERRITORY_COLORS[i], line: true })),
      },
      {
        span: 8,
        title: "Participação dos biomas na área queimada, por ano",
        subtitle: `${ini}–${fim} · cada ano soma 100% da área queimada do Brasil`,
        alterna: "linhas",
        chart: {
          kind: "stacked",
          percent: true,
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
        title: "Biomas: área queimada no período",
        subtitle: "Soma das áreas anuais e participação no Brasil",
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
  const cats = await cached(`${envId()}:cats`, () => fetchTerritoryCategories(c.env));
  const re = c.biomas.some((b) => b.id === territoryId) ? /^BIOMES$/ : /^ESTADOS|^POLITICAL_LEVEL_3$/;
  return cats.categories.find((x) => re.test(x.key)).id;
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
  const camada = MAP_LAYERS.find((c) => c.id === state.mapa.camada) || MAP_LAYERS[0];
  const legenda = await getLegend(camada.legend);
  const anos = camada.periodo ? [ini, fim] : fim;

  const [tileUrl, ativo, contornos] = await Promise.all([
    getMapUrl({
      subthemeKey: sub(camada.subtheme), legendKey: legenda.key, pixelValue: pixelValues(legenda.items),
      year: anos, ...territoryParams(t),
    }),
    getGeometry(t.ids, t.metodo).catch(() => null),
    referenceOutlines(state.mapa.visualizar),
  ]);

  const legendItems = levelItems(legenda, 0).map((i) => ({
    label: camada.id === "frequencia" ? frequencyLabel(i.name) : i.name,
    color: i.color,
  }));

  return {
    toolbar: [
      mapTypeControl(),
      {
        type: "select",
        label: "Camada",
        value: camada.id,
        options: MAP_LAYERS.map((c) => ({ value: c.id, label: c.label })),
        onChange: (v) => { state.mapa.camada = v; loadCurrentTab(); },
      },
      {
        type: "segmented",
        label: "Contornos",
        value: state.mapa.visualizar,
        options: MAP_VIEWS.map((v) => ({ value: v.id, label: v.label })),
        onChange: (v) => { state.mapa.visualizar = v; loadCurrentTab(); },
      },
    ],
    kpis: [],
    cards: [{
      span: 12,
      title: camada.label.replace("(ano fim)", `· ${fim}`).replace("(período)", `· ${ini}–${fim}`),
      subtitle: `Recorte: ${t.nome} · tiles oficiais do MapBiomas Fogo`,
      map: { tileUrl, ativo, contornos: contornos.geometrias, aviso: contornos.aviso },
      legend: legendItems,
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

  let linhas, features, escopoRotulo, aviso = "", emCalculo = false;
  if (unidade === "estados") {
    const catEstado = await categoriaDe(c.estados[0].id);
    const [rk, geoms] = await Promise.all([
      rankingPeriodo(catEstado, ini, fim, progresso("Ranking de estados (anos)")),
      mapLimit(c.estados, 6, (e) => getGeometry([e.id]).catch(() => null)),
    ]);
    linhas = rk.linhas;
    const valor = new Map(linhas.map((l) => [l.id, l.value]));
    features = c.estados.map((e, i) => ({
      type: "Feature", geometry: geoms[i], properties: { id: e.id, name: e.name, value: valor.get(e.id) || 0 },
    })).filter((f) => f.geometry);
    escopoRotulo = "Estados do Brasil";
    if (state.biomaId) aviso = "Estados inteiros: a API de ranking não recorta por bioma.";
  } else {
    const [escopo, rk, shapes] = await Promise.all([
      municipiosNoEscopo((msg) => setStatus(msg, true)),
      rankingPeriodo(c.catMunicipio, ini, fim, progresso("Ranking de municípios (anos)")),
      getMunicipiosShapes(progresso("Geometria dos municípios (tiles)")),
    ]);
    const r = aplicarRecorte(rk.linhas.filter(escopo.filtro), escopo, ini, fim);
    linhas = r.linhas;
    const valor = new Map(linhas.map((l) => [l.id, l]));
    // Com bioma: só a parte de cada município dentro dele (geometria recortada).
    const geometrias = escopo.recorte ? [...escopo.recorte.values()] : [...shapes.values()];
    // Um polígono por pedaço de tile: os pedaços de tiles vizinhos se sobrepõem na faixa de
    // buffer e, num mesmo MultiPolygon, o preenchimento even-odd do canvas abriria faixas vazias.
    features = geometrias
      .filter((m) => escopo.filtro({ id: m.id, name: m.name }))
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

  return {
    toolbar: [
      mapTypeControl(),
      {
        type: "segmented",
        label: "Pintar por",
        value: unidade,
        options: [{ value: "estados", label: "Estados" }, { value: "municipios", label: "Municípios" }],
        onChange: (v) => { state.mapa.unidade = v; loadCurrentTab(); },
      },
    ],
    kpis: [],
    cards: [
      {
        span: 8,
        title: `${escopoRotulo}: área queimada ${ini}–${fim}`,
        subtitle: `Soma das áreas anuais (ranking oficial da API) · 5 classes por quantis${unidade === "municipios" && !state.biomaId && !state.estadoId ? " · escolha um bioma ou estado no filtro para recortar" : ""}`,
        map: { choropleth: { features, contorno: recorte, bordas: unidade === "estados" }, aviso },
        legend: [
          ...classes.legend,
          { label: "Sem área queimada", color: "var(--surface-muted)" },
          ...(emCalculo ? [{ label: "Na divisa · área exata em cálculo", color: corEmCalculo }] : []),
        ],
      },
      {
        span: 4,
        title: unidade === "estados" ? "Estados que mais queimaram" : "Municípios que mais queimaram",
        subtitle: `Top 10 · cor = classe no mapa`,
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

async function referenceOutlines(visualizar) {
  const c = state.conn;
  if (visualizar === "biomas") {
    return { geometrias: await Promise.all(c.biomas.map((b) => getGeometry([b.id]).catch(() => null))) };
  }
  if (visualizar === "estados") {
    return { geometrias: await Promise.all(c.estados.map((e) => getGeometry([e.id]).catch(() => null))) };
  }
  if (state.municipioId) {
    // Com bioma ativo, só a parte do município dentro do bioma (interseção feita pela API).
    const geom = state.biomaId
      ? await getGeometry([state.municipioId, state.biomaId], "intersection").catch(() => null)
      : await getGeometry([state.municipioId]).catch(() => null);
    return { geometrias: [geom] };
  }
  return { geometrias: [], aviso: "Escolha um estado e um município nos filtros para recortar e destacar o município." };
}

// ---------------------------------------------------------------------------
// Renderização
// ---------------------------------------------------------------------------

function renderView(view) {
  destroyCharts();
  const tokens = themeTokens();
  aplicarPadroesChart(tokens);
  const t = activeTerritory();
  el.panel.innerHTML = `
    <div class="panel-head">
      <p class="context"><strong>${escapeHtml(t.nome)}</strong> · ${state.ini}–${state.fim}</p>
      ${headControls(view)}
    </div>
    ${view.kpis.length ? `<div class="kpis">${view.kpis.map(kpiHtml).join("")}</div>` : ""}
    <div class="grid">${view.cards.map((card, i) => cardHtml(card, i, view)).join("")}</div>`;

  view.toolbar?.forEach((ctl, i) => bindToolbar(ctl, i));
  const seq = state.seq;
  view.cards.forEach((card, i) => {
    const root = el.panel.querySelector(`[data-card="${i}"]`);
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
  anual: [["barras", "Barras"], ["linhas", "Linhas"], ["biomas", "Por bioma"]],
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
            paletaToggleHtml();
          registrarGrafico(root, new Chart(body.querySelector("canvas"), chartConfig(spec, tokens)));
          body.querySelector(".paleta-toggle").addEventListener("click", (e) => {
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
        ${ctl.options.map((o) => `<option value="${escapeHtml(o.value)}" ${o.value === ctl.value ? "selected" : ""}>${escapeHtml(o.label)}</option>`).join("")}
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
  const itens = (view.toolbar || []).map((ctl, i) => (ctl.inSection ? "" : toolbarHtml(ctl, i))).join("");
  return itens.trim() ? `<div class="toolbar">${itens}</div>` : "";
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
  return `
    <article class="card span-${card.span}${card.chart?.big ? " card-big" : ""}" data-card="${i}">
      <header class="card-head${toggle ? " has-toggle" : ""}">
        <div>
          <h2>${escapeHtml(card.title)}</h2>
          ${card.subtitle ? `<p>${escapeHtml(card.subtitle)}</p>` : ""}
        </div>
        ${toggle}
      </header>
      <div class="card-body">${cardBodyHtml(card, vistaDe(card, i))}</div>
    </article>`;
}

// Corpo do card na vista atual, com a legenda do card (a vista por bioma traz a sua ao carregar).
function cardBodyHtml(card, vista) {
  if (vista === "biomas") return `<p class="empty busy-text">Calculando biomas…</p>`;
  return conteudoHtml(card, vista) + (card.legend?.length ? legendRowHtml(card.legend) : "");
}

function conteudoHtml(card, vista) {
  if (vista === "rosca") return roscasHtml(roscasDe(card));
  if (card.topcards) return topcardsHtml(card.topcards);
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
  return `<ul class="legend-row">${items.map((it) => (it.note
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
        <div class="donut-center"><strong>${ha(total)}</strong><span>${escapeHtml(centerLabel)}</span></div>
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
  const map = L.map(box, { zoomSnap: 1, attributionControl: true });
  state.map = map;

  // Basemap sem chave de API (o da CARTO passou a exigir chave).
  const canvas = dark ? "World_Dark_Gray_Base" : "World_Light_Gray_Base";
  L.tileLayer(`https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/${canvas}/MapServer/tile/{z}/{y}/{x}`, {
    maxZoom: 16,
    attribution: "Tiles &copy; Esri &mdash; Esri, HERE, Garmin, &copy; OpenStreetMap",
  }).addTo(map);

  if (spec.choropleth) return renderChoropleth(map, spec.choropleth, tokens);

  L.tileLayer(spec.tileUrl, {
    maxZoom: 18,
    opacity: 0.95,
    attribution: `MapBiomas Fogo · Coleção ${escapeHtml(state.conn.colecao || "?")}`,
  }).addTo(map);

  const contornos = spec.contornos.filter(Boolean);
  if (contornos.length) {
    L.geoJSON(contornos, {
      interactive: false,
      style: { color: tokens.muted, weight: 0.8, fill: false },
    }).addTo(map);
  }

  let bounds = null;
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

function chartConfig(spec, t) {
  const options = baseOptions(t);
  const horizontal = spec.kind === "hbar";
  const stacked = spec.kind === "stacked" || spec.kind === "combo";
  const multi = stacked || spec.kind === "grouped";
  const valueOf = (ctx) => (horizontal ? ctx.parsed.x : ctx.parsed.y);
  const fmtValor = spec.percent ? (v) => pct(v)
    : spec.unit === "Mha" ? (v) => mha(v, 2)
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
  }
  if (spec.axisTitle) {
    valueAxis.title = { display: true, text: spec.axisTitle, color: t.text2, font: { size: fonte + 1, weight: "600" } };
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
  options.cutout = "70%";
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

function renderTabs() {
  el.tabs.innerHTML = TABS.map((t) => `
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
  el.bioma.innerHTML = optionsHtml(c.biomas, "Todos (Brasil)");
  el.estado.innerHTML = optionsHtml(c.estados, "Todos os estados");
  el.municipio.innerHTML = optionsHtml([], "Todos os municípios");

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
    el.municipio.innerHTML = optionsHtml([], "Todos os municípios");
    el.municipio.disabled = true;
    return;
  }
  el.municipio.disabled = true;
  el.municipio.innerHTML = `<option value="">Carregando municípios…</option>`;
  try {
    const porUf = await getMunicipiosPorUf();
    state.municipiosDoEstado = porUf.get(estado.uf) || [];
    el.municipio.innerHTML = optionsHtml(state.municipiosDoEstado, `Todos os municípios (${state.municipiosDoEstado.length})`);
    if (!state.municipiosDoEstado.some((m) => m.id === state.municipioId)) state.municipioId = "";
    el.municipio.value = state.municipioId;
    el.municipio.disabled = false;
  } catch (err) {
    console.error(err);
    state.municipioId = "";
    el.municipio.innerHTML = `<option value="">Falha ao carregar municípios</option>`;
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
  document.getElementById("colecao-titulo").textContent = `Col. ${conn.colecao || "?"}`;
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
  if (["estados", "municipios"].includes(p.get("unidade"))) state.mapa.unidade = p.get("unidade");
  if (MAP_LAYERS.some((c) => c.id === p.get("camada"))) state.mapa.camada = p.get("camada");
  if (MAP_VIEWS.some((v) => v.id === p.get("vis"))) state.mapa.visualizar = p.get("vis");
  if (["todas", "15", "9"].includes(p.get("freq"))) state.freqGrupo = p.get("freq");
}

function writeHash() {
  const p = new URLSearchParams({ aba: state.tab, ini: state.ini, fim: state.fim });
  if (state.biomaId) p.set("bioma", state.biomaId);
  if (state.estadoId) p.set("estado", state.estadoId);
  if (state.municipioId) p.set("municipio", state.municipioId);
  if (state.tab === "anual") p.set("nivel", state.nivel);
  if (state.tab === "mensal" && state.mensalAnos.length) p.set("anos", state.mensalAnos.join(","));
  if (state.tab === "frequencia") p.set("freq", state.freqGrupo);
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
  fillFilters();
  await fillMunicipios();
  for (const s of [el.bioma, el.estado, el.ini, el.fim]) s.disabled = false;
  await document.fonts?.ready;
  loadCurrentTab();
}

function init() {
  Chart.defaults.font.family = "'Montserrat', system-ui, -apple-system, 'Segoe UI', sans-serif";
  Chart.register(crosshairPlugin);
  readHash();
  renderTabs();

  el.tabs.addEventListener("click", (e) => {
    const btn = e.target.closest("[role=tab]");
    if (btn) selectTab(btn.dataset.tab);
  });
  el.tabs.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
    const i = TABS.findIndex((t) => t.id === state.tab);
    const next = (i + (e.key === "ArrowRight" ? 1 : -1) + TABS.length) % TABS.length;
    selectTab(TABS[next].id, true);
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
// Utilitários
// ---------------------------------------------------------------------------

const fmtFull = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 0 });
const fmtCompact = new Intl.NumberFormat("pt-BR", { notation: "compact", maximumFractionDigits: 1 });
const fmtPct = new Intl.NumberFormat("pt-BR", { style: "percent", maximumFractionDigits: 1 });

const ha = (v) => `${fmtCompact.format(v || 0)} ha`;
const fmtMhaEixo = new Intl.NumberFormat("pt-BR", { maximumFractionDigits: 1 });
// Milhões de hectares: 18.410.000 -> "18,4 Mha".
const mha = (v, casas = 1) =>
  `${new Intl.NumberFormat("pt-BR", { maximumFractionDigits: casas }).format((v || 0) / 1e6)} Mha`;
const pct = (v) => fmtPct.format(Number.isFinite(v) ? v : 0);
const sum = (arr) => arr.reduce((s, v) => s + v, 0);
const mean = (arr) => (arr.length ? sum(arr) / arr.length : 0);
const safeDiv = (a, b) => (b ? a / b : 0);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const range = (a, b) => Array.from({ length: Math.max(0, b - a + 1) }, (_, i) => a + i);
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

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
