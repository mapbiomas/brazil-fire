# CLAUDE.md · fire_global_monitor

Notas técnicas para quem mantém o monitor global. O que a ferramenta faz e como abrir está no `README.md`.

## O que é

Página estática (`app/index.html`, HTML + CSS + JS inline, sem build e sem bibliotecas externas). Gráficos e mapas são SVG desenhados à mão; o mapa usa a projeção Equal Earth implementada no próprio arquivo. Publicada pelo `.github/workflows/pages.yml` em `/global_analises/` (URL mantida da primeira versão, quando a pasta ficava na raiz como `global_analises/`).

## Como rodar e testar

- `python -m http.server` dentro de `app/` e abrir no navegador. Não há testes automatizados.
- Checagem de sintaxe do JS: extrair o `<script>` e rodar `node --check`.
- Scripts Python: `python -c "import ast; ast.parse(open(f).read())"`; os de GEE precisam de login (`scripts/ee_profiles.py`).

## Fonte da verdade

- GWIS country profile (ZIPs em `raw_gwis/`, fora do git): área queimada MCD64A1 e VNP64 por país e por estado GADM nível 1, com quebra por cobertura; emissões GFED. 2002–2025.
- Google Earth Engine: focos FIRMS (soma mensal de pixels-dia com `T21 > 0`, 1 km) e área queimada MCD64A1 depois do fim do GWIS; frequência de fogo. Conferido: MCD64A1 do GEE bate com o GWIS (Brasil set/2024 9,52 vs 9,54 Mha; mundo 100%).

## Decisões

- **Redução no GEE:** `reduceRegions` sobre polígonos globais falha na projeção sinusoidal do MODIS ("Unable to transform edge") e `getInfo` estoura o tempo. Usamos os limites rasterizados (`reduceToImage` com um id inteiro) e `Reducer.sum().repeat(n).group()` numa redução só, exportada como tabela. Export para asset exige geometria nas features (usamos um ponto fictício).
- **Frequência:** por período 2001–Y, soma das máscaras anuais (queimou em algum mês do ano), numa redução global única. Às vezes estoura o tempo ("Computation timed out"): reenviar só o período que falhou (o script pula os assets que já existem). **Não usar `--strips` > 1:** a divisão em faixas de longitude perdeu a faixa 135°E–180° inteira e áreas nas bordas (Austrália, Rússia oriental, Japão, Brasil…; 2026-11-17). Checagem obrigatória depois de cada fetch: a área acumulada do mundo tem que crescer com o ano final e a área total tem que ficar estável entre períodos. Estados guardam só as 5 classes (`c1…c5`) para o arquivo ficar leve; países guardam a frequência por número de anos (`f1…f25`).
- **Estados no mapa e na frequência:** os dados mensais de estado vêm do GWIS (GADM); o mapa e a frequência usam geoBoundaries ADM1 (GEE). O mapa casa os dois por nome (75% casam); a frequência por estado usa os nomes do geoBoundaries e a seleção nela vale só para o card.
- **Tarefas do GEE** sempre com descrição `WS_IPAM_GLOBAL_ANALISES-<arquivo>`. Projeto padrão `mapbiomas-brazil` (alternativa `mapbiomas-fire-485203`); `--account` escolhe o login salvo por `ee_profiles.py`.
- **Tradução:** textos em português no HTML/JS; `translateDOM` troca os nós de texto pelo dicionário `app/data/i18n.json` (frase inteira ou pedaços, com borda de palavra só nas pontas que são letras). Nomes de países vêm de `i18n/paises.csv`. Fora do português, as unidades viram k ha / kt / k para caber nas legendas.
- **Leitura de CSV no navegador:** sempre pelo `parseCSV` (respeita aspas); há nomes de países com vírgula.

## Problemas conhecidos

- Focos FIRMS em pixels-dia são bem maiores que as contagens de focos de boletins; servem para comparar países e meses.
- No chinês, alguns botões estreitos aparecem cortados.
- Unidades GADM sem par no geoBoundaries ficam cinza no mapa de estados.
