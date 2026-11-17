# Tradução do Monitor global de fogo

A página sai em português, inglês, espanhol, francês, indonésio e mandarim. As traduções ficam em:

- `traducoes.csv`: textos extraídos da página (`index.html`), com as colunas `en`, `es`, `fr`, `ind` e `zh`;
- `extra.csv`: palavras curtas e textos que vêm dos dados (meses, continentes, coberturas). A coluna `auto` = `nao` marca palavras que também são chaves no código e só são traduzidas onde o código chama `TR()`;
- `paises.csv`: nomes dos países. A base GWIS traz os nomes em inglês (`nome_gwis`); a coluna `en` já vem preenchida com eles.

Quando a página ganhar textos novos, rode `python i18n/extrair_textos.py` dentro de `global_analises/`. Ele grava `i18n/textos_novos.csv` só com o que ainda não tem tradução. Traduza esse arquivo com o texto abaixo e junte as linhas ao `traducoes.csv`.

Cole o texto abaixo junto com os três CSVs na IA que vai traduzir.

---

Preencha as colunas `en` (inglês), `es` (espanhol), `fr` (francês), `ind` (indonésio) e `zh` (mandarim, chinês simplificado) dos CSVs anexos com a tradução da coluna `pt`. É a interface de um painel científico sobre fogo no mundo: área queimada, focos de calor e emissões por continente, país e estado/província (dados MODIS MCD64A1 e GFED agregados pelo GWIS/JRC, e NASA FIRMS).

Regras:

1. Devolva os mesmos CSVs, com as mesmas colunas, linhas e ids, na mesma ordem. Altere só as colunas de idioma (e, no `paises.csv`, também a coluna `pt`).
2. Não altere as células que já estão preenchidas.
3. Muitas linhas são pedaços de frases que o código monta, como `' vs. '` ou `'. Setas: posição vs. '`. Mantenha os espaços do início e do fim exatamente como estão, e também a pontuação das pontas. Use a coluna `contexto` (o trecho de código ou a tag HTML de onde o texto saiu) para entender onde o pedaço entra. Se a ordem das palavras no outro idioma exigir outra montagem da frase, traduza o pedaço do melhor jeito possível e escreva `[REVISAR]` no fim da tradução.
4. Linhas que são só código, nome de variável ou caminho de arquivo (ex.: `global_analises/scripts/`) ficam iguais nas colunas de idioma.
5. Não traduza: siglas e nomes de produtos e instituições (MODIS, MCD64A1, VIIRS, VNP64, GFED, GWIS, JRC, Copernicus, FIRMS, NASA, Google Earth Engine, GEE, GADM, geoBoundaries, Natural Earth, MapBiomas), fórmulas químicas (CO₂, CH₄, PM2.5), unidades (ha, t, Mha, Mt) e símbolos (×, ·, ↗, ▲, ▼).
6. Em `paises.csv`: preencha `pt` com o nome do país em português do Brasil e as demais colunas com o nome usual em cada idioma. Mantenha `en` como está, a não ser que esteja claramente errado (nesse caso, corrija e escreva `[REVISAR]`).
7. Use o inglês dos EUA, com tom técnico e direto, e frases curtas como no original.
8. Números e datas dentro do texto ficam iguais. A formatação numérica é feita pelo código. Os meses abreviados (`jan`, `fev`…) devem continuar abreviados, com até 3 ou 4 letras.
9. Use sempre os mesmos termos:

| pt | en |
|---|---|
| Área queimada | Burned area |
| Focos de calor | Active fires (hotspots) |
| Focos | Hotspots |
| Emissão / emissões | Emissions |
| % do território | % of territory |
| Cobertura (do solo) | Land cover |
| Floresta / Savana / Campo / arbustivo / Agricultura | Forest / Savanna / Grassland / shrubland / Cropland |
| Continente / País / Estado / província | Continent / Country / State / province |
| América do Norte e Central | North and Central America |
| Recorte | Selection |
| Sazonalidade | Seasonality |
| Série anual | Annual series |
| Ano todo | Full year |
| Painel | Dashboard |
| Método | Methods |
| Página exploratória, não oficial | Unofficial, exploratory page |
| mil / milhões | thousand / million |
