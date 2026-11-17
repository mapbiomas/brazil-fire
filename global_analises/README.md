# global_analises — Monitor global de fogo

Painel estático com ranking mensal de **área queimada**, **focos de calor** e **emissões**
por país e por estado/província, no mundo inteiro.

Página: https://mapbiomas.github.io/brazil-fire/global_analises/

## Fontes

| Indicador | Fonte | Período | Nível |
|---|---|---|---|
| Área queimada MCD64A1 (+ quebra por cobertura) | GWIS / JRC country profile | 2002–2025 | país, estado (GADM 1) |
| Área queimada VIIRS (VNP64) | GWIS | 2013–2025 | país, estado |
| Emissões CO₂, CH₄, PM2.5, CO (GFED) | GWIS | 2002–2025 | país, estado |
| Focos de calor (FIRMS MODIS, pixels-dia) | Google Earth Engine | 2024+ | país |
| Área queimada MCD64A1 depois do fim do GWIS | Google Earth Engine | 2026+ | país |

## Como atualizar

```bash
# 1. GWIS: baixe os ZIPs de https://gwis.jrc.ec.europa.eu/apps/country.profile/downloads
#    para global_analises/raw_gwis/ (fora do git): MCD64A1, MVP64, emission_gfed
# 2. GEE: uma task por mês (focos + área queimada), exporta para assets
python scripts/gee_run_exports.py --level 0 --start 2024-01 --end 2026-09
# 3. quando as tasks terminarem
python scripts/fetch_assets_to_csv.py --level 0      # -> data/gee_adm0_monthly.csv
# 4. monta os arquivos do painel
python scripts/build_from_gwis.py
```

`scripts/gee_export_mcd64a1_monthly.js` é a versão do export para o Code Editor (exporta para o Drive).

## Arquivos do painel

- `data/adm0_monthly.csv`: série por país
- `data/adm1/<ano>.csv`: estados/províncias, carregado por ano
- `data/meta.json`: períodos de cada fonte
