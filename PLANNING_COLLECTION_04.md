# Planejamento: Estrutura da Collection 04 (CONCLUÍDO)

Este arquivo serve como guia para a organização e lapidação da pasta `collection_04`.

## 🗺️ Mapa da Estrutura Atual (Implementada Localmente)

```text
mapbiomas_fire_collections/collection_04/
├── 01-pre_classification/
│   ├── 01-toolkit_for_collection_samples_and_export_mosaics_to_google_cloud.js
│   └── auxiliar/
│       ├── module-blockList.js
│       └── toolkit-investigate-scenes.js
├── 02-classification_algorithms/
│   ├── 01-imports.py
│   ├── 02-train.py
│   └── 03-prediction.py
├── 03-post_classification/
│   ├── 00-annual_classification/
│   │   ├── merge_collections_pampa.js
│   │   ├── export_col4_masks-regions_lulc9.js
│   │   └── export_col41_masks-regions_lulc10.js
│   ├── 01-subproducts_collection_04/
│   │   ├── 01-export_annual_and_monthly_burned_area.js
│   │   ├── 02-mask_and_reclassification.js
│   │   ├── 03-gap_fill_and_spatial_filter.js
│   │   ├── 04-temporal_filter.js
│   │   ├── 05-scar_size_range_by_period.js
│   │   ├── 06-export_vectorization_annual_burned_area.js
│   │   ├── 07-scar_size_range_by_year.js
│   │   └── README.md
│   └── 02-subproducts_collection_04_1/
│       ├── 01-export_annual_and_monthly_burned_area.js
│       ├── 02-mask_and_reclassification.js
│       ├── 03-gap_fill_and_spatial_filter.js
│       ├── 04-temporal_filter.js
│       ├── 05-scar_size_range_by_period.js
│       ├── 06-export_vectorization_annual_burned_area.js
│       ├── 07-scar_size_range_by_year.js
│       └── README.md
└── 04-statistics/
    ├── 01-subproducts_collection_04/
    │   ├── toDrive-area-ano_ultimo_fogo-col4.js
    │   ├── toDrive-area-burned_cover_acumullated_unplished_fund-col4.js
    │   ├── toDrive-area-cobertura_queimada_acumulado-col4.js
    │   ├── toDrive-area-cobertura_queimada_acumulado_fundiario-col4.js
    │   ├── toDrive-area-cobertura_queimada_acumulado_fundiario-col4_24.js
    │   ├── toDrive-area-cobertura_queimada_anual_col4.js
    │   ├── toDrive-area-cobertura_queimada_frequencia-col4.js
    │   ├── toDrive-area-cobertura_queimada_mensal-col4.js
    │   ├── toDrive-area-tamanho_de_cicatrizes-col4.js
    │   ├── toDrive-area-tempo_apos_fogo-col4.js
    │   ├── toDrive-area_queimada_acumulado-areas_protegidas.js
    │   ├── toDrive-area_queimada_acumulado-base_fundiaria_ipam.js
    │   ├── toDrive-area_queimada_acumulado_municipios.js
    │   ├── toDrive-area_queimada_acumulado_pelo_vigor_das_passagens_por bioma.js
    │   └── toDrive-area_queimada_anual_municipios.js
    └── 02-subproducts_collection_04_1/
        ├── toDrive-area-ano_ultimo_fogo-col41.js
        ├── toDrive-area-cobertura_queimada_acumulado-col41.js
        ├── toDrive-area-cobertura_queimada_anual-col41.js
        ├── toDrive-area-cobertura_queimada_frequencia-col41.js
        ├── toDrive-area-cobertura_queimada_mensal-col41.js
        ├── toDrive-area-tamanho_de_cicatrizes-col41.js
        └── toDrive-area_queimada_anual_municipios.js
```