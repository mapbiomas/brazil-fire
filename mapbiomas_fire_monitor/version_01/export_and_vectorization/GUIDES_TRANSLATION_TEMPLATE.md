# Guias do app Export & Vectorization — Tradução PT → 6 idiomas

Este documento é um **template de tradução**: o conteúdo em Português (abaixo) é a
fonte e o objetivo é gerar as guias em **ES, EN, ID, FR, NL, ZH** no formato de
saída indicado. Envie este arquivo (ou só a seção "Conteúdo PT" + "Formato de
saída") para uma IA especialista em idiomas.

## Regras obrigatórias

1. Traduza o conteúdo **PT abaixo** para cada idioma, mantendo **TODAS as chaves**
   do schema.
2. **Preserve** as tags HTML (`<b>`, `<code>`) e emojis/símbolos (🔗, ⤓, ↓, └─).
   Os grafos são blocos de texto monoespaçado com `↓`/`└─` — traduza somente as
   palavras.
3. Mantenha os termos: **unidade** (banda/imagem), **Etapa**, **Load Data**,
   **Sync**, **Catálogo**, **config.py**, **catalog_cache.json**.
4. `name` = nome do idioma nativo; `tab_title` = "Guia: <Nome no idioma>".
5. `welcome_note` curto (1 parágrafo); as demais strings concisas.
6. **NÃO repita o conteúdo PT** e **NÃO acrescente explicações**. Saída única no
   formato abaixo (apenas `GUIDES_EXTRA`, sem PT).

## Schema (todas as chaves obrigatórias)

| chave | tipo | obs |
|-------|------|-----|
| `name` / `tab_title` | str | nome do idioma + título da aba |
| `welcome_note` | str | boas-vindas (1 parágrafo) |
| `what` | str | o que o app faz |
| `howto_title` | str | "Como usar" traduzido |
| `steps` | list[str] | 6 passos, com `<b>`/`<code>` |
| `cols_title` | str | "Colunas da grade" traduzido |
| `cols` | list[[col, desc]] | 7 pares (mesma ordem) |
| `links` | str | dica dos badges |
| `legend` | str | legenda OK/MISS/N/A |
| `graphs_title` | str | "Gráficos" traduzido |
| `graphs` | list[{title, lines}] | 5 grafos; `lines` = linhas do diagrama de texto |

## Conteúdo PT (fonte)

```
name: "Português"
tab_title: "Guia: Português"
welcome_note: "Bem-vindo! Esta guia em português explica o aplicativo Export &
  Vectorization do Monitor do Fogo MapBiomas: como navegar (país → tema → coleção →
  produto), descobrir as unidades, processar cada etapa e publicar os mapas. As abas
  acima mostram a interface; use esta guia sempre que precisar."
what: "O aplicativo exporta os mapas de área queimada/incêndio do MapBiomas Fire:
  do Earth Engine (GEE) para o GCS, monta mosaicos (COG) por unidade (banda ou imagem),
  vetoriza quando aplicável, publica no Earth Engine e no bucket público e remove os
  arquivos temporários."
howto_title: "Como usar"
steps:
  1. "<b>Navegue</b> pelas abas: país → tema → coleção → produto. Com muitos
     produtos, as guias quebram em várias linhas — verde = carregado, cinza =
     não carregado."
  2. "Clique em <b>Load Data</b> para carregar o produto atual (unidades da
     memória, descobrir dados novos e verificar o status, com limite de
     <code>SCAN_TIMEOUT</code>) — ou em <b>Load Collection</b> para carregar
     todos os produtos da coleção em fila."
  3. "Na grade (checkbox na primeira coluna), marque as unidades desejadas. Use
     <b>Select Pending</b> para selecionar por estágio (o título mostra o
     estágio-alvo e cada clique avança no ciclo), <b>Select All</b> ou
     <b>Select All Collection</b>. O filtro <code>Unit:</code> (padrão
     “All units”) restringe por prefixo; acima de 60 unidades inicia no
     prefixo recente."
  4. "Execute as etapas na ordem: <b>Export → Mosaico → Publicar mosaico → Limpar
     temp</b>. A vetorização é opcional (Steps 5–7, só produtos vetorizáveis,
     ex.: annual_burned): <b>Vetor GCS → Vetor GEE → Publicar vetor</b>."
  5. "Para refazer uma etapa, ative <code>FORCE_&lt;ETAPA&gt; = True</code> na célula
     da etapa e selecione as unidades na grade."
  6. "Para versionar a memória do catálogo (quando houver dados novos no
     <code>config.py</code>), use o botão <b>⤓ Catalog cache (.json)</b> na barra
     inferior e suba o arquivo no GitHub."
cols_title: "Colunas da grade"
cols:
  1. [Export, unidade exportada do GEE (temp/)]
  2. [Mosaic, COG montado]
  3. [Public mosaic, COG espelhado no bucket público]
  4. [Vector GCS, vetor zipado (só produtos vetorizáveis)]
  5. [Vector GEE, FeatureCollection no Earth Engine]
  6. [Public vector, ZIP espelhado no bucket público]
  7. [Clean temp, tiles temporários removidos após consolidação]
links: "Badges <b>🔗 OK</b> abrem o link de download; <b>Vector GEE</b> copia o asset ID."
legend: "OK = etapa concluída | MISS = etapa pendente | N/A = não se aplica"
graphs_title: "Gráficos"
graphs:
  G1 Navegação:
    País
    └─ Tema
       └─ Coleção
          └─ Produto
             └─ Unidades (bandas ou imagens)
  G2 Etapas (produto vetorizável):
    1 Export → tiles 0/1 (temp/)
     ↓
    2 Mosaico → COG
     ↓
    3 Publicar mosaico → bucket público
     ↓
    4 Limpar temp
     ↓
    5 Vetor GCS → ZIP
     ↓
    6 Vetor GEE → FeatureCollection
     ↓
    7 Publicar vetor → ZIP público
  G3 Etapas (demais produtos):
    1 Export
     ↓
    2 Mosaico
     ↓
    3 Publicar mosaico
     ↓
    4 Limpar temp
  G4 Memória do catálogo:
    config.py (dado cru)
     ↓
    Load Data (descobre unidades sob demanda)
     ↓
    catalog_cache.json (memória entre sessões)
     ↓
    Botão ⤓ Catalog cache (.json)
     ↓
    GitHub (versionar)
  G5 Fluxo de dados:
    GEE ImageCollection
     ↓
    Export → tiles 0/1 (temp/)
     ↓
    Mosaico → COG
     ↓
    Publicar → bucket público
     ↓
    (se vetorizável) Vetorização → ZIP + upload GEE
```

## Formato de saída (único bloco)

Retorne **apenas** um bloco de código Python com o dict `GUIDES_EXTRA` contendo os
6 idiomas (sem PT, sem explicações):

```python
GUIDES_EXTRA = {
    "ES": {
        "name": "Español[cite: 1]",
        "tab_title": "Guía: Español[cite: 1]",
        "welcome_note": "¡Bienvenido! Esta guía en español explica la aplicación Export & Vectorization del Monitor do Fogo de MapBiomas: cómo navegar (país → tema → colección → producto), descubrir las unidades, procesar cada Etapa y publicar los mapas. Las pestañas de arriba muestran la interfaz; use esta guía siempre que la necesite.[cite: 1]",
        "what": "La aplicación exporta los mapas de área quemada/incendio de MapBiomas Fire: de Earth Engine (GEE) a GCS, monta mosaicos (COG) por unidad (banda o imagen), vectoriza cuando corresponde, publica en Earth Engine y en el bucket público y elimina los archivos temporales.[cite: 1]",
        "howto_title": "Cómo usar[cite: 1]",
        "steps": [
            "<b>Navegue</b> por las pestañas: país → tema → colección → producto.[cite: 1]",
            "Haga clic en <b>Load Data</b> (botón rojo parpadeante) para descubrir las <b>unidades</b>: bandas (imagen multibanda) o imágenes (ImageCollection). El descubrimiento es bajo demanda; el caché no se llena con datos no cargados.[cite: 1]",
            "En la cuadrícula, marque las unidades deseadas. El filtro <code>Unit:</code> (predeterminado “All units”) restringe por prefijo de unidad.[cite: 1]",
            "Haga clic en <b>Sync</b> para verificar el estado de las etapas. El escaneo se ejecuta en segundo plano, con indicador de progreso (el kernel no se bloquea).[cite: 1]",
            "Ejecute las etapas en orden: <b>Export → Mosaico → Publicar mosaico → Vector GCS → Vector GEE → Publicar vector → Limpiar temp</b>. Etapas 4–6 solo para productos vectorizables (ej.: annual_burned); en los demás: <b>Export → Mosaico → Publicar mosaico → Limpiar temp</b>.[cite: 1]",
            "Para rehacer una Etapa, active <code>FORCE_&lt;ETAPA&gt; = True</code> en la celda de la Etapa y seleccione las unidades en la cuadrícula.[cite: 1]",
            "Para versionar la memoria del Catálogo (cuando haya datos nuevos en <code>config.py</code>), use el botón <b>⤓ Catalog cache (.json)</b> en la barra inferior y suba el archivo a GitHub.[cite: 1]"
        ],
        "cols_title": "Columnas de la cuadrícula[cite: 1]",
        "cols": [
            ["Export[cite: 1]", "unidade exportada de GEE (temp/)[cite: 1]"],
            ["Mosaic[cite: 1]", "COG ensamblado[cite: 1]"],
            ["Public mosaic[cite: 1]", "COG replicado en el bucket público[cite: 1]"],
            ["Vector GCS[cite: 1]", "vector comprimido (solo productos vectorizables)[cite: 1]"],
            ["Vector GEE[cite: 1]", "FeatureCollection en Earth Engine[cite: 1]"],
            ["Public vector[cite: 1]", "ZIP replicado en el bucket público[cite: 1]"],
            ["Clean temp[cite: 1]", "tiles temporales eliminados tras consolidación[cite: 1]"]
        ],
        "links": "Insignias <b>🔗 OK</b> abren el enlace de descarga; <b>Vector GEE</b> copia el asset ID.[cite: 1]",
        "legend": "OK = Etapa completada | MISS = Etapa pendiente | N/A = no aplica[cite: 1]",
        "graphs_title": "Gráficos[cite: 1]",
        "graphs": [
            {
                "title": "G1 Navegación:[cite: 1]",
                "lines": [
                    "País[cite: 1]",
                    "└─ Tema[cite: 1]",
                    "   └─ Colección[cite: 1]",
                    "      └─ Producto[cite: 1]",
                    "         └─ Unidades (bandas o imágenes)[cite: 1]"
                ]
            },
            {
                "title": "G2 Etapas (producto vectorizable):[cite: 1]",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mosaico → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publicar mosaico → bucket público[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Vector GCS → ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "5 Vector GEE → FeatureCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "6 Publicar vector → ZIP público[cite: 1]",
                    " ↓[cite: 1]",
                    "7 Limpiar temp[cite: 1]"
                ]
            },
            {
                "title": "G3 Etapas (demás productos):[cite: 1]",
                "lines": [
                    "1 Export[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mosaico[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publicar mosaico[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Limpiar temp[cite: 1]"
                ]
            },
            {
                "title": "G4 Memoria del Catálogo:[cite: 1]",
                "lines": [
                    "config.py (dato crudo)[cite: 1]",
                    " ↓[cite: 1]",
                    "Load Data (descubre unidades bajo demanda)[cite: 1]",
                    " ↓[cite: 1]",
                    "catalog_cache.json (memoria entre sesiones)[cite: 1]",
                    " ↓[cite: 1]",
                    "Botón ⤓ Catalog cache (.json)[cite: 1]",
                    " ↓[cite: 1]",
                    "GitHub (versionar)[cite: 1]"
                ]
            },
            {
                "title": "G5 Flujo de datos:[cite: 1]",
                "lines": [
                    "GEE ImageCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "Mosaico → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "Publicar → bucket público[cite: 1]",
                    " ↓[cite: 1]",
                    "(si vectorizable) Vectorización → ZIP + upload GEE[cite: 1]"
                ]
            }
        ]
    },
    "EN": {
        "name": "English[cite: 1]",
        "tab_title": "Guide: English[cite: 1]",
        "welcome_note": "Welcome! This English guide explains the MapBiomas Fire Export & Vectorization app: how to navigate (country → theme → collection → product), discover the unidades, process each Etapa, and publish the maps. The tabs above show the interface; use this guide whenever you need it.[cite: 1]",
        "what": "The app exports burned area/fire maps from MapBiomas Fire: from Earth Engine (GEE) to GCS, builds mosaics (COG) per unidade (band or image), vectorizes when applicable, publishes to Earth Engine and the public bucket, and removes temporary files.[cite: 1]",
        "howto_title": "How to use[cite: 1]",
        "steps": [
            "<b>Navigate</b> through the tabs: country → theme → collection → product.[cite: 1]",
            "Click <b>Load Data</b> (pulsing red button) to discover the <b>unidades</b>: bands (multiband image) or images (ImageCollection). Discovery is on-demand — the cache is not filled with unloaded data.[cite: 1]",
            "In the grid, check the desired unidades. The <code>Unit:</code> filter (default “All units”) restricts by unidade prefix.[cite: 1]",
            "Click <b>Sync</b> to check the status of the etapas. The scan runs in the background, with a progress indicator (the kernel does not block).[cite: 1]",
            "Execute the etapas in order: <b>Export → Mosaic → Publish mosaic → Vector GCS → Vector GEE → Publish vector → Clean temp</b>. Etapas 4–6 are only for vectorizable products (e.g., annual_burned); for others: <b>Export → Mosaic → Publish mosaic → Clean temp</b>.[cite: 1]",
            "To redo an Etapa, activate <code>FORCE_&lt;ETAPA&gt; = True</code> in the Etapa cell and select the unidades in the grid.[cite: 1]",
            "To version the Catálogo memory (when there is new data in <code>config.py</code>), use the <b>⤓ Catalog cache (.json)</b> button in the bottom bar and upload the file to GitHub.[cite: 1]"
        ],
        "cols_title": "Grid columns[cite: 1]",
        "cols": [
            ["Export[cite: 1]", "unidade exported from GEE (temp/)[cite: 1]"],
            ["Mosaic[cite: 1]", "assembled COG[cite: 1]"],
            ["Public mosaic[cite: 1]", "COG mirrored in the public bucket[cite: 1]"],
            ["Vector GCS[cite: 1]", "zipped vector (vectorizable products only)[cite: 1]"],
            ["Vector GEE[cite: 1]", "FeatureCollection in Earth Engine[cite: 1]"],
            ["Public vector[cite: 1]", "ZIP mirrored in the public bucket[cite: 1]"],
            ["Clean temp[cite: 1]", "temporary tiles removed after consolidation[cite: 1]"]
        ],
        "links": "<b>🔗 OK</b> badges open the download link; <b>Vector GEE</b> copies the asset ID.[cite: 1]",
        "legend": "OK = Etapa completed | MISS = Etapa pending | N/A = not applicable[cite: 1]",
        "graphs_title": "Graphs[cite: 1]",
        "graphs": [
            {
                "title": "G1 Navigation:[cite: 1]",
                "lines": [
                    "Country[cite: 1]",
                    "└─ Theme[cite: 1]",
                    "   └─ Collection[cite: 1]",
                    "      └─ Product[cite: 1]",
                    "         └─ Unidades (bands or images)[cite: 1]"
                ]
            },
            {
                "title": "G2 Etapas (vectorizable product):[cite: 1]",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mosaic → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publish mosaic → public bucket[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Vector GCS → ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "5 Vector GEE → FeatureCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "6 Publish vector → public ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "7 Clean temp[cite: 1]"
                ]
            },
            {
                "title": "G3 Etapas (other products):[cite: 1]",
                "lines": [
                    "1 Export[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mosaic[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publish mosaic[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Clean temp[cite: 1]"
                ]
            },
            {
                "title": "G4 Catálogo memory:[cite: 1]",
                "lines": [
                    "config.py (raw data)[cite: 1]",
                    " ↓[cite: 1]",
                    "Load Data (discovers unidades on demand)[cite: 1]",
                    " ↓[cite: 1]",
                    "catalog_cache.json (memory between sessions)[cite: 1]",
                    " ↓[cite: 1]",
                    "Button ⤓ Catalog cache (.json)[cite: 1]",
                    " ↓[cite: 1]",
                    "GitHub (versioning)[cite: 1]"
                ]
            },
            {
                "title": "G5 Data flow:[cite: 1]",
                "lines": [
                    "GEE ImageCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "Mosaic → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "Publish → public bucket[cite: 1]",
                    " ↓[cite: 1]",
                    "(if vectorizable) Vectorization → ZIP + upload GEE[cite: 1]"
                ]
            }
        ]
    },
    "ID": {
        "name": "Bahasa Indonesia[cite: 1]",
        "tab_title": "Panduan: Bahasa Indonesia[cite: 1]",
        "welcome_note": "Selamat datang! Panduan dalam bahasa Indonesia ini menjelaskan aplikasi Export & Vectorization dari MapBiomas Fire: cara navigasi (negara → tema → koleksi → produk), menemukan unidades, memproses setiap Etapa, dan memublikasikan peta. Tab di atas menampilkan antarmuka; gunakan panduan ini kapan pun Anda butuhkan.[cite: 1]",
        "what": "Aplikasi ini mengekspor peta area terbakar/kebakaran dari MapBiomas Fire: dari Earth Engine (GEE) ke GCS, membuat mosaik (COG) per unidade (band atau citra), melakukan vektorisasi jika berlaku, memublikasikan ke Earth Engine dan bucket publik, serta menghapus file sementara.[cite: 1]",
        "howto_title": "Cara penggunaan[cite: 1]",
        "steps": [
            "<b>Navigasi</b> melalui tab: negara → tema → koleksi → produk.[cite: 1]",
            "Klik <b>Load Data</b> (tombol merah berdenyut) untuk menemukan <b>unidades</b>: band (citra multiband) atau citra (ImageCollection). Penemuan dilakukan berdasarkan permintaan — cache tidak diisi dengan data yang tidak dimuat.[cite: 1]",
            "Pada kisi, centang unidades yang diinginkan. Filter <code>Unit:</code> (default “All units”) membatasi berdasarkan awalan unidade.[cite: 1]",
            "Klik <b>Sync</b> untuk memeriksa status etapas. Pemindaian berjalan di latar belakang, dengan indikator kemajuan (kernel tidak terblokir).[cite: 1]",
            "Jalankan etapas secara berurutan: <b>Export → Mosaik → Publikasikan mosaik → Vektor GCS → Vektor GEE → Publikasikan vektor → Bersihkan temp</b>. Etapas 4–6 hanya untuk produk yang dapat divektorisasi (misalnya, annual_burned); untuk yang lain: <b>Export → Mosaik → Publikasikan mosaik → Bersihkan temp</b>.[cite: 1]",
            "Untuk mengulang Etapa, aktifkan <code>FORCE_&lt;ETAPA&gt; = True</code> di sel Etapa dan pilih unidades di kisi.[cite: 1]",
            "Untuk membuat versi memori Catálogo (saat ada data baru di <code>config.py</code>), gunakan tombol <b>⤓ Catalog cache (.json)</b> di bilah bawah dan unggah file ke GitHub.[cite: 1]"
        ],
        "cols_title": "Kolom kisi[cite: 1]",
        "cols": [
            ["Export[cite: 1]", "unidade diekspor dari GEE (temp/)[cite: 1]"],
            ["Mosaic[cite: 1]", "COG yang digabungkan[cite: 1]"],
            ["Public mosaic[cite: 1]", "COG dicerminkan di bucket publik[cite: 1]"],
            ["Vector GCS[cite: 1]", "vektor dalam format ZIP (hanya produk yang dapat divektorisasi)[cite: 1]"],
            ["Vector GEE[cite: 1]", "FeatureCollection di Earth Engine[cite: 1]"],
            ["Public vector[cite: 1]", "ZIP dicerminkan di bucket publik[cite: 1]"],
            ["Clean temp[cite: 1]", "tile sementara dihapus setelah konsolidasi[cite: 1]"]
        ],
        "links": "Lencana <b>🔗 OK</b> membuka tautan unduhan; <b>Vector GEE</b> menyalin asset ID.[cite: 1]",
        "legend": "OK = Etapa selesai | MISS = Etapa tertunda | N/A = tidak berlaku[cite: 1]",
        "graphs_title": "Grafik[cite: 1]",
        "graphs": [
            {
                "title": "G1 Navigasi:[cite: 1]",
                "lines": [
                    "Negara[cite: 1]",
                    "└─ Tema[cite: 1]",
                    "   └─ Koleksi[cite: 1]",
                    "      └─ Produk[cite: 1]",
                    "         └─ Unidades (band atau citra)[cite: 1]"
                ]
            },
            {
                "title": "G2 Etapas (produk dapat divektorisasi):[cite: 1]",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mosaik → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publikasikan mosaik → bucket publik[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Vektor GCS → ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "5 Vektor GEE → FeatureCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "6 Publikasikan vektor → ZIP publik[cite: 1]",
                    " ↓[cite: 1]",
                    "7 Bersihkan temp[cite: 1]"
                ]
            },
            {
                "title": "G3 Etapas (produk lainnya):[cite: 1]",
                "lines": [
                    "1 Export[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mosaik[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publikasikan mosaik[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Bersihkan temp[cite: 1]"
                ]
            },
            {
                "title": "G4 Memori Catálogo:[cite: 1]",
                "lines": [
                    "config.py (data mentah)[cite: 1]",
                    " ↓[cite: 1]",
                    "Load Data (menemukan unidades berdasarkan permintaan)[cite: 1]",
                    " ↓[cite: 1]",
                    "catalog_cache.json (memori antar sesi)[cite: 1]",
                    " ↓[cite: 1]",
                    "Tombol ⤓ Catalog cache (.json)[cite: 1]",
                    " ↓[cite: 1]",
                    "GitHub (pembuatan versi)[cite: 1]"
                ]
            },
            {
                "title": "G5 Alur data:[cite: 1]",
                "lines": [
                    "GEE ImageCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "Mosaik → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "Publikasikan → bucket publik[cite: 1]",
                    " ↓[cite: 1]",
                    "(jika dapat divektorisasi) Vektorisasi → ZIP + upload GEE[cite: 1]"
                ]
            }
        ]
    },
    "FR": {
        "name": "Français[cite: 1]",
        "tab_title": "Guide : Français[cite: 1]",
        "welcome_note": "Bienvenue ! Ce guide en français explique l'application Export & Vectorization de MapBiomas Fire : comment naviguer (pays → thème → collection → produit), découvrir les unidades, traiter chaque Etapa et publier les cartes. Les onglets ci-dessus affichent l'interface ; utilisez ce guide chaque fois que nécessaire.[cite: 1]",
        "what": "L'application exporte les cartes des zones brûlées/incendies de MapBiomas Fire : de Earth Engine (GEE) vers GCS, assemble des mosaïques (COG) par unidade (bande ou image), vectorise le cas échéant, publie sur Earth Engine et le bucket public, et supprime les fichiers temporaires.[cite: 1]",
        "howto_title": "Comment utiliser[cite: 1]",
        "steps": [
            "<b>Naviguez</b> à travers les onglets : pays → thème → collection → produit.[cite: 1]",
            "Cliquez sur <b>Load Data</b> (bouton rouge clignotant) pour découvrir les <b>unidades</b> : bandes (image multibande) ou images (ImageCollection). La découverte est à la demande — le cache n'est pas rempli de données non chargées.[cite: 1]",
            "Dans la grille, cochez les unidades souhaitées. Le filtre <code>Unit:</code> (par défaut « All units ») restreint par préfixe d'unidade.[cite: 1]",
            "Cliquez sur <b>Sync</b> pour vérifier le statut des etapas. L'analyse s'exécute en arrière-plan avec un indicateur de progression (le noyau ne se bloque pas).[cite: 1]",
            "Exécutez les etapas dans l'ordre : <b>Export → Mosaïque → Publier mosaïque → Vecteur GCS → Vecteur GEE → Publier vecteur → Nettoyer temp</b>. Etapas 4–6 uniquement pour les produits vectorisables (ex. : annual_burned) ; pour les autres : <b>Export → Mosaïque → Publier mosaïque → Nettoyer temp</b>.[cite: 1]",
            "Pour refaire une Etapa, activez <code>FORCE_&lt;ETAPA&gt; = True</code> dans la cellule de l'Etapa et sélectionnez les unidades dans la grille.[cite: 1]",
            "Pour versionner la mémoire du Catálogo (lorsqu'il y a de nouvelles données dans <code>config.py</code>), utilisez le bouton <b>⤓ Catalog cache (.json)</b> dans la barre inférieure et téléchargez le fichier sur GitHub.[cite: 1]"
        ],
        "cols_title": "Colonnes de la grille[cite: 1]",
        "cols": [
            ["Export[cite: 1]", "unidade exportée depuis GEE (temp/)[cite: 1]"],
            ["Mosaic[cite: 1]", "COG assemblé[cite: 1]"],
            ["Public mosaic[cite: 1]", "COG mis en miroir dans le bucket public[cite: 1]"],
            ["Vector GCS[cite: 1]", "vecteur zippé (uniquement produits vectorisables)[cite: 1]"],
            ["Vector GEE[cite: 1]", "FeatureCollection dans Earth Engine[cite: 1]"],
            ["Public vector[cite: 1]", "ZIP mis en miroir dans le bucket public[cite: 1]"],
            ["Clean temp[cite: 1]", "tuiles temporaires supprimées après consolidation[cite: 1]"]
        ],
        "links": "Les badges <b>🔗 OK</b> ouvrent le lien de téléchargement ; <b>Vector GEE</b> copie l'asset ID.[cite: 1]",
        "legend": "OK = Etapa terminée | MISS = Etapa en attente | N/A = non applicable[cite: 1]",
        "graphs_title": "Graphiques[cite: 1]",
        "graphs": [
            {
                "title": "G1 Navigation :[cite: 1]",
                "lines": [
                    "Pays[cite: 1]",
                    "└─ Thème[cite: 1]",
                    "   └─ Collection[cite: 1]",
                    "      └─ Produit[cite: 1]",
                    "         └─ Unidades (bandes ou images)[cite: 1]"
                ]
            },
            {
                "title": "G2 Etapas (produit vectorisable) :[cite: 1]",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mosaïque → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publier mosaïque → bucket public[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Vecteur GCS → ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "5 Vecteur GEE → FeatureCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "6 Publier vecteur → ZIP public[cite: 1]",
                    " ↓[cite: 1]",
                    "7 Nettoyer temp[cite: 1]"
                ]
            },
            {
                "title": "G3 Etapas (autres produits) :[cite: 1]",
                "lines": [
                    "1 Export[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mosaïque[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publier mosaïque[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Nettoyer temp[cite: 1]"
                ]
            },
            {
                "title": "G4 Mémoire du Catálogo :[cite: 1]",
                "lines": [
                    "config.py (données brutes)[cite: 1]",
                    " ↓[cite: 1]",
                    "Load Data (découvre les unidades à la demande)[cite: 1]",
                    " ↓[cite: 1]",
                    "catalog_cache.json (mémoire entre sessions)[cite: 1]",
                    " ↓[cite: 1]",
                    "Bouton ⤓ Catalog cache (.json)[cite: 1]",
                    " ↓[cite: 1]",
                    "GitHub (versionnage)[cite: 1]"
                ]
            },
            {
                "title": "G5 Flux de données :[cite: 1]",
                "lines": [
                    "GEE ImageCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "Mosaïque → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "Publier → bucket public[cite: 1]",
                    " ↓[cite: 1]",
                    "(si vectorisable) Vectorisation → ZIP + upload GEE[cite: 1]"
                ]
            }
        ]
    },
    "NL": {
        "name": "Nederlands[cite: 1]",
        "tab_title": "Gids: Nederlands[cite: 1]",
        "welcome_note": "Welkom! Deze Nederlandstalige gids legt de Export & Vectorization app van MapBiomas Fire uit: hoe te navigeren (land → thema → collectie → product), de unidades te ontdekken, elke Etapa te verwerken en de kaarten te publiceren. De tabbladen hierboven tonen de interface; gebruik deze gids wanneer je hem nodig hebt.[cite: 1]",
        "what": "De app exporteert kaarten van verbrande gebieden/branden van MapBiomas Fire: van Earth Engine (GEE) naar GCS, bouwt mozaïeken (COG) per unidade (band of afbeelding), vectoriseert indien van toepassing, publiceert naar Earth Engine en de openbare bucket, en verwijdert tijdelijke bestanden.[cite: 1]",
        "howto_title": "Hoe te gebruiken[cite: 1]",
        "steps": [
            "<b>Navigeer</b> door de tabbladen: land → thema → collectie → product.[cite: 1]",
            "Klik op <b>Load Data</b> (kloppende rode knop) om de <b>unidades</b> te ontdekken: banden (multiband afbeelding) of afbeeldingen (ImageCollection). Ontdekking is on-demand — de cache wordt niet gevuld met ongeladen gegevens.[cite: 1]",
            "Vink in het raster de gewenste unidades aan. Het filter <code>Unit:</code> (standaard “All units”) beperkt op voorvoegsel van de unidade.[cite: 1]",
            "Klik op <b>Sync</b> om de status van de etapas te controleren. De scan draait op de achtergrond, met een voortgangsindicator (de kernel blokkeert niet).[cite: 1]",
            "Voer de etapas op volgorde uit: <b>Export → Mozaïek → Publiceer mozaïek → Vector GCS → Vector GEE → Publiceer vector → Wis temp</b>. Etapas 4–6 alleen voor vectoriseerbare producten (bijv. annual_burned); voor de rest: <b>Export → Mozaïek → Publiceer mozaïek → Wis temp</b>.[cite: 1]",
            "Om een Etapa opnieuw te doen, activeer je <code>FORCE_&lt;ETAPA&gt; = True</code> in de cel van de Etapa en selecteer je de unidades in het raster.[cite: 1]",
            "Om de geheugen van de Catálogo te versioneren (wanneer er nieuwe gegevens in <code>config.py</code> zijn), gebruik je de knop <b>⤓ Catalog cache (.json)</b> in de onderste balk en upload je het bestand naar GitHub.[cite: 1]"
        ],
        "cols_title": "Rasterkolommen[cite: 1]",
        "cols": [
            ["Export[cite: 1]", "unidade geëxporteerd van GEE (temp/)[cite: 1]"],
            ["Mosaic[cite: 1]", "geassembleerde COG[cite: 1]"],
            ["Public mosaic[cite: 1]", "COG gespiegeld in de openbare bucket[cite: 1]"],
            ["Vector GCS[cite: 1]", "gezipte vector (alleen vectoriseerbare producten)[cite: 1]"],
            ["Vector GEE[cite: 1]", "FeatureCollection in Earth Engine[cite: 1]"],
            ["Public vector[cite: 1]", "ZIP gespiegeld in de openbare bucket[cite: 1]"],
            ["Clean temp[cite: 1]", "tijdelijke tegels verwijderd na consolidatie[cite: 1]"]
        ],
        "links": "Badges <b>🔗 OK</b> openen de downloadlink; <b>Vector GEE</b> kopieert de asset ID.[cite: 1]",
        "legend": "OK = Etapa voltooid | MISS = Etapa in afwachting | N/A = niet van toepassing[cite: 1]",
        "graphs_title": "Grafieken[cite: 1]",
        "graphs": [
            {
                "title": "G1 Navigatie:[cite: 1]",
                "lines": [
                    "Land[cite: 1]",
                    "└─ Thema[cite: 1]",
                    "   └─ Collectie[cite: 1]",
                    "      └─ Product[cite: 1]",
                    "         └─ Unidades (banden of afbeeldingen)[cite: 1]"
                ]
            },
            {
                "title": "G2 Etapas (vectoriseerbaar product):[cite: 1]",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mozaïek → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publiceer mozaïek → openbare bucket[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Vector GCS → ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "5 Vector GEE → FeatureCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "6 Publiceer vector → openbare ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "7 Wis temp[cite: 1]"
                ]
            },
            {
                "title": "G3 Etapas (overige producten):[cite: 1]",
                "lines": [
                    "1 Export[cite: 1]",
                    " ↓[cite: 1]",
                    "2 Mozaïek[cite: 1]",
                    " ↓[cite: 1]",
                    "3 Publiceer mozaïek[cite: 1]",
                    " ↓[cite: 1]",
                    "4 Wis temp[cite: 1]"
                ]
            },
            {
                "title": "G4 Geheugen van de Catálogo:[cite: 1]",
                "lines": [
                    "config.py (ruwe data)[cite: 1]",
                    " ↓[cite: 1]",
                    "Load Data (ontdekt unidades on-demand)[cite: 1]",
                    " ↓[cite: 1]",
                    "catalog_cache.json (geheugen tussen sessies)[cite: 1]",
                    " ↓[cite: 1]",
                    "Knop ⤓ Catalog cache (.json)[cite: 1]",
                    " ↓[cite: 1]",
                    "GitHub (versioneren)[cite: 1]"
                ]
            },
            {
                "title": "G5 Dataflow:[cite: 1]",
                "lines": [
                    "GEE ImageCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "Mozaïek → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "Publiceer → openbare bucket[cite: 1]",
                    " ↓[cite: 1]",
                    "(indien vectoriseerbaar) Vectorisatie → ZIP + upload GEE[cite: 1]"
                ]
            }
        ]
    },
    "ZH": {
        "name": "中文[cite: 1]",
        "tab_title": "指南：中文[cite: 1]",
        "welcome_note": "欢迎！本中文指南说明了 MapBiomas Fire 的 Export & Vectorization 应用程序：如何导航（国家 → 主题 → 集合 → 产品），发现 unidades，处理每个 Etapa，并发布地图。上面的选项卡显示了界面；随时使用此指南。[cite: 1]",
        "what": "该应用程序从 MapBiomas Fire 导出烧毁面积/火灾地图：从 Earth Engine (GEE) 到 GCS，按 unidade（波段或图像）构建镶嵌图 (COG)，在适用时进行矢量化，发布到 Earth Engine 和公共存储桶，并删除临时文件。[cite: 1]",
        "howto_title": "如何使用[cite: 1]",
        "steps": [
            "通过选项卡<b>导航</b>：国家 → 主题 → 集合 → 产品。[cite: 1]",
            "单击 <b>Load Data</b>（闪烁的红色按钮）以发现 <b>unidades</b>：波段（多波段图像）或图像（ImageCollection）。发现是按需进行的 — 缓存不会被未加载的数据填满。[cite: 1]",
            "在网格中，勾选所需的 unidades。<code>Unit:</code> 过滤器（默认为“All units”）按 unidade 前缀进行限制。[cite: 1]",
            "单击 <b>Sync</b> 检查 etapas 的状态。扫描在后台运行，带有进度指示器（内核不会阻塞）。[cite: 1]",
            "按顺序执行 etapas：<b>Export → 镶嵌 → 发布镶嵌图 → 矢量 GCS → 矢量 GEE → 发布矢量 → 清理临时文件</b>。Etapas 4–6 仅适用于可矢量化的产品（例如 annual_burned）；对于其他产品：<b>Export → 镶嵌 → 发布镶嵌图 → 清理临时文件</b>。[cite: 1]",
            "要重做 Etapa，请在 Etapa 单元格中激活 <code>FORCE_&lt;ETAPA&gt; = True</code> 并在网格中选择 unidades。[cite: 1]",
            "要对 Catálogo 内存进行版本控制（当 <code>config.py</code> 中有新数据时），请使用底部栏中的 <b>⤓ Catalog cache (.json)</b> 按钮并将文件上传到 GitHub。[cite: 1]"
        ],
        "cols_title": "网格列[cite: 1]",
        "cols": [
            ["Export[cite: 1]", "从 GEE 导出的 unidade (temp/)[cite: 1]"],
            ["Mosaic[cite: 1]", "组装的 COG[cite: 1]"],
            ["Public mosaic[cite: 1]", "镜像在公共存储桶中的 COG[cite: 1]"],
            ["Vector GCS[cite: 1]", "压缩的矢量（仅限可矢量化产品）[cite: 1]"],
            ["Vector GEE[cite: 1]", "Earth Engine 中的 FeatureCollection[cite: 1]"],
            ["Public vector[cite: 1]", "镜像在公共存储桶中的 ZIP[cite: 1]"],
            ["Clean temp[cite: 1]", "合并后删除的临时切片[cite: 1]"]
        ],
        "links": "带有 <b>🔗 OK</b> 的徽章可打开下载链接；<b>Vector GEE</b> 复制 asset ID。[cite: 1]",
        "legend": "OK = Etapa 已完成 | MISS = Etapa 待处理 | N/A = 不适用[cite: 1]",
        "graphs_title": "图表[cite: 1]",
        "graphs": [
            {
                "title": "G1 导航：[cite: 1]",
                "lines": [
                    "国家[cite: 1]",
                    "└─ 主题[cite: 1]",
                    "   └─ 集合[cite: 1]",
                    "      └─ 产品[cite: 1]",
                    "         └─ Unidades（波段或图像）[cite: 1]"
                ]
            },
            {
                "title": "G2 Etapas（可矢量化产品）：[cite: 1]",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "2 镶嵌 → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "3 发布镶嵌图 → 公共存储桶[cite: 1]",
                    " ↓[cite: 1]",
                    "4 矢量 GCS → ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "5 矢量 GEE → FeatureCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "6 发布矢量 → 公共 ZIP[cite: 1]",
                    " ↓[cite: 1]",
                    "7 清理临时文件[cite: 1]"
                ]
            },
            {
                "title": "G3 Etapas（其他产品）：[cite: 1]",
                "lines": [
                    "1 Export[cite: 1]",
                    " ↓[cite: 1]",
                    "2 镶嵌[cite: 1]",
                    " ↓[cite: 1]",
                    "3 发布镶嵌图[cite: 1]",
                    " ↓[cite: 1]",
                    "4 清理临时文件[cite: 1]"
                ]
            },
            {
                "title": "G4 Catálogo 内存：[cite: 1]",
                "lines": [
                    "config.py（原始数据）[cite: 1]",
                    " ↓[cite: 1]",
                    "Load Data（按需发现 unidades）[cite: 1]",
                    " ↓[cite: 1]",
                    "catalog_cache.json（会话之间的内存）[cite: 1]",
                    " ↓[cite: 1]",
                    "按钮 ⤓ Catalog cache (.json)[cite: 1]",
                    " ↓[cite: 1]",
                    "GitHub（版本控制）[cite: 1]"
                ]
            },
            {
                "title": "G5 数据流：[cite: 1]",
                "lines": [
                    "GEE ImageCollection[cite: 1]",
                    " ↓[cite: 1]",
                    "Export → tiles 0/1 (temp/)[cite: 1]",
                    " ↓[cite: 1]",
                    "镶嵌 → COG[cite: 1]",
                    " ↓[cite: 1]",
                    "发布 → 公共存储桶[cite: 1]",
                    " ↓[cite: 1]",
                    "（如果可矢量化）矢量化 → ZIP + upload GEE[cite: 1]"
                ]
            }
        ]
    }
}
```