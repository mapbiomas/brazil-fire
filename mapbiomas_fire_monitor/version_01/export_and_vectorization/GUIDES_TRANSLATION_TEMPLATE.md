# Guias do app MapBiomas Export & Publish — Tradução PT → 6 idiomas

Este documento é um **template de tradução**: o conteúdo em Português (abaixo) é a
fonte e o objetivo é gerar as guias em **ES, EN, ID, FR, NL, ZH** no formato de
saída indicado. Envie este arquivo (ou só a seção "Conteúdo PT" + "Formato de
saída") para uma IA especialista em idiomas.

> ⚠️ **ATUALIZAÇÃO PENDENTE**: as traduções ES/EN/ID/FR/NL/ZH (bloco
> `GUIDES_EXTRA` ao final) estão **desatualizadas** — o **PT é a referência
> atual**. Precisam ser regeradas a partir do "Conteúdo PT" abaixo, cobrindo:
> novo nome **MapBiomas Export & Publish**, **Load Data / Load Collection**,
> **Select Pending / Select Pending Collection** (ciclo por estágio com
> `número · nome`), abas de produto em várias linhas, checkbox na primeira
> coluna e vetorização como **análise opcional**.

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
welcome_note: "Bem-vindo! Esta guia em português explica o aplicativo MapBiomas
  Export & Publish (catálogo de dados multipaís e multitemático): como navegar
  (país → tema → coleção → produto), descobrir as unidades, processar cada etapa
  e publicar os mapas. As abas acima mostram a interface; use esta guia sempre
  que precisar."
what: "O aplicativo exporta dados do catálogo MapBiomas (fogo, uso e cobertura,
  etc.): do Earth Engine (GEE) para o GCS, monta mosaicos (COG) por unidade
  (banda ou imagem), publica no bucket público e remove os arquivos temporários.
  A vetorização é uma análise opcional para produtos vetorizáveis (ex.:
  annual_burned)."
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
     <b>Select Pending</b> (número · nome do estágio no título) para selecionar
     por estágio — cada clique avança no ciclo. Use <b>Select Pending
     Collection</b> para o mesmo em todos os produtos da coleção. Também há
     <b>Select All</b> e <b>Select All Collection</b>. O filtro <code>Unit:</code>
     (padrão “All units”) restringe por prefixo; acima de 60 unidades inicia no
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
        "name": "Español",
        "tab_title": "Guía: Español",
        "welcome_note": "¡Bienvenido! Esta guía en español explica la aplicación MapBiomas Export & Publish (Catálogo de datos multipaís y multitemático): cómo navegar (país → tema → colección → producto), descubrir las unidades, procesar cada Etapa y publicar los mapas. Las pestañas de arriba muestran la interfaz; use esta guía siempre que la necesite.",
        "what": "La aplicación exporta datos del Catálogo MapBiomas (fuego, uso y cobertura, etc.): de Earth Engine (GEE) a GCS, ensambla mosaicos (COG) por unidade (banda o imagen), publica en el bucket público y elimina los archivos temporales. La vectorización es un análisis opcional para productos vectorizables (ej.: annual_burned).",
        "howto_title": "Cómo usar",
        "steps": [
            "<b>Navegue</b> por las pestañas: país → tema → colección → producto. Con muchos productos, las pestañas se dividen en varias líneas — verde = cargado, gris = no cargado.",
            "Haga clic en <b>Load Data</b> para cargar el producto actual (unidades de la memoria, descubrir datos nuevos y verificar el estado, con límite de <code>SCAN_TIMEOUT</code>) — o en <b>Load Collection</b> para cargar todos los productos de la colección en cola.",
            "En la cuadrícula (casilla de verificación en la primera columna), marque las unidades deseadas. Use <b>Select Pending</b> para seleccionar por Etapa (el título muestra la Etapa objetivo y cada clic avanza en el ciclo), <b>Select All</b> o <b>Select All Collection</b>. El filtro <code>Unit:</code> (predeterminado “All units”) restringe por prefijo; por encima de 60 unidades inicia en el prefijo reciente.",
            "Ejecute las etapas en orden: <b>Export → Mosaico → Publicar mosaico → Limpiar temp</b>. La vectorización es opcional (Steps 5–7, solo productos vectorizables, ej.: annual_burned): <b>Vector GCS → Vector GEE → Publicar vector</b>.",
            "Para rehacer una Etapa, active <code>FORCE_&lt;ETAPA&gt; = True</code> en la celda de la Etapa y seleccione las unidades en la cuadrícula.",
            "Para versionar la memoria del Catálogo (cuando haya datos nuevos en <code>config.py</code>), use el botón <b>⤓ Catalog cache (.json)</b> en la barra inferior y suba el archivo a GitHub."
        ],
        "cols_title": "Columnas de la cuadrícula",
        "cols": [
            ["Export", "unidade exportada de GEE (temp/)"],
            ["Mosaic", "COG ensamblado"],
            ["Public mosaic", "COG replicado en el bucket público"],
            ["Vector GCS", "vector comprimido en ZIP (solo productos vectorizables)"],
            ["Vector GEE", "FeatureCollection en Earth Engine"],
            ["Public vector", "ZIP replicado en el bucket público"],
            ["Clean temp", "tiles temporales eliminados tras consolidación"]
        ],
        "links": "Insignias <b>🔗 OK</b> abren el enlace de descarga; <b>Vector GEE</b> copia el asset ID.",
        "legend": "OK = Etapa completada | MISS = Etapa pendiente | N/A = no aplica",
        "graphs_title": "Gráficos",
        "graphs": [
            {
                "title": "G1 Navegación:",
                "lines": [
                    "País",
                    "└─ Tema",
                    "   └─ Colección",
                    "      └─ Producto",
                    "         └─ Unidades (bandas o imágenes)"
                ]
            },
            {
                "title": "G2 Etapas (producto vectorizable):",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)",
                    " ↓",
                    "2 Mosaico → COG",
                    " ↓",
                    "3 Publicar mosaico → bucket público",
                    " ↓",
                    "4 Limpiar temp",
                    " ↓",
                    "5 Vector GCS → ZIP",
                    " ↓",
                    "6 Vector GEE → FeatureCollection",
                    " ↓",
                    "7 Publicar vector → ZIP público"
                ]
            },
            {
                "title": "G3 Etapas (demás productos):",
                "lines": [
                    "1 Export",
                    " ↓",
                    "2 Mosaico",
                    " ↓",
                    "3 Publicar mosaico",
                    " ↓",
                    "4 Limpiar temp"
                ]
            },
            {
                "title": "G4 Memoria del Catálogo:",
                "lines": [
                    "config.py (dato crudo)",
                    " ↓",
                    "Load Data (descubre unidades bajo demanda)",
                    " ↓",
                    "catalog_cache.json (memoria entre sesiones)",
                    " ↓",
                    "Botón ⤓ Catalog cache (.json)",
                    " ↓",
                    "GitHub (versionar)"
                ]
            },
            {
                "title": "G5 Flujo de datos:",
                "lines": [
                    "GEE ImageCollection",
                    " ↓",
                    "Export → tiles 0/1 (temp/)",
                    " ↓",
                    "Mosaico → COG",
                    " ↓",
                    "Publicar → bucket público",
                    " ↓",
                    "(si vectorizable) Vectorización → ZIP + upload GEE"
                ]
            }
        ]
    },
    "EN": {
        "name": "English",
        "tab_title": "Guide: English",
        "welcome_note": "Welcome! This English guide explains the MapBiomas Export & Publish app (multi-country and multi-theme data Catálogo): how to navigate (country → theme → collection → product), discover the unidades, process each Etapa, and publish the maps. The tabs above show the interface; use this guide whenever you need it.",
        "what": "The app exports data from the MapBiomas Catálogo (fire, land use and cover, etc.): from Earth Engine (GEE) to GCS, builds mosaics (COG) per unidade (band or image), publishes to the public bucket, and removes temporary files. Vectorization is an optional analysis for vectorizable products (e.g., annual_burned).",
        "howto_title": "How to use",
        "steps": [
            "<b>Navigate</b> through the tabs: country → theme → collection → product. With many products, the tabs break into multiple lines — green = loaded, gray = not loaded.",
            "Click <b>Load Data</b> to load the current product (unidades from memory, discover new data, and check status, with a <code>SCAN_TIMEOUT</code> limit) — or <b>Load Collection</b> to load all products in the collection in a queue.",
            "In the grid (checkbox in the first column), check the desired unidades. Use <b>Select Pending</b> to select by stage (the title shows the target Etapa and each click advances the cycle), <b>Select All</b>, or <b>Select All Collection</b>. The <code>Unit:</code> filter (default “All units”) restricts by prefix; above 60 unidades it starts at the recent prefix.",
            "Execute the etapas in order: <b>Export → Mosaic → Publish mosaic → Clean temp</b>. Vectorization is optional (Steps 5–7, vectorizable products only, e.g., annual_burned): <b>Vector GCS → Vector GEE → Publish vector</b>.",
            "To redo an Etapa, activate <code>FORCE_&lt;ETAPA&gt; = True</code> in the Etapa cell and select the unidades in the grid.",
            "To version the Catálogo memory (when there is new data in <code>config.py</code>), use the <b>⤓ Catalog cache (.json)</b> button in the bottom bar and upload the file to GitHub."
        ],
        "cols_title": "Grid columns",
        "cols": [
            ["Export", "unidade exported from GEE (temp/)"],
            ["Mosaic", "assembled COG"],
            ["Public mosaic", "COG mirrored in the public bucket"],
            ["Vector GCS", "zipped vector (vectorizable products only)"],
            ["Vector GEE", "FeatureCollection in Earth Engine"],
            ["Public vector", "ZIP mirrored in the public bucket"],
            ["Clean temp", "temporary tiles removed after consolidation"]
        ],
        "links": "<b>🔗 OK</b> badges open the download link; <b>Vector GEE</b> copies the asset ID.",
        "legend": "OK = Etapa completed | MISS = Etapa pending | N/A = not applicable",
        "graphs_title": "Graphs",
        "graphs": [
            {
                "title": "G1 Navigation:",
                "lines": [
                    "Country",
                    "└─ Theme",
                    "   └─ Collection",
                    "      └─ Product",
                    "         └─ Unidades (bands or images)"
                ]
            },
            {
                "title": "G2 Etapas (vectorizable product):",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)",
                    " ↓",
                    "2 Mosaic → COG",
                    " ↓",
                    "3 Publish mosaic → public bucket",
                    " ↓",
                    "4 Clean temp",
                    " ↓",
                    "5 Vector GCS → ZIP",
                    " ↓",
                    "6 Vector GEE → FeatureCollection",
                    " ↓",
                    "7 Publish vector → public ZIP"
                ]
            },
            {
                "title": "G3 Etapas (other products):",
                "lines": [
                    "1 Export",
                    " ↓",
                    "2 Mosaic",
                    " ↓",
                    "3 Publish mosaic",
                    " ↓",
                    "4 Clean temp"
                ]
            },
            {
                "title": "G4 Catálogo memory:",
                "lines": [
                    "config.py (raw data)",
                    " ↓",
                    "Load Data (discovers unidades on demand)",
                    " ↓",
                    "catalog_cache.json (memory between sessions)",
                    " ↓",
                    "Button ⤓ Catalog cache (.json)",
                    " ↓",
                    "GitHub (versioning)"
                ]
            },
            {
                "title": "G5 Data flow:",
                "lines": [
                    "GEE ImageCollection",
                    " ↓",
                    "Export → tiles 0/1 (temp/)",
                    " ↓",
                    "Mosaic → COG",
                    " ↓",
                    "Publish → public bucket",
                    " ↓",
                    "(if vectorizable) Vectorization → ZIP + upload GEE"
                ]
            }
        ]
    },
    "ID": {
        "name": "Bahasa Indonesia",
        "tab_title": "Panduan: Bahasa Indonesia",
        "welcome_note": "Selamat datang! Panduan dalam bahasa Indonesia ini menjelaskan aplikasi MapBiomas Export & Publish (Catálogo data multi-negara dan multi-tema): cara navigasi (negara → tema → koleksi → produk), menemukan unidades, memproses setiap Etapa, dan memublikasikan peta. Tab di atas menampilkan antarmuka; gunakan panduan ini kapan pun Anda butuhkan.",
        "what": "Aplikasi ini mengekspor data dari Catálogo MapBiomas (api, penggunaan dan tutupan lahan, dll.): dari Earth Engine (GEE) ke GCS, membuat mosaik (COG) per unidade (band atau citra), memublikasikan ke bucket publik, dan menghapus file sementara. Vektorisasi adalah analisis opsional untuk produk yang dapat divektorisasi (misalnya, annual_burned).",
        "howto_title": "Cara penggunaan",
        "steps": [
            "<b>Navigasi</b> melalui tab: negara → tema → koleksi → produk. Dengan banyak produk, tab terbagi dalam beberapa baris — hijau = dimuat, abu-abu = belum dimuat.",
            "Klik <b>Load Data</b> untuk memuat produk saat ini (unidades dari memori, menemukan data baru dan memeriksa status, dengan batas <code>SCAN_TIMEOUT</code>) — atau <b>Load Collection</b> untuk memuat semua produk dalam koleksi ke dalam antrean.",
            "Pada kisi (kotak centang di kolom pertama), centang unidades yang diinginkan. Gunakan <b>Select Pending</b> untuk memilih berdasarkan tahapan (judul menampilkan Etapa target dan setiap klik memajukan siklus), <b>Select All</b>, atau <b>Select All Collection</b>. Filter <code>Unit:</code> (default “All units”) membatasi berdasarkan awalan; di atas 60 unidades dimulai pada awalan terbaru.",
            "Jalankan etapas secara berurutan: <b>Export → Mosaik → Publikasikan mosaik → Bersihkan temp</b>. Vektorisasi bersifat opsional (Steps 5–7, hanya produk yang dapat divektorisasi, mis.: annual_burned): <b>Vektor GCS → Vektor GEE → Publikasikan vektor</b>.",
            "Untuk mengulang Etapa, aktifkan <code>FORCE_&lt;ETAPA&gt; = True</code> di sel Etapa dan pilih unidades di kisi.",
            "Untuk membuat versi memori Catálogo (saat ada data baru di <code>config.py</code>), gunakan tombol <b>⤓ Catalog cache (.json)</b> di bilah bawah dan unggah file ke GitHub."
        ],
        "cols_title": "Kolom kisi",
        "cols": [
            ["Export", "unidade diekspor dari GEE (temp/)"],
            ["Mosaic", "COG yang digabungkan"],
            ["Public mosaic", "COG dicerminkan di bucket publik"],
            ["Vector GCS", "vektor dalam format ZIP (hanya produk yang dapat divektorisasi)"],
            ["Vector GEE", "FeatureCollection di Earth Engine"],
            ["Public vector", "ZIP dicerminkan di bucket publik"],
            ["Clean temp", "tile sementara dihapus setelah konsolidasi"]
        ],
        "links": "Lencana <b>🔗 OK</b> membuka tautan unduhan; <b>Vector GEE</b> menyalin asset ID.",
        "legend": "OK = Etapa selesai | MISS = Etapa tertunda | N/A = tidak berlaku",
        "graphs_title": "Grafik",
        "graphs": [
            {
                "title": "G1 Navigasi:",
                "lines": [
                    "Negara",
                    "└─ Tema",
                    "   └─ Koleksi",
                    "      └─ Produk",
                    "         └─ Unidades (band atau citra)"
                ]
            },
            {
                "title": "G2 Etapas (produk dapat divektorisasi):",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)",
                    " ↓",
                    "2 Mosaik → COG",
                    " ↓",
                    "3 Publikasikan mosaik → bucket publik",
                    " ↓",
                    "4 Bersihkan temp",
                    " ↓",
                    "5 Vektor GCS → ZIP",
                    " ↓",
                    "6 Vektor GEE → FeatureCollection",
                    " ↓",
                    "7 Publikasikan vektor → ZIP publik"
                ]
            },
            {
                "title": "G3 Etapas (produk lainnya):",
                "lines": [
                    "1 Export",
                    " ↓",
                    "2 Mosaik",
                    " ↓",
                    "3 Publikasikan mosaik",
                    " ↓",
                    "4 Bersihkan temp"
                ]
            },
            {
                "title": "G4 Memori Catálogo:",
                "lines": [
                    "config.py (data mentah)",
                    " ↓",
                    "Load Data (menemukan unidades berdasarkan permintaan)",
                    " ↓",
                    "catalog_cache.json (memori antar sesi)",
                    " ↓",
                    "Tombol ⤓ Catalog cache (.json)",
                    " ↓",
                    "GitHub (pembuatan versi)"
                ]
            },
            {
                "title": "G5 Alur data:",
                "lines": [
                    "GEE ImageCollection",
                    " ↓",
                    "Export → tiles 0/1 (temp/)",
                    " ↓",
                    "Mosaik → COG",
                    " ↓",
                    "Publikasikan → bucket publik",
                    " ↓",
                    "(jika dapat divektorisasi) Vektorisasi → ZIP + upload GEE"
                ]
            }
        ]
    },
    "FR": {
        "name": "Français",
        "tab_title": "Guide : Français",
        "welcome_note": "Bienvenue ! Ce guide en français explique l'application MapBiomas Export & Publish (Catálogo de données multi-pays et multi-thématique) : comment naviguer (pays → thème → collection → produit), découvrir les unidades, traiter chaque Etapa et publier les cartes. Les onglets ci-dessus affichent l'interface ; utilisez ce guide chaque fois que nécessaire.",
        "what": "L'application exporte les données du Catálogo MapBiomas (feu, utilisation et couverture des terres, etc.) : de Earth Engine (GEE) vers GCS, assemble des mosaïques (COG) par unidade (bande ou image), publie dans le bucket public et supprime les fichiers temporaires. La vectorisation est une analyse facultative pour les produits vectorisables (ex. : annual_burned).",
        "howto_title": "Comment utiliser",
        "steps": [
            "<b>Naviguez</b> à travers les onglets : pays → thème → collection → produit. S'il y a de nombreux produits, les onglets sont répartis sur plusieurs lignes — vert = chargé, gris = non chargé.",
            "Cliquez sur <b>Load Data</b> pour charger le produit actuel (unidades de la mémoire, découvrir de nouvelles données et vérifier l'état, avec une limite de <code>SCAN_TIMEOUT</code>) — ou sur <b>Load Collection</b> pour charger tous les produits de la collection en file d'attente.",
            "Dans la grille (case à cocher dans la première colonne), sélectionnez les unidades souhaitées. Utilisez <b>Select Pending</b> pour sélectionner par stade (le titre indique l'Etapa cible et chaque clic fait avancer le cycle), <b>Select All</b> ou <b>Select All Collection</b>. Le filtre <code>Unit:</code> (par défaut « All units ») restreint par préfixe ; au-delà de 60 unidades, il commence par le préfixe récent.",
            "Exécutez les étapes dans l'ordre : <b>Export → Mosaïque → Publier mosaïque → Nettoyer temp</b>. La vectorisation est facultative (Steps 5–7, uniquement pour les produits vectorisables, ex. : annual_burned) : <b>Vecteur GCS → Vecteur GEE → Publier vecteur</b>.",
            "Pour refaire une Etapa, activez <code>FORCE_&lt;ETAPA&gt; = True</code> dans la cellule de l'Etapa et sélectionnez les unidades dans la grille.",
            "Pour versionner la mémoire du Catálogo (lorsqu'il y a de nouvelles données dans <code>config.py</code>), utilisez le bouton <b>⤓ Catalog cache (.json)</b> dans la barre inférieure et téléchargez le fichier sur GitHub."
        ],
        "cols_title": "Colonnes de la grille",
        "cols": [
            ["Export", "unidade exportée depuis GEE (temp/)"],
            ["Mosaic", "COG assemblé"],
            ["Public mosaic", "COG mis en miroir dans le bucket public"],
            ["Vector GCS", "vecteur zippé (uniquement produits vectorisables)"],
            ["Vector GEE", "FeatureCollection dans Earth Engine"],
            ["Public vector", "ZIP mis en miroir dans le bucket public"],
            ["Clean temp", "tuiles temporaires supprimées après consolidation"]
        ],
        "links": "Les badges <b>🔗 OK</b> ouvrent le lien de téléchargement ; <b>Vector GEE</b> copie l'asset ID.",
        "legend": "OK = Etapa terminée | MISS = Etapa en attente | N/A = non applicable",
        "graphs_title": "Graphiques",
        "graphs": [
            {
                "title": "G1 Navigation :",
                "lines": [
                    "Pays",
                    "└─ Thème",
                    "   └─ Collection",
                    "      └─ Produit",
                    "         └─ Unidades (bandes ou images)"
                ]
            },
            {
                "title": "G2 Etapas (produit vectorisable) :",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)",
                    " ↓",
                    "2 Mosaïque → COG",
                    " ↓",
                    "3 Publier mosaïque → bucket public",
                    " ↓",
                    "4 Nettoyer temp",
                    " ↓",
                    "5 Vecteur GCS → ZIP",
                    " ↓",
                    "6 Vecteur GEE → FeatureCollection",
                    " ↓",
                    "7 Publier vecteur → ZIP public"
                ]
            },
            {
                "title": "G3 Etapas (autres produits) :",
                "lines": [
                    "1 Export",
                    " ↓",
                    "2 Mosaïque",
                    " ↓",
                    "3 Publier mosaïque",
                    " ↓",
                    "4 Nettoyer temp"
                ]
            },
            {
                "title": "G4 Mémoire du Catálogo :",
                "lines": [
                    "config.py (données brutes)",
                    " ↓",
                    "Load Data (découvre les unidades à la demande)",
                    " ↓",
                    "catalog_cache.json (mémoire entre sessions)",
                    " ↓",
                    "Bouton ⤓ Catalog cache (.json)",
                    " ↓",
                    "GitHub (versionnage)"
                ]
            },
            {
                "title": "G5 Flux de données :",
                "lines": [
                    "GEE ImageCollection",
                    " ↓",
                    "Export → tiles 0/1 (temp/)",
                    " ↓",
                    "Mosaïque → COG",
                    " ↓",
                    "Publier → bucket public",
                    " ↓",
                    "(si vectorisable) Vectorisation → ZIP + upload GEE"
                ]
            }
        ]
    },
    "NL": {
        "name": "Nederlands",
        "tab_title": "Gids: Nederlands",
        "welcome_note": "Welkom! Deze Nederlandstalige gids legt de MapBiomas Export & Publish app uit (multi-land en multi-thema data Catálogo): hoe te navigeren (land → thema → collectie → product), de unidades te ontdekken, elke Etapa te verwerken en de kaarten te publiceren. De tabbladen hierboven tonen de interface; gebruik deze gids wanneer je hem nodig hebt.",
        "what": "De app exporteert gegevens uit de MapBiomas Catálogo (brand, landgebruik en -bedekking, enz.): van Earth Engine (GEE) naar GCS, bouwt mozaïeken (COG) per unidade (band of afbeelding), publiceert in de openbare bucket en verwijdert tijdelijke bestanden. Vectorisatie is een optionele analyse voor vectoriseerbare producten (bijv. annual_burned).",
        "howto_title": "Hoe te gebruiken",
        "steps": [
            "<b>Navigeer</b> door de tabbladen: land → thema → collectie → product. Bij veel producten worden de tabbladen over meerdere regels verdeeld — groen = geladen, grijs = niet geladen.",
            "Klik op <b>Load Data</b> om het huidige product te laden (unidades uit het geheugen, nieuwe gegevens ontdekken en status controleren, met een limiet van <code>SCAN_TIMEOUT</code>) — of op <b>Load Collection</b> om alle producten in de collectie in de wachtrij te plaatsen en te laden.",
            "Vink in het raster (selectievakje in de eerste kolom) de gewenste unidades aan. Gebruik <b>Select Pending</b> om te selecteren op fase (de titel toont de doel-Etapa en elke klik brengt de cyclus vooruit), <b>Select All</b> of <b>Select All Collection</b>. Het filter <code>Unit:</code> (standaard “All units”) beperkt op voorvoegsel; boven de 60 unidades start het bij het recente voorvoegsel.",
            "Voer de etapas op volgorde uit: <b>Export → Mozaïek → Publiceer mozaïek → Wis temp</b>. Vectorisatie is optioneel (Steps 5–7, alleen voor vectoriseerbare producten, bijv. annual_burned): <b>Vector GCS → Vector GEE → Publiceer vector</b>.",
            "Om een Etapa opnieuw te doen, activeer je <code>FORCE_&lt;ETAPA&gt; = True</code> in de cel van de Etapa en selecteer je de unidades in het raster.",
            "Om de geheugen van de Catálogo te versioneren (wanneer er nieuwe gegevens in <code>config.py</code> zijn), gebruik je de knop <b>⤓ Catalog cache (.json)</b> in de onderste balk en upload je het bestand naar GitHub."
        ],
        "cols_title": "Rasterkolommen",
        "cols": [
            ["Export", "unidade geëxporteerd van GEE (temp/)"],
            ["Mosaic", "geassembleerde COG"],
            ["Public mosaic", "COG gespiegeld in de openbare bucket"],
            ["Vector GCS", "gezipte vector (alleen vectoriseerbare producten)"],
            ["Vector GEE", "FeatureCollection in Earth Engine"],
            ["Public vector", "ZIP gespiegeld in de openbare bucket"],
            ["Clean temp", "tijdelijke tegels verwijderd na consolidatie"]
        ],
        "links": "Badges <b>🔗 OK</b> openen de downloadlink; <b>Vector GEE</b> kopieert de asset ID.",
        "legend": "OK = Etapa voltooid | MISS = Etapa in afwachting | N/A = niet van toepassing",
        "graphs_title": "Grafieken",
        "graphs": [
            {
                "title": "G1 Navigatie:",
                "lines": [
                    "Land",
                    "└─ Thema",
                    "   └─ Collectie",
                    "      └─ Product",
                    "         └─ Unidades (banden of afbeeldingen)"
                ]
            },
            {
                "title": "G2 Etapas (vectoriseerbaar product):",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)",
                    " ↓",
                    "2 Mozaïek → COG",
                    " ↓",
                    "3 Publiceer mozaïek → openbare bucket",
                    " ↓",
                    "4 Wis temp",
                    " ↓",
                    "5 Vector GCS → ZIP",
                    " ↓",
                    "6 Vector GEE → FeatureCollection",
                    " ↓",
                    "7 Publiceer vector → openbare ZIP"
                ]
            },
            {
                "title": "G3 Etapas (overige producten):",
                "lines": [
                    "1 Export",
                    " ↓",
                    "2 Mozaïek",
                    " ↓",
                    "3 Publiceer mozaïek",
                    " ↓",
                    "4 Wis temp"
                ]
            },
            {
                "title": "G4 Geheugen van de Catálogo:",
                "lines": [
                    "config.py (ruwe data)",
                    " ↓",
                    "Load Data (ontdekt unidades on-demand)",
                    " ↓",
                    "catalog_cache.json (geheugen tussen sessies)",
                    " ↓",
                    "Knop ⤓ Catalog cache (.json)",
                    " ↓",
                    "GitHub (versioneren)"
                ]
            },
            {
                "title": "G5 Dataflow:",
                "lines": [
                    "GEE ImageCollection",
                    " ↓",
                    "Export → tiles 0/1 (temp/)",
                    " ↓",
                    "Mozaïek → COG",
                    " ↓",
                    "Publiceer → openbare bucket",
                    " ↓",
                    "(indien vectoriseerbaar) Vectorisatie → ZIP + upload GEE"
                ]
            }
        ]
    },
    "ZH": {
        "name": "中文",
        "tab_title": "指南：中文",
        "welcome_note": "欢迎！本中文指南说明了 MapBiomas Export & Publish 应用程序（多国和多主题数据 Catálogo）：如何导航（国家 → 主题 → 集合 → 产品），发现 unidades，处理每个 Etapa，并发布地图。上面的选项卡显示了界面；随时使用此指南。",
        "what": "该应用程序从 MapBiomas Catálogo（火灾、土地利用和覆盖等）导出数据：从 Earth Engine (GEE) 到 GCS，按 unidade（波段或图像）构建镶嵌图 (COG)，发布到公共存储桶，并删除临时文件。矢量化是可矢量化产品（例如 annual_burned）的可选分析。",
        "howto_title": "如何使用",
        "steps": [
            "通过选项卡<b>导航</b>：国家 → 主题 → 集合 → 产品。产品很多时，选项卡会分成多行 — 绿色 = 已加载，灰色 = 未加载。",
            "单击 <b>Load Data</b> 加载当前产品（从内存加载 unidades，发现新数据并检查状态，受 <code>SCAN_TIMEOUT</code> 限制）— 或单击 <b>Load Collection</b> 队列加载集合中的所有产品。",
            "在网格中（第一列的复选框），勾选所需的 unidades。使用 <b>Select Pending</b> 按阶段选择（标题显示目标 Etapa，每次单击都会推进循环），<b>Select All</b> 或 <b>Select All Collection</b>。<code>Unit:</code> 过滤器（默认为“All units”）按前缀限制；超过 60 个 unidades 时，从最近的前缀开始。",
            "按顺序执行 etapas：<b>Export → 镶嵌 → 发布镶嵌图 → 清理临时文件</b>。矢量化是可选的（Steps 5–7，仅限可矢量化产品，例如 annual_burned）：<b>矢量 GCS → 矢量 GEE → 发布矢量</b>。",
            "要重做 Etapa，请在 Etapa 单元格中激活 <code>FORCE_&lt;ETAPA&gt; = True</code> 并在网格中选择 unidades。",
            "要对 Catálogo 内存进行版本控制（当 <code>config.py</code> 中有新数据时），请使用底部栏中的 <b>⤓ Catalog cache (.json)</b> 按钮并将文件上传到 GitHub。"
        ],
        "cols_title": "网格列",
        "cols": [
            ["Export", "从 GEE 导出的 unidade (temp/)"],
            ["Mosaic", "组装的 COG"],
            ["Public mosaic", "镜像在公共存储桶中的 COG"],
            ["Vector GCS", "压缩的矢量（仅限可矢量化产品）"],
            ["Vector GEE", "Earth Engine 中的 FeatureCollection"],
            ["Public vector", "镜像在公共存储桶中的 ZIP"],
            ["Clean temp", "合并后删除的临时切片"]
        ],
        "links": "带有 <b>🔗 OK</b> 的徽章可打开下载链接；<b>Vector GEE</b> 复制 asset ID。",
        "legend": "OK = Etapa 已完成 | MISS = Etapa 待处理 | N/A = 不适用",
        "graphs_title": "图表",
        "graphs": [
            {
                "title": "G1 导航：",
                "lines": [
                    "国家",
                    "└─ 主题",
                    "   └─ 集合",
                    "      └─ 产品",
                    "         └─ Unidades（波段或图像）"
                ]
            },
            {
                "title": "G2 Etapas（可矢量化产品）：",
                "lines": [
                    "1 Export → tiles 0/1 (temp/)",
                    " ↓",
                    "2 镶嵌 → COG",
                    " ↓",
                    "3 发布镶嵌图 → 公共存储桶",
                    " ↓",
                    "4 清理临时文件",
                    " ↓",
                    "5 矢量 GCS → ZIP",
                    " ↓",
                    "6 矢量 GEE → FeatureCollection",
                    " ↓",
                    "7 发布矢量 → 公共 ZIP"
                ]
            },
            {
                "title": "G3 Etapas（其他产品）：",
                "lines": [
                    "1 Export",
                    " ↓",
                    "2 镶嵌",
                    " ↓",
                    "3 发布镶嵌图",
                    " ↓",
                    "4 清理临时文件"
                ]
            },
            {
                "title": "G4 Catálogo 内存：",
                "lines": [
                    "config.py（原始数据）",
                    " ↓",
                    "Load Data（按需发现 unidades）",
                    " ↓",
                    "catalog_cache.json（会话之间的内存）",
                    " ↓",
                    "按钮 ⤓ Catalog cache (.json)",
                    " ↓",
                    "GitHub（版本控制）"
                ]
            },
            {
                "title": "G5 数据流：",
                "lines": [
                    "GEE ImageCollection",
                    " ↓",
                    "Export → tiles 0/1 (temp/)",
                    " ↓",
                    "镶嵌 → COG",
                    " ↓",
                    "发布 → 公共存储桶",
                    " ↓",
                    "（如果可矢量化）矢量化 → ZIP + upload GEE"
                ]
            }
        ]
    }
}
```