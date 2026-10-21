# MapBiomas Fire Monitor - Version 01 (Perú)

Esta es la implementación técnica completa del ecosistema MapBiomas Fuego para la gestión automatizada y el procesamiento de imágenes en Perú.

A diferencia del planteamiento teórico, **todos los módulos (del M1 al M7) ya han sido implementados en el código base**, contando con su propia arquitectura distribuida, soporte para Deep Learning (TensorFlow) y herramientas nativas de geoprocesamiento (GDAL, Rasterio).

## Flujo de Trabajo y Etapas del Pipeline (M1 a M7)

El modelo asume un procesamiento híbrido: la nube (Google Earth Engine y Google Cloud Storage) se encarga del cómputo masivo inicial y el almacenamiento, y el procesamiento local se encarga del entrenamiento neuronal, inferencia y consolidación geográfica pesada.

1. **[M1] Exportación de Mosaicos** [✅ FINALIZADO]
   Generación de los mosaicos crudos (Landsat/Sentinel/MODIS/HLS/Planet) en Earth Engine y exportación masiva hacia GCS.
   *Rutas:* GEE -> `projects/mapbiomas-peru/assets/FIRE/MONITOR/VERSION_01/MOSAICS_RAW` | GCS -> `mapbiomas-fire/sudamerica/peru/monitor/mosaics/raw`

2. **[M2] Descarga y Mosaico Local** [✅ FINALIZADO]
   Sincronización desde GCS, unión de bandas (`gdalbuildvrt` / `gdal_translate`) para generar Cloud Optimized GeoTIFFs (COGs) y re-carga a Earth Engine.
   *Rutas:* GEE -> `projects/mapbiomas-peru/assets/FIRE/MONITOR/VERSION_01/MOSAICS_COG`

3. **[M3] Gestor de Muestras (`M3_toolkit.js`)** [✅ FINALIZADO]
   Dashboard integrado en GEE Code Editor para la recolección de firmas espectrales interactivas (Fuego / No Fuego). Incluye controles de dibujo y resumen en tiempo real.
   *Rutas:* GEE -> `projects/mapbiomas-peru/assets/FIRE/MONITOR/VERSION_01/LIBRARY_SAMPLES` | GCS -> `mapbiomas-fire/sudamerica/peru/monitor/library_samples`

4. **[M4] Entrenador del Modelo** [🚧 EN DESARROLLO]
   Despliegue de Red Neuronal Profunda (DNN) mediante TensorFlow. Toma las muestras del M3, gestiona el hold-out, entrena localmente y exporta los pesos (weights) a GCS.

5. **[M5] Clasificador** [🚧 EN DESARROLLO]
   Módulo de Inferencia. Recupera los pesos desde GCS, fragmenta la imagen (tiles/grids) para evitar saturación de RAM, corre las matrices sobre la DNN y escribe el resultado *dayOfYear* (día juliano).

6. **[M6] Publicador y Filtros LULC** [⏳ PENDIENTE]
   Aplica filtros morfológicos y cruza clasificaciones contra máscaras MapBiomas LULC para excluir agua, áreas urbanas, etc. Inyecta metadatos en GEE como *ImageCollection*.

7. **[M7] Curador de Colecciones** [⏳ PENDIENTE]
   Tablero de revisión para comparar versiones de modelos y filtros, determinando la versión "ganadora" que formará la colección oficial.

## Requisitos de Entorno

Debido a la magnitud del pipeline de IA:
```bash
conda env create -f environment.yml
conda activate fire_monitor
```
Es altamente estricto asegurar:
- Tener **TensorFlow 1.x / `tensorflow.compat.v1`** disponible.
- **`rasterio` y `scipy`** deben estar funcionales (para filtros espaciales como `sieve` y morfológicos).
- Presencia nativa de **GDAL** instalado.

## Demandas y Tareas Pendientes (Issues / TODO)

Al haber verificado el despliegue del software y sus clases (Python Scripts), listamos las optimizaciones o deudas técnicas a subsanar:

- [x] **Consolidación M1 a M3**: Interfaz de exportación (M1), montador de COGs (M2) y toolkit de muestras (M3) operativos y con rutas estandarizadas bajo `VERSION_01`.
- [ ] **Desarrollo de Notebooks y UIs (M4-M7)**: Integrar las clases lógicas de M4 y M5 en Jupyter Notebooks limpios y estructurados.
- [ ] **Validación TensorFlow (M4)**: Certificar el Entrenador DNN desarrollado usando las cuencas y ecorregiones peruanas. Si existen disonancias en RAM por TF1, evaluar upgrade a TF2/Keras.
- [ ] **Prueba End-to-End (M4-M5)**: Entrenar un modelo basado en las muestras (`LIBRARY_SAMPLES`) y aplicar la inferencia para validar el mapeo de *dayOfYear*.
