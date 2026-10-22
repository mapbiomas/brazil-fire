/* MAPBIOMAS FUEGO - COLECCIÓN 1 - PARAGUAY (REFERENCIA)
 Aplicación de máscaras de uso y cobertura del suelo (LULC) 
 sobre las cicatrices de fuego

 📅 FECHA: 05 de mayo de 2025

 EQUIPO:
 Red de mapeo de cicatrices de fuego - MapBiomas Fuego
 - Instituto de Pesquisa Ambiental da Amazônia (IPAM)
 - Wallace Silva y Vera Laisa

 🔗 REFERENCIAS:
 https://code.earthengine.google.com/4125bda1034925e6261d134e3c89a574
 https://code.earthengine.google.com/5c46e4b2a62b48349f01d35ac4cf063a

 -------------------------------------------------------------
 📌 ¿QUÉ HACE ESTE SCRIPT?
 1. Carga las cicatrices de fuego sin máscara (colección no-mask).
 2. Aplica máscaras de cobertura del suelo (MapBiomas) por región y año.
 3. Elimina píxeles solitarios y genera una versión final de la imagen.
 4. Exporta imágenes enmascaradas con codificación por mes (Landsat NBR).

 -------------------------------------------------------------
 🔧 ¿QUÉ DEBO MODIFICAR PARA USAR ESTE SCRIPT?
 ✅ Cambiar `landcover` y `region` según el país.
 ✅ Verificar las rutas de entrada (`col_nomask_id`) y salida (`col_mask_id`).
 ✅ Usar los filtros al final del script para seleccionar el año y la región
    antes de exportar (evita exportar todo de una vez).
 -------------------------------------------------------------
 ⚠️ RECOMENDACIÓN IMPORTANTE:
 ✅ Usa los filtros anteriores para seleccionar sólo una región o año por vez.
 Esto ayuda a evitar sobrecarga de tareas y facilita el control del proceso.
 📌 Cada imagen exportada genera una **tarea ("Task") individual**
 que aparece en la pestaña **"Tasks"** del Editor de Earth Engine.
 ▶️ Debes hacer clic en **"Run"** y luego confirmar manualmente 
 para cada tarea antes de que se ejecute.
       🧩 SUGERENCIA:
       Puedes instalar la extensión de navegador **Open Earth Engine** (para Google Chrome),
       que agrega un botón **"Run All Tasks"** para ejecutar todas las tareas pendientes
       de una sola vez, de forma más rápida.
***************************************************************/


// 🗂️ Colección de entrada (sin máscara)
var col_nomask_id = 'projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/CLASSIFICATION_COLLECTIONS/collection1_fire_no_mask_v1';

// 🗂️ Colección de salida (con máscara aplicada)
var col_mask_id = 'projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/CLASSIFICATION_COLLECTIONS/collection1_fire_mask_v1';

print('col_nomask_id', col_nomask_id, ee.ImageCollection(col_nomask_id).limit(10));


// 🗺️ Imágenes de cobertura del suelo por país (MapBiomas Integración)
var landcover_bolivia   = ee.Image('projects/mapbiomas-public/assets/bolivia/collection2/mapbiomas_bolivia_collection2_integration_v1'); // 1985–2023
var landcover_chile     = ee.Image('projects/mapbiomas-public/assets/chile/collection1/mapbiomas_chile_collection1_integration_v1');     // 2000–2022
var landcover_colombia  = ee.Image('projects/mapbiomas-public/assets/colombia/collection2/mapbiomas_colombia_collection2_integration_v1'); // 1985–2023
var landcover_paraguay  = ee.Image('projects/mapbiomas-public/assets/paraguay/collection2/mapbiomas_paraguay_collection2_integration_v1');  // 1985–2023
var landcover_peru      = ee.Image('projects/mapbiomas-public/assets/peru/collection2/mapbiomas_peru_collection2_integration_v1');          // 1985–2022

// 🧭 Divisiones regionales por país para el mapeo de fuego
var regions_bolivia   = ee.FeatureCollection('projects/mapbiomas-bolivia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_bolivia_v1');
var regions_chile     = ee.FeatureCollection('projects/mapbiomas-chile/assets/FIRE/AUXILIARY_DATA/regiones_fuego_chile_v1');
var regions_colombia  = ee.FeatureCollection('projects/mapbiomas-colombia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_colombia_v1');
var regions_paraguay  = ee.FeatureCollection('projects/mapbiomas-paraguay/assets/FIRE/AUXILIARY_DATA/regiones_fuego_paraguay_v1');
var regions_peru      = ee.FeatureCollection('projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1');

// 🛠️ Selección de país de trabajo (PARAGUAY como ejemplo)
var landcover = landcover_paraguay;
var region = regions_paraguay;

var geometry = landcover_paraguay.geometry()

// Añadir la banda de 2024 duplicando la de 2023 (si no existe aún)
landcover = landcover.addBands(
  landcover.select('classification_2023').rename('classification_2024')
);

// 🎯 Máscaras personalizadas por región (ajustar según el país)
// Ver códigos: https://paraguay.mapbiomas.org/en/codigos-de-la-leyenda
// 9: Forestaciones
// 22: Área sin vegetación
// 26: Cuerpo de agua
var masks = {
  'region1': [26,22],
  'region2': [26,22],
  'region3': [26,22],
  'region4': [26,22],
  'region5': [26,22],
  'region6': [26,22],
  'region7': [26,22],
};


// 📦 Biblioteca externa para mosaicos de calidad Landsat
var collection_landsat = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/require-landsat-collection');

// 🧰 Função principal para aplicar máscara y exportar
function exportImage(obj) {
  var image = ee.Image(obj.id);
  var name = obj.id.split('/').slice(-1)[0];
  var split = name.split('_');
  var year = split.slice(-1)[0];
  var region = split.slice(-2,-1)[0];
  var mask = masks[region];

  // Crear máscara basada en uso y cobertura del suelo
  mask = landcover
    .select(ee.String('classification_').cat(year))
    .eq(mask)
    .reduce('sum')
    .gte(1);

  
  // 🎯 Aplicar lógica adicional para una región específica, si es necesario
  if (region === 'region1') {
  
    // Por ejemplo: expandir clases con buffer, combinar con otras máscaras, etc.
    
    // 👉 Ejemplo específico:
    // Se aplica un buffer de 90 metros a los cuerpos de agua (clase 26)
    // para excluir áreas cercanas de forma más conservadora.
  
    var clasesEspeciales = [26];  // clase 26 = cuerpos de agua
  
    clasesEspeciales.forEach(function(clase) {
      var buffer = landcover
        .select(ee.String('classification_').cat(year))
        .eq(clase)
        .selfMask()
        .focalMax({radius: 90, units: 'meters'})
        .gte(1);
  
      // Combinar el buffer con la máscara original
      mask = mask.blend(buffer);
    });
  }
  
  var final_mask = mask.neq(1);
  var image_mask = image.updateMask(final_mask);

  // Remover píxeles solitários
  var connections = image_mask.connectedPixelCount({'maxSize': 100, 'eightConnected': false});
  var solitary_pixels = connections.lte(6);
  image_mask = image_mask.where(solitary_pixels, 0).selfMask().reproject('EPSG:4326', null, 30);

  // Criar imagem final com base no mês do menor NBR
  ee.Number.parse(year).int().evaluate(function(y) {
    var qualityMosaic = collection_landsat.landsat_year(y, geometry)
      .qualityMosaic('nbr')
      .select('monthOfYear')
      .byte();

    var startDate = ee.Date.fromYMD(y, 1, 1);
    var endDate = ee.Date.fromYMD(y + 1, 1, 1);

    var properties = {
      'source': 'mapbiomas-fuego',
      'pixel_unit': 'month',
      'name': name,
      'year': y,
      'region': region,
      "system:time_start": startDate.millis(),
      "system:time_end": endDate.millis()
    };

    var image_final = qualityMosaic
      .updateMask(image_mask)
      .set(properties);
    
  // 🧪 Visualización para verificación de máscaras (opcional)
  // Usa estos Map.addLayer sólo para validar visualmente si las máscaras están correctas.
  // ✅ Si ya confirmaste que el proceso está funcionando bien, se recomienda dejar estas líneas comentadas
  // para evitar sobrecargar el mapa o distraer la visualización.
    Map.addLayer(
      landcover.select(ee.String('classification_').cat(year)),
      {
        min: 0,
        max: 69,
        palette: require('users/mapbiomas/modules:Palettes.js').get('classification9')
      },
      'Cobertura del suelo ' + year,
      false
    );
    
    Map.addLayer(image, {min: 0, max: 1, palette: ['000000']}, 'Imagen sin máscara ' + year, false);
    
    Map.addLayer(mask, {palette: ['ffcccc','ff0000']}, 'Máscara aplicada ' + region + '-' + year, false);
    
    Map.addLayer(image_mask, {min: 0, max: 1, palette: ['ffff00']}, 'Imagen enmascarada ' + name, false);
    
    // Visualización de la imagen final con codificación por mes
    Map.addLayer(image_final, {
      min: 1,
      max: 12,
      palette: ['000000','ffffff']
    }, 'Imagen final (meses) ' + name, false);

    
    Export.image.toAsset({
      image: image_final,
      description: name,
      assetId: col_mask_id +'/'+ name,
      pyramidingPolicy: 'mode',
      region: geometry,
      scale: 30,
      maxPixels: 1e13,
    });
  });
}



// 🧾 FILTRAR LAS IMÁGENES A EXPORTAR
// Cada objeto representa una imagen dentro de la colección de entrada (`col_nomask_id`)
// Aquí filtramos primero por región, y opcionalmente por año, según el nombre del asset.
var col = ee.data.listAssets(col_nomask_id).assets;

col
  // 🔍 (Opcional) Filtrar por región (modificar 'region1' si deseas otra región)
  // .filter(function(obj) {
  //   var nombre = obj.id.split('/').slice(-1)[0];  // obtiene el nombre del asset
  //   var region = nombre.split('_').slice(-2, -1)[0];  // extrae la parte de la región
  // //   return region === 'region1';
  //   return region !== 'region1';
  // })

  // 🔍 (Opcional) Filtrar por año (descomentar para usar)
  
  // .filter(function(obj) {
  //   var nombre = obj.id.split('/').slice(-1)[0];
  //   var anio = nombre.split('_').slice(-1)[0];
  //   return anio === '2024';
  // })
  
  // 🚀 Exportar cada imagen filtrada
  .forEach(exportImage);
