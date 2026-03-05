/* MAPBIOMAS FUEGO - COLECCIÓN 1 - SCRIPT DE REFERENCIA (ex: PARAGUAY)
 *
 * Exportación final de la colección de cicatrices de fuego con y sin máscara de uso y cobertura del suelo (LULC).
 *
 * 📅 FECHA: 28 de julio de 2025
 *
 * EQUIPO:
 * Grupo de trabajo de mapeo de cicatrices de fuego - MapBiomas Fuego
 * - Instituto de Pesquisa Ambiental da Amazônia (IPAM)
 * - Wallace Silva y Vera Laisa
 *
 * -------------------------------------------------------------
 * 📌 ¿QUÉ HACE ESTE SCRIPT?
 * Este script automatiza el proceso de copiar las imágenes de cicatrices de fuego finales
 * a las colecciones de destino designadas en Google Earth Engine. Realiza las siguientes tareas:
 * 1. **Verificación y Creación de Colecciones**: Asegura que las colecciones de ImageCollection
 * de destino (con y sin máscara LULC) existan. Si no las encuentra, las crea automáticamente.
 * 2. **Iteración sobre Imágenes Finales**: Recorre una lista predefinida (`final_collection`)
 * de IDs de imágenes que representan las clasificaciones finales de cicatrices de fuego.
 * 3. **Copia de Assets**: Para cada ID en la lista, el script busca la imagen fuente
 * y la copia a la colección de destino correspondiente.
 * 4. **Manejo de Duplicados (Opcional)**: Si una imagen ya existe en la colección de destino,
 * el script puede, opcionalmente, eliminar la versión existente antes de copiar la nueva,
 * evitando duplicados o versiones desactualizadas (controlado por `ELIMINA_SI_YA_EXISTE`).
 * 5. **Visualización de IDs Disponibles (Opcional)**: Permite listar en la consola todos los
 * IDs de imágenes disponibles en la ruta de origen, facilitando la selección y adición
 * a la lista `final_collection`.
 *
 * -------------------------------------------------------------
 * 🔧 ¿QUÉ DEBO MODIFICAR PARA USAR ESTE SCRIPT?
 * ✅ **Editar la lista `final_collection`**: Reemplaza los IDs de ejemplo con los IDs de tus imágenes finales
 * que deseas copiar a las colecciones de destino.
 * ✅ **Verificar/Ajustar las rutas de destino**: Asegúrate de que `col_nomask_id` y `col_mask_id`
 * apuntan a las ubicaciones correctas de tus colecciones de destino.
 * ✅ **(Opcional) Activar la eliminación de assets existentes**: Si deseas que el script elimine
 * un asset en el destino antes de copiar uno nuevo con el mismo nombre, cambia la variable
 * `ELIMINA_SI_YA_EXISTE` a `true`. Por defecto, está en `false` para mayor seguridad.
 * ✅ **(Opcional) Habilitar la visualización de IDs**: Para ver una lista de todos los IDs de imágenes
 * disponibles en la ruta de origen y así poder copiarlos y pegarlos fácilmente en `final_collection`,
 * cambia `MOSTRAR_TODAS_LAS_VERSIONES_POR_REGION` a `true`.
 *
 * -------------------------------------------------------------
 * ⚠️ RECOMENDACIÓN IMPORTANTE:
 * Para evitar errores y facilitar la revisión del proceso,
 * se recomienda ejecutar el script **una región por vez**.
 * Para ello, comenta (con `//`) o elimina temporalmente las otras entradas
 * en la lista `final_collection`.
 * -------------------------------------------------------------
 * 🖱️ NOTA SOBRE LAS OPERACIONES:
 * Por cada imagen que se copie o elimine, el navegador mostrará
 * una ventana emergente (pop-up) pidiendo confirmación manual.
 * Asegúrate de aceptar o rechazar según corresponda para cada asset.
 ***************************************************************/

// IDs de las colecciones de destino
var col_nomask_id = 'projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/CLASSIFICATION_COLLECTIONS/collection1_fire_no_mask_v1/';
var col_mask_id   = 'projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/CLASSIFICATION_COLLECTIONS/collection1_fire_mask_v1/';

// 🔄 Función para crear la colección solo si no existe
function createAssetIfNotExists(assetId) {
  try {
    ee.data.getAsset(assetId);
    print('✅ La colección ya existe:', assetId);
  } catch (e) {
    print('🆕 Creando colección:', assetId);
    ee.data.createAsset({type:'ImageCollection'}, assetId);
  }
}

// Ejecutar para ambas colecciones
createAssetIfNotExists(col_nomask_id);
createAssetIfNotExists(col_mask_id);

// 📁 Ruta base de las imágenes fuente (clasificación final por región/año)
var path = 'projects/mapbiomas-peru/assets/FIRE/COLLECTION1/CLASSIFICATION/';

// ✅ Configuración: activa esta opción si deseas eliminar el asset de destino existente y reemplazarlo con una nueva copia.
var ELIMINA_SI_YA_EXISTE = false;

// 🛠 Configuración: habilite esta opción si desea ver todos los ID de imágenes disponibles en la consola
// para que pueda copiar y pegar en la lista "final_collection" declarada a continuación
var MOSTRAR_TODAS_LAS_VERSIONES_POR_REGION = true;

ee.Number(0).evaluate(function(a){

  if (MOSTRAR_TODAS_LAS_VERSIONES_POR_REGION !== true){return }

  var panels = ui.Panel();
  var regions = {};
  print('Copie y pegue los IDs a continuación en la variable "final_collection":', panels);

  ee.data.listAssets(path).assets.forEach(function(asset_col){
    // Este bucle itera sobre las colecciones de assets dentro de la ruta base.
    if (asset_col.type !== 'IMAGE_COLLECTION'){return } // Se asegura de que solo procesa colecciones de imágenes.
    var name_col = asset_col.id.split('/').slice(-1)[0]; // Extrae el nombre de la colección.
    
    try {
      ee.data.listAssets(asset_col.id).assets.forEach(function(asset){
        // Este bucle interno itera sobre las imágenes individuales dentro de cada colección.
        var name = asset.id.split('/').slice(-1)[0]; // Extrae el nombre de la imagen.
        var region = name.split('_').slice(-2,-1)[0]; // Determina la región a partir del nombre de la imagen.
  
        if(regions[region] === undefined){ regions[region] = []} // Inicializa el array si la región es nueva.
  
        regions[region].push('"' + name_col + '/' + name + '",'); // Agrega el ID completo de la imagen.
      });
    } catch (e){
      return ;
    }
  });

  // Ordena y muestra los IDs de las imágenes en la consola de una manera organizada.
  Object.keys(regions).sort().forEach(function(region){
    var _years = {};
    var list = regions[region].sort();
    var region_str = list[0].split('_').slice(-2,-1)[0];
    var panel = ui.Panel([ui.Label(''),ui.Label(''),ui.Label('// // --- --- '+region_str,{fontSize:'22px','font-weight': 'bold',color:'red',margin:'0px'})]);
    panels.add(panel);
    list.forEach(function(str){
      var _year = str.split('_').slice(-1)[0].slice(0,-2);
      if(_years[_year] === undefined){
        _years[_year] = [];
      }
      _years[_year].push(str);
    });

    Object.keys(_years).sort().forEach(function(_year){

      var list = _years[_year].sort();
      var panel_year = ui.Panel([ui.Label(''),ui.Label('// // --- '+ _year,{fontSize:'16px','font-weight': 'bold',color:'green',margin:'0px'})]);
      panel.add(panel_year);

      list.sort().forEach(function(str){
        panel_year.add(ui.Label('// ' + str,{margin:'0px'}));
      });
    });
  });
});


// 📌 Lista de imágenes finales a ser copiadas (MODIFICAR AQUÍ)
// Esta lista contiene los IDs de las imágenes de cicatrices de fuego que serán
// exportadas a las colecciones de destino (`col_nomask_id` y `col_mask_id`).
// Asegúrate de que los IDs aquí corresponden a las versiones finales y aprobadas
// de tus clasificaciones.
var final_collection = [
  // // --- --- region1
  // // --- 2005
  // "burned_area_paraguay_v3/burned_area_paraguay_l57_v3_region1_2005",
  // "burned_area_paraguay_v4/burned_area_paraguay_l57_v4_region1_2005",
  // "burned_area_paraguay_v6/burned_area_paraguay_l57_v6_region1_2005",
  // // --- 2013
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region1_2013",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2013",
  // // --- 2014
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region1_2014",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2014",
  // // --- 2015
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region1_2015",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2015",
  // // --- 2016
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region1_2016",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2016",
  // // --- 2017
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region1_2017",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2017",
  // // --- 2018
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region1_2018",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2018",
  // // --- 2019
  // "burned_area_paraguay_v1/burned_area_paraguay_l78_v1_region1_2019",
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region1_2019",
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region1_2019",
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region1_2019",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2019",
  // // --- 2020
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region1_2020",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2020",
  // // --- 2021
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region1_2021",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region1_2021",
  // // --- 2022
  // "burned_area_paraguay_v8/burned_area_paraguay_l89_v8_region1_2022",
  // "burned_area_paraguay_v9/burned_area_paraguay_l89_v9_region1_2022",
  // // --- 2023
  // "burned_area_paraguay_v9/burned_area_paraguay_l89_v9_region1_2023",
  // // --- 2024
  // "burned_area_paraguay_v9/burned_area_paraguay_l89_v9_region1_2024",
  // // --- --- region2
  // // --- 2005
  // "burned_area_paraguay_v4/burned_area_paraguay_l57_v4_region2_2005",
  // "burned_area_paraguay_v6/burned_area_paraguay_l57_v6_region2_2005",
  // // --- 2013
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2013",
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region2_2013",
  // // --- 2014
  // "burned_area_paraguay_v1/burned_area_paraguay_l78_v1_region2_2014",
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2014",
  // // --- 2015
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2015",
  // // --- 2016
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2016",
  // // --- 2017
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2017",
  // // --- 2018
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2018",
  // // --- 2019
  // "burned_area_paraguay_v15/burned_area_paraguay_l78_v15_region2_2019",
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2019",
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region2_2019",
  // "burned_area_paraguay_v22/burned_area_paraguay_l78_v22_region2_2019",
  // "burned_area_paraguay_v29/burned_area_paraguay_l78_v29_region2_2019",
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region2_2019",
  // // --- 2020
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2020",
  // // --- 2021
  // "burned_area_paraguay_v18/burned_area_paraguay_l78_v18_region2_2021",
  // // --- 2022
  // "burned_area_paraguay_v18/burned_area_paraguay_l89_v18_region2_2022",
  // // --- 2023
  // "burned_area_paraguay_v18/burned_area_paraguay_l89_v18_region2_2023",
  // // --- 2024
  // "burned_area_paraguay_v18/burned_area_paraguay_l89_v18_region2_2024",
  // // --- --- region3
  // // --- 2005
  // "burned_area_paraguay_v1/burned_area_paraguay_l57_v1_region3_2005",
  // "burned_area_paraguay_v2/burned_area_paraguay_l57_v2_region3_2005",
  // "burned_area_paraguay_v3/burned_area_paraguay_l57_v3_region3_2005",
  // "burned_area_paraguay_v4/burned_area_paraguay_l57_v4_region3_2005",
  // "burned_area_paraguay_v5/burned_area_paraguay_l57_v5_region3_2005",
  // // --- 2012
  // "burned_area_paraguay_v3/burned_area_paraguay_l57_v3_region3_2012",
  // // --- 2013
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2013",
  // // --- 2014
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2014",
  // // --- 2015
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2015",
  // // --- 2016
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2016",
  // // --- 2017
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2017",
  // // --- 2018
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2018",
  // // --- 2019
  // "burned_area_paraguay_v1/burned_area_paraguay_l78_v1_region3_2019",
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region3_2019",
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2019",
  // // --- 2020
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2020",
  // // --- 2021
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region3_2021",
  // // --- 2022
  // "burned_area_paraguay_v3/burned_area_paraguay_l89_v3_region3_2022",
  // // --- 2023
  // "burned_area_paraguay_v3/burned_area_paraguay_l89_v3_region3_2023",
  // // --- 2024
  // "burned_area_paraguay_v3/burned_area_paraguay_l89_v3_region3_2024",
  // // --- --- region4
  // // --- 2005
  // "burned_area_paraguay_v3/burned_area_paraguay_l57_v3_region4_2005",
  // // --- 2013
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2013",
  // // --- 2014
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2014",
  // // --- 2015
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2015",
  // // --- 2016
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2016",
  // // --- 2017
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2017",
  // // --- 2018
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2018",
  // // --- 2019
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region4_2019",
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region4_2019",
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2019",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region4_2019",
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region4_2019",
  // // --- 2020
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2020",
  // // --- 2021
  // "burned_area_paraguay_v4/burned_area_paraguay_l78_v4_region4_2021",
  // // --- 2022
  // "burned_area_paraguay_v4/burned_area_paraguay_l89_v4_region4_2022",
  // // --- 2023
  // "burned_area_paraguay_v4/burned_area_paraguay_l89_v4_region4_2023",
  // // --- 2024
  // "burned_area_paraguay_v4/burned_area_paraguay_l89_v4_region4_2024",
  // // --- --- region5
  // // --- 2005
  // "burned_area_paraguay_v2/burned_area_paraguay_l57_v2_region5_2005",
  // "burned_area_paraguay_v5/burned_area_paraguay_l57_v5_region5_2005",
  // // --- 2013
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2013",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2013",
  // // --- 2014
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2014",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2014",
  // // --- 2015
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2015",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2015",
  // // --- 2016
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2016",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2016",
  // // --- 2017
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2017",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2017",
  // // --- 2018
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2018",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2018",
  // // --- 2019
  // "burned_area_paraguay_v10/burned_area_paraguay_l78_v10_region5_2019",
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2019",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2019",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region5_2019",
  // // --- 2020
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2020",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2020",
  // // --- 2021
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region5_2021",
  // "burned_area_paraguay_v7/burned_area_paraguay_l78_v7_region5_2021",
  // // --- 2022
  // "burned_area_paraguay_v2/burned_area_paraguay_l89_v2_region5_2022",
  // "burned_area_paraguay_v7/burned_area_paraguay_l89_v7_region5_2022",
  // // --- 2023
  // "burned_area_paraguay_v2/burned_area_paraguay_l89_v2_region5_2023",
  // "burned_area_paraguay_v7/burned_area_paraguay_l89_v7_region5_2023",
  // // --- 2024
  // "burned_area_paraguay_v7/burned_area_paraguay_l89_v7_region5_2024",
  // // --- --- region6
  // // --- 2005
  // "burned_area_paraguay_v2/burned_area_paraguay_l57_v2_region6_2005",
  // "burned_area_paraguay_v4/burned_area_paraguay_l57_v4_region6_2005",
  // "burned_area_paraguay_v5/burned_area_paraguay_l57_v5_region6_2005",
  // // --- 2013
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2013",
  // // --- 2014
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2014",
  // // --- 2015
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2015",
  // // --- 2016
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2016",
  // // --- 2017
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2017",
  // // --- 2018
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2018",
  // // --- 2019
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region6_2019",
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region6_2019",
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2019",
  // // --- 2020
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2020",
  // // --- 2021
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region6_2021",
  // // --- 2022
  // "burned_area_paraguay_v8/burned_area_paraguay_l89_v8_region6_2022",
  // // --- 2023
  // "burned_area_paraguay_v8/burned_area_paraguay_l89_v8_region6_2023",
  // // --- 2024
  // "burned_area_paraguay_v8/burned_area_paraguay_l89_v8_region6_2024",
  // // --- --- region7
  // // --- 2006
  // "burned_area_paraguay_v2/burned_area_paraguay_l57_v2_region7_2006",
  // // --- 2013
  // "burned_area_paraguay_v1/burned_area_paraguay_l78_v1_region7_2013",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2013",
  // // --- 2014
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region7_2014",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2014",
  // // --- 2015
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region7_2015",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2015",
  // // --- 2016
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region7_2016",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2016",
  // // --- 2017
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region7_2017",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2017",
  // // --- 2018
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region7_2018",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2018",
  // // --- 2019
  // "burned_area_paraguay_v1/burned_area_paraguay_l78_v1_region7_2019",
  // "burned_area_paraguay_v2/burned_area_paraguay_l78_v2_region7_2019",
  // "burned_area_paraguay_v3/burned_area_paraguay_l78_v3_region7_2019",
  // "burned_area_paraguay_v5/burned_area_paraguay_l78_v5_region7_2019",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2019",
  // // --- 2020
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region7_2020",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2020",
  // // --- 2021
  // "burned_area_paraguay_v8/burned_area_paraguay_l78_v8_region7_2021",
  // "burned_area_paraguay_v9/burned_area_paraguay_l78_v9_region7_2021",
  // // --- 2022
  // "burned_area_paraguay_v8/burned_area_paraguay_l89_v8_region7_2022",
  // "burned_area_paraguay_v9/burned_area_paraguay_l89_v9_region7_2022",
  // // --- 2023
  // "burned_area_paraguay_v8/burned_area_paraguay_l89_v8_region7_2023",
  // "burned_area_paraguay_v9/burned_area_paraguay_l89_v9_region7_2023",
  // // --- 2024
  // "burned_area_paraguay_v9/burned_area_paraguay_l89_v9_region7_2024",  
];


// 🔁 Iterar sobre la lista de imágenes para copiar cada una
final_collection.forEach(function(id) {

  // 🧩 Preparación de nombres y rutas para la operación de copia.
  // Se extrae la versión del ID original.
  var version = id.split('_')[0].slice(-3); // Por ejemplo, de "burned_area_paraguay_v3/..." obtiene "v3".
  // Se crea un nuevo ID sin la información de la versión para el asset de destino.
  var new_id = id.replace(version, ''); // Elimina la parte de la versión del nombre del asset.
  // Construye la ruta completa del asset de origen.
  var source_asset = path + id;
  // Construye la ruta completa del asset de destino en la colección sin máscara.
  var destination_asset = col_nomask_id + new_id;

  try {
    // ✅ Paso 1: Verifica si la imagen fuente realmente existe en Earth Engine.
    // Esto previene errores si un ID en 'final_collection' es incorrecto o no existe.
    ee.data.getAsset(source_asset);
    print('ℹ️ Procesando imagen:', source_asset);

    try {
      // ⚠️ Paso 2: Intenta verificar si el asset ya existe en la colección de destino.
      ee.data.getAsset(destination_asset);
      print('✅ El asset ya existe en destino:', destination_asset);

      // Si existe y la opción de eliminación está activa, procede a eliminarlo.
      if (ELIMINA_SI_YA_EXISTE) {
        print('🗑️ Eliminando asset existente antes de copiar:', destination_asset);
        ee.data.deleteAsset(destination_asset); // Ejecuta la eliminación del asset.
        // Después de eliminar, procede a copiar la nueva versión.
        print('📤 Copiando nuevo asset al destino:', destination_asset);
        ee.data.copyAsset(source_asset, destination_asset);
      } else {
        // Si existe y la opción de eliminación no está activa, salta la copia.
        print('⏭️ Carga omitida (el asset ya existe y la eliminación automática no está activada):', destination_asset);
      }

    } catch (e2) {
      // 📤 Paso 3: Si el asset NO existe en el destino (el catch e2 se activa), cópialo directamente.
      print('📤 Copiando asset nuevo al destino:', destination_asset);
      ee.data.copyAsset(source_asset, destination_asset);
    }

  } catch (e1) {
    // ⚠️ Si la imagen fuente no se encuentra, se imprime una advertencia.
    print('⚠️ Imagen de origen no encontrada o ruta incorrecta:', source_asset);
  }
});
