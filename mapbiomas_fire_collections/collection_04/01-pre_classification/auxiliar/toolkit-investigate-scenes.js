var options = { 
  year:2025,
  month:1,
  region:'Brasil',
  geometry:ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na','Brazil')).geometry(),
  rota:'ANUAL'
};

//--- --- --- funçoes e variaveis auxiliares para o dataset

// Lista de imagens bloqueadas e excluidas da composição do moisaco

var blockList_landsat = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList').landsat();
var blockList_sentinel = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList').sentinel();

// recortando bordas de cenas landsat
function clipBoard_Landsat(image){
  return image
    .updateMask(ee.Image().paint(image.geometry().buffer(-3000)).eq(0));
}

// correção radiometrica das imagens landsat coleção 2
function  timeFlag_landsat (image) {

  var sensor = ee.String(image.getString('LANDSAT_PRODUCT_ID').split('_').get(0));
  sensor = ee.Number.parse(sensor.slice(-1));
  
  var path = ee.Number.parse(image.get('WRS_PATH'));
  var row = ee.Number.parse(image.get('WRS_ROW'));
  
  
  /* 
  // Symbol  Meaning                      Presentation  Examples
  // ------  -------                      ------------  -------
  // G       era                          text          AD
  // C       century of era (>=0)         number        20
  // Y       year of era (>=0)            year          1996
  
  // x       weekyear                     year          1996
  // w       week of weekyear             number        27
  // e       day of week                  number        2
  // E       day of week                  text          Tuesday; Tue
  
  // y       year                         year          1996
  // D       day of year                  number        189
  // M       month of year                month         July; Jul; 07
  // d       day of month                 number        10
  
  // a       halfday of day               text          PM
  // K       hour of halfday (0~11)       number        0
  // h       clockhour of halfday (1~12)  number        12
  
  // H       hour of day (0~23)           number        0
  // k       clockhour of day (1~24)      number        24
  // m       minute of hour               number        30
  // s       second of minute             number        55
  // S       fraction of second           number        978
  
  // z       time zone                    text          Pacific Standard Time; PST
  // Z       time zone offset/id          zone          -0800; -08:00; America/Los_Angeles
  
  // '       escape for text              delimiter
  // ''      single quote                 literal       '
  //*/    

  var dayOfYear = ee.Number.parse(ee.Date(image.get('system:time_start')).format('D'));
  var monthOfYear = ee.Number.parse(ee.Date(image.get('system:time_start')).format('M'));
  var year = ee.Number.parse(ee.Date(image.get('system:time_start')).format('Y'));
  var dayOfMonth = ee.Number.parse(ee.Date(image.get('system:time_start')).format('d'));
  // images: {
    dayOfYear = ee.Image(dayOfYear)
      .rename('dayOfYear')
      .int16();

    monthOfYear = ee.Image(monthOfYear)
      .rename('monthOfYear')
      .int16();

    year = ee.Image(year)
      .rename('year')
      .int16();

    dayOfMonth = ee.Image(dayOfMonth)
      .rename('dayOfMonth')
      .int16();

   sensor = ee.Image(sensor)
      .rename('sensor')
      .byte();

   path = ee.Image(path)
      .rename('path')
      .byte();

   row = ee.Image(row)
      .rename('row')
      .byte();
  // }
  
  return image
    .addBands(sensor)
    .addBands(dayOfYear)
    .addBands(monthOfYear)
    .addBands(year)
    .addBands(dayOfMonth)
    .addBands(path)
    .addBands(row);
    // .set({'string':string});
}


function address_Landsat (point,image){

  image.reduceRegion({
      reducer:ee.Reducer.sum(),
      geometry:point,
      scale:30,
      // crs:,
      // crsTransform:,
      // bestEffort:,
      maxPixels:1e11,
      // tileScale:
      })
    .evaluate(function(reduce){
      
      var sensorNames = {
        5:'LT05',
        7:'LE07',
        8:'LC08',
        9:'LC09'
      }
      var pathNames = {
        1:'00'+reduce['path'],
        2:'0'+reduce['path'],
        3:''+reduce['path'],
      }
      var rowNames = {
        1:'00'+reduce['row'],
        2:'0'+reduce['row'],
        3:''+reduce['row'],
      }
      var monthNames = {
        1:'0'+reduce['monthOfYear'],
        2:''+reduce['monthOfYear'],
      }
      var dayNames = {
        1:'0'+reduce['dayOfMonth'],
        2:''+reduce['dayOfMonth'],
      }
          // "LC08_006065_20190914",
  var landsatName = sensorNames[reduce.sensor] +
    '_' +
    pathNames[(''+reduce.path).length] +
    rowNames[(''+reduce.row).length] +
    '_' +
    reduce.year+
    monthNames[(''+reduce.monthOfYear).length] +
    dayNames[(''+reduce.dayOfMonth).length];
    
   //print('O pixel selecionado pertence a imagem: "'+landsatName+'"');
   print('"'+landsatName+'",');
  });
}

function  timeFlag_sentinel (image) {

  // - to flag Sentinel 2 path_row correlation
  var code_to_number =  ee.Dictionary({
      '22JFN':0,'22JGN':1,'22JGP':2,'22JGQ':3,'22JGR':4,'22JGS':5,'22JHS':6,'22JH':7,'22KHA':8,'22KHU':9,'22KHV':10,'23JKH':11,'23JKJ':12,'23JKK':13,'23JKL':14,'23JKM':15,'23JKN':16,'23JLG':17,'23JLH':18,'23JLJ':19,'23JLK':20,'23JLL':21,'23JLM':22,'23JLN':23,'23JMG':24,'23JMH':25,'23JMJ':26,'23JMK':27,'23JML':28,'23JMM':29,'23JMN':30,'23JNM':31,'23JNN':32,'23KKP':33,'23KKQ':34,'23KKR':35,'23KLA':36,'23KLP':37,'23KLQ':38,'23KLR':39,'23KLS':40,'23KL':41,'23KLU':42,'23KLV':43,'23KMA':44,'23KMB':45,'23KMP':46,'23KMQ':47,'23KMR':48,'23KMS':49,'23KM':50,'23KMU':51,'23KMV':52,'23KNA':53,'23KNB':54,'23KNP':55,'23KNQ':56,'23KNR':57,'23KNS':58,'23KN':59,'23KNU':60,'23KNV':61,'23KPA':62,'23KPB':63,'23KPR':64,'23KPS':65,'23KP':66,'23KPU':67,'23KPV':68,'23KQA':69,'23KQB':70,'23LMC':71,'23LMD':72,'23LME':73,'23LNC':74,'23LND':75,'23LNE':76,'23LNF':77,'23LNG':78,'23LNH':79,'23LNJ':80,'23LNK':81,'23LPC':82,'23LPD':83,'23LPE':84,'23LPF':85,'23LPG':86,'23LPH':87,'23LPJ':88,'23LPK':89,'23LPL':90,'23LQC':91,'23LQD':92,'23LQE':93,'23LQF':94,'23LQG':95,'23LQH':96,'23LQJ':97,'23LQK':98,'23LQL':99,
      '23LRE':100,'23LRF':101,'23LRG':102,'23LRH':103,'23LRJ':104,'23LRK':105,'23LRL':106,'23MPM':107,'23MPN':108,'23MPP':109,'23MQM':110,'23MQN':111,'23MQP':112,'23MQQ':113,'23MQR':114,'23MQS':115,'23MQ':116,'23MRM':117,'23MRN':118,'23MRP':119,'23MRQ':120,'23MRR':121,'23MRS':122,'23MR':123,'24L':124,'24M':125,'24MUA':126,'24MUB':127,'24MUC':128,'24MUS':129,'24MU':130,'24MUU':131,'24MUV':132,'24MVB':133,'24MVC':134,'20JRS':135,'20JR':136,'20KQU':137,'20KRA':138,'20KRB':139,'20KRC':140,'20KRU':141,'20KRV':142,'21H':143,'21HUE':144,'21J':145,'21JUF':146,'21JUG':147,'21JUH':148,'21JUJ':149,'21JUK':150,'21JUL':151,'21JUM':152,'21JUN':153,'21JVK':154,'21JVL':155,'21JVM':156,'21JVN':157,'21K':158,'21KUA':159,'21KUB':160,'21KUP':161,'21KUQ':162,'21KUR':163,'21KUS':164,'21KU':165,'21KUU':166,'21KUV':167,'21KVA':168,'21KVB':169,'21KVP':170,'21KVQ':171,'21KVR':172,'21KVS':173,'21KV':174,'21KVU':175,'21KVV':176,'21KWA':177,'21KWB':178,'21KWQ':179,'21KWR':180,'21KWS':181,'21KW':182,'21KWU':183,'21KWV':184,'21KXA':185,'21KXB':186,'21KXU':187,'21KXV':188,'21LUC':189,'21LUD':190,'21LVC':191,'21LVD':192,'21LVE':193,'21LVF':194,'21LVG':195,'21LVH':196,'21LWC':197,'21LWD':198,'21LWE':199,
      '21LWF':200,'21LWG':201,'21LWH':202,'21LWJ':203,'21LWK':204,'21LWL':205,'21LXC':206,'21LXD':207,'21LXE':208,'21LXF':209,'21LXG':210,'21LXH':211,'21LXJ':212,'21LXK':213,'21LXL':214,'21LYD':215,'21LYE':216,'21LYF':217,'21LYG':218,'21LYH':219,'21LYJ':220,'21LYK':221,'21LYL':222,'21LZH':223,'21LZJ':224,'21LZK':225,'21LZL':226,'21MWM':227,'21MXM':228,'21MXN':229,'21MXP':230,'21MXQ':231,'21MXR':232,'21MXS':233,'21MYM':234,'21MYN':235,'21MYP':236,'21MYQ':237,'21MYR':238,'21MYS':239,'21MY':240,'21MYU':241,'21MYV':242,'21MZM':243,'21MZN':244,'21MZP':245,'21MZQ':246,'21MZR':247,'21MZS':248,'21MZ':249,'21MZU':250,'21MZV':251,'21NYA':252,'21NZA':253,'21NZB':254,'21NZC':255,'21NZD':256,'21NZE':257,'21NZF':258,'22LBR':259,'22MBA':260,'22MBB':261,'22MBC':262,'22MBD':263,'22MBE':264,'22MBS':265,'22MB':266,'22MBU':267,'22MBV':268,'22MCA':269,'22MCB':270,'22MCC':271,'22MCD':272,'22MCE':273,'22MCV':274,'22MDE':275,'22NBF':276,'22NBG':277,'22NBH':278,'22NBJ':279,'22NBK':280,'22NBL':281,'22NBM':282,'22NCF':283,'22NCG':284,'22NCH':285,'22NCJ':286,'22NCK':287,'22NCL':288,'22NCM':289,'22NDF':290,'22NDG':291,'22NDH':292,'22NDJ':293,'22NDK':294,'22NDL':295,'22NDM':296,'22NEJ':297,'22NEK':298,'22NEL':299,
      '19LCF':300,'19LDF':301,'19LDG':302,'19LDH':303,'19LDJ':304,'19LDK':305,'19LDL':306,'19LEF':307,'19LEG':308,'19LEH':309,'19LEJ':310,'19LEK':311,'19LEL':312,'19LFE':313,'19LFF':314,'19LFG':315,'19LFH':316,'19LFJ':317,'19LFK':318,'19LFL':319,'19LGF':320,'19LGG':321,'19LGH':322,'19LGJ':323,'19LGK':324,'19LGL':325,'19LHL':326,'19MEM':327,'19MEN':328,'19MEP':329,'19MEQ':330,'19MFM':331,'19MFN':332,'19MFP':333,'19MFQ':334,'19MFR':335,'19MFS':336,'19MF':337,'19MFU':338,'19MFV':339,'19MGM':340,'19MGN':341,'19MGP':342,'19MGQ':343,'19MGR':344,'19MGS':345,'19MG':346,'19MGU':347,'19MGV':348,'19MHM':349,'19MHN':350,'19MHP':351,'19MHQ':352,'19MHR':353,'19MHS':354,'19MH':355,'19MHU':356,'19MHV':357,'19NGA':358,'19NGB':359,'19NGC':360,'19NGD':361,'19NHA':362,'19NHB':363,'19NHC':364,'19NHD':365,'19NHE':366,'19NHF':367,'19NHG':368,'19NHH':369,'20MKA':370,'20MKB':371,'20MKC':372,'20MKD':373,'20MKE':374,'20MKU':375,'20MKV':376,'20MLC':377,'20MLD':378,'20MLE':379,'20NKF':380,'20NKG':381,'20NKH':382,'20NKJ':383,'20NKK':384,'20NKL':385,'20NKM':386,'20NKN':387,'20NLF':388,'20NLG':389,'20NLH':390,'20NLJ':391,'20NLK':392,'20NLL':393,'20NLM':394,'20NLN':395,'20NMH':396,'20NMJ':397,'20NMK':398,'20NML':399,
      '20NMM':400,'20NMN':401,'20NNM':402,'20NNN':403,'20NNP':404,'24LYP':405,'24LYQ':406,'24LYR':407,'24LZP':408,'24LZQ':409,'24LZR':410,'24MZS':411,'24MZ':412,'24MZU':413,'24MZV':414,'25LBJ':415,'25LBK':416,'25LBL':417,'25LCH':418,'25LCJ':419,'25LCK':420,'25LCL':421,'25LDJ':422,'25LDK':423,'25LDL':424,'25MBM':425,'25MBN':426,'25MBP':427,'25MBQ':428,'25MCM':429,'25MCN':430,'25MCP':431,'25MCQ':432,'25MDM':433,'25MDN':434,'25MDP':435,'25MDQ':436,'25MEP':437,'25MEQ':438,'24LYM':439,'24LYN':440,'24LZL':441,'24LZM':442,'24LZN':443,'25LBF':444,'25LBG':445,'25LBH':446,'25LCF':447,'25LCG':448,'21HWA':449,'21HWV':450,'21HXA':451,'21HXB':452,'21HXC':453,'21HXD':454,'21HXE':455,'21HXV':456,'21HYA':457,'21HYB':458,'21HYC':459,'21HYD':460,'21HYE':461,'21HYV':462,'21JYF':463,'21JYG':464,'21JYH':465,'21JZM':466,'22HBE':467,'22HBF':468,'22HBG':469,'22HBH':470,'22HBJ':471,'22HBK':472,'22HCE':473,'22HCF':474,'22HCG':475,'22HCH':476,'22HCJ':477,'22HCK':478,'22HDH':479,'22HDJ':480,'22HDK':481,'22JBL':482,'22JBM':483,'22JBN':484,'22JBP':485,'22JBQ':486,'22JBR':487,'22JBS':488,'22JCL':489,'22JCM':490,'22JCN':491,'22JCP':492,'22JCQ':493,'22JCR':494,'22JCS':495,'22JC':496,'22JDL':497,'22JDM':498,'22JDN':499,
      '22JDP':500,'22JDQ':501,'22JDR':502,'22JDS':503,'22JD':504,'22JEN':505,'22JEP':506,'22JEQ':507,'22JER':508,'22JES':509,'22JE':510,'22JFS':511,'22JF':512,'22KCA':513,'22KCU':514,'22KCV':515,'22KDA':516,'22KDB':517,'22KDC':518,'22KDD':519,'22KDE':520,'22KDF':521,'22KDU':522,'22KDV':523,'22KEA':524,'22KEB':525,'22KEC':526,'22KED':527,'22KEE':528,'22KEF':529,'22KEG':530,'22KEU':531,'22KEV':532,'22KFA':533,'22KFB':534,'22KFC':535,'22KFD':536,'22KFE':537,'22KFF':538,'22KFG':539,'22KFU':540,'22KFV':541,'22KGA':542,'22KGB':543,'22KGC':544,'22KGD':545,'22KGE':546,'22KGF':547,'22KGG':548,'22KHF':549,'22KHG':550,'22LEH':551,'22LEJ':552,'22LEK':553,'22LFH':554,'22LFJ':555,'22LFK':556,'22LFL':557,'22LFM':558,'22LFN':559,'22LFP':560,'22LGH':561,'22LGJ':562,'22LGK':563,'22LGL':564,'22LGM':565,'22LGN':566,'22LGP':567,'22LGQ':568,'22LGR':569,'22LHH':570,'22LHJ':571,'22LHK':572,'22LHL':573,'22LHM':574,'22LHN':575,'22LHP':576,'22LHQ':577,'22LHR':578,'22MGS':579,'22MG':580,'22MGU':581,'22MHA':582,'22MHB':583,'22MHC':584,'22MHS':585,'22MH':586,'22MHU':587,'22MHV':588,'23KKB':589,'23LKC':590,'23LKD':591,'23LKE':592,'23LKF':593,'23LKG':594,'23LKH':595,'23LKJ':596,'23LKK':597,'23LKL':598,'23LLG':599,
      '23LLH':600,'23LLJ':601,'23LLK':602,'23LLL':603,'23MKM':604,'23MKN':605,'23MKP':606,'23MKQ':607,'23MKR':608,'23MKS':609,'23MK':610,'23MKU':611,'23MKV':612,'23MLM':613,'23MLN':614,'23MLP':615,'23MLQ':616,'23MLR':617,'23MLS':618,'23ML':619,'23MLU':620,'23MLV':621,'23MMM':622,'23MMN':623,'23MMP':624,'23MMQ':625,'23MMR':626,'23MMS':627,'23MM':628,'23MMU':629,'23MMV':630,'23MNR':631,'23MNS':632,'23MN':633,'23MNU':634,'23MNV':635,'20KMG':636,'20KNG':637,'20KPG':638,'20KQG':639,'20LMH':640,'20LNH':641,'20LNJ':642,'20LNK':643,'20LNL':644,'20LNM':645,'20LNN':646,'20LPH':647,'20LPJ':648,'20LPK':649,'20LPL':650,'20LPM':651,'20LPN':652,'20LPP':653,'20LPQ':654,'20LPR':655,'20LQH':656,'20LQJ':657,'20LQK':658,'20LQL':659,'20LQM':660,'20LQN':661,'20LQP':662,'20LQQ':663,'20LQR':664,'20LRH':665,'20LRJ':666,'20LRK':667,'20LRL':668,'20LRM':669,'20LRN':670,'20LRP':671,'20LRQ':672,'20LRR':673,'20MPS':674,'20MQA':675,'20MQS':676,'20MQ':677,'20MQU':678,'20MQV':679,'20MRA':680,'20MRB':681,'20MRC':682,'20MRD':683,'20MRE':684,'20MRS':685,'20MR':686,'20MRU':687,'20MRV':688,'20NRF':689,'21L':690,'21LUK':691,'21LUL':692,'21M':693,'21MUM':694,'21MUN':695,'21MUP':696,'21MUQ':697,'21MUR':698,'21MUS':699,
      '21MU':700,'21MUU':701,'21MUV':702,'21MVQ':703,'21MVR':704,'21MVS':705,'21MV':706,'21MVU':707,'21MVV':708,'21MWU':709,'21MWV':710,'21N':711,'21NUA':712,'21NUB':713,'21NUC':714,'21NUD':715,'21NUE':716,'21NUF':717,'21NUG':718,'21NUH':719,'21NUJ':720,'21NVA':721,'21NVB':722,'21NVC':723,'21NVD':724,'21NVE':725,'21NVF':726,'21NVG':727,'21NVH':728,'21NVJ':729,'21NWA':730,'21NWB':731,'21NWC':732,'21NWD':733,'21NWE':734,'21NWF':735,'21NWG':736,'21NWH':737,'21NWJ':738,'21NXD':739,'21NXE':740,'21NXF':741,'21NXG':742,'21NXH':743,'21NXJ':744,'21NYJ':745,'21PVK':746,'21PWK':747,'21PXK':748,'21PYK':749,'20KMD':750,'20KME':751,'20KMF':752,'20KNC':753,'20KND':754,'20KNE':755,'20KNF':756,'20KPB':757,'20KPC':758,'20KPD':759,'20KPE':760,'20KPF':761,'20KQD':762,'20KQE':763,'20KQF':764,'18LWN':765,'18LWP':766,'18LWQ':767,'18LWR':768,'18LXM':769,'18LXN':770,'18LXP':771,'18LXQ':772,'18LXR':773,'18LYM':774,'18LYN':775,'18LYP':776,'18LYQ':777,'18LYR':778,'18LZM':779,'18LZN':780,'18LZP':781,'18LZQ':782,'18LZR':783,'18MXS':784,'18MX':785,'18MXU':786,'18MXV':787,'18MYA':788,'18MYB':789,'18MYC':790,'18MYD':791,'18MYS':792,'18MY':793,'18MYU':794,'18MYV':795,'18MZA':796,'18MZB':797,'18MZC':798,'18MZD':799,
      '18MZE':800,'18MZS':801,'18MZ':802,'18MZU':803,'18MZV':804,'18NZF':805,'18NZG':806,'18NZH':807,'19LBJ':808,'19LBK':809,'19LBL':810,'19MBM':811,'19MBN':812,'19MBP':813,'19MBQ':814,'19MBR':815,'19MBS':816,'19MB':817,'19MBU':818,'19MBV':819,'19MCN':820,'19MCP':821,'19MCQ':822,'19MCR':823,'19MCS':824,'19MC':825,'19MCU':826,'19MCV':827,'19MD':828,'19MDU':829,'19MDV':830,'19NBA':831,'19NBB':832,'19NBC':833,'19NBD':834,'19NCA':835,'19NCB':836,'19NCC':837,'19NCD':838,'19NCE':839,'19NDA':840,'19NDB':841,'19NDC':842,'19NDD':843,'19NDE':844,'19NEB':845,'19NEC':846,'19NED':847,'19NEE':848,'19NEF':849,'24MVA':850,'24MVV':851,'24MWA':852,'24MWB':853,'24MWC':854,'24MWV':855,'24MXA':856,'24MXB':857,'24MXV':858,'23JNL':859,'23JPL':860,'23JPM':861,'23JPN':862,'23JQL':863,'23JQM':864,'23JQN':865,'23KPP':866,'23KPQ':867,'23KQP':868,'23KQQ':869,'23KQR':870,'23KQS':871,'23KQ':872,'23KQU':873,'23KQV':874,'23KRA':875,'23KRB':876,'23KRP':877,'23KRQ':878,'23KRR':879,'23KRS':880,'23KR':881,'23KRU':882,'23KRV':883,'23LRC':884,'23LRD':885,'24K':886,'24KUD':887,'24KUE':888,'24KUF':889,'24KUG':890,'24LUH':891,'24LUJ':892,'24LUK':893,'24LUL':894,'24LUM':895,'24LUN':896,'24LUP':897,'24LUQ':898,'24LUR':899,
      '24LVJ':900,'24LVK':901,'24LVL':902,'24LVM':903,'24LVN':904,'24LVP':905,'24LVQ':906,'24LVR':907,'24LWP':908,'24LWQ':909,'24LWR':910,'24MVS':911,'24MV':912,'24MVU':913,'24MWS':914,'24MW':915,'24MWU':916,'24MX':917,'24MXU':918,'21HUC':919,'21HUD':920,'21HVB':921,'21HVC':922,'21HVD':923,'21HVE':924,'21HWC':925,'21HWD':926,'21HWE':927,'21JVF':928,'21JVG':929,'21JVH':930,'21JVJ':931,'21JWF':932,'21JWG':933,'21JWH':934,'21JWJ':935,'21JWK':936,'21JWL':937,'21JWM':938,'21JWN':939,'21JXH':940,'21JXJ':941,'21JXK':942,'21JXL':943,'21JXM':944,'21JXN':945,'21JYM':946,'21JYN':947,'21KWP':948,'21KXP':949,'21KXQ':950,'21KXR':951,'21KXS':952,'21KX':953,'21KYA':954,'21KYB':955,'21KYP':956,'21KYQ':957,'21KYR':958,'21KYS':959,'21KY':960,'21KYU':961,'21KYV':962,'21KZA':963,'21KZB':964,'21KZR':965,'21KZS':966,'21KZ':967,'21KZU':968,'21KZV':969,'21LYC':970,'21LZC':971,'21LZD':972,'21LZE':973,'21LZF':974,'21LZG':975,'22KBB':976,'22KBC':977,'22KBD':978,'22KBE':979,'22KBF':980,'22KBG':981,'22KCG':982,'22LBH':983,'22LBJ':984,'22LBK':985,'22LBL':986,'22LBM':987,'22LBN':988,'22LBP':989,'22LBQ':990,'22LCH':991,'22LCJ':992,'22LCK':993,'22LCL':994,'22LCM':995,'22LCN':996,'22LCP':997,'22LCQ':998,'22LCR':999,
      '22LDM':1000,'22LDN':1001,'22LDP':1002,'22LDQ':1003,'22LDR':1004,'22MCS':1005,'22MC':1006,'22MCU':1007,'22MDA':1008,'22MDB':1009,'22MDC':1010,'22MDD':1011,'22MDS':1012,'22MD':1013,'22MDU':1014,'22MDV':1015,'22MEA':1016,'22MEB':1017,'22MEC':1018,'22MED':1019,'22MEE':1020,'22MES':1021,'22ME':1022,'22MEU':1023,'22MEV':1024,'22MFA':1025,'22MFB':1026,'22MFC':1027,'22MFD':1028,'22MFE':1029,'22NEF':1030,'22NEG':1031,'22NEH':1032,'22NFF':1033,'22NFG':1034,'22NFH':1035,'22NFJ':1036,'22NFK':1037,'22NGF':1038,'22NGG':1039,'22NGH':1040,'22NGJ':1041,'22NGK':1042,'19LHG':1043,'19LHH':1044,'19LHJ':1045,'19LHK':1046,'20LKM':1047,'20LKN':1048,'20LKP':1049,'20LKQ':1050,'20LKR':1051,'20LLL':1052,'20LLM':1053,'20LLN':1054,'20LLP':1055,'20LLQ':1056,'20LLR':1057,'20LMQ':1058,'20LMR':1059,'20MKS':1060,'20MK':1061,'20MLA':1062,'20MLB':1063,'20MLS':1064,'20ML':1065,'20MLU':1066,'20MLV':1067,'20MMA':1068,'20MMB':1069,'20MMC':1070,'20MMD':1071,'20MME':1072,'20MMS':1073,'20MM':1074,'20MMU':1075,'20MMV':1076,'20MNA':1077,'20MNB':1078,'20MNC':1079,'20MND':1080,'20MNE':1081,'20MNV':1082,'20MPD':1083,'20MPE':1084,'20NMF':1085,'20NMG':1086,'20NNF':1087,'20NNG':1088,'20NNH':1089,'20NNJ':1090,'20NNK':1091,'20NNL':1092,'20NPF':1093,'20NPG':1094,'20NPH':1095,'20NPJ':1096,'20NPK':1097,'20NPL':1098,'20NPM':1099,
      '20NPN':1100,'20NPP':1101,'20NQJ':1102,'20NQK':1103,'20NQL':1104,'20NQM':1105,'20NQN':1106,'20NQP':1107,'20NRN':1108,'20NRP':1109,'19KHB':1110,'19LGC':1111,'19LGD':1112,'19LGE':1113,'19LHC':1114,'19LHD':1115,'19LHE':1116,'19LHF':1117,'20KKG':1118,'20LKH':1119,'20LKJ':1120,'20LKK':1121,'20LKL':1122,'20LLK':1123,'18LUQ':1124,'18LUR':1125,'18MUS':1126,'18MU':1127,'18MUU':1128,'18MUV':1129,'18MVA':1130,'18MVB':1131,'18MV':1132,'18MVU':1133,'18MVV':1134,'22HEG':1135,'22HEH':1136,'22HEJ':1137,'22HEK':1138,'22HFG':1139,'22HFH':1140,'22HFJ':1141,'22HFK':1142,'22JEL':1143,'22JEM':1144,'22JFL':1145,'22JFM':1146,'22JFP':1147,'22JFQ':1148,'22JFR':1149,'22JGL':1150,'22JGM':1151,'22JG':1152,'22KGU':1153,'22KGV':1154,'22KHB':1155,'22KHC':1156,'22KHD':1157,'22KHE':1158,'23KKA':1159,'23KKS':1160,'23KK':1161,'23KKU':1162,'23KKV':1163,'23KLB':1164,'23LLC':1165,'23LLD':1166,'23LLE':1167,'23LLF':1168,'23LMF':1169,'23LMG':1170,'23LMH':1171,'23LMJ':1172,'23LMK':1173,'23LML':1174,'23LNL':1175,'23MNM':1176,'23MNN':1177,'23MNP':1178,'23MNQ':1179,'23MPQ':1180,'23MPR':1181,'23MPS':1182,'23MP':1183,'23MPU':1184,'23MQU':1185,'23MRU':1186,'20KPA':1187,'20KQA':1188,'20KQB':1189,'20KQC':1190,'20KQV':1191,'20KRD':1192,'20KRE':1193,'20KRF':1194,'20KRG':1195,'21LUE':1196,'21LUF':1197,'21LUG':1198,'21LUH':1199,
      '21LUJ':1200,'21LVJ':1201,'21LVK':1202,'21LVL':1203,'21MVM':1204,'21MVN':1205,'21MVP':1206,'21MWN':1207,'21MWP':1208,'21MWQ':1209,'21MWR':1210,'21MWS':1211,'21MW':1212,'21MX':1213,'21MXU':1214,'21MXV':1215,'21NXA':1216,'21NXB':1217,'21NXC':1218,'21NYB':1219,'21NYC':1220,'21NYD':1221,'21NYE':1222,'21NYF':1223,'21NYG':1224,'21NYH':1225,'21NZG':1226,'19LBG':1227,'19LBH':1228,'19LCG':1229,'19LCH':1230,'19LCJ':1231,'19LCK':1232,'19LCL':1233,'19MCM':1234,'19MDM':1235,'19MDN':1236,'19MDP':1237,'19MDQ':1238,'19MDR':1239,'19MDS':1240,'19MER':1241,'19MES':1242,'19ME':1243,'19MEU':1244,'19MEV':1245,'19NEA':1246,'19NFA':1247,'19NFB':1248,'19NFC':1249,'19NFD':1250,'19NFE':1251,'19NFF':1252,'19NGE':1253,'19NGF':1254,'19NGG':1255,'24KUA':1256,'24KUB':1257,'24KUC':1258,'24KUU':1259,'24KUV':1260,'24KVA':1261,'24KVB':1262,'24KVC':1263,'24KVD':1264,'24KVE':1265,'24KVF':1266,'24KVG':1267,'24KVU':1268,'24KVV':1269,'24KWA':1270,'24KWB':1271,'24KWC':1272,'24KWD':1273,'24KWE':1274,'24KWF':1275,'24KWG':1276,'24KXF':1277,'24KXG':1278,'24LVH':1279,'24LWH':1280,'24LWJ':1281,'24LWK':1282,'24LWL':1283,'24LWM':1284,'24LWN':1285,'24LXH':1286,'24LXJ':1287,'24LXK':1288,'24LXL':1289,'24LXM':1290,'24LXN':1291,'24LXP':1292,'24LXQ':1293,'24LXR':1294,'24LYK':1295,'24LYL':1296,'24MXS':1297,'24MYA':1298,'24MYS':1299,
      '24MY':1300,'24MYU':1301,'24MYV':1302,'24MZA':1303,'25MBR':1304,'21HWB':1305,'21JXF':1306,'21JXG':1307,'21JYJ':1308,'21JYK':1309,'21JYL':1310,'21JZN':1311,'21KZP':1312,'21KZQ':1313,'22JB':1314,'22KBA':1315,'22KBU':1316,'22KBV':1317,'22KCB':1318,'22KCC':1319,'22KCD':1320,'22KCE':1321,'22KCF':1322,'22KDG':1323,'22LDH':1324,'22LDJ':1325,'22LDK':1326,'22LDL':1327,'22LEL':1328,'22LEM':1329,'22LEN':1330,'22LEP':1331,'22LEQ':1332,'22LER':1333,'22LFQ':1334,'22LFR':1335,'22MFS':1336,'22MF':1337,'22MFU':1338,'22MFV':1339,'22MGA':1340,'22MGB':1341,'22MGC':1342,'22MGD':1343,'22MGE':1344,'22MGV':1345,'22MHD':1346,'22MHE':1347,'22NHF':1348,'23NKA':1349,'23NLA':1350,'20KLF':1351,'20KLG':1352,'20LLH':1353,'20LLJ':1354,'20LMJ':1355,'20LMK':1356,'20LML':1357,'20LMM':1358,'20LMN':1359,'20LMP':1360,'20LNP':1361,'20LNQ':1362,'20LNR':1363,'20MNS':1364,'20MN':1365,'20MNU':1366,'20MPA':1367,'20MPB':1368,'20MPC':1369,'20MP':1370,'20MPU':1371,'20MPV':1372,'20MQB':1373,'20MQC':1374,'20MQD':1375,'20MQE':1376,'20NQF':1377,'20NQG':1378,'20NQH':1379,'20NRG':1380,'20NRH':1381,'20NRJ':1382,'20NRK':1383,'20NRL':1384,'20NRM':1385,'21P':1386,'21PUK':1387,'20KLE':1388,'18LUN':1389,'18LUP':1390,'18LVN':1391,'18LVP':1392,'18LVQ':1393,'18LVR':1394,'18MVC':1395,'18MVS':1396,'18MWA':1397,'18MWB':1398,'18MWC':1399,
      '18MWD':1400,'18MWS':1401,'18MW':1402,'18MWU':1403,'18MWV':1404,'18MXA':1405,'18MXB':1406,'18MXC':1407,'18MXD':1408,'18MXE':1409,'18MYE':1410,'18NXF':1411,'18NYF':1412,'18NYG':1413,'27MYN':1414,'27MZN':1415,'27MZP':1416,'27MZQ':1417,'27MZR':1418,'28MBA':1419,'28MBU':1420,'28MBV':1421,'22NEM':1422,'25MEN':1423,'21PYL':1424,'23MRV':1425,'23NRA':1426,'23NRB':1427,'23NRC':1428,'24MUD':1429,'24MUE':1430,'24MVD':1431,'24MVE':1432,'24N':1433,'24NUF':1434,'24NUG':1435,'24NUH':1436,'24NUJ':1437,'24NUK':1438,'24NUL':1439,'24NVF':1440,'24NVG':1441,'24NVH':1442,'24NVJ':1443,'24NVK':1444,'24NVL':1445,'24NWF':1446,'24NWG':1447,'24NWH':1448,'24NWJ':1449,'24NWK':1450,'24NWL':1451,'23JKG':1452,'22NBN':1453,'22NBP':1454,'22NCN':1455,'22NCP':1456,'22NDN':1457,'22NDP':1458,'22NEN':1459,'22NEP':1460,'23NLB':1461,'23NLC':1462,'23NLD':1463,'23NLE':1464,'23NLF':1465,'23NMA':1466,'23NMB':1467,'23NMC':1468,'23NMD':1469,'23NME':1470,'23NMF':1471,'23NMG':1472,'23NNA':1473,'23NNB':1474,'23NNC':1475,'23NND':1476,'23NNE':1477,'23NNF':1478,'23NNG':1479,'23NPA':1480,'23NPB':1481,'23NPC':1482,'23NPD':1483,'23NPE':1484,'23NPF':1485,'23NPG':1486,'23NQF':1487,'23NQG':1488,'22NFL':1489,'22NFM':1490,'22NFN':1491,'22NFP':1492,'22NGL':1493,'22NGM':1494,'22NGN':1495,'22NGP':1496,'22NHK':1497,'22NHL':1498,'22NHM':1499,
      '22NHN':1500,'23NKH':1501,'23MPV':1502,'23MQV':1503,'23NQA':1504,'23NQB':1505,'23NQC':1506,'23NQD':1507,'23NQE':1508,'23NRD':1509,'23NRE':1510,'23NRF':1511,'23NRG':1512,'22NHG':1513,'22NHH':1514,'22NHJ':1515,'23NKB':1516,'23NKC':1517,'23NKD':1518,'23NKE':1519,'23NKF':1520,'23NKG':1521,'23NLG':1522,'23NLH':1523,
      '23NMH':1524,
  });
 
  var dayOfYear = ee.Number.parse(ee.Date(image.get('system:time_start')).format('D'));
  var monthOfYear = ee.Number.parse(ee.Date(image.get('system:time_start')).format('M'));
  var year = ee.Number.parse(ee.Date(image.get('system:time_start')).format('Y'));
  var dayOfMonth = ee.Number.parse(ee.Date(image.get('system:time_start')).format('d'));
  // images: {
    dayOfYear = ee.Image(dayOfYear)
      .rename('dayOfYear')
      .int16();

    monthOfYear = ee.Image(monthOfYear)
      .rename('monthOfYear')
      .int16();

    year = ee.Image(year)
      .rename('year')
      .int16();

    dayOfMonth = ee.Image(dayOfMonth)
      .rename('dayOfMonth')
      .int16();
  // } 
  // other flags

  var id = image.getString('system:index');
  
  var list = ee.List(id.split('_'))
    .map(function(str){return ee.String(str).split('T')})
    .flatten();

                  // 20190102T140049_20190102T140050_T21KUT
  // example id:  20170328T083601_20170328T084228_T35RNK
  // split                T               T      _T           igual em todas as imagens
  // split        20170328                                    propertie_0     -   8 caracteres
  // split                 083601                             propertie_1     -   6 caracteres
  // split                        20170328                    propertie_2     -   8 caracteres
  // split                                 084228             propertie_3     -   6 caracteres
  // split                                         35RNK      propertie_4     -   convertido para numerico, ocupando 4 caracteres
  //                                                 
  // var other_flag = {
    
    var propertie_0 = ee.Number.parse(list.get(0));
    var band_propertie_0 = ee.Image(propertie_0)
      .rename('propertie_0');
    
    var propertie_1 = ee.Number.parse(list.get(1));
    var band_propertie_1 = ee.Image(propertie_1)
      .rename('propertie_1');
    
    var propertie_2 = ee.Number.parse(list.get(2));
    var band_propertie_2 = ee.Image(propertie_2)
      .rename('propertie_2');
    
    var propertie_3 = ee.Number.parse(list.get(3));
    var band_propertie_3 = ee.Image(propertie_3)
      .rename('propertie_3');
    
    var propertie_4 = ee.Number.parse(code_to_number.get(list.get(5)));
    var band_propertie_4 = ee.Image(propertie_4)
      .rename('propertie_4');
    
  // };
  
  var flags = image.addBands(dayOfYear)
    .addBands(monthOfYear)
    .addBands(year)
    .addBands(dayOfMonth)
    .addBands(band_propertie_0)
    .addBands(band_propertie_1)
    .addBands(band_propertie_2)
    .addBands(band_propertie_3)
    .addBands(band_propertie_4)
    .int32();

  
  return image
    .addBands(flags);

}


function corrections_LS57_col2 (image){
  var opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  // - return 
  
  image = image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
              
  // mascara de nuvem
  var cloudShadowBitMask = (1 << 3);
  var cloudsBitMask = (1 << 5);


  var qa = image.select('QA_PIXEL');
  var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
      .and(qa.bitwiseAnd(cloudsBitMask).eq(0));

  // mascara de ruidos, saturação radiométrica
  function bitwiseExtract(value, fromBit, toBit) {
    if (toBit === undefined)
      toBit = fromBit;
    var maskSize = ee.Number(1).add(toBit).subtract(fromBit);
    var mask = ee.Number(1).leftShift(maskSize).subtract(1);
    return value.rightShift(fromBit).bitwiseAnd(mask);
  }

  var clear = bitwiseExtract(qa, 6); // 1 if clear
  var water = bitwiseExtract(qa, 7); // 1 if water

  var radsatQA = image.select('QA_RADSAT');
  var band5Saturated = bitwiseExtract(radsatQA, 4); // 0 if band 5 is not saturated
  var anySaturated = bitwiseExtract(radsatQA, 0, 6); // 0 if no bands are saturated

  var mask_saturation = clear
    .or(water)
    .and(anySaturated.not());
  
  // is visible bands with negative reflectance? 
  var negative_mask = image.select(['SR_B1']).gt(0).and(
    image.select(['SR_B2']).gt(0)).and(
      image.select(['SR_B3']).gt(0)).and(
        image.select(['SR_B4']).gt(0)).and(
          image.select(['SR_B5']).gt(0)).and(
            image.select(['SR_B7']).gt(0));
  
  // - return
  image = image
    .updateMask(mask)
    .updateMask(mask_saturation)
    .updateMask(negative_mask);
        

  var oldBands = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7',];
  var newBands = ['blue', 'green','red',  'nir',  'swir1','swir2'];
  // - return
  image = image.select(oldBands,newBands);

  // -
  return timeFlag_landsat(image);
}

function address_Sentinel (point,image){
  
    // - to flag Sentinel 2 path_row correlation
    var number_to_code =  {
            0: '22JFN',   1:  '22JGN',   2: '22JGP',   3: '22JGQ',   4: '22JGR',   5: '22JGS',   6: '22JHS',   7: '22JH',   8: '22KHA',   9: '22KHU',  10: '22KHV',  11: '23JKH',  12: '23JKJ',  13: '23JKK',  14: '23JKL',  15: '23JKM',  16: '23JKN',  17: '23JLG',  18: '23JLH',  19: '23JLJ',  20: '23JLK',  21: '23JLL',  22: '23JLM',  23: '23JLN',  24: '23JMG',  25: '23JMH',  26: '23JMJ',  27: '23JMK',  28: '23JML',  29: '23JMM',  30: '23JMN',  31: '23JNM',  32: '23JNN',  33: '23KKP',  34: '23KKQ',  35: '23KKR',  36: '23KLA',  37: '23KLP',  38: '23KLQ',  39: '23KLR',  40: '23KLS',  41: '23KL',  42: '23KLU',  43: '23KLV',  44: '23KMA',  45: '23KMB',  46: '23KMP',  47: '23KMQ',  48: '23KMR',  49: '23KMS',  50: '23KM',  51: '23KMU',  52: '23KMV',  53: '23KNA',  54: '23KNB',  55: '23KNP',  56: '23KNQ',  57: '23KNR',  58: '23KNS',  59: '23KN',  60: '23KNU',  61: '23KNV',  62: '23KPA',  63: '23KPB',  64: '23KPR',  65: '23KPS',  66: '23KP',  67: '23KPU',  68: '23KPV',  69: '23KQA',  70: '23KQB',  71: '23LMC',  72: '23LMD',  73: '23LME',  74: '23LNC',  75: '23LND',  76: '23LNE',  77: '23LNF',  78: '23LNG',  79: '23LNH',  80: '23LNJ',  81: '23LNK',  82: '23LPC',  83: '23LPD',  84: '23LPE',  85: '23LPF',  86: '23LPG',  87: '23LPH',  88: '23LPJ',  89: '23LPK',  90: '23LPL',  91: '23LQC',  92: '23LQD',  93: '23LQE',  94: '23LQF',  95: '23LQG',  96: '23LQH',  97: '23LQJ',  98: '23LQK',  99: '23LQL',
          100: '23LRE', 101:  '23LRF', 102: '23LRG', 103: '23LRH', 104: '23LRJ', 105: '23LRK', 106: '23LRL', 107: '23MPM', 108: '23MPN', 109: '23MPP', 110: '23MQM', 111: '23MQN', 112: '23MQP', 113: '23MQQ', 114: '23MQR', 115: '23MQS', 116: '23MQ', 117: '23MRM', 118: '23MRN', 119: '23MRP', 120: '23MRQ', 121: '23MRR', 122: '23MRS', 123: '23MR', 124: '24L', 125: '24M', 126: '24MUA', 127: '24MUB', 128: '24MUC', 129: '24MUS', 130: '24MU', 131: '24MUU', 132: '24MUV', 133: '24MVB', 134: '24MVC', 135: '20JRS', 136: '20JR', 137: '20KQU', 138: '20KRA', 139: '20KRB', 140: '20KRC', 141: '20KRU', 142: '20KRV', 143: '21H', 144: '21HUE', 145: '21J', 146: '21JUF', 147: '21JUG', 148: '21JUH', 149: '21JUJ', 150: '21JUK', 151: '21JUL', 152: '21JUM', 153: '21JUN', 154: '21JVK', 155: '21JVL', 156: '21JVM', 157: '21JVN', 158: '21K', 159: '21KUA', 160: '21KUB', 161: '21KUP', 162: '21KUQ', 163: '21KUR', 164: '21KUS', 165: '21KU', 166: '21KUU', 167: '21KUV', 168: '21KVA', 169: '21KVB', 170: '21KVP', 171: '21KVQ', 172: '21KVR', 173: '21KVS', 174: '21KV', 175: '21KVU', 176: '21KVV', 177: '21KWA', 178: '21KWB', 179: '21KWQ', 180: '21KWR', 181: '21KWS', 182: '21KW', 183: '21KWU', 184: '21KWV', 185: '21KXA', 186: '21KXB', 187: '21KXU', 188: '21KXV', 189: '21LUC', 190: '21LUD', 191: '21LVC', 192: '21LVD', 193: '21LVE', 194: '21LVF', 195: '21LVG', 196: '21LVH', 197: '21LWC', 198: '21LWD', 199: '21LWE',
          200: '21LWF', 201:  '21LWG', 202: '21LWH', 203: '21LWJ', 204: '21LWK', 205: '21LWL', 206: '21LXC', 207: '21LXD', 208: '21LXE', 209: '21LXF', 210: '21LXG', 211: '21LXH', 212: '21LXJ', 213: '21LXK', 214: '21LXL', 215: '21LYD', 216: '21LYE', 217: '21LYF', 218: '21LYG', 219: '21LYH', 220: '21LYJ', 221: '21LYK', 222: '21LYL', 223: '21LZH', 224: '21LZJ', 225: '21LZK', 226: '21LZL', 227: '21MWM', 228: '21MXM', 229: '21MXN', 230: '21MXP', 231: '21MXQ', 232: '21MXR', 233: '21MXS', 234: '21MYM', 235: '21MYN', 236: '21MYP', 237: '21MYQ', 238: '21MYR', 239: '21MYS', 240: '21MY', 241: '21MYU', 242: '21MYV', 243: '21MZM', 244: '21MZN', 245: '21MZP', 246: '21MZQ', 247: '21MZR', 248: '21MZS', 249: '21MZ', 250: '21MZU', 251: '21MZV', 252: '21NYA', 253: '21NZA', 254: '21NZB', 255: '21NZC', 256: '21NZD', 257: '21NZE', 258: '21NZF', 259: '22LBR', 260: '22MBA', 261: '22MBB', 262: '22MBC', 263: '22MBD', 264: '22MBE', 265: '22MBS', 266: '22MB', 267: '22MBU', 268: '22MBV', 269: '22MCA', 270: '22MCB', 271: '22MCC', 272: '22MCD', 273: '22MCE', 274: '22MCV', 275: '22MDE', 276: '22NBF', 277: '22NBG', 278: '22NBH', 279: '22NBJ', 280: '22NBK', 281: '22NBL', 282: '22NBM', 283: '22NCF', 284: '22NCG', 285: '22NCH', 286: '22NCJ', 287: '22NCK', 288: '22NCL', 289: '22NCM', 290: '22NDF', 291: '22NDG', 292: '22NDH', 293: '22NDJ', 294: '22NDK', 295: '22NDL', 296: '22NDM', 297: '22NEJ', 298: '22NEK', 299: '22NEL',
          300: '19LCF', 301:  '19LDF', 302: '19LDG', 303: '19LDH', 304: '19LDJ', 305: '19LDK', 306: '19LDL', 307: '19LEF', 308: '19LEG', 309: '19LEH', 310: '19LEJ', 311: '19LEK', 312: '19LEL', 313: '19LFE', 314: '19LFF', 315: '19LFG', 316: '19LFH', 317: '19LFJ', 318: '19LFK', 319: '19LFL', 320: '19LGF', 321: '19LGG', 322: '19LGH', 323: '19LGJ', 324: '19LGK', 325: '19LGL', 326: '19LHL', 327: '19MEM', 328: '19MEN', 329: '19MEP', 330: '19MEQ', 331: '19MFM', 332: '19MFN', 333: '19MFP', 334: '19MFQ', 335: '19MFR', 336: '19MFS', 337: '19MF', 338: '19MFU', 339: '19MFV', 340: '19MGM', 341: '19MGN', 342: '19MGP', 343: '19MGQ', 344: '19MGR', 345: '19MGS', 346: '19MG', 347: '19MGU', 348: '19MGV', 349: '19MHM', 350: '19MHN', 351: '19MHP', 352: '19MHQ', 353: '19MHR', 354: '19MHS', 355: '19MH', 356: '19MHU', 357: '19MHV', 358: '19NGA', 359: '19NGB', 360: '19NGC', 361: '19NGD', 362: '19NHA', 363: '19NHB', 364: '19NHC', 365: '19NHD', 366: '19NHE', 367: '19NHF', 368: '19NHG', 369: '19NHH', 370: '20MKA', 371: '20MKB', 372: '20MKC', 373: '20MKD', 374: '20MKE', 375: '20MKU', 376: '20MKV', 377: '20MLC', 378: '20MLD', 379: '20MLE', 380: '20NKF', 381: '20NKG', 382: '20NKH', 383: '20NKJ', 384: '20NKK', 385: '20NKL', 386: '20NKM', 387: '20NKN', 388: '20NLF', 389: '20NLG', 390: '20NLH', 391: '20NLJ', 392: '20NLK', 393: '20NLL', 394: '20NLM', 395: '20NLN', 396: '20NMH', 397: '20NMJ', 398: '20NMK', 399: '20NML',
          400: '20NMM', 401:  '20NMN', 402: '20NNM', 403: '20NNN', 404: '20NNP', 405: '24LYP', 406: '24LYQ', 407: '24LYR', 408: '24LZP', 409: '24LZQ', 410: '24LZR', 411: '24MZS', 412: '24MZ', 413: '24MZU', 414: '24MZV', 415: '25LBJ', 416: '25LBK', 417: '25LBL', 418: '25LCH', 419: '25LCJ', 420: '25LCK', 421: '25LCL', 422: '25LDJ', 423: '25LDK', 424: '25LDL', 425: '25MBM', 426: '25MBN', 427: '25MBP', 428: '25MBQ', 429: '25MCM', 430: '25MCN', 431: '25MCP', 432: '25MCQ', 433: '25MDM', 434: '25MDN', 435: '25MDP', 436: '25MDQ', 437: '25MEP', 438: '25MEQ', 439: '24LYM', 440: '24LYN', 441: '24LZL', 442: '24LZM', 443: '24LZN', 444: '25LBF', 445: '25LBG', 446: '25LBH', 447: '25LCF', 448: '25LCG', 449: '21HWA', 450: '21HWV', 451: '21HXA', 452: '21HXB', 453: '21HXC', 454: '21HXD', 455: '21HXE', 456: '21HXV', 457: '21HYA', 458: '21HYB', 459: '21HYC', 460: '21HYD', 461: '21HYE', 462: '21HYV', 463: '21JYF', 464: '21JYG', 465: '21JYH', 466: '21JZM', 467: '22HBE', 468: '22HBF', 469: '22HBG', 470: '22HBH', 471: '22HBJ', 472: '22HBK', 473: '22HCE', 474: '22HCF', 475: '22HCG', 476: '22HCH', 477: '22HCJ', 478: '22HCK', 479: '22HDH', 480: '22HDJ', 481: '22HDK', 482: '22JBL', 483: '22JBM', 484: '22JBN', 485: '22JBP', 486: '22JBQ', 487: '22JBR', 488: '22JBS', 489: '22JCL', 490: '22JCM', 491: '22JCN', 492: '22JCP', 493: '22JCQ', 494: '22JCR', 495: '22JCS', 496: '22JC', 497: '22JDL', 498: '22JDM', 499: '22JDN', 
          500: '22JDP', 501:  '22JDQ', 502: '22JDR', 503: '22JDS', 504: '22JD', 505: '22JEN', 506: '22JEP', 507: '22JEQ', 508: '22JER', 509: '22JES', 510: '22JE', 511: '22JFS', 512: '22JF', 513: '22KCA', 514: '22KCU', 515: '22KCV', 516: '22KDA', 517: '22KDB', 518: '22KDC', 519: '22KDD', 520: '22KDE', 521: '22KDF', 522: '22KDU', 523: '22KDV', 524: '22KEA', 525: '22KEB', 526: '22KEC', 527: '22KED', 528: '22KEE', 529: '22KEF', 530: '22KEG', 531: '22KEU', 532: '22KEV', 533: '22KFA', 534: '22KFB', 535: '22KFC', 536: '22KFD', 537: '22KFE', 538: '22KFF', 539: '22KFG', 540: '22KFU', 541: '22KFV', 542: '22KGA', 543: '22KGB', 544: '22KGC', 545: '22KGD', 546: '22KGE', 547: '22KGF', 548: '22KGG', 549: '22KHF', 550: '22KHG', 551: '22LEH', 552: '22LEJ', 553: '22LEK', 554: '22LFH', 555: '22LFJ', 556: '22LFK', 557: '22LFL', 558: '22LFM', 559: '22LFN', 560: '22LFP', 561: '22LGH', 562: '22LGJ', 563: '22LGK', 564: '22LGL', 565: '22LGM', 566: '22LGN', 567: '22LGP', 568: '22LGQ', 569: '22LGR', 570: '22LHH', 571: '22LHJ', 572: '22LHK', 573: '22LHL', 574: '22LHM', 575: '22LHN', 576: '22LHP', 577: '22LHQ', 578: '22LHR', 579: '22MGS', 580: '22MG', 581: '22MGU', 582: '22MHA', 583: '22MHB', 584: '22MHC', 585: '22MHS', 586: '22MH', 587: '22MHU', 588: '22MHV', 589: '23KKB', 590: '23LKC', 591: '23LKD', 592: '23LKE', 593: '23LKF', 594: '23LKG', 595: '23LKH', 596: '23LKJ', 597: '23LKK', 598: '23LKL', 599: '23LLG', 
          600: '23LLH', 601:  '23LLJ', 602: '23LLK', 603: '23LLL', 604: '23MKM', 605: '23MKN', 606: '23MKP', 607: '23MKQ', 608: '23MKR', 609: '23MKS', 610: '23MK', 611: '23MKU', 612: '23MKV', 613: '23MLM', 614: '23MLN', 615: '23MLP', 616: '23MLQ', 617: '23MLR', 618: '23MLS', 619: '23ML', 620: '23MLU', 621: '23MLV', 622: '23MMM', 623: '23MMN', 624: '23MMP', 625: '23MMQ', 626: '23MMR', 627: '23MMS', 628: '23MM', 629: '23MMU', 630: '23MMV', 631: '23MNR', 632: '23MNS', 633: '23MN', 634: '23MNU', 635: '23MNV', 636: '20KMG', 637: '20KNG', 638: '20KPG', 639: '20KQG', 640: '20LMH', 641: '20LNH', 642: '20LNJ', 643: '20LNK', 644: '20LNL', 645: '20LNM', 646: '20LNN', 647: '20LPH', 648: '20LPJ', 649: '20LPK', 650: '20LPL', 651: '20LPM', 652: '20LPN', 653: '20LPP', 654: '20LPQ', 655: '20LPR', 656: '20LQH', 657: '20LQJ', 658: '20LQK', 659: '20LQL', 660: '20LQM', 661: '20LQN', 662: '20LQP', 663: '20LQQ', 664: '20LQR', 665: '20LRH', 666: '20LRJ', 667: '20LRK', 668: '20LRL', 669: '20LRM', 670: '20LRN', 671: '20LRP', 672: '20LRQ', 673: '20LRR', 674: '20MPS', 675: '20MQA', 676: '20MQS', 677: '20MQ', 678: '20MQU', 679: '20MQV', 680: '20MRA', 681: '20MRB', 682: '20MRC', 683: '20MRD', 684: '20MRE', 685: '20MRS', 686: '20MR', 687: '20MRU', 688: '20MRV', 689: '20NRF', 690: '21L', 691: '21LUK', 692: '21LUL', 693: '21M', 694: '21MUM', 695: '21MUN', 696: '21MUP', 697: '21MUQ', 698: '21MUR', 699: '21MUS', 
          700: '21MU',  701:  '21MUU', 702: '21MUV', 703: '21MVQ', 704: '21MVR', 705: '21MVS', 706: '21MV', 707: '21MVU', 708: '21MVV', 709: '21MWU', 710: '21MWV', 711: '21N', 712: '21NUA', 713: '21NUB', 714: '21NUC', 715: '21NUD', 716: '21NUE', 717: '21NUF', 718: '21NUG', 719: '21NUH', 720: '21NUJ', 721: '21NVA', 722: '21NVB', 723: '21NVC', 724: '21NVD', 725: '21NVE', 726: '21NVF', 727: '21NVG', 728: '21NVH', 729: '21NVJ', 730: '21NWA', 731: '21NWB', 732: '21NWC', 733: '21NWD', 734: '21NWE', 735: '21NWF', 736: '21NWG', 737: '21NWH', 738: '21NWJ', 739: '21NXD', 740: '21NXE', 741: '21NXF', 742: '21NXG', 743: '21NXH', 744: '21NXJ', 745: '21NYJ', 746: '21PVK', 747: '21PWK', 748: '21PXK', 749: '21PYK', 750: '20KMD', 751: '20KME', 752: '20KMF', 753: '20KNC', 754: '20KND', 755: '20KNE', 756: '20KNF', 757: '20KPB', 758: '20KPC', 759: '20KPD', 760: '20KPE', 761: '20KPF', 762: '20KQD', 763: '20KQE', 764: '20KQF', 765: '18LWN', 766: '18LWP', 767: '18LWQ', 768: '18LWR', 769: '18LXM', 770: '18LXN', 771: '18LXP', 772: '18LXQ', 773: '18LXR', 774: '18LYM', 775: '18LYN', 776: '18LYP', 777: '18LYQ', 778: '18LYR', 779: '18LZM', 780: '18LZN', 781: '18LZP', 782: '18LZQ', 783: '18LZR', 784: '18MXS', 785: '18MX', 786: '18MXU', 787: '18MXV', 788: '18MYA', 789: '18MYB', 790: '18MYC', 791: '18MYD', 792: '18MYS', 793: '18MY', 794: '18MYU', 795: '18MYV', 796: '18MZA', 797: '18MZB', 798: '18MZC', 799: '18MZD', 
          800: '18MZE', 801:  '18MZS', 802: '18MZ', 803: '18MZU', 804: '18MZV', 805: '18NZF', 806: '18NZG', 807: '18NZH', 808: '19LBJ', 809: '19LBK', 810: '19LBL', 811: '19MBM', 812: '19MBN', 813: '19MBP', 814: '19MBQ', 815: '19MBR', 816: '19MBS', 817: '19MB', 818: '19MBU', 819: '19MBV', 820: '19MCN', 821: '19MCP', 822: '19MCQ', 823: '19MCR', 824: '19MCS', 825: '19MC', 826: '19MCU', 827: '19MCV', 828: '19MD', 829: '19MDU', 830: '19MDV', 831: '19NBA', 832: '19NBB', 833: '19NBC', 834: '19NBD', 835: '19NCA', 836: '19NCB', 837: '19NCC', 838: '19NCD', 839: '19NCE', 840: '19NDA', 841: '19NDB', 842: '19NDC', 843: '19NDD', 844: '19NDE', 845: '19NEB', 846: '19NEC', 847: '19NED', 848: '19NEE', 849: '19NEF', 850: '24MVA', 851: '24MVV', 852: '24MWA', 853: '24MWB', 854: '24MWC', 855: '24MWV', 856: '24MXA', 857: '24MXB', 858: '24MXV', 859: '23JNL', 860: '23JPL', 861: '23JPM', 862: '23JPN', 863: '23JQL', 864: '23JQM', 865: '23JQN', 866: '23KPP', 867: '23KPQ', 868: '23KQP', 869: '23KQQ', 870: '23KQR', 871: '23KQS', 872: '23KQ', 873: '23KQU', 874: '23KQV', 875: '23KRA', 876: '23KRB', 877: '23KRP', 878: '23KRQ', 879: '23KRR', 880: '23KRS', 881: '23KR', 882: '23KRU', 883: '23KRV', 884: '23LRC', 885: '23LRD', 886: '24K', 887: '24KUD', 888: '24KUE', 889: '24KUF', 890: '24KUG', 891: '24LUH', 892: '24LUJ', 893: '24LUK', 894: '24LUL', 895: '24LUM', 896: '24LUN', 897: '24LUP', 898: '24LUQ', 899: '24LUR', 
          900: '24LVJ', 901:  '24LVK', 902: '24LVL', 903: '24LVM', 904: '24LVN', 905: '24LVP', 906: '24LVQ', 907: '24LVR', 908: '24LWP', 909: '24LWQ', 910: '24LWR', 911: '24MVS', 912: '24MV', 913: '24MVU', 914: '24MWS', 915: '24MW', 916: '24MWU', 917: '24MX', 918: '24MXU', 919: '21HUC', 920: '21HUD', 921: '21HVB', 922: '21HVC', 923: '21HVD', 924: '21HVE', 925: '21HWC', 926: '21HWD', 927: '21HWE', 928: '21JVF', 929: '21JVG', 930: '21JVH', 931: '21JVJ', 932: '21JWF', 933: '21JWG', 934: '21JWH', 935: '21JWJ', 936: '21JWK', 937: '21JWL', 938: '21JWM', 939: '21JWN', 940: '21JXH', 941: '21JXJ', 942: '21JXK', 943: '21JXL', 944: '21JXM', 945: '21JXN', 946: '21JYM', 947: '21JYN', 948: '21KWP', 949: '21KXP', 950: '21KXQ', 951: '21KXR', 952: '21KXS', 953: '21KX', 954: '21KYA', 955: '21KYB', 956: '21KYP', 957: '21KYQ', 958: '21KYR', 959: '21KYS', 960: '21KY', 961: '21KYU', 962: '21KYV', 963: '21KZA', 964: '21KZB', 965: '21KZR', 966: '21KZS', 967: '21KZ', 968: '21KZU', 969: '21KZV', 970: '21LYC', 971: '21LZC', 972: '21LZD', 973: '21LZE', 974: '21LZF', 975: '21LZG', 976: '22KBB', 977: '22KBC', 978: '22KBD', 979: '22KBE', 980: '22KBF', 981: '22KBG', 982: '22KCG', 983: '22LBH', 984: '22LBJ', 985: '22LBK', 986: '22LBL', 987: '22LBM', 988: '22LBN', 989: '22LBP', 990: '22LBQ', 991: '22LCH', 992: '22LCJ', 993: '22LCK', 994: '22LCL', 995: '22LCM', 996: '22LCN', 997: '22LCP', 998: '22LCQ', 999: '22LCR',
         1000: '22LDM', 1001: '22LDN',1002: '22LDP',1003: '22LDQ',1004: '22LDR',1005: '22MCS',1006: '22MC',1007: '22MCU',1008: '22MDA',1009: '22MDB',1010: '22MDC',1011: '22MDD',1012: '22MDS',1013: '22MD',1014: '22MDU',1015: '22MDV',1016: '22MEA',1017: '22MEB',1018: '22MEC',1019: '22MED',1020: '22MEE',1021: '22MES',1022: '22ME',1023: '22MEU',1024: '22MEV',1025: '22MFA',1026: '22MFB',1027: '22MFC',1028: '22MFD',1029: '22MFE',1030: '22NEF',1031: '22NEG',1032: '22NEH',1033: '22NFF',1034: '22NFG',1035: '22NFH',1036: '22NFJ',1037: '22NFK',1038: '22NGF',1039: '22NGG',1040: '22NGH',1041: '22NGJ',1042: '22NGK',1043: '19LHG',1044: '19LHH',1045: '19LHJ',1046: '19LHK',1047: '20LKM',1048: '20LKN',1049: '20LKP',1050: '20LKQ',1051: '20LKR',1052: '20LLL',1053: '20LLM',1054: '20LLN',1055: '20LLP',1056: '20LLQ',1057: '20LLR',1058: '20LMQ',1059: '20LMR',1060: '20MKS',1061: '20MK',1062: '20MLA',1063: '20MLB',1064: '20MLS',1065: '20ML',1066: '20MLU',1067: '20MLV',1068: '20MMA',1069: '20MMB',1070: '20MMC',1071: '20MMD',1072: '20MME',1073: '20MMS',1074: '20MM',1075: '20MMU',1076: '20MMV',1077: '20MNA',1078: '20MNB',1079: '20MNC',1080: '20MND',1081: '20MNE',1082: '20MNV',1083: '20MPD',1084: '20MPE',1085: '20NMF',1086: '20NMG',1087: '20NNF',1088: '20NNG',1089: '20NNH',1090: '20NNJ',1091: '20NNK',1092: '20NNL',1093: '20NPF',1094: '20NPG',1095: '20NPH',1096: '20NPJ',1097: '20NPK',1098: '20NPL',1099: '20NPM',
         1100: '20NPN', 1101: '20NPP',1102: '20NQJ',1103: '20NQK',1104: '20NQL',1105: '20NQM',1106: '20NQN',1107: '20NQP',1108: '20NRN',1109: '20NRP',1110: '19KHB',1111: '19LGC',1112: '19LGD',1113: '19LGE',1114: '19LHC',1115: '19LHD',1116: '19LHE',1117: '19LHF',1118: '20KKG',1119: '20LKH',1120: '20LKJ',1121: '20LKK',1122: '20LKL',1123: '20LLK',1124: '18LUQ',1125: '18LUR',1126: '18MUS',1127: '18MU',1128: '18MUU',1129: '18MUV',1130: '18MVA',1131: '18MVB',1132: '18MV',1133: '18MVU',1134: '18MVV',1135: '22HEG',1136: '22HEH',1137: '22HEJ',1138: '22HEK',1139: '22HFG',1140: '22HFH',1141: '22HFJ',1142: '22HFK',1143: '22JEL',1144: '22JEM',1145: '22JFL',1146: '22JFM',1147: '22JFP',1148: '22JFQ',1149: '22JFR',1150: '22JGL',1151: '22JGM',1152: '22JG',1153: '22KGU',1154: '22KGV',1155: '22KHB',1156: '22KHC',1157: '22KHD',1158: '22KHE',1159: '23KKA',1160: '23KKS',1161: '23KK',1162: '23KKU',1163: '23KKV',1164: '23KLB',1165: '23LLC',1166: '23LLD',1167: '23LLE',1168: '23LLF',1169: '23LMF',1170: '23LMG',1171: '23LMH',1172: '23LMJ',1173: '23LMK',1174: '23LML',1175: '23LNL',1176: '23MNM',1177: '23MNN',1178: '23MNP',1179: '23MNQ',1180: '23MPQ',1181: '23MPR',1182: '23MPS',1183: '23MP',1184: '23MPU',1185: '23MQU',1186: '23MRU',1187: '20KPA',1188: '20KQA',1189: '20KQB',1190: '20KQC',1191: '20KQV',1192: '20KRD',1193: '20KRE',1194: '20KRF',1195: '20KRG',1196: '21LUE',1197: '21LUF',1198: '21LUG',1199: '21LUH',
         1200: '21LUJ', 1201: '21LVJ',1202: '21LVK',1203: '21LVL',1204: '21MVM',1205: '21MVN',1206: '21MVP',1207: '21MWN',1208: '21MWP',1209: '21MWQ',1210: '21MWR',1211: '21MWS',1212: '21MW',1213: '21MX',1214: '21MXU',1215: '21MXV',1216: '21NXA',1217: '21NXB',1218: '21NXC',1219: '21NYB',1220: '21NYC',1221: '21NYD',1222: '21NYE',1223: '21NYF',1224: '21NYG',1225: '21NYH',1226: '21NZG',1227: '19LBG',1228: '19LBH',1229: '19LCG',1230: '19LCH',1231: '19LCJ',1232: '19LCK',1233: '19LCL',1234: '19MCM',1235: '19MDM',1236: '19MDN',1237: '19MDP',1238: '19MDQ',1239: '19MDR',1240: '19MDS',1241: '19MER',1242: '19MES',1243: '19ME',1244: '19MEU',1245: '19MEV',1246: '19NEA',1247: '19NFA',1248: '19NFB',1249: '19NFC',1250: '19NFD',1251: '19NFE',1252: '19NFF',1253: '19NGE',1254: '19NGF',1255: '19NGG',1256: '24KUA',1257: '24KUB',1258: '24KUC',1259: '24KUU',1260: '24KUV',1261: '24KVA',1262: '24KVB',1263: '24KVC',1264: '24KVD',1265: '24KVE',1266: '24KVF',1267: '24KVG',1268: '24KVU',1269: '24KVV',1270: '24KWA',1271: '24KWB',1272: '24KWC',1273: '24KWD',1274: '24KWE',1275: '24KWF',1276: '24KWG',1277: '24KXF',1278: '24KXG',1279: '24LVH',1280: '24LWH',1281: '24LWJ',1282: '24LWK',1283: '24LWL',1284: '24LWM',1285: '24LWN',1286: '24LXH',1287: '24LXJ',1288: '24LXK',1289: '24LXL',1290: '24LXM',1291: '24LXN',1292: '24LXP',1293: '24LXQ',1294: '24LXR',1295: '24LYK',1296: '24LYL',1297: '24MXS',1298: '24MYA',1299: '24MYS',
         1300: '24MY',  1301: '24MYU',1302: '24MYV',1303: '24MZA',1304: '25MBR',1305: '21HWB',1306: '21JXF',1307: '21JXG',1308: '21JYJ',1309: '21JYK',1310: '21JYL',1311: '21JZN',1312: '21KZP',1313: '21KZQ',1314: '22JB',1315: '22KBA',1316: '22KBU',1317: '22KBV',1318: '22KCB',1319: '22KCC',1320: '22KCD',1321: '22KCE',1322: '22KCF',1323: '22KDG',1324: '22LDH',1325: '22LDJ',1326: '22LDK',1327: '22LDL',1328: '22LEL',1329: '22LEM',1330: '22LEN',1331: '22LEP',1332: '22LEQ',1333: '22LER',1334: '22LFQ',1335: '22LFR',1336: '22MFS',1337: '22MF',1338: '22MFU',1339: '22MFV',1340: '22MGA',1341: '22MGB',1342: '22MGC',1343: '22MGD',1344: '22MGE',1345: '22MGV',1346: '22MHD',1347: '22MHE',1348: '22NHF',1349: '23NKA',1350: '23NLA',1351: '20KLF',1352: '20KLG',1353: '20LLH',1354: '20LLJ',1355: '20LMJ',1356: '20LMK',1357: '20LML',1358: '20LMM',1359: '20LMN',1360: '20LMP',1361: '20LNP',1362: '20LNQ',1363: '20LNR',1364: '20MNS',1365: '20MN',1366: '20MNU',1367: '20MPA',1368: '20MPB',1369: '20MPC',1370: '20MP',1371: '20MPU',1372: '20MPV',1373: '20MQB',1374: '20MQC',1375: '20MQD',1376: '20MQE',1377: '20NQF',1378: '20NQG',1379: '20NQH',1380: '20NRG',1381: '20NRH',1382: '20NRJ',1383: '20NRK',1384: '20NRL',1385: '20NRM',1386: '21P',1387: '21PUK',1388: '20KLE',1389: '18LUN',1390: '18LUP',1391: '18LVN',1392: '18LVP',1393: '18LVQ',1394: '18LVR',1395: '18MVC',1396: '18MVS',1397: '18MWA',1398: '18MWB',1399: '18MWC',
         1400: '18MWD', 1401: '18MWS',1402: '18MW',1403: '18MWU',1404: '18MWV',1405: '18MXA',1406: '18MXB',1407: '18MXC',1408: '18MXD',1409: '18MXE',1410: '18MYE',1411: '18NXF',1412: '18NYF',1413: '18NYG',1414: '27MYN',1415: '27MZN',1416: '27MZP',1417: '27MZQ',1418: '27MZR',1419: '28MBA',1420: '28MBU',1421: '28MBV',1422: '22NEM',1423: '25MEN',1424: '21PYL',1425: '23MRV',1426: '23NRA',1427: '23NRB',1428: '23NRC',1429: '24MUD',1430: '24MUE',1431: '24MVD',1432: '24MVE',1433: '24N',1434: '24NUF',1435: '24NUG',1436: '24NUH',1437: '24NUJ',1438: '24NUK',1439: '24NUL',1440: '24NVF',1441: '24NVG',1442: '24NVH',1443: '24NVJ',1444: '24NVK',1445: '24NVL',1446: '24NWF',1447: '24NWG',1448: '24NWH',1449: '24NWJ',1450: '24NWK',1451: '24NWL',1452: '23JKG',1453: '22NBN',1454: '22NBP',1455: '22NCN',1456: '22NCP',1457: '22NDN',1458: '22NDP',1459: '22NEN',1460: '22NEP',1461: '23NLB',1462: '23NLC',1463: '23NLD',1464: '23NLE',1465: '23NLF',1466: '23NMA',1467: '23NMB',1468: '23NMC',1469: '23NMD',1470: '23NME',1471: '23NMF',1472: '23NMG',1473: '23NNA',1474: '23NNB',1475: '23NNC',1476: '23NND',1477: '23NNE',1478: '23NNF',1479: '23NNG',1480: '23NPA',1481: '23NPB',1482: '23NPC',1483: '23NPD',1484: '23NPE',1485: '23NPF',1486: '23NPG',1487: '23NQF',1488: '23NQG',1489: '22NFL',1490: '22NFM',1491: '22NFN',1492: '22NFP',1493: '22NGL',1494: '22NGM',1495: '22NGN',1496: '22NGP',1497: '22NHK',1498: '22NHL',1499: '22NHM',
         1500: '22NHN', 1501: '23NKH',1502: '23MPV',1503: '23MQV',1504: '23NQA',1505: '23NQB',1506: '23NQC',1507: '23NQD',1508: '23NQE',1509: '23NRD',1510: '23NRE',1511: '23NRF',1512: '23NRG',1513: '22NHG',1514: '22NHH',1515: '22NHJ',1516: '23NKB',1517: '23NKC',1518: '23NKD',1519: '23NKE',1520: '23NKF',1521: '23NKG',1522: '23NLG',1523: '23NLH',1524: '23NMH',
    };

    image.reduceRegion({
      reducer:ee.Reducer.sum(),
      geometry:point,
      scale:10,
      // crs:,
      // crsTransform:,
      // bestEffort:,
      maxPixels:1e11,
      // tileScale:
      }).evaluate(function(reduce){
                  
    //  20190102T140049_20190102T140050_T21KUT
    // example id:  20170328T083601_20170328T084228_T35RNK
    // split                T               T      _T           igual em todas as imagens
    // split        20170328                                    propertie_0     -   8 caracteres
    // split                 083601                             propertie_1     -   6 caracteres
    // split                        20170328                    propertie_2     -   8 caracteres
    // split                                 084228             propertie_3     -   6 caracteres
    // split                                         35RNK      propertie_4     -   convertido para numerico, ocupando 4 caracteres

    var sentinelName = reduce.propertie_0
        + 'T'
        + reduce.propertie_1
        + '_'
        + reduce.propertie_2
        + 'T'
        + reduce.propertie_3
        + '_T'
        + number_to_code[reduce.propertie_4];
  
    print('O pixel selecionado pertence a imagem:"'+ sentinelName + '"');
  });
}


function corrections_LS8_col2 (image){

  // - radiometric correction
  var opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2);
  // rectfy to dark corpse reflectance == -0.0000000001
  opticalBands = opticalBands.multiply(10000).subtract(0.0000275 * 0.2 * 1e5 * 100).round();
  
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  
  // - return 
  image = image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
    
  // - masks
  // If the cloud bit (3) is set and the cloud confidence (9) is high
  // or the cloud shadow bit is set (3), then it's a bad pixel.
  var qa = image.select('QA_PIXEL');
      var cloud = qa.bitwiseAnd(1 << 3)
      .and(qa.bitwiseAnd(1 << 9))
      .or(qa.bitwiseAnd(1 << 4));
  
  // If the clear bit (6) is set 
  // or water bit is set (7), then it's a good pixel 
  var good_pixel  = qa.bitwiseAnd(1 << 6)
      .or(qa.bitwiseAnd(1 << 7));

  // read radsat 
  var radsatQA = image.select('QA_RADSAT');
  // Is any band saturated? 
  var saturated = radsatQA.bitwiseAnd(1 << 0)
    .or(radsatQA.bitwiseAnd(1 << 1))
      .or(radsatQA.bitwiseAnd(1 << 2))
        .or(radsatQA.bitwiseAnd(1 << 3))
          .or(radsatQA.bitwiseAnd(1 << 4))
            .or(radsatQA.bitwiseAnd(1 << 5))
              .or(radsatQA.bitwiseAnd(1 << 6));

  // is any band with negative reflectance? 
  var negative_mask = image.select(['SR_B1']).gt(0).and(
    image.select(['SR_B2']).gt(0)).and(
      image.select(['SR_B3']).gt(0)).and(
        image.select(['SR_B4']).gt(0)).and(
          image.select(['SR_B5']).gt(0)).and(
            image.select(['SR_B7']).gt(0));

  
  // -return 
  image = image
  .updateMask(cloud.not())
  .updateMask(good_pixel)
  .updateMask(saturated.not())
  .updateMask(negative_mask);
  
  
  // correction bandnames to default
  var oldBands = ['SR_B2','SR_B3','SR_B4','SR_B5','SR_B6','SR_B7',];
  var newBands = ['blue', 'green','red',  'nir',  'swir1','swir2'];
  // - return
  image = image.select(oldBands,newBands);

  // -
  return timeFlag_landsat(image);
}

function corrections_sentinel (image){
  // return- 
  image = image // --- funtion maskEdge ref ->  https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY 
    .updateMask(image.select('B8A').mask()
        .updateMask(image.select('B9').mask()));
        
  // --- funtion maskClouds ref ->  https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY 
  var max_cloud_probability = 65;
  var clouds = ee.Image(image.get('cloud_mask')).select('probability');
  var isNotCloud = clouds.lt(max_cloud_probability);
  // - return
  image = image.updateMask(isNotCloud);
  
  // var oldBands = ['QA60', 'B1', 'B2',   'B3',    'B4',  'B5',  'B6',  'B7', 'B8'  ,  'B8A',  'B9',           'B11',   'B12',  'B12'];
  // var newBands = ['QA60', 'cb', 'blue', 'green', 'red', 'red1','red2','red3','nir'  ,'nir2', 'waterVapor',   'swir1', 'swir2','cloudShadowMask']
  var oldBands = ['B2',   'B3',    'B4',   'B8',  'B11',    'B12'];
  var newBands = ['blue', 'green', 'red',  'nir', 'swir1',  'swir2'];
  // -return
  image = image.select(oldBands,newBands);

  // -
  return timeFlag_sentinel(image);
}

function corrections_modis (image){
  var oldBands = ['sur_refl_b01','sur_refl_b02','sur_refl_b03','sur_refl_b04','sur_refl_b05','sur_refl_b06','sur_refl_b07',];
  var newBands = ['red',         'nir',         'blue',        'green',       'nir2',        'swir1',       'swir2',       ];

  return image
  
  .select(oldBands,newBands);
}

function addBand_NBR (image){
  var exp = '( b("nir") - b("swir2") ) / ( b("nir") + b("swir2") )';
  var minimoNBR = image
    .expression(exp)
    .add(1)
    .multiply(1000)
    .multiply(-1)
    .int16()
    .rename("nbr");
  return image
    .addBands(minimoNBR);
}

// visParams = {
// };


options.satellites = [
  {
    'type':'ImageCollection',
    'id':'LANDSAT/LT05/C02/T1_L2',
    'name':'landsat5_C02',
    'time_start':'1984-03-16',
    'time_end':'2012-05-05',
    'years':[1984,1985,1986,1987,1988,1989,1990,1991,1992,1993,1994,1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011],
    allProcess:function(col){
      col = ee.ImageCollection(col).filter(ee.Filter.inList('system:index', blockList_landsat).not());
      return ee.ImageCollection(col).map(function(image){
        
      image = clipBoard_Landsat (image);
      image = corrections_LS57_col2(image);
      image = addBand_NBR(image);
      return image;
    });
    },
    reduceProcess:function(col){
      return col.qualityMosaic('nbr');
    },
    'vis':{
        min:0.03,
        max:0.4,
        bands:['swir1','nir','red'],
    },
  },
  {
    'type':'ImageCollection',
    'id':'LANDSAT/LE07/C02/T1_L2',
    'name':'landsat7_C02',
    'time_start':'1999-05-28',
    'time_end':'2022-04-06',
    'years':[1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    
    allProcess:function(col){
      col = ee.ImageCollection(col).filter(ee.Filter.inList('system:index', blockList_landsat).not());
      return col.map(function(image){
        
      image = clipBoard_Landsat (image);
      image = corrections_LS57_col2(image);
      image = addBand_NBR(image);
      return image;
    });
    },
    reduceProcess:function(col){
      return col.qualityMosaic('nbr');
    },
    'vis':{
      min:0.03,
      max:0.4,
      bands:['swir1','nir','red'],
    },
  },
  {
    'type':'ImageCollection',
    'id':'LANDSAT/LC08/C02/T1_L2',
    'name':'landsat8_col2',
    'time_start':'2013-03-18',
    'time_end':'-1m', //
    'years':[2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025],
    allProcess:function(col){
      col = ee.ImageCollection(col).filter(ee.Filter.inList('system:index', blockList_landsat).not());
      return col.map(function(image){
        
      image = clipBoard_Landsat (image);
      image = corrections_LS8_col2(image);
      image = addBand_NBR(image);
      return image;
    });
    },
    reduceProcess:function(col){
      return col.qualityMosaic('nbr');
    },
    'vis':{
        min:300,
        max:4000,
        bands:['swir1','nir','red'],
      }
  }, 
  {
    'type':'ImageCollection',
    'id':'LANDSAT/LC09/C02/T1_L2',
    'name':'landsat9_col2',
    'time_start':'2021-10-31',
    'time_end':'-1m', //
    'years':[2021,2022,2023,2024,2025],
    allProcess:function(col){
      col = ee.ImageCollection(col).filter(ee.Filter.inList('system:index', blockList_landsat).not());
      return col.map(function(image){
        
      image = clipBoard_Landsat (image);
      image = corrections_LS8_col2(image);
      image = addBand_NBR(image);
      return image;
    });
    },
    reduceProcess:function(col){
      return col.qualityMosaic('nbr');
    },
    'vis':{
        min:300,
        max:4000,
        bands:['swir1','nir','red'],
      }
  }, 
  {
    'type':'ImageCollection',
    'id':'COPERNICUS/S2_SR_HARMONIZED',
    'name':'Sentinel2',
    'time_start':'2017-03-28',
    'time_end':'-1m',
    'years':[2017,2018,2019,2020,2021,2022,2023,2024,2025],
    allProcess:function(col){
      col = ee.ImageCollection(ee.Join.saveFirst('cloud_mask').apply({
          primary: ee.ImageCollection(col),
          secondary: ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY"),
          condition:ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
      })).filter(ee.Filter.inList('system:index', blockList_sentinel).not());
      
      return col.map(function(image){
        
      // image = clipBoard_Landsat (image);
      image = corrections_sentinel(image);
      image = addBand_NBR(image);
      return image;
    });
    },
    reduceProcess:function(col){
      return col.qualityMosaic('nbr');
    },
    'vis':{
        min:300,
        max:4000,
        bands:['swir1','nir','red'],
      }
  },
  {
    'type':'ImageCollection',
    'id':'MODIS/006/MOD09A1',
    'name':'MODIS_TERRA',
    'time_start':'2000-03-05',
    'time_end':'-1m', 
    'years':[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    allProcess:function(col){
      return ee.ImageCollection(col).map(function(image){
        image = corrections_modis(image);
        image = addBand_NBR(image);
        
        return image;
      });
    },
    reduceProcess:function(col){
      return col.qualityMosaic('nbr');
    },
    'vis':{
        min:300,
        max:4000,
        bands:['swir1','nir','red'],
      }
  },
  {
    'type':'ImageCollection',
    'id':'MODIS/006/MYD09A1',
    'name':'MODIS_AQUA',
    'time_start':'2002-07-04',
    'time_end':'-1m',
    'years':[2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023],
    allProcess:function(col){
      return ee.ImageCollection(col).map(function(image){
        image = corrections_modis(image);
        image = addBand_NBR(image);
        return image;
      });
    },
    reduceProcess:function(col){
      return col.qualityMosaic('nbr');
    },
    'vis':{
        min:300,
        max:4000,
        bands:['swir1','nir','red'],
      }
  },
];

options.fire_collections = [
    {
    'type':'ImageCollection',
    'id':'MODIS/006/MCD64A1',
    'name':'MCD64A1',
    'time_start':'2000-11-01',
    'time_end':'-4m', 
    allProcess:function(col){
      return col.select('BurnDate','Área').gte(1);
    },
    'vis':[
      {
        min:0,
        max:1,
      },
    ]
  },
    {
    'type':'ImageCollection',
    'id':'FIRMS',
    'name':'FIRMS',
    'time_start':'2000-11-01',
    'time_end':'-1m', 
    allProcess:function(col){
      return col.select('BurnDate','Área').gte(1);
    },
    'vis':[
      {
        min:0,
        max:1,
      },
    ]
  },

];

// interface
var styles = {
  'comum':{
    'margin':'0px 0px 0px 0px',
  },
  'control_panel':{
    'margin':'0px 0px 0px 0px',
    'position':'bottom-left'
  },
  'comum_strech':{
    'margin':'0px 0px 0px 0px',
    'stretch':'horizontal'
  },
  'select':{
    'margin':'0px 0px 0px 0px',
    'stretch':'both'
  },
  
  'logo':{
    'margin':'0px 0px 0px 0px',
    'width':'200px'
  },
  'logo_miniature':{
    'margin':'0px 0px 0px 0px',
    'width':'150px',
    'position':'bottom-left',
  },
  
};

function setLayout (){
  
  // base
  options.mapp = ui.root.widgets().get(0); // -> capturando a Map principal
  options.panel = ui.Panel({
    // widgets:,
    layout:ui.Panel.Layout.Flow('horizontal'),
    style:styles.comum
  });
  
  options.splitPanel = ui.SplitPanel({
    firstPanel:options.mapp,
    secondPanel:options.panel,
    orientation:'horizontal',
    wipe:false,
    style:styles.comum
  });
  
  ui.root.widgets().reset([options.splitPanel]);
  
  // legenda / painel de controle
  options.control_panel = ui.Panel({
    // widgets:,
    layout:ui.Panel.Layout.Flow('vertical'),
    style:styles.control_panel
  });

  options.mapp.add(options.control_panel);
  
    
  // panel control
  // cabeçalho -> logo marca
  var logomarca = ee.Image('projects/ee-wallacesilva/assets/logo-mapbiomasfogo-1_georeferenced');
  logomarca = logomarca.updateMask(logomarca.select(0).neq(0)) // comentar esta linha deixa a logo com fundo preto
  .visualize({});
  
  var option_box = ee.Image().paint(ee.FeatureCollection([ 
    ee.Geometry.Polygon(
            [[[0.29307103076535057, 3.725014863596575],
              [0.29307103076535057, 2.9792280610229556],
              [1.1609909526403506, 2.9792280610229556],
              [1.1609909526403506, 3.725014863596575]]], null, false)
  ]),'vazio',0.25).visualize({});
  
  var option_min = ee.Image().paint(ee.FeatureCollection([ 
    ee.Geometry.Polygon(
            [[[0.38096165576535057, 3.4399288093250107],
              [0.38096165576535057, 3.286385157544812],
              [1.0840866557653506, 3.286385157544812],
              [1.0840866557653506, 3.4399288093250107]]], null, false)
  ])).visualize({});
  
  option_min = option_box.blend(option_min);
  
  var option_max = ee.Image().paint(ee.FeatureCollection([ 
    ee.Geometry.Polygon(
            [[[0.39194798389035057, 3.6483631516114237],
              [0.39194798389035057, 3.0780603900662418],
              [1.0731003276403506, 3.0780603900662418],
              [1.0731003276403506, 3.6483631516114237]]], null, false)
  ]),'vazio',2).visualize({});
  
  option_max = option_box.blend(option_max);

  var logo = ui.Thumbnail({
    image:logomarca.blend(option_min),
    params:{
      dimensions:1000,
    },
    // onClick:,
    style:styles.logo
  });
  
  var logo_miniature = ui.Thumbnail({
    image:logomarca.blend(option_max),
    params:{
      dimensions:1000,
    },
    // onClick:,
    style:styles.logo_miniature
  });
  
  logo_miniature.onClick(function(){
      options.mapp.remove(logo_miniature);
      options.mapp.add(options.control_panel);
  });
  
  logo.onClick(function(){
      options.mapp.remove(options.control_panel);
      options.mapp.add(logo_miniature);
  });
  
  options.control_panel
    .insert(0,logo);
  
  // firstline -> escolher o painel
  var select_dinamic = ui.Select({
    items:[
      'ANUAL',
      'MENSAL',
    ],
    // placeholder:,
    value:options.rota,
    onChange:function(value){
      print(value);
    },
    // disabled:,
    style:styles.select
  });
  
  var slider_annual = ui.Slider({
    min:1985,
    max:2025,
    value:options.year,
    step:1,
    onChange:function(value){
      options.year = value;
      setSubtitle();
    },
    // direction:,
    // disabled:,
    style:styles.comum_strech,
  });
  var slider_month = ui.Slider({
    min:1,
    max:12,
    value:options.month,
    step:1,
    onChange:function(value){
      options.month = value;
      setSubtitle();
    },
    // direction:,
    disabled:options.filter_month,
    style:styles.comum_strech,
  });
  
  var head_panel = ui.Panel({
    widgets:[select_dinamic],
    layout:ui.Panel.Layout.Flow('horizontal'),
    style:styles.comum
  });

  var slider_panel = ui.Panel({
    widgets:[head_panel,slider_annual],
    layout:ui.Panel.Layout.Flow('vertical'),
    style:styles.comum
  });
  
  
  // ->
  var SWITCH ={
    'ANUAL':function(){
        slider_panel.remove(slider_month);
        options.filter_month = false;
        setSubtitle();
    },
    'MENSAL':function(){
      slider_panel.insert(2,slider_month);
      options.filter_month = true;
      setSubtitle();
    },
  };
  
  SWITCH[options.rota]();
  
  select_dinamic.onChange(function(value){
        SWITCH[value]();
  });
  
  options.control_panel.insert(1,slider_panel);
    
  // select region
  var select = ui.Select({
    items:[
      'Amazônia',
      'Caatinga',
      'Cerrado',
      'Mata Atlântica',
      'Pampa',
      'Pantanal',
      'Brasil',
    ], 
    // placeholder:,
    value:options.region,
    onChange:function(value){
      
      options.mapp.layers().forEach(function(layer){
        
        var name = layer.getName().split('-')[0];
        
        if (name === 'region'){
          options.mapp.remove(layer);
        }
        return ;
      });      
      
      if (value === 'Bolivia'){
       options.geometry = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee.Filter.eq('country_na','Bolivia')).geometry();
        
       options.mapp.addLayer({
         eeObject:ee.Image().paint(options.geometry,'vazio',2),
        // visParams:,
         name:'region-Bolivia',
         shown:true,
        // opacity:
        });
      setSubtitle();

         return ;
      }
      if (value === 'Indonesia'){
       options.geometry = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee.Filter.eq('country_na','Indonesia')).geometry();
        
       options.mapp.addLayer({
         eeObject:ee.Image().paint(options.geometry,'vazio',2),
        // visParams:,
         name:'region-indonesia',
         shown:true,
        // opacity:
        });
      setSubtitle();

         return ;
      }
      
      if (value === 'Brasil'){
       options.geometry = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017")
        .filter(ee.Filter.eq('country_na','Brazil')).geometry();
       
      options.mapp.addLayer({
         eeObject:ee.Image().paint(options.geometry,'vazio',2),
        // visParams:,
         name:'region-Brasil',
         shown:true,
        // opacity:
      });
      setSubtitle();

       return  ;
      }

      options.geometry = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/biomas_IBGE_250mil')
        .filter(ee.Filter.eq('Bioma',value));
      options.mapp.addLayer({
         eeObject:ee.Image().paint(options.geometry,'vazio',2),
        // visParams:,
         name:'region-'+value,
         shown:true,
        // opacity:
      });
      setSubtitle();

    },
    // disabled:,
    style:styles.select
  });
  
  head_panel.insert(1,select);
  
    

}

function setSubtitle (){
  if (options.subtitle === undefined){
      options.subtitle = ui.Panel({
        // widgets:,
        layout:ui.Panel.Layout.Flow('vertical'),
        style:styles.comum
      });
      
      options.control_panel.add(options.subtitle);
  } else {
    options.subtitle.clear();
  }
  
  options.satellites.forEach(function(obj){
    var situation = obj.years.indexOf(options.year);
    
    if (situation === -1){
      obj.show === false;
      options.mapp.layers().forEach(function(ly){
        if (ly.getName().split('-')[0] === obj.name){
          options.mapp.layers().remove(ly);
        }
      });
      return ;
    }
    
    
    var collection = obj.allProcess(obj.id);
    
    
    // filterDate
    var start, end, name;
    if (options.filter_month !== true){
      start = ''+options.year+'-01-01';
      end = ''+(options.year+1)+'-01-01';
      name = obj.name + '-' + options.year;
    } else {
      start = ''+options.year+'-'+ options.month + '-01';
      end = ''+options.year+'-'+ (options.month+1) + '-01';
      
      if(options.month === 12){
        end = ''+(options.year+1)+'-01-01';
      }
      
      name = obj.name + '-' + options.year + '-' + options.month;
    }
    
    
    // print(start,end);
    // mosaicing
    var col_filter = collection
      .filterBounds(options.geometry)
      .filterDate(start,end);
    
    var image = obj.reduceProcess(col_filter);
    
    var layer = ui.Map.Layer({
      eeObject:image,
      visParams:obj.vis,
      name:name,
      shown:true,
      opacity:1
    });
    
    var panel = ui.Panel({
      // widgets:,
      layout:ui.Panel.Layout.Flow('vertical'),
      style:styles.comum
    });
    
    var switchs = {
      true:function(){
        options.mapp.layers().forEach(function(ly){
          if (ly.getName().split('-')[0] === obj.name){
            options.mapp.layers().remove(ly);
          }
        });

        options.mapp.add(layer);
        
      },
      false:function(){
        options.mapp.remove(layer);
        options.panel.widgets().forEach(function(widget){
            if (widget.widgets().get(0).getValue() === obj.name){
              options.panel.widgets().remove(widget);
            }
          });
      }
    };
    
    switchs[obj.show || false]();
    
    panel.add(ui.Checkbox({
      label:name,
      value:obj.show || false,
      onChange:function(value){
        // print(value);
        switchs[value]();
        obj.show = value;
        
      },
      // disabled:,
      style:styles.comum
    }));
    
    options.subtitle.add(panel);
    
  });
  
}

function onClick_map (){
  options.mapp.onClick(function(obj_click){
    
    options.point = ee.Geometry.Point([obj_click.lon,obj_click.lat]);
    
    options.mapp.layers()
      .forEach(function(layer){
        
        var name = layer.getName();
        if (name.split('-')[0] === 'region'){
          // options.mapp.remove(layer);
          return ;
        }        
        if (name === 'point'){
          options.mapp.remove(layer);
          
          return ;
        } else {
          var satellite = name.split('-')[0];
          
          options.panel.widgets().forEach(function(widget){
            if (widget.widgets().get(0).getValue() === satellite){
              options.panel.widgets().remove(widget);
            }
          });
          // print('satellite',satellite);
          var obj = options.satellites.filter(function(objt){return objt.name === satellite})[0];
          // print('obj',options.satellites,satellite,obj)
           // filterDate
          var start, end;
          if (options.filter_month !== true){
            start = ''+options.year+'-01-01';
            end = ''+(options.year+1)+'-01-01';
          } else {
            start = ''+options.year+'-'+ options.month + '-01';
            end = ''+options.year+'-'+ (options.month+1) + '-01';
            
            if(options.month === 12){
              end = ''+(options.year+1)+'-01-01';
            }
          }
          // print('obj',obj);
          var collection = obj.allProcess(obj.id);
          collection = collection
            .filterBounds(options.point)
            .filterDate(start,end);
            
          collection.aggregate_array('system:index').evaluate(function(list){
            var comum_strech = list.map(function(id){
              var image = collection
                .filter(ee.Filter.inList('system:index', [id]))
                .first();
              
              var panel = ui.Panel({
                // widgets:,
                layout:ui.Panel.Layout.Flow('vertical'),
                style:styles.comum
              });
              
              var switchs = {
                true:function(){
                  // print(layer);
                  options.mapp.add(layer);
                },
                false:function(){
                  options.mapp.remove(layer);
                }
              };
              
              panel.add(ui.Checkbox({
                label:id,
                value:false,
                onChange:function(value){
                  // print(value);
                  
                  switchs[value]();
                },
                // disabled:,
                style:styles.comum
              }));
              
              var geom_thumbnail = null;

              if (obj.name.split('_')[0] === 'MODIS'){
                geom_thumbnail = options.point.buffer(100000).bounds();
                
                image = image.updateMask(ee.Image().paint(geom_thumbnail).neq(1));
              }
              
              var layer = ui.Map.Layer({
                eeObject:image,
                visParams:obj.vis,
                name:obj.name+'/'+id,
                shown:true,
                // opacity:
              });

              
              panel.add(ui.Thumbnail({
                image:image.visualize(obj.vis),
                params:{
                  // dimensions:
                    region:geom_thumbnail
                },
                onClick:function(){
                  print('"'+ id + '",');
                },
                style:styles.comum
              }));
              
              return panel;
            });
          
          
          comum_strech = ui.Panel({
            widgets:comum_strech,
            layout:ui.Panel.Layout.Flow('horizontal',true),
            style:styles.comum_strech
          });
          
          comum_strech = ui.Panel({
            widgets:[ui.Label(obj.name),comum_strech],
            layout:ui.Panel.Layout.Flow('vertical',true),
            style:styles.comum_strech
          });
          

          options.panel.add(comum_strech);
            
            
          });
          
        }
        
        
        
      });
    
    options.mapp.add(ui.Map.Layer({
      eeObject:options.point,
      // visParams:,
      name:'point',
      shown:true,
      // opacity:
    }));
    
    options.mapp.layers().forEach(function(layer){
      var name = layer.getName();
      
      if (name === 'point' || name.split('-')[0] === 'region'){
        // print(name);
        return ;
      }
      
      if (name.slice(0,7) === 'landsat'){
        address_Landsat(options.point,layer.getEeObject());
        return ;
      }
      
      if (name.slice(0,8) === 'Sentinel'){
        address_Sentinel(options.point,layer.getEeObject());
        return ;
      }
      
    });
    
    
  });
}

function start (){
  
  setLayout();
  
  setSubtitle();

  onClick_map();

  print(options);
}

start();
