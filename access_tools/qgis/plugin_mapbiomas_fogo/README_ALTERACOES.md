# Modificações - MapBiomas Fogo Plugin
## Download de Feições do Fogo

### ✅ Funcionalidade Implementada

O plugin agora aplica **intersect automático** ao baixar feições do fogo (FGB) conforme o recorte territorial ou geometria personalizada selecionada.

---

## 📋 Comportamento por Recorte

### 1️⃣ **Brasil**
- ✔️ **Ação**: Carrega a camada **SEM processamento**
- **Resultado**: Feições de fogo de todo o Brasil

### 2️⃣ **Outro Recorte Territorial** (Ex: Cerado, Pantanal, etc.)
- ✔️ **Ação**: 
  - Obtém os **limites completos** do território da API (não apenas bounding box)
  - Faz **INTERSECT** entre FGBs e esses limites
- **Resultado**: Apenas feições dentro do território selecionado

### 3️⃣ **Geometria Personalizada**
- ✔️ **Ação**:
  - Aplica tratamentos (dissolve, seleção de feições)
  - Faz **INTERSECT** com FGBs **localmente** (SEM upload para API)
- **Resultado**: Apenas feições dentro da geometria customizada

### 4️⃣ **Demais Consultas**
- ✔️ Funcionamento mantém-se **igual** (sem alterações)

---

## 🚀 Como Usar

### Baixar com Território Específico

1. Abra o plugin
2. Aba **Download**: Selecione anos
3. Aba **Recortes Territoriais**: Selecione um território
4. Clique **OK** → Intersect automático!

### Baixar com Geometria Personalizada

1. Abra o plugin
2. Aba **Download**: Selecione anos
3. Aba **Geometria Personalizada**: Selecione sua camada
4. (Opcional) Marque "Apenas feições selecionadas"
5. Clique **OK** → Intersect automático!

---

## 🔧 Detalhes Técnicos

### Novos Métodos
- `_build_qgs_geometry_from_geojson()` - Converte GeoJSON → QgsGeometry
- `_intersect_with_territory_geometry()` - Intersect com território da API
- `_intersect_with_user_geometry()` - Intersect com geometria do usuário
- `_get_dissolved_geometry()` - Combina múltiplas feições
- `_perform_intersect()` - Executa o intersect

### Métodos Modificados
- `_load_selected_fgb_layers()` - Coleta territórios e passa api_client
- `_load_fgb_layer()` - Implementa lógica de intersect
- `accept()` - Passa api_client ao carregar

### Arquivos Alterados
- `MapBiomas_fogo_dialog.py` - Todas as alterações principais

---

## ✨ Características

- ✅ Sem circular imports
- ✅ Compatível com QGIS 3.x e 4.x
- ✅ Fallback automático em caso de erro
- ✅ Suporte para múltiplos territórios
- ✅ Suporte para feições selecionadas
- ✅ Logging detalhado
- ✅ Performance otimizada

---

## 📝 Logs Esperados

Ao usar intersect, você verá mensagens como:

```
[MapBiomas Fogo] Iniciando intersect com territórios: ['1-2-3']
[MapBiomas Fogo] Realizando intersect com Território (2024)
[MapBiomas Fogo] Intersect concluído: 1234 feições para 2024 (Território)
```

---

**Plugin**: MapBiomas Fogo  
**Versão**: 1.0  
**Data**: Julho 2025
