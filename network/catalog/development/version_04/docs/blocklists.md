# Blocklists — fontes, verificação e atualização

As listas de cenas excluídas são **opcionais**. Uma fonte ausente ou fora do ar
não cancela o export: o sensor segue sem filtro e o status aparece no relatório.

## Fontes atuais

O padrão é o mesmo usado pelo Toolkit Monitor Fuego:

| Sensor | Fonte GEE | Export | Uso |
|---|---|---|---|
| `landsat` | `users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList` | `landsat` | Brasil |
| `landsat` | `users/mapbiomasworkspace1/mapbiomas-fire:00_Tools/require-blockList` | `landsat` | América do Sul (agrega país + Brasil) |
| `sentinel2` | `users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList` | `sentinel` | Brasil |
| `sentinel2` | `users/mapbiomasworkspace1/mapbiomas-fire:00_Tools/module-blockList` | `sentinel` | América do Sul |

Sem fonte configurada por enquanto:

| Chave | Aplicação em M1 | Observação |
|---|---|---|
| `hls` / `hls_l30` / `hls_s30` | `HLSL30` e `HLSS30` | Não existe lista HLS no repositório GEE |
| `modis` / `modis_terra` / `modis_aqua` | `MOD09A1` e `MYD09A1` | Não existe lista MODIS; confirmar semântica dos IDs |
| `viirs` | futuro | Sensor ainda não implementado |

O endereço antigo `users/mapbiomasworkspace1/mapbiomas-fire:00_Tools/require-landsat-collection`
não é usado pelo Monitor (≈9.671 IDs na cópia local). Mantenha-o apenas como
referência de comparação.

## Resolução e fallback

`BLOCKLIST_ALIASES` define a ordem de fallback:

- `hls_l30` → `hls_l30`, depois `hls`
- `hls_s30` → `hls_s30`, depois `hls`
- `modis_terra` → `modis_terra`, depois `modis`
- `modis_aqua` → `modis_aqua`, depois `modis`

As listas encontradas são unidas e deduplicadas. Uma lista vazia é tratada como
"sem filtro" (não gera `Filter.inList` vazio).

## Verificação

```python
from M_blocklist import blocklist_report
import json
print(json.dumps(blocklist_report(), indent=2))
```

O relatório traz, por sensor:

- `count`: total após união/deduplicação;
- `sample`: primeiros IDs;
- `sources`: endereço, `status` (`ok` ou `error`), contagem por fonte, erro e
  commit Git quando disponível.

Confira também o formato dos IDs:

| Sensor | Formato esperado |
|---|---|
| Landsat | `^L[COT]0[5-9]_\d{6}_\d{8}$` (ex.: `LC08_002064_20260329`) |
| Sentinel-2 | `^\d{8}T\d{6}_\d{8}T\d{6}_T[A-Z0-9]{5}$` |
| HLS | `^HLS\.(L30|S30)\.T[A-Z0-9]{5}\.\d{7}T\d{6}` |
| MODIS | Verificar antes de ativar; o `system:index` pode ser um índice temporal global (`YYYY_DOY`) |

## Atualização

1. Atualize o endereço em `BLOCKLIST_SOURCES` (`M_blocklist.py`).
2. Rode `blocklist_report()` e confira `status`, `count`, `sample` e `commit`.
3. Compare com a execução anterior (variação grande de IDs merece revisão).
4. Valide os IDs contra o `system:index` da coleção do sensor.
5. Commit apenas do registro/documentação.

## Colab

O clone usa o token OAuth do ambiente (ADC), porque o Colab normalmente não
tem `~/.gitcookies`. Antes do M1:

1. rode a célula de autenticação do M0;
2. reinicie o runtime após atualizar o repositório, para recarregar os módulos;
3. se o erro de autenticação persistir, gere uma senha de Git em
   `https://earthengine.googlesource.com/new-password` e configure o
   `~/.gitcookies` conforme as instruções da página.

Sem token e sem cookie, o `git` não tem como autenticar no
`earthengine.googlesource.com`. Com o código atual o export **não cancela**:
ele registra um warning e segue sem as listas.

## Teste offline

```powershell
python -m unittest discover -s tests -v
```

Os testes usam `read_source` injetável: nenhum clone, GEE, GCS ou rede é
necessário. Para validar o parser contra cópias reais dos módulos, use os
arquivos em `githubs/mapbiomas-fire/00_Tools/` e `githubs/mapbiomas-fire/01_Mosaics/`.
