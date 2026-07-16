# -*- coding: utf-8 -*-
"""Cliente da API MapBiomas Fogo usado pelo plugin.

Este módulo concentra as chamadas HTTP da nova API para evitar que a lógica
de rede fique misturada com a lógica da interface do QGIS.
"""

import os
import json
import time

import requests


class MapBiomasApiClient:
    def __init__(
        self,
        base_url="https://stg.plataforma.mapbiomas.org", #"https://plataforma.mapbiomas.org",
        upload_base_url="https://stg.plataforma.mapbiomas.org", #"https://prd.plataforma.mapbiomas.org"
        timeout=30,
    ):
        """Inicializa o cliente da API.

        Parâmetros:
            base_url: Endereço base da API do MapBiomas.
            timeout: Tempo máximo de espera, em segundos, para cada requisição.
        """
        self.base_url = base_url.rstrip("/")
        self.upload_base_url = upload_base_url.rstrip("/")
        self.timeout = timeout
        # Guarda a lista de legendas por região para evitar consultas repetidas.
        self._legends_cache_by_region = {}
        # Guarda o detalhe de cada legenda já consultada por id.
        self._legend_detail_cache = {}
        # Guarda categorias de territorio por regiao.
        self._territory_categories_cache = {}
        # Guarda listas de territorios por combinacao de filtros.
        self._territory_list_cache = {}
        # Guarda dados do tema fire por regiao para reuso no plugin.
        self._fire_theme_cache = {}

    def _build_url(self, path):
        """Monta a URL completa a partir do caminho do endpoint."""
        return f"{self.base_url}{path}"

    def _build_upload_url(self, path):
        """Monta a URL completa para rotas de upload de geometria customizada."""
        return f"{self.upload_base_url}{path}"

    def _get(self, path, params=None, timeout=None, max_attempts=4):
        """Executa um GET simples e devolve o JSON já convertido.

        Parâmetros:
            path: Caminho do endpoint, por exemplo /api/v1/brazil/themes.
            params: Dicionário com query params enviados na URL.
        """
        effective_timeout = self.timeout if timeout is None else timeout
        attempt = 1
        while True:
            try:
                response = requests.get(self._build_url(path), params=params, timeout=effective_timeout)
                response.raise_for_status()
                # A API devolve JSON UTF-8; em alguns ambientes o requests pode inferir
                # encoding incorreto e gerar textos com caracteres quebrados (mojibake).
                response.encoding = "utf-8"
                return json.loads(response.text)
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                retryable_status = {429, 500, 502, 503, 504}
                should_retry = status_code in retryable_status
                if not should_retry or attempt >= max_attempts:
                    raise

                sleep_seconds = 0.8 * (2 ** (attempt - 1))
                print(
                    "[MapBiomas Fogo][API] GET falhou, nova tentativa -> "
                    f"path={path}, status={status_code}, tentativa={attempt}/{max_attempts}, "
                    f"aguardando={sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
                attempt += 1
            except requests.RequestException:
                if attempt >= max_attempts:
                    raise

                sleep_seconds = 0.8 * (2 ** (attempt - 1))
                print(
                    "[MapBiomas Fogo][API] Erro de rede, nova tentativa -> "
                    f"path={path}, tentativa={attempt}/{max_attempts}, aguardando={sleep_seconds:.1f}s"
                )
                time.sleep(sleep_seconds)
                attempt += 1

    def upload_territory_zip(self, zip_file_path, region_name="brazil"):
        """Faz upload de um ZIP (SHP) e retorna o territoryId UUID criado na API.

        Esse é o mesmo fluxo usado pela plataforma web para recortes de
        "My geometry", que depois são consumidos no maps/map via territoryId.
        """
        if not zip_file_path or not os.path.exists(zip_file_path):
            raise ValueError("Arquivo ZIP de geometria nao encontrado para upload.")

        upload_path = f"/api/v1/{region_name}/territories/upload"
        with open(zip_file_path, "rb") as zip_file:
            files = {
                "file": (os.path.basename(zip_file_path), zip_file, "application/zip"),
            }
            response = requests.post(
                self._build_upload_url(upload_path),
                files=files,
                timeout=max(self.timeout, 120),
            )

        response.raise_for_status()
        payload = response.json()
        territory = payload.get("territory", {}) if isinstance(payload, dict) else {}
        territory_id = territory.get("id")
        if not territory_id:
            raise ValueError("Upload concluido sem territoryId na resposta da API.")

        return str(territory_id)

    @staticmethod
    def _normalize_territory_ids(territory_id):
        """Normaliza territoryId para lista de strings ou None."""
        if territory_id is None:
            return None

        if isinstance(territory_id, str):
            value = territory_id.strip()
            if not value:
                return None
            return [value]

        if isinstance(territory_id, (list, tuple, set)):
            normalized = [str(item).strip() for item in territory_id if str(item).strip()]
            return normalized or None

        value = str(territory_id).strip()
        return [value] if value else None

    def _apply_territory_params(self, params, territory_id=None, spatial_method="union"):
        """Aplica territoryId e spatialMethod quando necessario.

        Quando ha mais de um territoryId, alguns endpoints da API exigem
        explicitamente spatialMethod. Nos demais, o parametro e aceito e manter
        o mesmo comportamento evita divergencia entre chamadas do plugin.
        """
        normalized_territory_ids = self._normalize_territory_ids(territory_id)
        if not normalized_territory_ids:
            return None

        params["territoryId"] = normalized_territory_ids
        if len(normalized_territory_ids) > 1:
            params["spatialMethod"] = str(spatial_method or "union")

        return normalized_territory_ids

    def _get_fire_theme(self, region_name="brazil"):
        """Retorna o objeto do tema fire para a regiao informada."""
        if region_name in self._fire_theme_cache:
            return self._fire_theme_cache[region_name]

        payload = self._get(
            f"/api/v1/{region_name}/themes",
            params={"page": 1, "pageSize": 100},
        )
        themes = payload.get("themes", [])
        fire_theme = next((theme for theme in themes if theme.get("key") == "fire"), None)

        if not fire_theme:
            raise ValueError("Tema 'fire' nao encontrado na API.")

        self._fire_theme_cache[region_name] = fire_theme
        return fire_theme

    def _get_fire_subtheme_keys(self, region_name="brazil"):
        """Busca as chaves dos subtemas do tema fire.

        Parâmetros:
            region_name: Região da API, como brazil.

        Retorno:
            Lista de chaves de subtemas usadas nas próximas consultas.
        """
        fire_theme = self._get_fire_theme(region_name=region_name)

        subtheme_keys = [
            item.get("key")
            for item in fire_theme.get("subthemes", [])
            if item.get("key")
        ]

        if not subtheme_keys:
            raise ValueError("Subtemas do tema 'fire' nao encontrados na API.")

        print(f"[MapBiomas Fogo][DEBUG] Subtemas do fire theme: {subtheme_keys}")
        return subtheme_keys

    def _list_legends(self, region_name="brazil"):
        """Lista todas as legendas do tema fire para a região informada.

        Parâmetros:
            region_name: Região usada na URL da API.

        Observação:
            O resultado fica em cache porque várias opções do plugin reutilizam
            a mesma lista de legendas.
        """
        if region_name in self._legends_cache_by_region:
            return self._legends_cache_by_region[region_name]

        subtheme_keys = self._get_fire_subtheme_keys(region_name=region_name)
        payload = self._get(
            f"/api/v1/{region_name}/legends",
            params={"page": 1, "pageSize": 500, "subthemeKey": subtheme_keys},
        )
        legends = payload.get("legends", [])
        self._legends_cache_by_region[region_name] = legends
        return legends

    @staticmethod
    def _extract_pixel_values(legend_detail):
        """Extrai pixelValues e itens da legenda a partir do JSON detalhado.

        Parâmetros:
            legend_detail: Resposta completa do endpoint de detalhe da legenda.

        Retorno:
            Dicionário com:
            - pixelValues: lista sem duplicados dos valores de pixel.
            - legendItems: itens de legenda relevantes para uso futuro.
        """
        pixel_values = []
        legend_items = []

        def _build_item(node):
            if not isinstance(node, dict):
                return None

            item = {
                "id": node.get("id"),
                "key": node.get("key"),
                "pixelValue": node.get("pixelValue"),
                "name": node.get("name"),
                "color": node.get("color"),
                "order": node.get("order"),
            }

            children = node.get("children")
            if isinstance(children, list) and children:
                parsed_children = []
                for child in children:
                    child_item = _build_item(child)
                    if child_item is not None:
                        parsed_children.append(child_item)
                if parsed_children:
                    item["children"] = parsed_children

            return item

        def _collect_pixels(item):
            pixel_value = item.get("pixelValue")
            if pixel_value is not None:
                pixel_values.append(pixel_value)

            for child in item.get("children", []):
                _collect_pixels(child)

        items_root = legend_detail.get("items") if isinstance(legend_detail, dict) else None
        if isinstance(items_root, list) and items_root:
            for node in items_root:
                built_item = _build_item(node)
                if built_item is None:
                    continue
                legend_items.append(built_item)
                _collect_pixels(built_item)
        else:
            # Fallback para formatos antigos sem campo raiz "items".
            def _walk(node):
                if isinstance(node, dict):
                    if "pixelValue" in node and node.get("pixelValue") is not None:
                        built_item = _build_item(node)
                        if built_item is not None:
                            legend_items.append(built_item)
                            _collect_pixels(built_item)
                    for value in node.values():
                        _walk(value)
                elif isinstance(node, list):
                    for item in node:
                        _walk(item)

            _walk(legend_detail)

        unique_pixel_values = list(dict.fromkeys(pixel_values))
        return {
            "pixelValues": unique_pixel_values,
            "legendItems": legend_items,
        }

    def _get_legend_pixel_values_by_id(self, legend_id, region_name="brazil"):
        """Consulta o detalhe de uma legenda e extrai seus pixelValues.

        Parâmetros:
            legend_id: Identificador interno da legenda na API.
            region_name: Região usada no endpoint.
        """
        cache_key = (region_name, legend_id)
        if cache_key in self._legend_detail_cache:
            return self._legend_detail_cache[cache_key]

        legend_detail = self._get(f"/api/v1/{region_name}/legends/{legend_id}")
        parsed = self._extract_pixel_values(legend_detail)
        self._legend_detail_cache[cache_key] = parsed
        return parsed

    def _get_legend_by_key(self, legend_key, region_name="brazil"):
        """Obtém os dados completos de uma legenda diretamente pela chave.
        
        Parâmetros:
            legend_key: Chave da legenda, ex: fire_monthly_mapbiomas_monthly.
            region_name: Região da API.
            
        Retorno:
            Dicionário com os dados completos da legenda, incluindo items com pixelValues.
        """
        cache_key = (region_name, legend_key)
        if cache_key in self._legend_detail_cache:
            return self._legend_detail_cache[cache_key]
            
        legend_detail = self._get(f"/api/v1/{region_name}/legends/by/key/{legend_key}")
        parsed = self._extract_pixel_values(legend_detail)
        self._legend_detail_cache[cache_key] = parsed
        return parsed

    def get_pixel_values_by_key(self, legend_key, region_name="brazil"):
        """Localiza uma legenda pela key e devolve seus pixelValues.

        Parâmetros:
            legend_key: Chave da legenda usada pelo plugin, por exemplo
                fire_total_burned_mapbiomas_total_burned.
            region_name: Região usada na API.

        Retorno:
            Dicionário com id da legenda, key, pixelValues e legendItems.

        Fluxo:
            1. Busca a lista de legendas do tema fire.
            2. Encontra a legenda com a key desejada.
            3. Se não encontrar na lista, tenta o endpoint direto /legends/by/key/{legend_key}.
            4. Extrai os pixelValues que serão enviados ao endpoint do mapa.
        """
        # Primeiro, tenta encontrar na lista
        legends = self._list_legends(region_name=region_name)
        legend_match = next(
            (legend for legend in legends if legend.get("key") == legend_key),
            None,
        )
        
        legend_id = None
        if legend_match:
            legend_id = legend_match.get("id")
            print(f"[MapBiomas Fogo][DEBUG] Legenda '{legend_key}' encontrada na lista com id={legend_id}")
        else:
            # Fallback: tenta acessar diretamente via endpoint /legends/by/key/{legend_key}
            try:
                print(f"[MapBiomas Fogo][DEBUG] Legenda '{legend_key}' não encontrada na lista. Tentando endpoint direto...")
                legend_response = self._get(f"/api/v1/{region_name}/legends/by/key/{legend_key}")
                
                # O endpoint direto pode retornar a legenda em um campo 'legend' ou diretamente
                if isinstance(legend_response, dict) and "legend" in legend_response:
                    legend_match = legend_response["legend"]
                else:
                    legend_match = legend_response
                    
                legend_id = legend_match.get("id")
                print(f"[MapBiomas Fogo][DEBUG] ✓ Legenda obtida via endpoint direto! id={legend_id}")
            except Exception as fallback_error:
                # Se o fallback também falhar, mostra mensagem de erro com legendas disponíveis
                available_keys = [legend.get("key") for legend in legends if legend.get("key")]
                print(f"[MapBiomas Fogo][DEBUG] Legenda '{legend_key}' não encontrada em nenhum endpoint.")
                print(f"[MapBiomas Fogo][DEBUG] Legendas disponíveis ({len(available_keys)}): {available_keys}")
                print(f"[MapBiomas Fogo][DEBUG] Erro no fallback: {fallback_error}")
                raise ValueError(
                    f"Legenda nao encontrada para key='{legend_key}'. "
                    f"Legendas disponíveis: {', '.join(available_keys[:5])}{'...' if len(available_keys) > 5 else ''}"
                )
        
        if not legend_id:
            raise ValueError(f"Legenda encontrada sem id para key='{legend_key}'.")

        # Agora obtém os detalhes com pixelValues
        detail = self._get_legend_pixel_values_by_id(legend_id, region_name=region_name)
        return {
            "legendId": legend_id,
            "legendKey": legend_match.get("key"),
            "pixelValues": detail.get("pixelValues", []),
            "legendItems": detail.get("legendItems", []),
        }

    def get_fire_informative_note(self, region_name="brazil"):
        """Retorna o bloco informativeNote do tema fire."""
        fire_theme = self._get_fire_theme(region_name=region_name)
        return fire_theme.get("informativeNote") or {}

    def list_available_legend_keys(self, region_name="brazil"):
        """Lista todas as chaves de legenda disponíveis para debug.
        
        Retorno:
            Lista de dicionários com id, key e subthemeKey de cada legenda.
        """
        legends = self._list_legends(region_name=region_name)
        return [
            {
                "id": legend.get("id"),
                "key": legend.get("key"),
                "subthemeKey": legend.get("subthemeKey"),
                "name": legend.get("name"),
            }
            for legend in legends
        ]

    def list_territory_categories(self, region_name="brazil"):
        """Lista as categorias de territorios disponiveis para a regiao."""
        if region_name in self._territory_categories_cache:
            return self._territory_categories_cache[region_name]

        payload = self._get(
            f"/api/v1/{region_name}/territories/categories",
            params={"page": 1, "pageSize": 500},
        )
        categories = payload.get("categories", [])
        self._territory_categories_cache[region_name] = categories
        return categories

    def list_territories(self, region_name="brazil", category_id=None, parent_id=None, page_size=1000):
        """Lista territorios filtrando por categoria e/ou parentId com cache."""
        cache_key = (region_name, category_id, parent_id, page_size)
        if cache_key in self._territory_list_cache:
            return self._territory_list_cache[cache_key]

        page = 1
        territories = []
        while True:
            params = {
                "page": page,
                "pageSize": page_size,
            }
            if category_id is not None:
                params["categoryId"] = category_id
            if parent_id:
                params["parentId"] = parent_id

            payload = self._get(
                f"/api/v1/{region_name}/territories",
                params=params,
            )
            page_items = payload.get("territories", [])
            territories.extend(page_items)

            if len(page_items) < page_size:
                break
            page += 1

        self._territory_list_cache[cache_key] = territories
        return territories

    def get_territory_geometry(
        self,
        region_name="brazil",
        territoryId=None,
        territoryIds=None,
        simplify=True,
        spatialMethod="union",
    ):
        """Retorna a geometria GeoJSON do territoryId informado."""
        normalized_territory_ids = self._normalize_territory_ids(
            territoryIds if territoryIds is not None else territoryId
        )
        if not normalized_territory_ids:
            return None

        params = {
            "simplify": str(bool(simplify)).lower(),
        }
        self._apply_territory_params(
            params,
            normalized_territory_ids,
            spatial_method=spatialMethod,
        )

        payload = self._get(
            f"/api/v1/{region_name}/territories/geometry",
            params=params,
        )
        geometry = payload.get("geometry") if isinstance(payload, dict) else None
        return geometry

    def get_map_url(
        self,
        region,
        subthemeKey,
        legendKey,
        pixelValue,
        year,
        territoryId=None,
        territoryIds=None,
        spatialMethod="union",
    ):
        """Consulta a URL do mapa raster para um recorte temático específico.

        Parâmetros:
            region: Região da API, como brazil.
            subthemeKey: Subtema do produto de fogo, por exemplo fire_total_burned.
            legendKey: Legenda que define como o dado será renderizado.
            pixelValue: Valor ou lista de valores de pixel usados pela API.
            year: Ano do produto a ser carregado.
            territoryId: Identificador opcional de território (ou lista).
            territoryIds: Alias opcional para lista de territorios.

        Retorno:
            URL de tiles do mapa, normalmente no formato XYZ.
        """
        params = {
            "region": region,
            "subthemeKey": subthemeKey,
            "legendKey": legendKey,
            "pixelValue": pixelValue,
            "year": year,
        }
        self._apply_territory_params(
            params,
            territoryIds if territoryIds is not None else territoryId,
            spatial_method=spatialMethod,
        )

        payload = self._get(
            f"/api/v1/{region}/maps/map",
            params=params,
        )
        return payload.get("url")

    def get_area_statistics(
        self,
        region,
        subthemeKey,
        legendKey,
        pixelValue,
        year,
        territoryId=None,
        territoryIds=None,
        spatialMethod="union",
        yearSeriesAnchor=None,
        propertyCode=None,
        territoryCategoryId=None,
        requestTimeout=None,
        requestMaxAttempts=5,
    ):
        """Consulta estatistica de area no endpoint /statistics/area.

        Usa o mesmo contexto da camada (recorte, legenda, ano e pixels), garantindo
        que os valores do plugin sigam os da plataforma para o mesmo filtro.
        """
        normalized_territory_ids = self._normalize_territory_ids(
            territoryIds if territoryIds is not None else territoryId
        )

        params = {
            "region": region,
            "subthemeKey": subthemeKey,
            "legendKey": legendKey,
            "pixelValue": pixelValue,
            "year": year,
        }
        if yearSeriesAnchor:
            params["yearSeriesAnchor"] = str(yearSeriesAnchor)
        if propertyCode is not None:
            params["propertyCode"] = str(propertyCode)
        if territoryCategoryId is not None:
            params["territoryCategoryId"] = territoryCategoryId

        if normalized_territory_ids:
            applied_ids = self._apply_territory_params(
                params,
                normalized_territory_ids,
                spatial_method=spatialMethod,
            )
            if applied_ids and len(applied_ids) == 1:
                params["territoryId"] = applied_ids[0]
            if spatialMethod is not None:
                params["spatialMethod"] = str(spatialMethod or "union")
        elif str(region or "").lower() == "brazil":
            # Mesmo comportamento da plataforma quando nao ha recorte explicito.
            params["territoryId"] = "1-1-1"
            params["spatialMethod"] = str(spatialMethod or "union")

        return self._get(
            f"/api/v1/{region}/statistics/area",
            params=params,
            timeout=requestTimeout if requestTimeout is not None else max(self.timeout, 60),
            max_attempts=max(1, int(requestMaxAttempts or 1)),
        )
