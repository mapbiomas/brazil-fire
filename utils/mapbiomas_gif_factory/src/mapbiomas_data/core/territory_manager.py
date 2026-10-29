import math
from typing import Any, Dict, List, Optional, Tuple
from ..config import ConfigLoader


class TerritoryManager:
    """Gerenciar territórios geográficos a partir da configuração."""

    TYPES = ["countries", "biomes", "states", "regions", "departments"]
    TYPE_ALIASES = {
        "ufs": "states",
        "custom_regions": "regions",
        "paraguay": "departments",
    }

    def __init__(self, config: ConfigLoader):
        self.config = config
        self._territories = config.territories

    def _resolve_type(self, type_name: str) -> str:
        return self.TYPE_ALIASES.get(type_name, type_name)

    def list_types(self) -> List[str]:
        return self.TYPES

    def list_territories(self, territory_type: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        territories = self._territories
        if territory_type:
            resolved = self._resolve_type(territory_type)
            if resolved not in territories:
                return []
            for tid, tinfo in territories.get(resolved, {}).items():
                result.append(self._format_territory(tid, tinfo, resolved))
        else:
            for ttype in self.TYPES:
                for tid, tinfo in territories.get(ttype, {}).items():
                    result.append(self._format_territory(tid, tinfo, ttype))
        return sorted(result, key=lambda x: (x["type"], x["name"]))

    def _format_territory(self, tid: str, tinfo: Dict, ttype: str) -> Dict[str, Any]:
        return {
            "id": tid,
            "name": tinfo.get("name", tid),
            "name_en": tinfo.get("name_en", ""),
            "type": ttype,
            "source": tinfo.get("source", ""),
            "filter": tinfo.get("filter", ""),
            "geometry": tinfo.get("geometry"),
            "bbox": tinfo.get("bbox"),
        }

    def _find_territory_entry(self, territory_id: str) -> Dict[str, Any]:
        for ttype in self.TYPES:
            group = self._territories.get(ttype, {})
            if territory_id in group and not isinstance(group.get(territory_id), dict):
                continue
            if territory_id in group and isinstance(group.get(territory_id), dict):
                return {'entry': group[territory_id], 'type': ttype}
            for sub_key, sub_group in group.items():
                if isinstance(sub_group, dict) and territory_id in sub_group:
                    return {'entry': sub_group[territory_id], 'type': ttype}
        raise KeyError(f"Território '{territory_id}' não encontrado")

    def _find_in_group(self, group: dict, territory_id: str):
        """Procura recursivamente um territorio em um grupo (incluindo sub-grupos)."""
        if territory_id in group and isinstance(group.get(territory_id), dict):
            return group[territory_id]
        for sub_key, sub_group in group.items():
            if isinstance(sub_group, dict) and territory_id in sub_group:
                return sub_group[territory_id]
        return None

    def get_territory(self, territory_id: str) -> Dict[str, Any]:
        found = self._find_territory_entry(territory_id)
        return self._format_territory(territory_id, found['entry'], found['type'])

    def get_territory_info(self, territory_type: str, territory_id: str) -> Dict[str, Any]:
        territories = self._territories.get(territory_type, {})
        entry = self._find_in_group(territories, territory_id)
        if entry is None:
            raise KeyError(f"Território '{territory_id}' não encontrado em '{territory_type}'")
        return self._format_territory(territory_id, entry, territory_type)

    def get_territory_name(self, territory_type: str, territory_id: str) -> str:
        return self.get_territory_info(territory_type, territory_id)["name"]

    def get_feature_collection(self, territory_id: str):
        found = self._find_territory_entry(territory_id)
        info = found['entry']
        try:
            import ee
        except ImportError:
            return None
        if info.get("geometry"):
            return self._build_geometry_fc(ee, info["geometry"])
        fc = ee.FeatureCollection(info["source"])
        if info.get("filter"):
            parsed = self._parse_filter(info["filter"])
            if parsed:
                fc = fc.filter(ee.Filter.equals(parsed["field"], parsed["value"]))
        filter_in = info.get("filter_in")
        if filter_in:
            fc = fc.filter(ee.Filter.inList(filter_in["field"], filter_in["values"]))
        return fc

    def _build_geometry_fc(self, ee, geometry: dict):
        """Constroi um FeatureCollection a partir de bbox ou ponto + buffer.

        geometry:
          bbox: [lon_min, lat_min, lon_max, lat_max] (retângulo direto)
          center: [lon, lat]
          buffer_m: raio do buffer em metros (define a altura)
          aspect: relacao largura/altura para o quadro (ex: 1.78 p/ paisagem 16:9)
        O retangulo resultante (center): altura = 2*buffer_m, largura = altura*aspect.
        """
        if "bbox" in geometry:
            rect = ee.Geometry.Rectangle(geometry["bbox"])
            return ee.FeatureCollection([ee.Feature(rect)])
        center = geometry["center"]
        buf_m = float(geometry.get("buffer_m", 2500))
        aspect = float(geometry.get("aspect", 1.0))
        lat = center[1]
        half_h = buf_m / 111320.0
        half_w = half_h * aspect / max(math.cos(math.radians(lat)), 0.01)
        lon_min = center[0] - half_w
        lon_max = center[0] + half_w
        lat_min = center[1] - half_h
        lat_max = center[1] + half_h
        rect = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
        return ee.FeatureCollection([ee.Feature(rect)])

    def _parse_filter(self, expression: str) -> Optional[Dict[str, str]]:
        if "==" in expression:
            parts = expression.split("==")
            return {
                "field": parts[0].strip(),
                "value": parts[1].strip().strip("'\""),
            }
        return None

    def get_overlay_fc(self, territory_id: str):
        info = self.get_raw_territory(territory_id)
        overlay_source = info.get("overlay_source")
        if not overlay_source:
            return None
        try:
            import ee
        except ImportError:
            return None
        return ee.FeatureCollection(overlay_source)

    def get_raw_territory(self, territory_id: str) -> Dict[str, Any]:
        found = self._find_territory_entry(territory_id)
        info = found['entry']
        info["id"] = territory_id
        info["type"] = found['type']
        return info

    def get_bbox(self, territory_id: str) -> Optional[List[float]]:
        info = self.get_territory(territory_id)
        return info.get("bbox")

    def validate_territory(self, territory_id: str) -> Tuple[bool, str]:
        try:
            info = self.get_territory(territory_id)
            if not info.get("source") and not info.get("geometry"):
                return False, "Território sem fonte definida"
            return True, "OK"
        except KeyError as e:
            return False, str(e)
