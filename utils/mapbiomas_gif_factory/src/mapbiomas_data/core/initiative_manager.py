from typing import Any, Dict, List, Optional
from ..config import ConfigLoader


class InitiativeManager:
    """Gerenciar iniciativas (Brasil, Paraguay, etc.) e seus grupos/coleções."""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self._initiatives = config.initiatives

    def list_initiatives(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": iid,
                "name": info.get("name", iid),
                "description": info.get("description", ""),
                "collection_count": len(info.get("collections", [])),
                "territory_group_count": len(info.get("territory_groups", [])),
            }
            for iid, info in sorted(self._initiatives.items())
        ]

    def get_initiative(self, initiative_id: str) -> Dict[str, Any]:
        info = self._initiatives.get(initiative_id)
        if not info:
            raise KeyError(f"Initiative '{initiative_id}' not found")
        return info

    def list_collections(self, initiative_id: str) -> List[Dict[str, Any]]:
        info = self.get_initiative(initiative_id)
        return info.get("collections", [])

    def list_territory_groups(self, initiative_id: str) -> List[Dict[str, Any]]:
        info = self.get_initiative(initiative_id)
        return info.get("territory_groups", [])

    def list_territories(self, initiative_id: str, group_id: str) -> List[Dict[str, Any]]:
        info = self.get_initiative(initiative_id)
        groups = info.get("territory_groups", [])
        group = next((g for g in groups if g["id"] == group_id), None)
        if not group:
            raise KeyError(f"Group '{group_id}' not found in initiative '{initiative_id}'")
        territory_type = group.get("type", group_id)
        terr_ids = group.get("territory_ids", [])
        territory_dict = self.config.territories.get(territory_type, {})
        result = []
        for tid in terr_ids:
            tinfo = territory_dict.get(tid)
            if tinfo:
                result.append({
                    "id": tid,
                    "name": tinfo.get("name", tid),
                    "name_en": tinfo.get("name_en", ""),
                    "type": territory_type,
                })
        return result

    def get_collection_dataset_id(self, initiative_id: str, collection_id: str) -> str:
        info = self.get_initiative(initiative_id)
        collections = info.get("collections", [])
        coll = next((c for c in collections if c["id"] == collection_id), None)
        if not coll:
            raise KeyError(f"Collection '{collection_id}' not found in initiative '{initiative_id}'")
        return coll["dataset_id"]

    def get_group_territory_ids(self, initiative_id: str, group_id: str) -> List[str]:
        info = self.get_initiative(initiative_id)
        groups = info.get("territory_groups", [])
        group = next((g for g in groups if g["id"] == group_id), None)
        if not group:
            raise KeyError(f"Group '{group_id}' not found in initiative '{initiative_id}'")
        return group.get("territory_ids", [])

    def get_group_type(self, initiative_id: str, group_id: str) -> str:
        info = self.get_initiative(initiative_id)
        groups = info.get("territory_groups", [])
        group = next((g for g in groups if g["id"] == group_id), None)
        if not group:
            raise KeyError(f"Group '{group_id}' not found in initiative '{initiative_id}'")
        return group.get("type", group_id)

    def resolve_dataset(self, dataset_id: str) -> Optional[str]:
        """Find which initiative a dataset belongs to."""
        for iid, info in self._initiatives.items():
            for coll in info.get("collections", []):
                if coll["dataset_id"] == dataset_id:
                    return iid
        return None

    def resolve_territory(self, territory_id: str) -> Optional[str]:
        """Find which initiative a territory belongs to."""
        for iid, info in self._initiatives.items():
            for group in info.get("territory_groups", []):
                if territory_id in group.get("territory_ids", []):
                    return iid
        return None
