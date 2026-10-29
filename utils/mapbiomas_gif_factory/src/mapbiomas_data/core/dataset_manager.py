from typing import Any, Dict, List, Optional
from ..config import ConfigLoader


class DatasetManager:
    """Gerenciar datasets do Earth Engine a partir da configuração YAML."""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self._datasets = config.datasets
        self._categories = config.categories

    def list_categories(self) -> List[Dict[str, Any]]:
        result = []
        seen = set()
        for ds_id, ds in self._datasets.items():
            cat = ds.get("category", "other")
            if cat not in seen:
                seen.add(cat)
                cat_info = self._categories.get(cat, {})
                result.append({
                    "id": cat,
                    "label": cat_info.get("label", cat.title()),
                    "color": cat_info.get("color", "#999999"),
                })
        return sorted(result, key=lambda x: x["label"])

    def list_datasets(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        result = []
        for ds_id, ds in self._datasets.items():
            if category and ds.get("category") != category:
                continue
            products = list(ds.get("products", {}).keys()) if ds.get("products") else []
            result.append({
                "id": ds_id,
                "description": ds.get("description", ""),
                "category": ds.get("category", ""),
                "source": ds.get("source", ""),
                "products": products,
            })
        return sorted(result, key=lambda x: x["id"])

    def list_products(self, dataset_id: str) -> List[Dict[str, Any]]:
        ds = self._datasets.get(dataset_id)
        if not ds:
            raise KeyError(f"Dataset '{dataset_id}' não encontrado")
        products = ds.get("products", {})
        result = []
        for pid, pinfo in products.items():
            result.append({
                "id": pid,
                "name": pinfo.get("name", pid),
                "bands": pinfo.get("bands", []),
                "temporal_range": pinfo.get("temporal_range"),
                "visualization": pinfo.get("visualization"),
            })
        return result

    def get_product(self, dataset_id: str, product_id: str) -> Dict[str, Any]:
        ds = self._datasets.get(dataset_id)
        if not ds:
            raise KeyError(f"Dataset '{dataset_id}' não encontrado")
        products = ds.get("products", {})
        product = products.get(product_id)
        if not product:
            raise KeyError(f"Produto '{product_id}' não encontrado em '{dataset_id}'")
        result = dict(product)
        result["id"] = product_id
        result["dataset_id"] = dataset_id
        result["dataset_description"] = ds.get("description", "")
        result["collection_title"] = ds.get("title_collection", "")
        result["dataset_source"] = ds.get("source", "")
        result["asset"] = product.get("asset", "")
        result["asset_type"] = product.get("asset_type", "image")
        result["mosaic"] = product.get("mosaic", False)
        result["mask_value"] = product.get("mask_value")
        result["post_processing"] = product.get("post_processing")
        result["bands_slice"] = product.get("bands_slice")
        return result

    def get_asset_id(self, dataset_id: str, product_id: str) -> str:
        product = self.get_product(dataset_id, product_id)
        return product.get("asset", "")

    def search(self, query: str) -> List[Dict[str, Any]]:
        query = query.lower()
        results = []
        for ds_id, ds in self._datasets.items():
            for pid, pinfo in ds.get("products", {}).items():
                if query in pid.lower() or query in pinfo.get("name", "").lower():
                    results.append({
                        "dataset_id": ds_id,
                        "product_id": pid,
                        "name": pinfo.get("name", pid),
                        "description": pinfo.get("description", ""),
                    })
        return results
