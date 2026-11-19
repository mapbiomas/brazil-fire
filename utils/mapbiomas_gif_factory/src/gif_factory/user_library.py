"""'Meus mapas' e 'Meus territórios': itens do analista que voltam nas próximas sessões.

Ficam num JSON no Drive (MyDrive/MapBiomas_GIF_Factory/meus_itens.json) quando o
Drive está conectado; sem Drive, só nesta sessão do Colab.
"""
import json
import os
import tempfile
from pathlib import Path
from threading import Lock

DRIVE_FILE = Path("/content/drive/MyDrive/MapBiomas_GIF_Factory/meus_itens.json")
SESSION_FILE = Path("/content/mapbiomas_gif_factory_meus_itens.json")


def default_path():
    return DRIVE_FILE if DRIVE_FILE.parents[1].is_dir() else SESSION_FILE


class UserLibrary:
    def __init__(self, path=None):
        self.path = Path(path) if path else default_path()
        self.lock = Lock()
        self.data = {"maps": [], "territories": []}
        self.error = ""
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                self.data["maps"] = list(loaded.get("maps") or [])
                self.data["territories"] = list(loaded.get("territories") or [])
            except (OSError, ValueError) as exc:
                self.error = f"Não foi possível ler {self.path}: {exc}"

    @property
    def persistent(self):
        return self.path == DRIVE_FILE or not str(self.path).startswith("/content/")

    @property
    def maps(self):
        return [dict(item) for item in self.data["maps"]]

    @property
    def territories(self):
        return [dict(item) for item in self.data["territories"]]

    def _write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Grava num arquivo temporário e troca, para nunca deixar o JSON pela metade.
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(self.data, handle, ensure_ascii=False, indent=1)
        os.replace(tmp, self.path)

    def _upsert(self, key, item):
        with self.lock:
            items = [x for x in self.data[key] if x.get("id") != item["id"]]
            items.append(dict(item))
            self.data[key] = items
            self._write()

    def _remove(self, key, item_id):
        with self.lock:
            before = len(self.data[key])
            self.data[key] = [x for x in self.data[key] if x.get("id") != item_id]
            if len(self.data[key]) != before:
                self._write()

    def save_map(self, layer):
        self._upsert("maps", layer)

    def remove_map(self, layer_id):
        self._remove("maps", layer_id)

    def save_territory(self, territory):
        self._upsert("territories", territory)

    def remove_territory(self, territory_id):
        self._remove("territories", territory_id)
