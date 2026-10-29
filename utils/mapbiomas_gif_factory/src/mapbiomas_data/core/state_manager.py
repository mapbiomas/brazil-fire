import os
from datetime import datetime
from typing import List, Optional


class StateManager:
    """Gerencia marcadores de estado para checkpoint/resume da pipeline.

    Cada etapa concluida cria um arquivo `.state_<etapa>` no diretorio
    de saida. Na proxima execucao, as etapas ja marcadas sao puladas.
    """

    STEPS = [
        "download",
        "resize",
        "overlay_legend",
        "frames_clean",
        "collage",
        "special_collages",
        "collage_labels",
        "frames_maps",
        "gif",
        "special_gifs",
    ]

    def __init__(self, output_dir: str):
        self.output_dir = output_dir

    def _path(self, state: str) -> str:
        return os.path.join(self.output_dir, f".state_{state}")

    def is_complete(self, state: str) -> bool:
        return os.path.exists(self._path(state))

    def mark_complete(self, state: str):
        os.makedirs(self.output_dir, exist_ok=True)
        with open(self._path(state), "w") as f:
            f.write(datetime.now().isoformat())

    def clear_one(self, state: str):
        path = self._path(state)
        if os.path.exists(path):
            os.remove(path)

    def clear_all(self):
        for state in self.STEPS:
            self.clear_one(state)

    def get_resume_point(self, steps: Optional[List[str]] = None) -> Optional[str]:
        steps = steps or self.STEPS
        for step in steps:
            if not self.is_complete(step):
                return step
        return None

    def get_completed(self) -> List[str]:
        return [s for s in self.STEPS if self.is_complete(s)]
