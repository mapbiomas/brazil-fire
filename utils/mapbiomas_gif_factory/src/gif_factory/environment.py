"""Descobre onde a Fábrica está rodando e quantas combinações gerar ao mesmo tempo.

O gargalo é a memória: cada combinação monta dezenas de quadros em alta resolução.
Por isso o número de tarefas simultâneas segue a RAM (cerca de 6 GB por tarefa),
limitado pelos núcleos e a no máximo 4, para não estourar as cotas do Earth Engine.
O Colab gratuito (2 núcleos, ~12 GB) fica com 2 ao mesmo tempo.
"""
import os
from pathlib import Path

GB_PER_TASK = 6
MAX_WORKERS = 4


def _ram_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) / 1024 / 1024
    except (OSError, ValueError, IndexError):
        pass
    try:
        import psutil
        return psutil.virtual_memory().total / 1024 ** 3
    except Exception:
        return None


def _in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return bool(os.environ.get("COLAB_RELEASE_TAG") or os.environ.get("COLAB_GPU"))


def _gpu():
    return bool(os.environ.get("COLAB_GPU")) or Path("/dev/nvidia0").exists()


def suggested_workers(cpus, ram_gb):
    by_ram = int((ram_gb or GB_PER_TASK) // GB_PER_TASK) or 1
    return max(1, min(MAX_WORKERS, cpus or 1, by_ram))


def detect():
    """{'colab', 'cpus', 'ram_gb', 'gpu', 'label', 'workers'}"""
    cpus = os.cpu_count() or 1
    ram = _ram_gb()
    colab = _in_colab()
    gpu = _gpu()
    resources = f"{cpus} núcleo(s)" + (f", {ram:.0f} GB de memória" if ram else "")
    if colab:
        if ram and ram >= 30:
            kind = "Colab com mais memória"
        elif gpu:
            kind = "Colab com GPU"
        else:
            kind = "Colab padrão (gratuito)"
    else:
        kind = "Este computador"
    return {"colab": colab, "cpus": cpus, "ram_gb": ram, "gpu": gpu,
            "label": f"{kind} · {resources}", "workers": suggested_workers(cpus, ram)}
