"""Estimativas de tempo e de volume de um lote, para a pessoa pensar antes de gerar.

Os números são médias grosseiras (território médio, 2000 px de altura). O tempo por
quadro é recalibrado com o que já rodou na sessão (`Calibration`).
"""
from threading import Lock

SECONDS_PER_FRAME = 10.0      # baixar do Earth Engine + montar o quadro
SECONDS_PER_COMBO = 30.0      # consultas, legenda, GIF e publicação
DEFAULT_FRAMES = 40           # quando ainda não se sabe quantos anos o mapa tem
MB_PER_FRAME = {"gif": 0.35, "finais": 1.2, "preparados": 1.0, "brutos": 0.6}
MB_PER_COMBO = {"colagem": 6.0}
WORK_MB_PER_FRAME = 3.5       # quadros intermediários guardados na pasta temporária
DRIVE_FREE_GB = 15


class Calibration:
    """Média de segundos por quadro observada nesta sessão."""

    def __init__(self):
        self.lock = Lock()
        self.seconds = 0.0
        self.frames = 0

    def add(self, seconds, frames):
        if frames > 0 and seconds > 0:
            with self.lock:
                self.seconds += seconds
                self.frames += frames

    def per_frame(self):
        with self.lock:
            return self.seconds / self.frames if self.frames >= 3 else None


def batch(frames_per_combo, kinds, workers=1, per_frame=None, to_drive=False):
    """{'seconds', 'saved_mb', 'work_mb', 'frames', 'calibrated'} para uma lista de nº de quadros por combinação."""
    frames = [f or DEFAULT_FRAMES for f in frames_per_combo]
    rate = per_frame or SECONDS_PER_FRAME
    serial = sum(f * rate + SECONDS_PER_COMBO for f in frames)
    seconds = serial / max(1, min(workers, len(frames) or 1))
    saved = sum(sum(MB_PER_FRAME.get(k, 0) for k in kinds) * f + sum(MB_PER_COMBO.get(k, 0) for k in kinds)
                for f in frames)
    work = sum(f * WORK_MB_PER_FRAME for f in frames)
    return {"seconds": seconds, "saved_mb": saved * (2 if to_drive else 1), "deliver_mb": saved,
            "work_mb": work, "frames": sum(frames), "calibrated": per_frame is not None}


def human_time(seconds):
    seconds = int(round(seconds))
    if seconds < 60:
        return "menos de 1 minuto"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours} h" + (f" {minutes:02d} min" if minutes else "")
    days, hours = divmod(hours, 24)
    return f"{days} dia(s)" + (f" e {hours} h" if hours else "")


def about(seconds):
    """'~25 min', ou 'menos de 1 minuto' sem o til."""
    text = human_time(seconds)
    return text if seconds < 60 else "~" + text


def human_size(mb):
    if mb < 1:
        return "menos de 1 MB"
    if mb < 1024:
        return f"{mb:.0f} MB"
    return f"{mb / 1024:.1f} GB"


def mood(seconds):
    """(emoji, frase, nível) — nível 0 tranquilo, 1 pense bem, 2 repense."""
    minutes = seconds / 60
    if minutes < 3:
        return "☕", "Mais rápido que passar um café.", 0
    if minutes < 20:
        return "☕", "Dá tempo de um café e uma boa conversa.", 0
    if minutes < 60:
        return "🍲", "Dá para almoçar enquanto os mapas brotam.", 0
    if minutes < 4 * 60:
        return "⏳", ("Uma tarde inteira. Deixe a aba do Colab aberta: o Colab gratuito desconecta "
                     "depois de cerca de 90 min sem interação."), 1
    if minutes < 12 * 60:
        return "🌙", ("Uma noite inteira. Pense bem: o Colab gratuito costuma encerrar sessões em até "
                     "12 h. Dividir em lotes menores é mais seguro."), 2
    return "🗓️", ("Dias! Mais do que uma sessão do Colab aguenta. Escolha menos anos, territórios ou "
                  "tipos de arquivo, ou divida em vários lotes."), 2


def size_note(saved_mb, to_drive):
    if to_drive and saved_mb / 1024 > DRIVE_FREE_GB * 0.5:
        return (f"Isso é mais da metade dos {DRIVE_FREE_GB} GB do Drive gratuito. Confira o espaço livre antes.", 2)
    if saved_mb > 1024:
        return ("Mais de 1 GB: baixar tudo em .zip pode demorar.", 1)
    return ("", 0)
