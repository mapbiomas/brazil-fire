"""Estado de progresso independente da interface, com relógio injetável."""
from threading import RLock
from time import monotonic

from .i18n import _


class ProgressState:
    def __init__(self, total, clock=monotonic):
        self.lock = RLock()
        self.clock = clock
        self.started = clock()
        self.combo_started = self.started
        self.total = total
        self.completed = 0
        self.number = 0
        self.header = ""
        self.stage = "Preparando lote"
        self.detail = ""
        self.frame_index = 0
        self.frame_total = 0
        self.frame_done = 0
        self.samples = []

    def begin(self, number, header):
        with self.lock:
            self.number, self.header = number, header
            self.combo_started = self.clock()
            self.stage, self.detail = "Preparando combinação", ""
            self.frame_index = self.frame_total = self.frame_done = 0
            self.samples = []

    def update(self, kind, *args):
        with self.lock:
            if kind == "discovered":
                self.stage, self.detail = "Preparando download", _("{n} unidades encontradas", n=len(args[0]))
                self.frame_total = len(args[0])
            elif kind == "frame_start":
                index, total, band = args
                self.stage = "Baixando frames"
                self.frame_index, self.frame_total = index, total
                self.detail = _("Frame {i} de {n}", i=index, n=total) + f" — {band}"
            elif kind == "frame_done":
                index, total, band, elapsed = args
                self.frame_done += 1
                self.samples.append(elapsed)
                self.detail = _("Frame {i} de {n} concluído", i=index, n=total) + f" — {band}"
            elif kind == "frame_retry":
                index, total, band, attempt, maximum, wait, error = args
                self.detail = (_("Frame {i} de {n}", i=index, n=total) + f" — {band}: "
                               + _("tentativa {a} de {m} falhou; nova tentativa em {s}s:", a=attempt, m=maximum, s=wait)
                               + f" {error}")
            elif kind == "frame_failed":
                index, total, band, error = args
                self.detail = (_("Frame {i} de {n}", i=index, n=total) + f" — {band}: "
                               + _("falhou após as tentativas:") + f" {error}")
            elif kind == "stage":
                self.stage, self.detail = args[0], ""

    def finish_combo(self):
        with self.lock:
            self.completed += 1

    def snapshot(self):
        with self.lock:
            now = self.clock()
            eta = None
            if self.stage == "Baixando frames" and len(self.samples) >= 2 and self.frame_done < self.frame_total:
                eta = sum(self.samples) / len(self.samples) * (self.frame_total - self.frame_done)
            return {"total": self.total, "completed": self.completed, "number": self.number,
                    "header": self.header, "stage": self.stage, "detail": self.detail,
                    "elapsed": now - self.started, "combo_elapsed": now - self.combo_started,
                    "frame_done": self.frame_done, "frame_total": self.frame_total, "eta": eta}
