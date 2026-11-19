"""Vitrine de resultados: um cartão por combinação (mapa × território).

Serve para acompanhar o lote enquanto ele roda (fila, gerando, pronto, falha) e,
depois, para navegar na biblioteca já gerada (nesta sessão ou no Drive): ver a
miniatura, abrir a animação, copiar o caminho e baixar arquivo por arquivo ou a
pasta inteira.
"""
import io
import zipfile
from html import escape
from pathlib import Path
from threading import RLock

import ipywidgets as w

from .i18n import N_, _, ui_text
from .library import load_manifest

STATES = {"fila": (N_("Na fila"), "var(--muted)", "var(--sel)"),
          "gerando": (N_("Gerando"), "var(--accent-ink)", "var(--accent-soft)"),
          "pronto": (N_("Pronto"), "var(--green)", "var(--green-soft)"),
          "existe": (N_("Já existia"), "var(--green)", "var(--green-soft)"),
          "falha": (N_("Falhou"), "var(--danger)", "var(--danger-soft)")}
FILTERS = [(N_("Todos"), ""), (N_("Prontos"), "pronto"), (N_("Em andamento"), "gerando"), (N_("Na fila"), "fila"),
           (N_("Falhas"), "falha")]
KIND_ORDER = [("gif", N_("GIF animado")), ("colagem", N_("Colagem")), ("finais", N_("Mapa completo de cada ano")),
              ("preparados", N_("Cada ano com ano e escala")), ("brutos", N_("Cada ano sem moldura")),
              ("outros", N_("Outros"))]
FRAME_FOLDERS = {"Mapa_completo": "finais", "Mapa_com_ano_e_escala": "preparados", "Mapa_basico": "brutos"}
MAX_INLINE_GIF = 25 * 1024 * 1024


def kind_of(path):
    parts = Path(path).parts
    for i, part in enumerate(parts):
        if part == "GIFs":
            return "gif"
        if part == "Colagens":
            return "colagem"
        if part == "Frames" and i + 1 < len(parts):
            return FRAME_FOLDERS.get(parts[i + 1], "outros")
    return "outros"


def group_files(files):
    groups = {}
    for f in files:
        groups.setdefault(kind_of(f), []).append(Path(f))
    return {kind: sorted(groups[kind]) for kind, _x in KIND_ORDER if kind in groups}


def thumbnail(path, width=300):
    """PNG pequeno do último quadro (o ano mais recente), ou None."""
    try:
        from PIL import Image
        with Image.open(path) as image:
            frames = getattr(image, "n_frames", 1)
            if frames > 1:
                image.seek(frames - 1)
            frame = image.convert("RGB")
        frame.thumbnail((width, width * 2))
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return None


def colab_files():
    try:
        from google.colab import files
        return files
    except ImportError:
        return None


def zip_files(files, target, base=None):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w") as archive:
        for f in files:
            f = Path(f)
            if not f.is_file():
                continue
            try:
                name = f.relative_to(base).as_posix() if base else f.name
            except ValueError:
                name = f.name
            archive.write(f, name)
    return target


def scan_library(root):
    """Combinações já gravadas em `root`, lidas dos metadados (mais recentes primeiro)."""
    root = Path(root)
    entries = []
    if not root.is_dir():
        return entries
    for meta in root.rglob("metadados/*.json"):
        folder = meta.parent.parent
        manifest = load_manifest(folder, meta.stem)
        if not manifest:
            continue
        files = [folder / r["path"] for records in manifest["artifacts"].values() for r in records]
        files = [f for f in files if f.is_file()]
        if not files:
            continue
        layer, territory = manifest.get("layer") or {}, manifest.get("territory") or {}
        variant = "" if meta.stem == "completo" else f" · {meta.stem}"
        entries.append({"key": str(meta), "title": layer.get("name") or folder.parent.name,
                        "subtitle": (territory.get("name") or folder.name) + variant,
                        "folder": folder, "files": files, "time": meta.stat().st_mtime})
    return sorted(entries, key=lambda e: -e["time"])


class Card:
    def __init__(self, gallery, key, title, subtitle):
        self.gallery, self.key = gallery, key
        self.state, self.files, self.folder = "fila", [], None
        self.thumb = w.HTML(self._placeholder("⏳"), layout=w.Layout(width="100%"))
        self.title = w.HTML(f"<div style='font-weight:600;font-size:13px;line-height:1.25'>{escape(title)}</div>"
                            f"<div style='font-size:12px;color:var(--muted)'>{escape(subtitle)}</div>")
        self.status = w.HTML()
        self.bar = w.IntProgress(value=0, min=0, max=1, style={"description_width": "0px"},
                                 layout=w.Layout(width="auto", height="12px", margin="2px 0", display="none"))
        self.open = w.ToggleButton(description=_("Abrir"), icon="folder-open", disabled=True,
                                   layout=w.Layout(width="110px"))
        self.details = w.VBox(layout=w.Layout(display="none"))
        self.open.observe(self._toggle, names="value")
        self.widget = w.VBox([self.thumb, self.title, self.status, self.bar, self.open],
                             layout=w.Layout(padding="8px", margin="0", overflow="hidden", min_width="0"))
        self.widget.add_class("gf-card")
        self.update("fila")

    @staticmethod
    def _placeholder(symbol):
        return (f"<div style='height:120px;display:flex;align-items:center;justify-content:center;"
                f"background:var(--sel);color:var(--muted);font-size:28px'>{symbol}</div>")

    def update(self, state, detail="", done=0, total=0):
        self.state = state
        label, color, background = STATES[state]
        label = _(label)
        extra = f" · {escape(_(detail))}" if detail else ""
        self.status.value = (f"<div style='font-size:12px'><span style='background:{background};color:{color};"
                             f"border-radius:8px;padding:1px 8px;font-weight:600'>{label}</span>{extra}</div>")
        downloading = state == "gerando" and total > 0
        self.bar.layout.display = "" if downloading else "none"
        if downloading:
            self.bar.max = max(1, total)
            self.bar.value = min(done, self.bar.max)
        if state == "gerando" and not self.files:
            self.thumb.value = self._placeholder("🌱")
        if state == "falha" and not self.files:
            self.thumb.value = self._placeholder("✕")
        self.gallery.changed()

    def set_files(self, files, folder=None):
        self.files = [Path(f) for f in files]
        self.folder = Path(folder) if folder else None
        self.open.disabled = not self.files
        groups = group_files(self.files)
        first = next((files[0] for files in groups.values() if files), None)
        data = thumbnail(first) if first else None
        if data:
            self.thumb = w.Image(value=data, format="png", layout=w.Layout(width="100%", object_fit="contain",
                                                                          max_height="220px"))
            self.widget.children = (self.thumb,) + tuple(self.widget.children[1:])
        if self.open.value:
            self._build_details()

    def _toggle(self, change):
        if change["new"]:
            self._build_details()
            self.gallery.show_details(self)
        else:
            self.gallery.show_details(None)

    def _build_details(self):
        rows = []
        groups = group_files(self.files)
        gif = (groups.get("gif") or [None])[0]
        if gif is not None:
            try:
                if gif.stat().st_size <= MAX_INLINE_GIF:
                    rows.append(w.Image(value=gif.read_bytes(), format="gif",
                                        layout=w.Layout(max_width="100%", max_height="520px", object_fit="contain")))
            except OSError:
                pass
        files_api = colab_files()
        if self.folder is not None:
            rows.append(w.HTML(f"<div style='font-size:12px'>{_('Pasta:')} <code style='user-select:all'>"
                               f"{escape(str(self.folder))}</code></div>"))
        if files_api is not None:
            whole = w.Button(description=_("Baixar tudo desta combinação (.zip)"), icon="download",
                             layout=w.Layout(width="300px"))
            whole.on_click(lambda _b: self._download_zip())
            rows.append(whole)
        for kind, label in KIND_ORDER:
            files = groups.get(kind)
            if not files:
                continue
            rows.append(w.HTML(f"<b style='font-size:13px'>{_(label)}</b> <span style='color:var(--muted);font-size:12px'>"
                               f"({len(files)})</span>"))
            for f in files[:60]:
                name = w.HTML(f"<code style='user-select:all;font-size:11px' title='{escape(str(f))}'>"
                              f"{escape(f.name)}</code>", layout=w.Layout(flex="1 1 auto"))
                items = [name]
                if files_api is not None:
                    button = w.Button(icon="download", tooltip=_("Baixar"), layout=w.Layout(width="36px"))
                    button.on_click(lambda _b, f=f: files_api.download(str(f)))
                    items.insert(0, button)
                rows.append(w.HBox(items))
            if len(files) > 60:
                rows.append(w.HTML(f"<i style='font-size:12px'>{_('… e mais {n} (no .zip)', n=len(files) - 60)}</i>"))
        self.details.children = rows

    def _download_zip(self):
        files_api = colab_files()
        base = self.folder or (self.files[0].parent if self.files else None)
        name = "_".join(p for p in (base.parent.name, base.name) if p) if base else "resultado"
        target = zip_files(self.files, self.gallery.zip_root / f"{name}.zip", base)
        if files_api is not None:
            files_api.download(str(target))


class Gallery:
    """Grade de cartões com contagem por estado, filtro e painel de detalhes."""

    def __init__(self, zip_root="/content", empty=None, filters=True):
        self.zip_root = Path(zip_root)
        self.use_filters = filters
        self.lock = RLock()
        self.cards = {}
        self.empty_text = empty or _("Nada por aqui ainda.")
        self.counts = w.HTML()
        self.filter = w.ToggleButtons(options=[(_(label), value) for label, value in FILTERS], value="",
                                      style={"button_width": "auto"})
        self.filter.add_class("gf-tabs")
        self.filter.observe(lambda _c: self._render(), names="value")
        self.download_all = w.Button(description=_("Baixar tudo (.zip)"), icon="download",
                                     layout=w.Layout(display="none", width="190px"))
        self.download_all.on_click(lambda _b: self._download_all())
        self.grid = w.GridBox(layout=w.Layout(grid_template_columns="repeat(auto-fill, minmax(230px, 1fr))",
                                              grid_gap="8px", width="100%"))
        self.detail_title = w.HTML()
        self.detail_box = w.VBox(layout=w.Layout(display="none", border="2px solid var(--green)", padding="10px",
                                                 margin="8px 0"))
        self.widget = w.VBox([w.HBox([self.counts, self.download_all],
                                     layout=w.Layout(justify_content="space-between", align_items="center")),
                              self.filter, self.detail_box, self.grid])
        self._render()

    def reset(self):
        with self.lock:
            self.cards = {}
            self.show_details(None)
            self._render()

    def add(self, key, title, subtitle):
        with self.lock:
            card = Card(self, key, title, subtitle)
            self.cards[key] = card
            self._render()
            return card

    def card(self, key):
        return self.cards.get(key)

    def files(self):
        return [f for card in self.cards.values() for f in card.files]

    def show_details(self, card):
        for other in self.cards.values():
            if other is not card and other.open.value:
                other.open.value = False
        if card is None:
            self.detail_box.layout.display = "none"
            self.detail_box.children = []
            return
        close = w.Button(description=_("Fechar"), icon="times", layout=w.Layout(width="100px"))
        close.on_click(lambda _b: setattr(card.open, "value", False))
        self.detail_box.children = [w.HBox([card.title, close], layout=w.Layout(justify_content="space-between")),
                                    *card.details.children]
        self.detail_box.layout.display = ""

    def changed(self):
        """Atualiza a contagem (chamado pelos cartões a cada mudança de estado)."""
        with self.lock:
            counts = {state: 0 for state in STATES}
            for card in self.cards.values():
                counts[card.state] += 1
            total = len(self.cards)
            done = counts["pronto"] + counts["existe"]
            if not total:
                self.counts.value = f"<i style='color:var(--muted)'>{escape(self.empty_text)}</i>"
            else:
                parts = [f"<b>{_('{feitos} de {total} prontos', feitos=done, total=total)}</b>"]
                for state in ("gerando", "fila", "falha"):
                    if counts[state]:
                        parts.append(f"{counts[state]} {_(STATES[state][0]).lower()}")
                self.counts.value = "<span style='font-size:13px'>" + " · ".join(parts) + "</span>"
            self.download_all.layout.display = "" if (colab_files() and self.files()) else "none"
            if self.filter.value:
                self._filter_grid()

    def _filter_grid(self):
        wanted = self.filter.value
        cards = [c for c in self.cards.values()
                 if not wanted or c.state == wanted or (wanted == "pronto" and c.state == "existe")]
        self.grid.children = [c.widget for c in cards]
        self.filter.layout.display = "" if self.use_filters and len(self.cards) > 1 else "none"

    def _render(self):
        with self.lock:
            self._filter_grid()
            self.changed()

    def _download_all(self):
        import os
        files = [Path(f) for f in self.files() if Path(f).is_file()]
        if not files:
            return
        base = Path(os.path.commonpath([str(f.parent) for f in files]))
        target = zip_files(files, self.zip_root / "mapbiomas_gifs.zip", base)
        files_api = colab_files()
        if files_api is not None:
            files_api.download(str(target))
