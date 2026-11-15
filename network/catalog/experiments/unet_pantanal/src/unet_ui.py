"""Interfaces (ipywidgets) do notebook, uma por etapa, como no catálogo (production / development).

Cada etapa mostra só a escolha que ela usa:

    Passo 1  mostrar_mosaicos(cfg)              lista os meses e as bandas existentes (sem escolher nada)
    Passo 2  ui = run_ui_treino(cfg)            aba "Novo treino": amostras (CSVs), bandas, índices,
             cfg = apply_treino(ui)               ID da versão, shortname e comentário (como o M4)
                                                aba "Versões existentes": reabre um treino já feito
    Passo 4  ui = run_ui_opcoes(cfg)            épocas e batch
             cfg = apply_opcoes(ui)
    Passo 6  ui = run_ui_classificar(cfg)       versão do modelo e períodos de CLASSIFICAÇÃO
             cfg = apply_classificar(ui)

Cada versão (``training_{id}_{shortname}_{sensor}``) tem uma pasta própria em
``{EXPERIMENT}/`` com chips, estatísticas, modelo, relatórios, mapas e o
``config.json`` que permite reabri-la.

A lógica (listar, montar a configuração) fica em funções sem widgets, para
poder ser testada sem Colab.
"""
import re

from unet_config import (FULL_INDICES, INDEX_BANDS, PERIOD_FOLDERS, DEFAULTS, make_config, paths,
                         period_from_file_name, period_info, region_slug, version_name)

# Bandas que o mosaico mensal pode ter (ordem das camadas de entrada).
ALL_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]
VERSION_RE = re.compile(r"^training_(\d{3})_")

# ---------------------------------------------------------------------------
# Lógica (sem widgets)
# ---------------------------------------------------------------------------


def _ls(ls):
    if ls is None:
        import unet_env
        return unet_env.ls
    return ls


def available_periods(cfg, ls=None):
    """Períodos encontrados no GCS: ``{periodo: {"bands": {...}, "samples": [...]}}``.

    - ``bands``: bandas que têm COG exportado naquele período (mosaico).
    - ``samples``: arquivos CSV de amostras cujo nome termina naquele período.
    """
    ls = _ls(ls)
    P = paths(cfg)
    sensor, mosaic = cfg["SENSOR"].lower(), cfg["MOSAIC"].lower()
    out = {}
    base = (f"{cfg['BUCKET']}/{cfg['GCS_PREFIX']}/LIBRARY_IMAGES/{sensor.upper()}/"
            f"{PERIOD_FOLDERS['monthly']}/{mosaic.upper()}")
    for name in ls(base):
        try:
            key = period_info(name)[0]
        except ValueError:
            continue
        files = ls(f"{base}/{name}/COG")
        pat = re.compile(rf"image_.*_{mosaic}_(?P<band>[A-Za-z0-9]+)_{re.escape(key)}_cog\.tif$")
        bands = {m.group("band") for m in (pat.match(f) for f in files) if m}
        out.setdefault(key, {"bands": set(), "samples": []})["bands"] |= bands
    for name in ls(f"{P['bucket']}/{P['samples']}"):
        if name.endswith(".csv"):
            key = period_from_file_name(name)
            if key:
                out.setdefault(key, {"bands": set(), "samples": []})["samples"].append(name)
    return dict(sorted(out.items()))


def train_keys(listing):
    """Meses que servem para treino: têm mosaico E amostras."""
    return [k for k, i in listing.items() if i["bands"] and i["samples"]]


def predict_keys(listing):
    """Meses que podem ser classificados: têm mosaico."""
    return [k for k, i in listing.items() if i["bands"]]


def sample_choices(listing):
    """CSVs que podem treinar (o mês deles tem mosaico): [(arquivo, período)], ordenados."""
    out = []
    for key, info in listing.items():
        if info["bands"]:
            out += [(name, key) for name in info["samples"]]
    return sorted(out)


def _needed(cfg):
    return list(cfg["BANDS"]) + ([cfg["VALID_BAND"]] if cfg.get("VALID_BAND") else [])


def period_label(key, info, bands_needed, with_samples=True):
    """Texto de uma linha da lista de períodos (com marcadores)."""
    have = info["bands"]
    if not have:
        mos = "🛰️ sem mosaico"
    elif set(bands_needed) <= have:
        mos = "🛰️ mosaico ✅"
    else:
        mos = f"🛰️ faltam bandas: {', '.join(sorted(set(bands_needed) - have))} ⚠️"
    if not with_samples:
        return f"{key}   |   {mos}"
    amo = f"📄 {len(info['samples'])} arquivo(s) de amostras" if info["samples"] else "📄 sem amostras"
    return f"{key}   |   {mos}   |   {amo}"


# --- versões --------------------------------------------------------------------------------

def list_versions(cfg, ls=None):
    """Pastas de versão em {EXPERIMENT}/ (training_NNN_...), ordenadas."""
    names = _ls(ls)(f"{cfg['BUCKET']}/{paths(cfg)['experiment']}")
    return sorted(n for n in names if VERSION_RE.match(n))


def next_training_id(versions):
    """Menor ID livre (001, 002, ...), como o M4 do catálogo."""
    used = {int(VERSION_RE.match(v).group(1)) for v in versions if VERSION_RE.match(v)}
    return next(f"{i:03d}" for i in range(1, 1000) if i not in used)


def auto_shortname(cfg, n_channels):
    """'pantanal_r1_10c' = região + número de canais (bandas + índices)."""
    return f"{region_slug(cfg)}_{n_channels}c"


def build_treino(cfg, sample_files, bands, indices, training_id=None, shortname=None, comment=""):
    """Configuração de um NOVO treino (valida antes de devolver).

    Os períodos de treino saem das datas no nome dos CSVs escolhidos.
    """
    if not sample_files:
        raise ValueError("Escolha pelo menos um arquivo de amostras.")
    if not bands:
        raise ValueError("Marque pelo menos uma banda.")
    bad = [i for i in indices if INDEX_BANDS.get(i, set()) - set(bands)]
    if bad:
        need = {i: sorted(INDEX_BANDS[i] - set(bands)) for i in bad}
        raise ValueError(f"Índices sem as bandas necessárias: {need}. Marque as bandas ou desmarque os índices.")
    periods = sorted({period_from_file_name(f) for f in sample_files} - {""})
    if not periods:
        raise ValueError("Os nomes dos CSVs escolhidos não terminam com o período (ex.: ..._2026_10.csv).")
    tid = str(training_id).strip() if training_id else None
    if tid and not tid.isdigit():
        raise ValueError("O ID do treino deve ser um número (ex.: 001).")
    over = dict(cfg)
    over.update(SAMPLE_FILES=sorted(sample_files), PERIODS=periods, PREDICT_PERIODS=periods,
                BANDS=[b for b in ALL_BANDS if b in bands] + [b for b in bands if b not in ALL_BANDS],
                INDICES=[i for i in FULL_INDICES if i in indices],
                TRAINING_ID=f"{int(tid):03d}" if tid else None,
                SHORTNAME=shortname or None, COMMENT=comment or "")
    return make_config(**over)


def config_from_saved(base_cfg, saved):
    """Configuração de uma versão a partir do config.json salvo (chaves desconhecidas são ignoradas)."""
    over = dict(base_cfg)
    over.update({k: v for k, v in saved.items() if k in DEFAULTS})
    if isinstance(over.get("SPLIT"), list):
        over["SPLIT"] = tuple(over["SPLIT"])
    return make_config(**over)


def save_version(cfg):
    """Grava {versão}/config.json (o que permite reabrir a versão depois)."""
    if not cfg.get("TRAINING_ID"):
        return None
    import datetime
    from unet_io import write_json
    data = dict(cfg)
    data["_saved_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_json(cfg, paths(cfg)["config"], data)
    return paths(cfg)["config"]


def load_version(base_cfg, name):
    """Configuração salva de uma versão (pelo nome da pasta)."""
    from unet_io import read_json
    P = paths(base_cfg)
    return config_from_saved(base_cfg, read_json(f"{P['experiment']}/{name}/config.json"))


def build_opcoes(cfg, epochs, batch):
    if int(epochs) < 1 or int(batch) < 1:
        raise ValueError("Épocas e batch precisam ser maiores que zero.")
    over = dict(cfg)
    over.update(EPOCHS=int(epochs), BATCH=int(batch))
    return make_config(**over)


def build_classificar(cfg, predict_periods):
    if not predict_periods:
        raise ValueError("Marque pelo menos um período de CLASSIFICAÇÃO.")
    over = dict(cfg)
    over.update(PREDICT_PERIODS=sorted(predict_periods))
    return make_config(**over)


def mosaic_table(listing, bands=None):
    """Linhas de texto: período x bandas do mosaico (✅ existe / ·  falta)."""
    cols = list(bands or ALL_BANDS) + ["dayOfYear"]
    lines = ["período   " + " ".join(f"{c[:7]:>8}" for c in cols)]
    for key, info in listing.items():
        if info["bands"]:
            lines.append(f"{key:<9} " + " ".join(f"{('✅' if c in info['bands'] else '·'):>7}" for c in cols))
    return lines


def mostrar_mosaicos(cfg, listing=None):
    """Passo 1: mostra quais meses têm mosaico e quais bandas foram exportadas (só leitura)."""
    from unet_resumo import _cab
    listing = listing if listing is not None else available_periods(cfg)
    _cab(f"📋 MOSAICOS DISPONÍVEIS  |  {cfg['SENSOR']} {cfg['MOSAIC']}  |  {cfg['COUNTRY']}")
    print("Para que serve: ver quais meses já têm imagem exportada (M2) e com quais bandas, antes de escolher o treino.\n")
    rows = mosaic_table(listing)
    if len(rows) == 1:
        print("❌ Nenhum mosaico encontrado nessa pasta do GCS. Confira país, sensor e mosaico na configuração.")
        return listing
    print("\n".join("   " + r for r in rows))
    print("\n💡 Quando faltar uma banda, o mês só serve se você desmarcar essa banda no passo 2.")
    return listing


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


def _box(w, title, hint, items):
    return w.VBox([w.HTML(f"<b>{title}</b><br><span style='color:#666'>{hint}</span>"), *items],
                  layout=w.Layout(border="1px solid #ddd", padding="8px", margin="4px 0"))


class _Base:
    def show(self):
        from IPython.display import display
        display(self.root)
        return self


class TreinoUI(_Base):
    """Passo 2: aba "Novo treino" (amostras, bandas, índices, versão) e aba "Versões existentes"."""

    def __init__(self, cfg, listing=None, versions=None):
        import ipywidgets as w
        self.w, self.cfg = w, cfg
        L = w.Layout
        self.listing = listing if listing is not None else available_periods(cfg)
        self.versions = versions if versions is not None else list_versions(cfg)
        self.choices = sample_choices(self.listing)
        self.period_of = dict(self.choices)
        pre = set(cfg.get("SAMPLE_FILES") or [])
        if not pre:  # sem escolha anterior: arquivos dos períodos da configuração
            pre = {n for n, p in self.choices if p in cfg["PERIODS"]}
        self.selected_samples = {n for n in pre if n in self.period_of}

        # --- amostras: dois painéis (disponíveis / selecionadas), como o M4 da development ---
        self.txt_search = w.Text(placeholder="filtrar amostras (ex.: r1, 2026_10)", layout=L(width="260px"))
        self.txt_search.observe(lambda _: self._refresh(), names="value")
        b_all = w.Button(description="Todas", icon="check-square", button_style="info", layout=L(width="90px"))
        b_none = w.Button(description="Limpar", icon="square-o", button_style="warning", layout=L(width="90px"))
        b_all.on_click(lambda _: self._select_all())
        b_none.on_click(lambda _: self._clear())
        self.pane_avail = w.VBox(layout=L(border="1px solid #ddd", height="170px", overflow_y="auto"))
        self.pane_sel = w.VBox(layout=L(border="1px solid #ddd", height="170px", overflow_y="auto"))
        self.lbl_periods = w.HTML()
        dual = w.HBox([w.VBox([w.HTML("<b>Disponíveis</b> (clique para incluir)"), self.pane_avail], layout=L(flex="1")),
                       w.VBox([w.HTML("<b>Selecionadas ✅</b> (clique para tirar)"), self.pane_sel], layout=L(flex="1"))])
        samples_box = _box(w, "📄 Amostras de treino",
                           "só aparecem CSVs cujo mês tem mosaico; os períodos de treino saem do nome dos arquivos",
                           [w.HBox([self.txt_search, b_all, b_none]), dual, self.lbl_periods])

        # --- bandas e índices ---
        self.chk_bands = [w.Checkbox(value=b in cfg["BANDS"], description=b, indent=False, layout=L(width="95px"))
                          for b in ALL_BANDS]
        self.chk_idx = [w.Checkbox(value=i in cfg["INDICES"], description=i, indent=False, layout=L(width="95px"))
                        for i in FULL_INDICES]
        for c in self.chk_bands + self.chk_idx:
            c.observe(lambda _: self._refresh(), names="value")

        # --- versão (destino), como o M4: ID sugerido + shortname + comentário ---
        self.w_id = w.Text(value=next_training_id(self.versions), description="ID do treino:",
                           style={"description_width": "110px"}, layout=L(width="230px"))
        self.w_short = w.Text(value="", description="Shortname:", layout=L(width="330px"))
        self.w_comment = w.Textarea(placeholder="comentário (o que muda nesta versão)", layout=L(width="98%", height="50px"))
        self.lbl_version = w.HTML()
        for c in (self.w_id, self.w_short):
            c.observe(lambda _: self._refresh_version(), names="value")
        self._auto_short = True
        self.w_short.observe(self._on_short_typed, names="value")
        version_box = _box(w, "🏷️ Versão (destino no GCS)",
                           "cada versão tem a sua pasta: chips, modelo, métricas e mapas ficam separados",
                           [w.HBox([self.w_id, self.w_short]), self.w_comment, self.lbl_version])

        new_tab = w.VBox([
            samples_box,
            _box(w, "🛰️ Bandas", "só as que existem no mosaico (veja o passo 1)",
                 [w.HBox(self.chk_bands, layout=L(flex_flow="row wrap"))]),
            _box(w, "🧮 Índices espectrais", "calculados das bandas; precisam das bandas de cada fórmula",
                 [w.HBox(self.chk_idx, layout=L(flex_flow="row wrap"))]),
            version_box])

        # --- aba de versões existentes ---
        opts = self.versions or ["(nenhuma versão ainda)"]
        self.dd_version = w.Dropdown(options=opts, description="Versão:", layout=L(width="520px"),
                                     disabled=not self.versions)
        old_tab = _box(w, "📚 Versões existentes",
                       "reabre um treino já feito (amostras, bandas e modelo salvos) para avaliar, classificar ou comparar",
                       [self.dd_version])

        self.tabs = w.Tab(children=[new_tab, old_tab])
        self.tabs.set_title(0, "Novo treino")
        self.tabs.set_title(1, "Versões existentes")
        self.root = w.VBox([self.tabs])
        self._refresh()

    # --- painéis ---
    def _btn(self, name, add):
        w = self.w
        b = w.Button(description=f"{name}", tooltip=name, layout=w.Layout(width="100%", min_height="26px"),
                     style={"button_color": "#f8f9fa" if add else "#e3f2fd"})
        b.on_click(lambda _: (self.selected_samples.add(name) if add else self.selected_samples.discard(name),
                              self._refresh()))
        return b

    def _filtered(self):
        q = (self.txt_search.value or "").lower()
        return [n for n, _ in self.choices if q in n.lower()]

    def _select_all(self):
        self.selected_samples |= set(self._filtered())
        self._refresh()

    def _clear(self):
        self.selected_samples.clear()
        self._refresh()

    def _on_short_typed(self, change):
        if change["new"] != getattr(self, "_last_auto", None):
            self._auto_short = False

    def _n_channels(self):
        return sum(c.value for c in self.chk_bands) + sum(c.value for c in self.chk_idx)

    def _refresh(self):
        avail = [n for n in self._filtered() if n not in self.selected_samples]
        self.pane_avail.children = [self._btn(n, True) for n in avail]
        self.pane_sel.children = [self._btn(n, False) for n in sorted(self.selected_samples)]
        periods = sorted({self.period_of[n] for n in self.selected_samples})
        self.lbl_periods.value = (f"Períodos de treino: <b>{', '.join(periods)}</b>" if periods
                                  else "<span style='color:#b00'>Nenhuma amostra selecionada.</span>")
        if self._auto_short:
            self._last_auto = auto_shortname(self.cfg, self._n_channels())
            self.w_short.value = self._last_auto
        self._refresh_version()

    def _refresh_version(self):
        tid = (self.w_id.value or "").strip()
        if not tid.isdigit():
            self.lbl_version.value = "<span style='color:#b00'>❌ O ID deve ser um número (ex.: 001).</span>"
            return
        name = version_name({**self.cfg, "TRAINING_ID": tid, "SHORTNAME": self.w_short.value})
        clash = [v for v in self.versions if v.startswith(f"training_{int(tid):03d}_")]
        warn = (f" <span style='color:#b60'>⚠️ o ID {int(tid):03d} já é usado por {', '.join(clash)}</span>"
                if clash else "")
        self.lbl_version.value = f"Pasta: <code>{paths(self.cfg)['experiment']}/{name}/</code>{warn}"

    # --- leitura ---
    @property
    def mode(self):
        return "existente" if self.tabs.selected_index == 1 else "novo"

    def selected(self):
        return dict(sample_files=sorted(self.selected_samples),
                    bands=[c.description for c in self.chk_bands if c.value],
                    indices=[c.description for c in self.chk_idx if c.value],
                    training_id=self.w_id.value, shortname=self.w_short.value, comment=self.w_comment.value)


class OpcoesUI(_Base):
    """Passo 4: épocas e batch."""

    def __init__(self, cfg):
        import ipywidgets as w
        self.w, self.cfg = w, cfg
        L = w.Layout
        self.w_epochs = w.IntText(value=cfg["EPOCHS"], description="Épocas:", layout=L(width="170px"))
        self.w_batch = w.IntText(value=cfg["BATCH"], description="Batch:", layout=L(width="170px"))
        self.root = _box(w, "⚙️ Treino", "mais épocas = mais tempo; batch 8 cabe na GPU T4 do Colab",
                         [w.HBox([self.w_epochs, self.w_batch])])

    def selected(self):
        return dict(epochs=self.w_epochs.value, batch=self.w_batch.value)


class ClassificarUI(_Base):
    """Passo 6: versão do modelo e períodos de classificação (só precisam do mosaico)."""

    def __init__(self, cfg, listing=None, versions=None, loader=None):
        import ipywidgets as w
        self.w, self.cfg = w, cfg
        L = w.Layout
        self.listing = listing if listing is not None else available_periods(cfg)
        self.versions = versions if versions is not None else list_versions(cfg)
        self.loader = loader or (lambda name: load_version(cfg, name))
        self.keys = predict_keys(self.listing)
        self.cfg0 = cfg
        current = cfg["MODEL_NAME"] if cfg.get("TRAINING_ID") else "(configuração atual)"
        opts = list(self.versions)
        if current not in opts:
            opts = [current] + opts
        self.dd_version = w.Dropdown(options=opts, value=current, description="Modelo:",
                                     layout=L(width="520px"), disabled=len(opts) < 2)
        self.dd_version.observe(lambda _: self._on_version(), names="value")
        self.chk_pred = [w.Checkbox(value=k in cfg["PREDICT_PERIODS"], indent=False, layout=L(width="auto"))
                         for k in self.keys]
        self.root = w.VBox([
            _box(w, "🏷️ Versão do modelo", "o treino (pasta training_...) cujo modelo vai classificar", [self.dd_version]),
            _box(w, "🗺️ Períodos de CLASSIFICAÇÃO",
                 "meses com mosaico; precisam ter as bandas do modelo escolhido", self.chk_pred or
                 [w.HTML("❌ Nenhum mês tem mosaico no GCS.")])])
        self._on_version()

    def _on_version(self):
        name = self.dd_version.value
        if name in self.versions and name != self.cfg0.get("MODEL_NAME"):
            self.cfg = self.loader(name)
        else:
            self.cfg = self.cfg0
        need = _needed(self.cfg)
        for k, c in zip(self.keys, self.chk_pred):
            c.description = period_label(k, self.listing[k], need, with_samples=False)

    def selected(self):
        return dict(predict_periods=[k for k, c in zip(self.keys, self.chk_pred) if c.value])


def run_ui_treino(cfg, listing=None, versions=None):
    return TreinoUI(cfg, listing, versions).show()


def run_ui_opcoes(cfg):
    return OpcoesUI(cfg).show()


def run_ui_classificar(cfg, listing=None, versions=None):
    return ClassificarUI(cfg, listing, versions).show()


def apply_treino(ui, save=True):
    """Novo treino: monta e grava a versão. Versão existente: reabre o config.json salvo."""
    from unet_resumo import print_config
    if ui.mode == "existente":
        if not ui.versions:
            raise ValueError("Ainda não há versões salvas; use a aba 'Novo treino'.")
        cfg = load_version(ui.cfg, ui.dd_version.value)
        print(f"📚 Versão reaberta: {cfg['MODEL_NAME']} (os passos 3 a 7 usam os arquivos dela).")
    else:
        cfg = build_treino(ui.cfg, **ui.selected())
        if save:
            save_version(cfg)
    print_config(cfg)
    return cfg


def apply_opcoes(ui, save=True):
    cfg = build_opcoes(ui.cfg, **ui.selected())
    if save:
        save_version(cfg)
    print(f"⚙️ Treino: {cfg['EPOCHS']} épocas, batch {cfg['BATCH']}.")
    return cfg


def apply_classificar(ui):
    cfg = build_classificar(ui.cfg, **ui.selected())
    print(f"🗺️ Modelo {cfg['MODEL_NAME']} | períodos a classificar: {', '.join(cfg['PREDICT_PERIODS'])}")
    return cfg
