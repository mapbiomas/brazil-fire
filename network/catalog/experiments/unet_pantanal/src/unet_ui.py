"""Interfaces (ipywidgets) do notebook, uma por etapa, como no catálogo de produção.

Cada etapa mostra só a escolha que ela usa:

    Passo 1  mostrar_mosaicos(cfg)                    lista os meses e as bandas existentes (sem escolher nada)
    Passo 2  ui = run_ui_treino(cfg)                  períodos de TREINO, bandas e índices
             cfg = apply_treino(ui)
    Passo 4  ui = run_ui_opcoes(cfg)                  épocas e batch
             cfg = apply_opcoes(ui)
    Passo 6  ui = run_ui_classificar(cfg)             períodos de CLASSIFICAÇÃO
             cfg = apply_classificar(ui)

A lógica (listar períodos, montar a configuração) fica em funções sem widgets,
para poder ser testada sem Colab.
"""
import re

from unet_config import (FULL_INDICES, INDEX_BANDS, PERIOD_FOLDERS, make_config,
                         period_from_file_name, period_info)

# Bandas que o mosaico mensal pode ter (ordem das camadas de entrada).
ALL_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]

# ---------------------------------------------------------------------------
# Lógica (sem widgets)
# ---------------------------------------------------------------------------


def available_periods(cfg, ls=None):
    """Períodos encontrados no GCS: ``{periodo: {"bands": {...}, "samples": [...]}}``.

    - ``bands``: bandas que têm COG exportado naquele período (mosaico).
    - ``samples``: arquivos CSV de amostras cujo nome termina naquele período.
    """
    if ls is None:
        import unet_env
        ls = unet_env.ls
    from unet_config import paths
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


def build_treino(cfg, train_periods, bands, indices):
    """Configuração com períodos de treino, bandas e índices escolhidos (valida antes de devolver)."""
    if not train_periods:
        raise ValueError("Marque pelo menos um período de TREINO.")
    if not bands:
        raise ValueError("Marque pelo menos uma banda.")
    bad = [i for i in indices if INDEX_BANDS.get(i, set()) - set(bands)]
    if bad:
        need = {i: sorted(INDEX_BANDS[i] - set(bands)) for i in bad}
        raise ValueError(f"Índices sem as bandas necessárias: {need}. Marque as bandas ou desmarque os índices.")
    over = dict(cfg)
    over.update(PERIODS=sorted(train_periods),
                BANDS=[b for b in ALL_BANDS if b in bands] + [b for b in bands if b not in ALL_BANDS],
                INDICES=[i for i in FULL_INDICES if i in indices])
    return make_config(**over)


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


class _Base:
    def __init__(self, cfg, listing=None):
        import ipywidgets as widgets
        self.w = widgets
        self.cfg = cfg
        self.listing = listing if listing is not None else available_periods(cfg)
        self.out = widgets.Output()

    def _box(self, title, hint, items):
        w = self.w
        return w.VBox([w.HTML(f"<b>{title}</b><br><span style='color:#666'>{hint}</span>"), *items],
                      layout=w.Layout(border="1px solid #ddd", padding="8px", margin="4px 0"))

    def _period_checks(self, selected, with_samples):
        w = self.w
        keys = train_keys(self.listing) if with_samples else predict_keys(self.listing)
        self.keys = keys
        need = _needed(self.cfg)
        return [w.Checkbox(value=k in selected,
                           description=period_label(k, self.listing.get(k, {"bands": set(), "samples": []}),
                                                    need, with_samples),
                           indent=False, layout=w.Layout(width="auto")) for k in keys]

    def _picked(self, checks):
        return [k for k, c in zip(self.keys, checks) if c.value]

    def show(self):
        from IPython.display import display
        display(self.root)
        return self


class TreinoUI(_Base):
    """Passo 2: períodos de treino (com mosaico e amostras), bandas e índices."""

    def __init__(self, cfg, listing=None):
        super().__init__(cfg, listing)
        w = self.w
        L = w.Layout
        self.chk_train = self._period_checks(cfg["PERIODS"], with_samples=True)
        if not self.chk_train:
            self.chk_train_view = [w.HTML("❌ Nenhum mês tem mosaico <b>e</b> amostras no GCS ainda.")]
        else:
            self.chk_train_view = self.chk_train
        self.chk_bands = [w.Checkbox(value=b in cfg["BANDS"], description=b, indent=False, layout=L(width="95px"))
                          for b in ALL_BANDS]
        self.chk_idx = [w.Checkbox(value=i in cfg["INDICES"], description=i, indent=False, layout=L(width="95px"))
                        for i in FULL_INDICES]
        self.root = w.VBox([
            self._box("🏫 Períodos de TREINO",
                      "só aparecem os meses com mosaico <b>e</b> amostras; os chips de todos treinam um só modelo",
                      self.chk_train_view),
            self._box("🛰️ Bandas", "só as que existem no mosaico (veja os marcadores acima)",
                      [w.HBox(self.chk_bands, layout=L(flex_flow="row wrap"))]),
            self._box("🧮 Índices espectrais", "calculados das bandas; precisam das bandas de cada fórmula",
                      [w.HBox(self.chk_idx, layout=L(flex_flow="row wrap"))]),
            self.out])

    def selected(self):
        return dict(train_periods=self._picked(self.chk_train),
                    bands=[c.description for c in self.chk_bands if c.value],
                    indices=[c.description for c in self.chk_idx if c.value])


class OpcoesUI(_Base):
    """Passo 4: épocas e batch."""

    def __init__(self, cfg, listing=None):
        import ipywidgets as widgets
        self.w = widgets
        self.cfg = cfg
        L = widgets.Layout
        self.w_epochs = widgets.IntText(value=cfg["EPOCHS"], description="Épocas:", layout=L(width="170px"))
        self.w_batch = widgets.IntText(value=cfg["BATCH"], description="Batch:", layout=L(width="170px"))
        self.root = self._box("⚙️ Treino", "mais épocas = mais tempo; batch 8 cabe na GPU T4 do Colab",
                              [widgets.HBox([self.w_epochs, self.w_batch])])

    def selected(self):
        return dict(epochs=self.w_epochs.value, batch=self.w_batch.value)


class ClassificarUI(_Base):
    """Passo 6: períodos de classificação (só precisam do mosaico)."""

    def __init__(self, cfg, listing=None):
        super().__init__(cfg, listing)
        self.chk_pred = self._period_checks(cfg["PREDICT_PERIODS"], with_samples=False)
        self.root = self._box("🗺️ Períodos de CLASSIFICAÇÃO",
                              "meses a mapear com o modelo treinado; só precisam do mosaico com as bandas do modelo",
                              self.chk_pred)

    def selected(self):
        return dict(predict_periods=self._picked(self.chk_pred))


def run_ui_treino(cfg, listing=None):
    return TreinoUI(cfg, listing).show()


def run_ui_opcoes(cfg):
    return OpcoesUI(cfg).show()


def run_ui_classificar(cfg, listing=None):
    return ClassificarUI(cfg, listing).show()


def apply_treino(ui):
    cfg = build_treino(ui.cfg, **ui.selected())
    from unet_resumo import print_config
    print_config(cfg)
    return cfg


def apply_opcoes(ui):
    cfg = build_opcoes(ui.cfg, **ui.selected())
    print(f"⚙️ Treino: {cfg['EPOCHS']} épocas, batch {cfg['BATCH']}.")
    return cfg


def apply_classificar(ui):
    cfg = build_classificar(ui.cfg, **ui.selected())
    print(f"🗺️ Períodos a classificar: {', '.join(cfg['PREDICT_PERIODS'])}")
    return cfg
