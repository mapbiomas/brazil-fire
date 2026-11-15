"""Interface (ipywidgets) para escolher períodos, bandas e opções antes de rodar a U-Net.

Segue o padrão do catálogo de produção (``run_ui`` → escolher → célula de disparo):

    ui = run_ui(UCFG)          # mostra a interface
    UCFG = apply_ui(ui)        # lê o que foi marcado e devolve a configuração

O que é lógica pura (listar períodos, montar a configuração) fica em funções
sem widgets, para poder ser testada sem Colab.
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
    for periodicity in ("monthly",):
        base = (f"{cfg['BUCKET']}/{cfg['GCS_PREFIX']}/LIBRARY_IMAGES/{sensor.upper()}/"
                f"{PERIOD_FOLDERS[periodicity]}/{mosaic.upper()}")
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


def period_label(key, info, bands_needed):
    """Texto de uma linha da lista de períodos (com marcadores)."""
    have = info["bands"]
    if not have:
        mos = "🛰️ sem mosaico"
    elif set(bands_needed) <= have:
        mos = "🛰️ mosaico ✅"
    else:
        mos = f"🛰️ faltam bandas: {', '.join(sorted(set(bands_needed) - have))} ⚠️"
    amo = f"📄 {len(info['samples'])} arquivo(s) de amostras" if info["samples"] else "📄 sem amostras"
    return f"{key}   |   {mos}   |   {amo}"


def build_config(base_cfg, train_periods, predict_periods, bands, indices, epochs, batch,
                 valid_band_ok=True):
    """Nova configuração a partir da escolha na interface (valida antes de devolver)."""
    if not train_periods:
        raise ValueError("Marque pelo menos um período de TREINO.")
    if not bands:
        raise ValueError("Marque pelo menos uma banda.")
    bad = [i for i in indices if INDEX_BANDS.get(i, set()) - set(bands)]
    if bad:
        need = {i: sorted(INDEX_BANDS[i] - set(bands)) for i in bad}
        raise ValueError(f"Índices sem as bandas necessárias: {need}. Marque as bandas ou desmarque os índices.")
    over = {k: v for k, v in base_cfg.items()}
    over.update(PERIODS=sorted(train_periods), PREDICT_PERIODS=sorted(predict_periods or train_periods),
                BANDS=[b for b in ALL_BANDS if b in bands] + [b for b in bands if b not in ALL_BANDS],
                INDICES=[i for i in FULL_INDICES if i in indices], EPOCHS=int(epochs), BATCH=int(batch))
    return make_config(**over)


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------


class ConfigUI:
    def __init__(self, cfg, listing=None):
        import ipywidgets as widgets
        self.w = widgets
        self.cfg = cfg
        self.listing = listing if listing is not None else available_periods(cfg)
        L = widgets.Layout
        needed = list(cfg["BANDS"]) + ([cfg["VALID_BAND"]] if cfg.get("VALID_BAND") else [])

        keys = list(self.listing) or sorted(set(cfg["PERIODS"]) | set(cfg["PREDICT_PERIODS"]))
        self.listing = {k: self.listing.get(k, {"bands": set(), "samples": []}) for k in keys}

        def checks(selected):
            return [widgets.Checkbox(value=k in selected, description=period_label(k, self.listing[k], needed),
                                     indent=False, layout=L(width="auto")) for k in keys]

        self.chk_train = checks(cfg["PERIODS"])
        self.chk_pred = checks(cfg["PREDICT_PERIODS"])
        self.chk_bands = [widgets.Checkbox(value=b in cfg["BANDS"], description=b, indent=False,
                                           layout=L(width="95px")) for b in ALL_BANDS]
        self.chk_idx = [widgets.Checkbox(value=i in cfg["INDICES"], description=i, indent=False,
                                         layout=L(width="95px")) for i in FULL_INDICES]
        self.w_epochs = widgets.IntText(value=cfg["EPOCHS"], description="Épocas:", layout=L(width="160px"))
        self.w_batch = widgets.IntText(value=cfg["BATCH"], description="Batch:", layout=L(width="160px"))
        self.out = widgets.Output()

        def box(title, hint, items):
            return widgets.VBox([widgets.HTML(f"<b>{title}</b><br><span style='color:#666'>{hint}</span>"),
                                 *items], layout=L(border="1px solid #ddd", padding="8px", margin="4px 0"))

        self.root = widgets.VBox([
            box("🏫 Períodos de TREINO", "meses com mosaico <b>e</b> amostras; os chips de todos treinam um só modelo",
                self.chk_train),
            box("🗺️ Períodos de CLASSIFICAÇÃO", "meses a mapear com o modelo treinado; só precisam do mosaico",
                self.chk_pred),
            box("🛰️ Bandas", "só as que existem no mosaico (veja os marcadores acima)",
                [widgets.HBox(self.chk_bands, layout=L(flex_flow="row wrap"))]),
            box("🧮 Índices espectrais", "calculados das bandas; precisam das bandas de cada fórmula",
                [widgets.HBox(self.chk_idx, layout=L(flex_flow="row wrap"))]),
            box("⚙️ Treino", "valores maiores = mais tempo", [widgets.HBox([self.w_epochs, self.w_batch])]),
            self.out,
        ])

    def selected(self):
        keys = list(self.listing)
        pick = lambda cs: [k for k, c in zip(keys, cs) if c.value]
        return dict(train_periods=pick(self.chk_train), predict_periods=pick(self.chk_pred),
                    bands=[c.description for c in self.chk_bands if c.value],
                    indices=[c.description for c in self.chk_idx if c.value],
                    epochs=self.w_epochs.value, batch=self.w_batch.value)

    def show(self):
        from IPython.display import display
        display(self.root)
        return self


def run_ui(cfg, listing=None):
    """Mostra a interface de escolha (lê no GCS os períodos e bandas existentes)."""
    return ConfigUI(cfg, listing).show()


def apply_ui(ui):
    """Devolve a configuração com o que está marcado na interface e imprime o resumo."""
    cfg = build_config(ui.cfg, **ui.selected())
    from unet_resumo import print_config
    print_config(cfg)
    return cfg
