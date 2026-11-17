"""
Notebook em três blocos, como o catálogo de produção (um módulo por fase, cada um com abas):

    APP = abrir_config(UCFG)       1. Configuração: país -> bioma -> região -> treino (novo ou existente)
    abrir_treino(APP)              2. Treino: amostras e versão | parâmetros | treinar | métricas
    abrir_classificacao(APP)       3. Classificação: modelo e meses | classificar | juntar | publicar no GEE

O treino e a classificação não dividem widgets: a classificação parte de um treino já feito e tem a sua própria
configuração (`Estado.cfg_class`). Os botões rodam as etapas dentro das abas (saída na própria aba).

A lógica sem widgets (montar parâmetros, cartões de métricas) fica em funções puras, testáveis sem Colab.
"""
import copy
import os
import traceback

from unet_config import biome_slug, make_config, paths
import unet_ui as ui

# ---------------------------------------------------------------------------
# Lógica (sem widgets)
# ---------------------------------------------------------------------------

COUNTRIES = [("Brasil", "brazil")]  # os caminhos do GCS/GEE de outros países ainda não estão ligados aqui
FILTER_PRESETS = {
    "leve (16-32-64-128)": [16, 32, 64, 128],
    "padrão (32-64-128-256)": [32, 64, 128, 256],
    "grande (64-128-256-512)": [64, 128, 256, 512],
}
SPLIT_PRESETS = {"70 / 15 / 15": (0.70, 0.15, 0.15), "80 / 10 / 10": (0.80, 0.10, 0.10),
                 "60 / 20 / 20": (0.60, 0.20, 0.20)}
# Parâmetros que mudam os chips: se mudarem, os chips têm de ser refeitos.
CHIP_KEYS = ("CHIP", "POINTS_PER_POLYGON", "BLOCK_KM", "SPLIT", "SEED")


def split_region(name):
    """('Pantanal_r1') -> ('Pantanal', 'Pantanal_r1'): o bioma é o texto antes de '_r{id}'."""
    return (name.rsplit("_r", 1)[0] if "_r" in name else name), name


def regions_by_biome(names):
    out = {}
    for n in sorted(set(names)):
        b, _ = split_region(n)
        out.setdefault(b, []).append(n)
    return out


def list_regions(cfg):
    """Nomes de região do asset de regiões (precisa do Earth Engine já iniciado)."""
    from unet_io import regions_collection
    return sorted(set(map(str, regions_collection(cfg).aggregate_array(cfg["REGION_PROPERTY"]).getInfo())))


def build_params(cfg, **v):
    """Aplica os parâmetros do passo 'Parâmetros' e devolve (nova config, chaves que obrigam refazer os chips)."""
    for k in ("EPOCHS", "BATCH", "PATIENCE", "CHIP", "BLOCK_KM", "POINTS_PER_POLYGON", "INFER_TILE", "INFER_BATCH"):
        if int(v[k]) < 1:
            raise ValueError(f"{k} precisa ser maior que zero.")
    over = dict(cfg)
    filters = list(v["FILTERS"])
    over.update(
        EPOCHS=int(v["EPOCHS"]), BATCH=int(v["BATCH"]), LR=float(v["LR"]), PATIENCE=int(v["PATIENCE"]),
        FILTERS=filters, BOTTOM_FILTERS=filters[-1] * 2, MIXED_PRECISION=bool(v["MIXED_PRECISION"]),
        CHIP=int(v["CHIP"]), POINTS_PER_POLYGON=int(v["POINTS_PER_POLYGON"]), BLOCK_KM=int(v["BLOCK_KM"]),
        SPLIT=tuple(v["SPLIT"]), SEED=int(v["SEED"]),
        OVERLAP=int(v["OVERLAP"]), INFER_TILE=int(v["INFER_TILE"]), INFER_BATCH=int(v["INFER_BATCH"]))
    new = make_config(**over)  # valida (múltiplos de 16, soma do SPLIT...)
    norm = lambda x: [str(i) for i in (x if isinstance(x, (list, tuple)) else [x])]  # noqa: E731
    changed = [k for k in CHIP_KEYS if norm(cfg[k]) != norm(new[k])]
    return new, changed


def _pct(x):
    return f"{100 * x:.0f}%"


def cards_html(res):
    """Cartões de métricas em linguagem simples, a partir de um resumo de `evaluate_unet` (um período ou 'all')."""
    t = res["test_at_best"]
    thr = res["val_best_threshold"]
    t5 = res["test_at_0_5"]

    def card(titulo, valor, texto, cor):
        return (f"<div style='flex:1;min-width:190px;border:1px solid #ddd;border-left:6px solid {cor};"
                f"border-radius:6px;padding:8px 10px;margin:4px'>"
                f"<div style='color:#666;font-size:12px'>{titulo}</div>"
                f"<div style='font-size:26px;font-weight:600'>{valor}</div>"
                f"<div style='color:#444;font-size:12px'>{texto}</div></div>")

    tot_fire = t["tp"] + t["fn"]
    cards = [
        card("Fogo encontrado (recall)", _pct(t["recall"]),
             f"de cada 100 pixels que queimaram, o modelo achou {100 * t['recall']:.0f} "
             f"(perdeu {100 * t['omission']:.0f}: omissão)", "#2e7d32"),
        card("Acerto do alarme (precisão)", _pct(t["precision"]),
             f"de cada 100 pixels marcados como fogo, {100 * t['precision']:.0f} eram fogo "
             f"({100 * t['commission']:.0f} eram alarme falso: comissão)", "#1565c0"),
        card("F1", f"{t['f1']:.2f}", "equilíbrio entre os dois (1 = perfeito)", "#6a1b9a"),
        card("IoU", f"{t['iou']:.2f}", "sobreposição entre o previsto e o real (1 = perfeito)", "#ef6c00"),
        card("Limiar escolhido", f"{thr:.2f}",
             f"probabilidade mínima para contar como fogo; no 0,50 o F1 seria {t5['f1']:.2f}", "#455a64"),
    ]
    nota = (f"<div style='color:#666;font-size:12px;margin:4px'>Nota calculada em pixels de teste (nunca vistos no "
            f"treino); {tot_fire:,} deles são fogo de verdade. O limiar vem da validação.</div>")
    return f"<div style='display:flex;flex-wrap:wrap'>{''.join(cards)}</div>{nota}"


def diagnostico(res):
    """Lista de frases (ícone, texto) que explicam o resultado."""
    t, t5 = res["test_at_best"], res["test_at_0_5"]
    out = []
    if t5["recall"] > 0.95 and t5["precision"] < 0.4:
        out.append(("⚠️", "No limiar 0,50 o modelo marca quase tudo como fogo; por isso o limiar da validação importa."))
    if t["precision"] > 0.9 and t["recall"] < 0.6:
        out.append(("⚠️", f"Modelo cauteloso: acerta o que marca, mas perde {100 * (1 - t['recall']):.0f}% do fogo real."))
    if t["precision"] < 0.5 and t["recall"] > 0.8:
        out.append(("⚠️", "Modelo marca fogo demais (alarmes falsos). Confira água escura, sombras e solo escuro."))
    if t["iou"] < 0.5:
        out.append(("⚠️", f"IoU {t['iou']:.2f} (< 0,5): resultado inicial. Mais meses e amostras são o caminho."))
    elif t["iou"] >= 0.7:
        out.append(("✅", f"IoU {t['iou']:.2f}: bom resultado para esta base. Confira em mais blocos de teste."))
    return out or [("ℹ️", "Resultado intermediário: veja a matriz para saber se o erro é de omissão ou comissão.")]


def pick_summary(results):
    """O resumo 'all' (todos os períodos juntos) ou o único período avaliado."""
    if "all" in results:
        return results["all"]
    return next(iter(results.values()))


# ---------------------------------------------------------------------------
# Gráficos (matplotlib)
# ---------------------------------------------------------------------------

def fig_limiar(res):
    """F1, precisão e recall em função do limiar (validação), com o corte escolhido marcado."""
    import matplotlib.pyplot as plt
    c = res.get("val_curve")
    if not c:
        print("   (resultado salvo sem a curva do limiar: rode 'Calcular métricas' de novo)")
        return
    thr = res["val_best_threshold"]
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(c["threshold"], c["f1"], label="F1", lw=2)
    ax.plot(c["threshold"], c["precision"], label="precisão", ls="--")
    ax.plot(c["threshold"], c["recall"], label="recall", ls=":")
    ax.axvline(thr, color="gray", lw=1)
    ax.annotate(f"limiar {thr:.2f}", (thr, 0.02), rotation=90, va="bottom", ha="right", color="gray")
    ax.set_xlabel("limiar de probabilidade")
    ax.set_ylim(0, 1.02)
    ax.set_title("Como o corte muda os acertos (validação)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


def fig_matriz(res):
    """Matriz de confusão do teste, com a porcentagem por linha."""
    import matplotlib.pyplot as plt
    import numpy as np
    t = res["test_at_best"]
    m = np.array([[t["tp"], t["fn"]], [t["fp"], t["tn"]]], dtype=float)
    frac = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(4.8, 3.8))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    nomes = ["fogo", "não-fogo"]
    ax.set_xticks([0, 1], [f"previsto: {n}" for n in nomes])
    ax.set_yticks([0, 1], [f"real: {n}" for n in nomes])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{100 * frac[i, j]:.0f}%\n{int(m[i, j]):,}", ha="center", va="center",
                    color="white" if frac[i, j] > 0.5 else "black")
    ax.set_title("Matriz de confusão (teste)")
    fig.tight_layout()
    plt.show()


def fig_comparacao(rows):
    """Barras de F1, IoU, precisão e recall de cada modelo (U-Net x MLP) no teste."""
    import matplotlib.pyplot as plt
    metricas = [("f1", "F1"), ("iou", "IoU"), ("precision", "precisão"), ("recall", "recall")]
    nomes = {"A_mlp_bands": "MLP (bandas)", "B_mlp_bands_indices": "MLP (bandas+índices)", "C_unet": "U-Net"}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    n = len(rows)
    for i, (name, r) in enumerate(rows.items()):
        vals = [r["test_at_best"][k] for k, _ in metricas]
        xs = [j + (i - (n - 1) / 2) * 0.8 / n for j in range(len(metricas))]
        ax.bar(xs, vals, width=0.8 / n, label=nomes.get(name, name))
    ax.set_xticks(range(len(metricas)), [m[1] for m in metricas])
    ax.set_ylim(0, 1.02)
    ax.set_title("Mesmos pixels de teste para todos os modelos")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

class Estado:
    """Estado compartilhado entre os blocos (a escolha da Configuração e os resultados já calculados)."""

    def __init__(self, cfg):
        self.base = cfg            # configuração do projeto (UCFG)
        self.cfg = cfg             # configuração do treino em andamento
        self.cfg_class = None      # configuração do modelo usado na classificação
        self.confirmado = False
        self.existente = False     # True = a Configuração escolheu um treino já feito
        self.results = None        # métricas de teste (evaluate_unet)
        self.meta = None


def _run(out, fn, *args, **kw):
    """Roda `fn` mostrando tudo (prints e figuras) dentro do Output da aba; erros viram uma mensagem clara."""
    out.clear_output()
    with out:
        try:
            return fn(*args, **kw)
        except Exception as e:  # noqa: BLE001
            print(f"❌ {type(e).__name__}: {e}")
            if os.environ.get("UNET_DEBUG"):
                traceback.print_exc()
    return None


def _button(w, text, style="primary", icon="play", width="230px"):
    return w.Button(description=text, button_style=style, icon=icon, layout=w.Layout(width=width))


def _bind(w, buttons, out, btn, fn, *args):
    """Liga o botão a `fn`, desabilitando todos os botões da aba enquanto roda (evita clique duplo)."""
    def go(_):
        for b in buttons:
            b.disabled = True
        try:
            _run(out, fn, *args)
        finally:
            for b in buttons:
                b.disabled = False
    btn.on_click(go)


def _aviso(w, texto="Rode a célula de <b>Configuração</b> e clique em <b>Confirmar</b> antes."):
    from IPython.display import display
    display(w.HTML(f"<div style='border:1px solid #e0a800;background:#fff8e1;padding:8px'>⚠️ {texto}</div>"))


class ConfigUI(ui._Base):
    """Bloco 1: país -> bioma -> região -> treino (novo ou existente). Tudo o que vem depois depende disto."""

    def __init__(self, estado, regions=None, versions_fn=None):
        import ipywidgets as w
        self.w, self.estado = w, estado
        self.versions_fn = versions_fn or ui.list_versions
        L = w.Layout
        base = estado.base
        if regions is None:
            try:
                regions = list_regions(base)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Não consegui listar as regiões do GEE ({e}); digite o nome da região.")
                regions = []
        self.by_biome = regions_by_biome(regions)
        self.dd_pais = w.Dropdown(options=COUNTRIES, value=base["COUNTRY"], description="País:",
                                  layout=L(width="260px"), disabled=len(COUNTRIES) < 2)
        self.out = w.Output()
        self.btn = _button(w, "Confirmar configuração", icon="check")
        self.lbl = w.HTML()
        if self.by_biome:
            b0 = split_region(base["REGION"])[0]
            if b0 not in self.by_biome:
                b0 = sorted(self.by_biome)[0]
            self.dd_bioma = w.Dropdown(options=sorted(self.by_biome), value=b0, description="Bioma:",
                                       layout=L(width="260px"))
            self.dd_regiao = w.Dropdown(options=self.by_biome[b0], description="Região:", layout=L(width="260px"))
            if base["REGION"] in self.by_biome[b0]:
                self.dd_regiao.value = base["REGION"]
            self.dd_bioma.observe(self._on_bioma, names="value")
            self.txt_regiao = None
        else:
            self.dd_bioma = None
            self.dd_regiao = None
            self.txt_regiao = w.Text(value=base["REGION"], description="Região:", layout=L(width="300px"))
        self.dd_treino = w.Dropdown(options=[("➕ Novo treino", "")], description="Treino:", layout=L(width="560px"))
        for c in (self.dd_regiao, self.txt_regiao):
            if c is not None:
                c.observe(lambda _: self._refresh_versions(), names="value")
        self._refresh_versions()
        self.btn.on_click(lambda _: self.confirmar())
        self.root = w.VBox([
            w.HTML("<b>1. Escolha onde trabalhar</b><br><span style='color:#666'>país → bioma → região → treino. "
                   "O treino e a classificação só funcionam depois de confirmar.</span>"),
            w.HBox([self.dd_pais] + ([self.dd_bioma, self.dd_regiao] if self.dd_bioma else [self.txt_regiao])),
            self.dd_treino, self.btn, self.lbl, self.out],
            layout=L(border="1px solid #ddd", padding="8px"))

    def region(self):
        return self.dd_regiao.value if self.dd_regiao is not None else self.txt_regiao.value.strip()

    def _on_bioma(self, _):
        self.dd_regiao.options = self.by_biome[self.dd_bioma.value]

    def _cfg_region(self):
        over = dict(self.estado.base)
        over.update(REGION=self.region(), SAMPLE_REGION=self.region(), TRAINING_ID=None, SHORTNAME=None,
                    BIOME=None)
        return make_config(**over)

    def _refresh_versions(self):
        try:
            versions = self.versions_fn(self._cfg_region())
        except Exception:  # noqa: BLE001
            versions = []
        self.dd_treino.options = [("➕ Novo treino", "")] + [(v, v) for v in versions]
        self.dd_treino.value = ""

    def confirmar(self):
        def run():
            cfg = self._cfg_region()
            name = self.dd_treino.value
            if name:
                cfg = ui.load_version(cfg, name)
            self.estado.cfg = cfg
            self.estado.cfg_class = None
            self.estado.existente = bool(name)
            self.estado.confirmado = True
            self.estado.results = self.estado.meta = None
            P = paths(cfg)
            treino = name or "novo treino (as amostras e a versão são escolhidas no bloco Treino)"
            self.lbl.value = (f"✅ <b>{self.dd_pais.label}</b> › <b>{biome_slug(cfg)}</b> › <b>{cfg['REGION']}</b> › "
                              f"<b>{treino}</b><br><span style='color:#666'>pasta no GCS: "
                              f"<code>gs://{P['bucket']}/{P['versions']}/</code></span>")
        _run(self.out, run)


def abrir_config(cfg, regions=None, versions_fn=None):
    """Bloco 1. Devolve o `APP` (estado) que os outros blocos usam."""
    estado = Estado(cfg)
    estado.ui = ConfigUI(estado, regions, versions_fn).show()
    return estado


# --- bloco 2: treino ---------------------------------------------------------------------------------------

class ParamsUI(ui._Base):
    """Aba 'Parâmetros' da U-Net: cada campo com uma frase que diz o efeito."""

    def __init__(self, cfg):
        import ipywidgets as w
        self.w = w
        L = w.Layout
        st = {"description_width": "150px"}

        def row(widget, texto):
            return w.HBox([widget, w.HTML(f"<span style='color:#666;font-size:12px'>&nbsp;{texto}</span>")])

        def num(value, desc, **kw):
            return w.IntText(value=value, description=desc, style=st, layout=L(width="250px"), **kw)

        cur_filters = next((k for k, v in FILTER_PRESETS.items() if v == list(cfg["FILTERS"])), None)
        self.w_epochs = num(cfg["EPOCHS"], "Épocas máx.:")
        self.w_batch = w.Dropdown(options=sorted({2, 4, 8, 16, 32, int(cfg["BATCH"])}), value=int(cfg["BATCH"]),
                                  description="Batch:", style=st, layout=L(width="250px"))
        lrs = sorted({1e-3, 5e-4, 1e-4, 5e-5, 1e-5, float(cfg["LR"])}, reverse=True)
        self.w_lr = w.Dropdown(options=[(f"{x:g}", x) for x in lrs], value=float(cfg["LR"]),
                               description="Taxa de aprendizado:", style=st, layout=L(width="250px"))
        self.w_patience = w.IntSlider(value=int(cfg["PATIENCE"]), min=2, max=30, description="Paciência:", style=st,
                                      layout=L(width="320px"))
        self.w_filters = w.Dropdown(options=list(FILTER_PRESETS.items()) if cur_filters else
                                    [(str(cfg["FILTERS"]), list(cfg["FILTERS"]))] + list(FILTER_PRESETS.items()),
                                    value=list(cfg["FILTERS"]), description="Tamanho da rede:", style=st,
                                    layout=L(width="330px"))
        self.w_mixed = w.Checkbox(value=bool(cfg["MIXED_PRECISION"]), description="Precisão mista (mais rápido na T4)",
                                  indent=False)
        self.w_chip = w.Dropdown(options=[128, 256, 512], value=int(cfg["CHIP"]), description="Chip (px):", style=st,
                                 layout=L(width="250px"))
        self.w_points = w.IntSlider(value=int(cfg["POINTS_PER_POLYGON"]), min=1, max=20, description="Pontos/polígono:",
                                    style=st, layout=L(width="320px"))
        self.w_block = w.IntSlider(value=int(cfg["BLOCK_KM"]), min=5, max=100, step=5, description="Bloco (km):",
                                   style=st, layout=L(width="320px"))
        split_opts = list(SPLIT_PRESETS.items())
        if tuple(cfg["SPLIT"]) not in SPLIT_PRESETS.values():
            split_opts.insert(0, (" / ".join(f"{100 * s:.0f}" for s in cfg["SPLIT"]), tuple(cfg["SPLIT"])))
        self.w_split = w.Dropdown(options=split_opts, value=tuple(cfg["SPLIT"]), description="Treino/val./teste (%):",
                                  style=st, layout=L(width="300px"))
        self.w_seed = num(cfg["SEED"], "Semente:")
        self.w_overlap = w.Dropdown(options=sorted({8, 16, 32, int(cfg["OVERLAP"])}), value=int(cfg["OVERLAP"]),
                                    description="Sobreposição (px):", style=st, layout=L(width="250px"))
        self.w_tile = w.Dropdown(options=sorted({256, 512, 1024, int(cfg["INFER_TILE"])}), value=int(cfg["INFER_TILE"]),
                                 description="Janela (px):", style=st, layout=L(width="250px"))
        self.w_ibatch = w.Dropdown(options=sorted({4, 8, 16, 32, int(cfg["INFER_BATCH"])}),
                                   value=int(cfg["INFER_BATCH"]), description="Janelas por lote:", style=st,
                                   layout=L(width="250px"))
        self.root = w.VBox([
            ui._box(w, "🧠 Treino da U-Net", "mais épocas e rede maior = mais tempo; a T4 do Colab aguenta batch 8",
                    [row(self.w_epochs, "teto de épocas; o treino para antes se a validação parar de melhorar"),
                     row(self.w_batch, "chips por passo; menor se faltar memória da GPU"),
                     row(self.w_lr, "passo de aprendizado; 1e-4 é um bom começo"),
                     row(self.w_patience, "épocas sem melhora até parar"),
                     row(self.w_filters, "mais filtros = rede maior e mais lenta"),
                     self.w_mixed]),
            ui._box(w, "🧩 Chips e divisão", "se mudar algo daqui, refaça os chips no passo Treinar",
                    [row(self.w_chip, "tamanho do recorte de treino (256 px = 2,56 km a 10 m)"),
                     row(self.w_points, "pontos sorteados dentro de cada polígono (mais chips por amostra)"),
                     row(self.w_block, "lado dos blocos que separam treino, validação e teste"),
                     row(self.w_split, "proporção de blocos de cada divisão"),
                     row(self.w_seed, "mesma semente = mesma divisão")]),
            ui._box(w, "🗺️ Classificação (janela deslizante)", "só muda o tempo e a memória; o mapa é o mesmo",
                    [row(self.w_tile, "pedaço do mapa gravado por janela (múltiplo de 16)"),
                     row(self.w_overlap, "borda lida a mais em cada lado e descartada, para não ter emenda"),
                     row(self.w_ibatch, "janelas por passo da GPU")])])

    def values(self):
        return dict(EPOCHS=self.w_epochs.value, BATCH=self.w_batch.value, LR=self.w_lr.value,
                    PATIENCE=self.w_patience.value, FILTERS=self.w_filters.value, MIXED_PRECISION=self.w_mixed.value,
                    CHIP=self.w_chip.value, POINTS_PER_POLYGON=self.w_points.value, BLOCK_KM=self.w_block.value,
                    SPLIT=self.w_split.value, SEED=self.w_seed.value, OVERLAP=self.w_overlap.value,
                    INFER_TILE=self.w_tile.value, INFER_BATCH=self.w_ibatch.value)


def _tab_amostras(w, estado):
    """Aba 1: novo treino (amostras, bandas, índices, versão) ou resumo da versão aberta."""
    out = w.Output()
    cfg = estado.cfg
    if estado.existente:
        txt = (f"<b>Versão aberta:</b> <code>{cfg['MODEL_NAME']}</code><br>"
               f"amostras: {', '.join(cfg.get('SAMPLE_FILES') or []) or '(padrão)'}<br>"
               f"bandas: {', '.join(cfg['BANDS'])} | índices: {', '.join(cfg['INDICES']) or '(nenhum)'}<br>"
               f"períodos de treino: {', '.join(cfg['PERIODS'])}<br>"
               "<span style='color:#666'>Para mudar amostras, bandas ou índices, crie um novo treino na Configuração. "
               "Os parâmetros ainda podem ser ajustados na aba seguinte.</span>")
        return w.VBox([w.HTML(txt)]), out
    tui = ui.TreinoUI(cfg)
    btn = _button(w, "Aplicar e salvar versão", icon="save")

    def aplicar():
        new = ui.build_treino(estado.cfg, **tui.selected())
        ui.save_version(new)
        estado.cfg = new
        from unet_resumo import print_config
        print_config(new)

    _bind(w, [btn], out, btn, aplicar)
    return w.VBox([tui.tabs.children[0], btn, out]), out


def _passo_conferir(cfg):
    from unet_io import MosaicReader, list_sample_files, load_samples
    from unet_resumo import print_inventory
    for period in cfg["PERIODS"]:
        reader = MosaicReader(cfg, period)
        try:
            inv = reader.inventory()
            files = list_sample_files(cfg, period)
            samples, report = load_samples(cfg, period, reader.crs)
            print_inventory(cfg, inv, files, report)
        finally:
            reader.close()


def _passo_chips(cfg, refazer):
    from IPython.display import Image, display
    from unet_chips import build_chips
    from unet_io import local_dir
    from unet_resumo import resumo_chips
    for period in cfg["PERIODS"]:
        summary = build_chips(cfg, period, overwrite=refazer)
        resumo_chips(cfg, summary)
        if not summary.get("skipped"):
            png = os.path.join(local_dir("reports"), f"chips_preview_{period}.png")
            if os.path.exists(png):
                display(Image(png))


def _passo_stats(cfg):
    from unet_chips import compute_stats
    from unet_resumo import resumo_estatisticas
    resumo_estatisticas(compute_stats(cfg))


def _passo_treinar(estado, cfg):
    from unet_model import train
    from unet_resumo import figura_treino, resumo_treino
    model, meta = train(cfg)
    estado.meta = meta
    resumo_treino(cfg, meta, model)
    figura_treino(meta)


def _tab_treinar(w, estado):
    out = w.Output()
    chk = w.Checkbox(value=False, indent=False, description="Refazer os chips (marque se mudou chip, bloco ou divisão)")
    b1 = _button(w, "1. Conferir mosaico e amostras", style="info", icon="search")
    b2 = _button(w, "2. Criar chips", icon="th")
    b3 = _button(w, "3. Estatísticas de normalização", icon="calculator")
    b4 = _button(w, "4. Treinar a U-Net (GPU)", style="success", icon="rocket")
    b5 = _button(w, "▶ Fazer 2, 3 e 4", style="warning", icon="forward")
    btns = [b1, b2, b3, b4, b5]
    _bind(w, btns, out, b1, lambda: _passo_conferir(estado.cfg))
    _bind(w, btns, out, b2, lambda: _passo_chips(estado.cfg, chk.value))
    _bind(w, btns, out, b3, lambda: _passo_stats(estado.cfg))
    _bind(w, btns, out, b4, lambda: _passo_treinar(estado, estado.cfg))

    def tudo():
        _passo_chips(estado.cfg, chk.value)
        _passo_stats(estado.cfg)
        _passo_treinar(estado, estado.cfg)
    _bind(w, btns, out, b5, tudo)
    hint = w.HTML("<span style='color:#666'>Use GPU (T4): na CPU o treino é muito lento. Cada etapa grava o resultado "
                  "no GCS, então dá para repetir só a que falhou.</span>")
    return w.VBox([hint, chk, w.HBox([b1, b2, b3]), w.HBox([b4, b5]), out])


def _show_results(estado, results):
    from IPython.display import HTML, display
    res = pick_summary(results)
    display(HTML(cards_html(res)))
    for ico, txt in diagnostico(res):
        print(f"{ico} {txt}")
    fig_limiar(res)
    fig_matriz(res)


def _get_results(estado):
    if estado.results:
        return estado.results
    from unet_io import read_json
    P = paths(estado.cfg)
    estado.results = {"all": read_json(f"{P['reports']}/metrics_all.json")}
    return estado.results


def _passo_metricas(estado):
    from unet_model import load_model
    from unet_report import evaluate_unet
    cfg = estado.cfg
    estado.results = evaluate_unet(cfg, load_model(cfg), logger=print)
    _show_results(estado, estado.results)


def _passo_mostrar(estado):
    _show_results(estado, _get_results(estado))


def _passo_curvas(estado):
    from unet_io import read_json
    from unet_resumo import figura_treino
    meta = estado.meta
    if meta is None:
        P = paths(estado.cfg)
        meta = read_json(f"{P['models']}/{estado.cfg['MODEL_NAME']}_metadata.json")
    figura_treino(meta)


def _passo_mlp(estado):
    from IPython.display import Markdown, display
    from unet_report import compare_with_mlp
    from unet_resumo import resumo_comparacao
    rows, table = compare_with_mlp(estado.cfg, _get_results(estado), logger=print)
    resumo_comparacao(estado.cfg, rows)
    display(Markdown(table))
    fig_comparacao(rows)


def _tab_metricas(w, estado):
    out = w.Output()
    b0 = _button(w, "Ver curvas de aprendizado", style="info", icon="line-chart")
    b1 = _button(w, "Calcular métricas de teste", icon="bar-chart")
    b2 = _button(w, "Ver métricas já salvas", style="info", icon="folder-open")
    b3 = _button(w, "Comparar com a MLP (demora)", style="warning", icon="balance-scale")
    btns = [b0, b1, b2, b3]
    _bind(w, btns, out, b0, _passo_curvas, estado)
    _bind(w, btns, out, b1, _passo_metricas, estado)
    _bind(w, btns, out, b2, _passo_mostrar, estado)
    _bind(w, btns, out, b3, _passo_mlp, estado)
    hint = w.HTML("<span style='color:#666'>O limiar com melhor F1 é escolhido na <b>validação</b>; a nota é dada no "
                  "<b>teste</b>, com pixels que o modelo nunca viu.</span>")
    return w.VBox([hint, w.HBox([b0, b1]), w.HBox([b2, b3]), out])


def abrir_treino(estado):
    """Bloco 2: amostras e versão | parâmetros | treinar | métricas."""
    import ipywidgets as w
    from IPython.display import display
    if not estado.confirmado:
        return _aviso(w)
    estado.cfg_class = None
    amostras, _ = _tab_amostras(w, estado)
    out_params = w.Output()
    pui = ParamsUI(estado.cfg)
    b = _button(w, "Aplicar parâmetros", icon="save")

    def aplicar():
        new, chips = build_params(estado.cfg, **pui.values())
        ui.save_version(new)
        estado.cfg = new
        print(f"⚙️ Parâmetros aplicados: {new['EPOCHS']} épocas, batch {new['BATCH']}, LR {new['LR']:g}, "
              f"rede {new['FILTERS']}.")
        if chips:
            print(f"⚠️ Mudou {', '.join(chips)}: marque 'Refazer os chips' na aba Treinar.")

    _bind(w, [b], out_params, b, aplicar)
    tabs = w.Tab(children=[amostras, w.VBox([pui.root, b, out_params]), _tab_treinar(w, estado),
                           _tab_metricas(w, estado)])
    for i, t in enumerate(["1 Amostras e versão", "2 Parâmetros", "3 Treinar", "4 Métricas"]):
        tabs.set_title(i, t)
    cab = w.HTML(f"<b>Treino</b> · {estado.cfg['REGION']} · "
                 f"{'versão ' + estado.cfg['MODEL_NAME'] if estado.existente else 'novo treino'}")
    display(w.VBox([cab, tabs]))
    return tabs


# --- bloco 3: classificação -----------------------------------------------------------------------------

def _passo_classificar(estado, refazer):
    from unet_io import read_json
    from unet_model import load_model
    from unet_predict import predict_mosaic
    cfg = estado.cfg_class
    if not cfg.get("PREDICT_PERIODS"):
        raise ValueError("Escolha os meses na aba 1 e clique em 'Aplicar modelo e meses'.")
    P = paths(cfg)
    print(f"Modelo: {cfg['MODEL_NAME']}")
    model = load_model(cfg)
    stats = read_json(P["stats"])
    for period in cfg["PREDICT_PERIODS"]:
        res = predict_mosaic(cfg, period, model, stats, logger=print, overwrite=refazer, assemble=False)
        if res.get("skipped"):
            print(f"Mapa de {period}: já existia em {res['gcs']} (pulado)")
        else:
            print(f"✅ Faixas de {period} prontas ({res['strips']} faixas, {res['minutes']} min). "
                  f"Junte na aba 3.")


def _holder_tab(w, estado, build, apply, titulo_btn, hint):
    """Aba com uma lista (que se atualiza) e um botão de ação; `build()` devolve a UI, `apply(ui)` roda."""
    out = w.Output()
    box = w.VBox([])
    state = {}

    def refresh():
        state["ui"] = build()
        box.children = [state["ui"].root]

    def run():
        apply(state["ui"])
        refresh()

    b_ref = _button(w, "Atualizar lista", style="info", icon="refresh", width="170px")
    b_run = _button(w, titulo_btn, icon="play")
    btns = [b_ref, b_run]
    _bind(w, btns, out, b_ref, refresh)
    _bind(w, btns, out, b_run, run)
    try:
        refresh()
    except Exception as e:  # noqa: BLE001
        box.children = [w.HTML(f"⚠️ não foi possível listar: {e}")]
    return w.VBox([w.HTML(f"<span style='color:#666'>{hint}</span>"), box, w.HBox([b_ref, b_run]), out])


def abrir_classificacao(estado):
    """Bloco 3: modelo e meses | classificar | juntar | publicar no GEE (parte de um treino já feito)."""
    import ipywidgets as w
    from IPython.display import display
    if not estado.confirmado:
        return _aviso(w)
    if estado.cfg_class is None:
        estado.cfg_class = estado.cfg
    out1 = w.Output()
    cui = ui.ClassificarUI(estado.cfg_class)
    b1 = _button(w, "Aplicar modelo e meses", icon="check")

    def aplicar():
        estado.cfg_class = ui.build_classificar(cui.cfg, **cui.selected())
        c = estado.cfg_class
        print(f"🗺️ Modelo {c['MODEL_NAME']} | meses a classificar: {', '.join(c['PREDICT_PERIODS'])}")

    _bind(w, [b1], out1, b1, aplicar)
    tab1 = w.VBox([cui.root, b1, out1])

    out2 = w.Output()
    chk = w.Checkbox(value=False, indent=False, description="Classificar de novo, mesmo se o mapa do mês já existe")
    b2 = _button(w, "Classificar (GPU)", style="success", icon="rocket")
    _bind(w, [b2], out2, b2, lambda: _passo_classificar(estado, chk.value))
    tab2 = w.VBox([w.HTML("<span style='color:#666'>Cada faixa pronta é salva no GCS. Se o Colab desconectar, abra "
                          "o bloco de novo, aplique o modelo e rode esta aba: ela continua da faixa seguinte.</span>"),
                   chk, b2, out2])

    tab3 = _holder_tab(w, estado, lambda: ui.JuntarUI(estado.cfg_class),
                       lambda u: ui.apply_juntar(u), "Juntar os meses marcados",
                       "Junta as faixas do mês em um mapa único (COG). Não precisa de GPU.")
    tab4 = _holder_tab(w, estado, lambda: ui.PublicarUI(estado.cfg_class),
                       lambda u: ui.apply_publicar(u), "Subir os meses marcados ao GEE",
                       "Cada mapa vira uma imagem na coleção da versão. A ingestão roda na aba Tasks do GEE.")
    tabs = w.Tab(children=[tab1, tab2, tab3, tab4])
    for i, t in enumerate(["1 Modelo e meses", "2 Classificar", "3 Juntar", "4 Publicar no GEE"]):
        tabs.set_title(i, t)
    cab = w.HTML(f"<b>Classificação</b> · {estado.cfg_class['REGION']} · modelo {estado.cfg_class['MODEL_NAME']}")
    display(w.VBox([cab, tabs]))
    return tabs
