"""
Resumos didáticos, em português, dos passos do notebook (inventário, chips,
normalização, treino e teste).

Cada função só imprime texto (e, no treino, uma figura); nenhuma calcula algo
novo. O idioma fica concentrado aqui, o que facilita oferecer uma escolha de
idioma depois.
"""
import os

LINHA = "=" * 76

SPLIT_NOME = {"train": "treino", "val": "validação", "test": "teste"}
SPLIT_PARA_QUE = {
    "train": "o modelo aprende com estes",
    "val": "define quando parar o treino e o limiar",
    "test": "nota final, nunca vista no treino",
}


def _cab(titulo):
    print(LINHA)
    print(titulo)
    print(LINHA)


# ---------------------------------------------------------------------------
# Passo 1 - Inventário
# ---------------------------------------------------------------------------

def print_inventory(cfg, inv, sample_files, report):
    """Resumo do inventário: mosaico (de onde vem a imagem) e amostras (o que o modelo aprende)."""
    res_deg = next(iter(inv["bands"].values()))["res"][0]
    geographic = str(inv["crs"]).upper() in ("EPSG:4326", "OGC:CRS84")
    res_m = res_deg * 111320.0 if geographic else res_deg
    _cab(f"INVENTÁRIO  |  período {inv['period']}  |  região {cfg['REGION']}  |  {cfg['SENSOR']} {cfg['MOSAIC']}")
    print("Para que serve: conferir, sem treinar nada, se as imagens (mosaico) e as amostras")
    print("(polígonos de fogo e não-fogo) existem, estão na mesma grade e são suficientes.\n")

    print("1) MOSAICO (as imagens que entram no modelo)")
    print(f"   Grade: {inv['width']:,} x {inv['height']:,} pixels | CRS {inv['crs']} | pixel ≈ {res_m:.1f} m")
    print("   (o mosaico cobre a América do Sul inteira; o código lê só a janela da região)")
    print(f"   {'banda':<11}{'tipo':<8}{'nodata':<9}{'pixel (graus)':<16}arquivo")
    for name, b in inv["bands"].items():
        print(f"   {name:<11}{b['dtype']:<8}{str(b['nodata']):<9}{b['res'][0]:<16.3e}{b['path'].rsplit('/', 1)[-1]}")
    n_feat = len(cfg["BANDS"]) + len(cfg["INDICES"])
    print(f"   -> o modelo recebe {n_feat} canais: bandas {cfg['BANDS']}"
          + (f" + índices {cfg['INDICES']}" if cfg["INDICES"] else " (sem índices)"))
    if cfg.get("VALID_BAND"):
        print(f"   -> pixel válido: onde {inv['valid_band']} > 0; a banda 2 da saída traz o dia do ano da queimada")
    else:
        print("   -> sem dayOfYear: pixel válido = bandas não-nodata e não todas 0; a banda 2 da saída fica vazia")
    res_ok = len({round(b["res"][0], 12) for b in inv["bands"].values()}) == 1
    print(f"   -> todas as bandas na mesma grade: {'SIM' if res_ok else 'NÃO - verifique os COGs'}\n")

    print("2) AMOSTRAS (o que o modelo aprende)")
    print(f"   Arquivo(s): {', '.join(os.path.basename(str(f)) for f in sample_files) or 'nenhum'}")
    print(f"   Região filtrada: {cfg.get('SAMPLE_REGION') or cfg['REGION']} | períodos no arquivo: {report['periods_found']}")
    corr = report["period_corrected_rows"]
    fixed = sum(corr.values()) if isinstance(corr, dict) else corr
    print(f"   Linhas com período corrigido pela regra do M4: {fixed}")
    f, n = report["by_class"]["1"], report["by_class"]["0"]
    total = f["polygons"] + n["polygons"]
    print(f"   {'classe':<22}{'polígonos':>10}{'área (ha)':>13}{'média/polígono (ha)':>22}")
    for label, c in (("fogo (1)", f), ("não-fogo (0)", n)):
        mean = c["area_ha"] / c["polygons"] if c["polygons"] else 0
        print(f"   {label:<22}{c['polygons']:>10}{c['area_ha']:>13,.0f}{mean:>22,.1f}")
    print(f"   {'total':<22}{total:>10}\n")

    print("3) O QUE OBSERVAR")
    notes = []
    if total < 100:
        notes.append(f"Poucos polígonos ({total}): com blocos de {cfg['BLOCK_KM']} km, a validação e o teste "
                     "podem ficar com 1 ou 2 blocos. Acrescente meses em PERIODS, se possível.")
    if n["polygons"] < 10:
        notes.append(f"Só {n['polygons']} polígonos de não-fogo: o modelo pode confundir água, sombra e "
                     "solo escuro com cicatriz de fogo.")
    if f["polygons"] and n["polygons"] and (f["area_ha"] / max(n["area_ha"], 1e-9) < 0.2
                                             or f["area_ha"] / max(n["area_ha"], 1e-9) > 5):
        notes.append("As áreas de fogo e de não-fogo são muito diferentes; os chips são balanceados "
                     "pelo número de centros, não pela área.")
    if not cfg["INDICES"]:
        notes.append("Sem índices espectrais (faltam nir/swir1): a comparação com o MLP terá só as variantes A e C.")
    if not notes:
        notes.append("Nada de especial: pode seguir para a etapa de chips.")
    for t in notes:
        print(f"   - {t}")
    print(LINHA)


# ---------------------------------------------------------------------------
# Passo 2 - Chips
# ---------------------------------------------------------------------------

def resumo_chips(cfg, summary):
    """Passo 2: blocos, centros, chips por divisão, balanço de classes e descartes."""
    if summary.get("skipped"):
        print(f"[CHIPS] {summary['period']}: já existem {len(summary['shards'])} arquivo(s) de chips no GCS "
              "e nada foi refeito.")
        print("        Para refazer (por exemplo, depois de mudar bandas, amostras ou região), "
              "use REFAZER_CHIPS = True.")
        return
    c = summary["chips"]
    b = summary["blocks"]
    cen = summary.get("centers_by_split", {})
    d = summary["dropped"]
    _cab(f"CHIPS  |  período {summary['period']}  |  chip {cfg['CHIP']}x{cfg['CHIP']} px")
    print("O que é um chip: um recorte quadrado da imagem, com o seu mapa de rótulos. A U-Net não vê")
    print("a região inteira de uma vez; ela aprende com muitos recortes pequenos e depois é")
    print("aplicada, janela a janela, sobre toda a região.")
    print(f"Como a divisão é feita: a região é cortada em blocos de {cfg['BLOCK_KM']} km. Cada bloco com amostras")
    print("vai INTEIRO para treino, validação ou teste. Assim, o teste nunca 'espia' lugares que o")
    print("modelo viu no treino (evita vazamento espacial e notas otimistas demais).\n")
    print(f"   {'divisão':<11}{'blocos':>7}{'centros fogo':>14}{'centros não-fogo':>18}{'chips':>8}"
          f"{'px fogo':>10}{'px não-fogo':>13}   para que serve")
    for s in ("train", "val", "test"):
        cc = cen.get(s, {})
        print(f"   {SPLIT_NOME[s]:<11}{b.get(s, 0):>7}{cc.get('1', 0):>14}{cc.get('0', 0):>18}"
              f"{c[s]['chips']:>8}{c[s]['fire_px']:>10,}{c[s]['notfire_px']:>13,}   {SPLIT_PARA_QUE[s]}")

    print("\n   Balanço de classes (pixels rotulados):")
    for s in ("train", "val", "test"):
        f_px, n_px = c[s]["fire_px"], c[s]["notfire_px"]
        tot = f_px + n_px
        if tot:
            razao = (n_px / f_px) if f_px else float("inf")
            txt = f"{100 * f_px / tot:5.1f}% fogo | {100 * n_px / tot:5.1f}% não-fogo"
            txt += f" | {razao:.1f} pixels de não-fogo para cada pixel de fogo" if f_px else " | sem fogo!"
            print(f"   - {SPLIT_NOME[s]:<10}{txt}")
    print("     (O fogo costuma ser a classe rara. Por isso a perda combina BCE + Dice, que não deixa o")
    print("      modelo 'ganhar' prevendo tudo como não-fogo.)")

    print(f"\n   Chips descartados: {d['cross_block']} cruzavam blocos de divisões diferentes | "
          f"{d['few_labels']} com poucos pixels rotulados | {d['nodata']} com muito nodata")
    db = summary.get("dropped_by_split_class")
    if db:
        partes = [f"{SPLIT_NOME[s]}: {db[s]['1']} de fogo, {db[s]['0']} de não-fogo" for s in ("train", "val", "test")]
        print("   Descartados por divisão: " + " | ".join(partes))
    ss = summary.get("split_search") or {}
    if ss.get("min_centers_per_class") is not None:
        print(f"   Sorteio dos blocos: menor número de centros de uma classe numa divisão, depois dos "
              f"descartes = {ss['min_centers_per_class']} (meta: {cfg['MIN_CENTERS_PER_CLASS']})")
    ov = summary["fire_notfire_overlaps"]
    if ov["chips"]:
        print(f"   Polígonos de fogo e não-fogo se sobrepõem em {ov['chips']} chip(s) "
              f"({ov['pixels']} pixels): vale o fogo.")
    print("\n   Rótulos: 1 = fogo, 0 = não-fogo, 255 = sem informação (fora dos polígonos ou nodata).")
    print("   Os pixels 255 são ignorados na perda e nas métricas: o modelo só é cobrado onde há rótulo.")

    print("\nO QUE OBSERVAR")
    avisos = []
    for s in ("val", "test"):
        if c[s]["chips"] == 0:
            cc = cen.get(s, {})
            if b.get(s, 0) == 0:
                motivo = "nenhum bloco foi sorteado para esta divisão (poucos blocos com amostras)"
            elif not cc.get("1") or not cc.get("0"):
                motivo = "os blocos desta divisão têm só uma classe (só fogo ou só não-fogo)"
            else:
                motivo = "todos os chips foram descartados (veja a linha de descartes)"
            efeito = ("O treino não roda sem validação." if s == "val"
                      else "Sem teste não há nota final honesta.")
            avisos.append(f"{SPLIT_NOME[s].upper()} SEM CHIPS: {motivo}. {efeito} "
                          "Opções: acrescentar meses em PERIODS, reduzir BLOCK_KM (ex.: 10) ou "
                          "conferir os polígonos.")
        elif c[s]["fire_px"] == 0 or c[s]["notfire_px"] == 0:
            avisos.append(f"{SPLIT_NOME[s]} tem só uma classe nos pixels rotulados: as métricas dessa "
                          "divisão não são confiáveis.")
    if c["train"]["chips"] < 100:
        avisos.append(f"Só {c['train']['chips']} chips de treino: é pouco para uma U-Net; espere resultado "
                      "instável. Mais meses ou mais amostras ajudam.")
    if c["val"]["chips"] and 0 < c["val"]["fire_px"] < 20000:
        avisos.append(f"A validação tem só {c['val']['fire_px']:,} pixels de fogo "
                      f"(≈ {c['val']['fire_px'] / 100:.0f} ha): a nota de validação oscila bastante.")
    if not avisos:
        avisos.append("Divisões com as duas classes e chips em todas: pode seguir para o passo 3.")
    for a in avisos:
        print(f"   - {a}")
    print("   - Confira a figura abaixo: vermelho = fogo, azul = não-fogo. Cuidado com lagoas "
          "(água escura parece cicatriz), nuvens e sombras.")
    print(LINHA)


# ---------------------------------------------------------------------------
# Passo 3 - Normalização
# ---------------------------------------------------------------------------

def resumo_estatisticas(stats):
    """Passo 3: o que é a normalização e os valores por canal."""
    _cab("NORMALIZAÇÃO  |  estatísticas calculadas só nos chips de treino")
    print("Para que serve: a rede aprende melhor quando todos os canais têm escala parecida.")
    print("Cada valor x vira  (x_cortado − média) / desvio.  'Cortado' = limitado entre o")
    print("percentil 1 e o 99, para que pixels extremos (nuvem, brilho) não distorçam a escala.")
    print("Usar só o treino evita que informação do teste vaze para o modelo.\n")
    print(f"   Pixels válidos de treino lidos: {stats.get('train_pixels_seen', stats['n_pixels']):,} "
          f"(amostra usada: {stats['n_pixels']:,})")
    print("   Valores em reflectância (0 a 1); 0,05 = 5% da luz refletida.\n")
    print(f"   {'canal':<8}{'corte mín (p1)':>16}{'corte máx (p99)':>17}{'média':>9}{'desvio':>9}")
    for n, lo, hi, m, s in zip(stats["channels"], stats["clip_low"], stats["clip_high"],
                               stats["mean"], stats["std"]):
        print(f"   {n:<8}{lo:>16.4f}{hi:>17.4f}{m:>9.4f}{s:>9.4f}")
    print("\n   Como ler: um canal com desvio pequeno varia pouco entre pixels. O swir2 costuma ter média")
    print("   maior que as bandas visíveis (solo e vegetação seca refletem mais no infravermelho).")
    print("   Estes números ficam salvos e são reaplicados na classificação (stats/norm_stats.json).")
    print(LINHA)


# ---------------------------------------------------------------------------
# Passo 4 - Treino
# ---------------------------------------------------------------------------

def _curvas(h):
    return len(h["loss"]), h["loss"], h.get("val_loss", []), h["iou_fire"], h["val_iou_fire"]


def figura_treino(meta):
    """Curvas de aprendizado: perda e IoU do fogo, treino x validação, por época."""
    import matplotlib.pyplot as plt
    ep, loss, vloss, iou, viou = _curvas(meta["history"])
    xs = list(range(1, ep + 1))
    melhor = max(range(ep), key=lambda i: viou[i]) + 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    ax[0].plot(xs, loss, label="treino")
    if vloss:
        ax[0].plot(xs, vloss, label="validação")
    ax[0].set_title("Perda (menor é melhor)")
    ax[1].plot(xs, iou, label="treino")
    ax[1].plot(xs, viou, label="validação")
    ax[1].axvline(melhor, color="gray", ls="--", lw=1, label=f"melhor época ({melhor})")
    ax[1].set_title("IoU do fogo (maior é melhor)")
    for a in ax:
        a.set_xlabel("época")
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle("Curvas de aprendizado: se o treino sobe e a validação não, o modelo está decorando")
    fig.tight_layout()
    plt.show()


def resumo_treino(cfg, meta, model=None):
    """Passo 4: arquitetura, épocas, melhor época, leitura das curvas e diagnóstico."""
    h = meta["history"]
    ep, loss, vloss, iou, viou = _curvas(h)
    melhor = max(range(ep), key=lambda i: viou[i])
    _cab(f"TREINO  |  modelo {meta['model']}  |  {len(meta['channels'])} canais de entrada")
    print("Como a U-Net funciona, em poucas palavras: um 'codificador' reduz a imagem em etapas para entender")
    print("o contexto (o que é cicatriz, o que é água); um 'decodificador' volta ao tamanho original,")
    print("reaproveitando por atalhos (skip connections) os detalhes finos de cada etapa. A saída é um")
    print("mapa, pixel a pixel, com a probabilidade de fogo (0 a 1).")
    print(f"   Filtros por nível: {list(cfg['FILTERS'])} + fundo {cfg['BOTTOM_FILTERS']}"
          + (f" | parâmetros: {model.count_params():,}" if model is not None else ""))
    print(f"   Chips: {len(meta['train_shards'])} arquivo(s) de treino, {len(meta['val_shards'])} de validação")
    print(f"   Lote (batch): {cfg['BATCH']} chips por passo | taxa de aprendizado inicial: {cfg['LR']}")
    print(f"   Épocas rodadas: {ep} (máximo {cfg['EPOCHS']}); parada antecipada após {cfg['PATIENCE']} "
          "épocas sem melhorar o IoU de validação\n")

    print(f"   {'época':>6}{'perda':>9}{'perda val':>11}{'IoU':>8}{'IoU val':>9}{'F1 val':>8}")
    passo = max(1, ep // 12)
    f1s = h.get("val_f1_fire", [float("nan")] * ep)
    for i in range(ep):
        if i % passo == 0 or i == melhor or i == ep - 1:
            marca = "  <- melhor" if i == melhor else ""
            vl = vloss[i] if vloss else float("nan")
            print(f"   {i + 1:>6}{loss[i]:>9.3f}{vl:>11.3f}{iou[i]:>8.3f}{viou[i]:>9.3f}{f1s[i]:>8.3f}{marca}")

    print(f"\n   Melhor época: {melhor + 1} | IoU do fogo na validação: {viou[melhor]:.3f}")
    print("\n   DIAGNÓSTICO")
    if max(viou) <= 0:
        print("   *** TREINO INVÁLIDO: o IoU de validação ficou em 0 em todas as épocas. ***")
        print("   A validação não tem pixels de fogo (ou o modelo não acerta nenhum), então a 'melhor")
        print("   época' é arbitrária e o modelo não deve ser usado. Volte ao passo 2, confira se a")
        print("   validação tem fogo e não-fogo na tabela, refaça os chips (REFAZER_CHIPS = True) e treine de novo.")
    else:
        lacuna = iou[melhor] - viou[melhor]
        if lacuna > 0.25:
            print(f"   - Treino ({iou[melhor]:.2f}) muito acima da validação ({viou[melhor]:.2f}): sinal de "
                  "decoração (overfitting). Com poucos chips isso é esperado; mais meses ou amostras ajudam.")
        elif viou[melhor] < 0.3:
            print("   - IoU de validação baixo (< 0,3): o modelo ainda erra muito. Confira os chips na figura "
                  "do passo 2 (rótulos, água, sombras) antes de culpar o modelo.")
        else:
            print("   - Treino e validação andam juntos e o IoU é razoável: bom sinal.")
        if ep >= cfg["EPOCHS"]:
            print("   - O treino usou todas as épocas: o modelo ainda podia melhorar. Aumente EPOCHS.")
    print("\n   Como ler:")
    print("   - IoU = área acertada / (área prevista + área real − área acertada). 1,0 é perfeito.")
    print("   - F1 = equilíbrio entre precisão (o que marcou é fogo mesmo) e recall (fogo que encontrou).")
    print("   - Perda: quanto menor, melhor; compare treino e validação, não só o valor.")
    print(LINHA)


# ---------------------------------------------------------------------------
# Passo 5 - Teste
# ---------------------------------------------------------------------------

def _matriz(m):
    print("                        previsto: FOGO        previsto: NÃO-FOGO")
    print(f"   real: FOGO        {m['tp']:>14,} (acertou)   {m['fn']:>14,} (perdeu fogo)")
    print(f"   real: NÃO-FOGO    {m['fp']:>14,} (alarme falso) {m['tn']:>12,} (acertou)")


def resumo_teste(unet_results):
    """Passo 5: métricas de teste, matriz de confusão e leitura em linguagem simples."""
    _cab("TESTE  |  nota final, em pixels que o modelo nunca viu")
    print("Limiar = probabilidade a partir da qual um pixel conta como fogo. Foi escolhido na")
    print("VALIDAÇÃO (melhor F1), nunca no teste, para que a nota seja honesta.\n")
    for per, res in unet_results.items():
        t = res["test_at_best"]
        t5 = res["test_at_0_5"]
        nome = "todos os períodos juntos" if per == "all" else per
        thr = res["val_best_threshold"]
        print(f"   {nome}  |  limiar escolhido na validação: {thr:.2f}")
        print(f"   {'':<4}{'limiar':<10}{'IoU':>7}{'F1':>7}{'precisão':>10}{'recall':>8}{'omissão':>9}{'comissão':>10}")
        for rot, m in ((f"{thr:.2f}", t), ("0.50", t5)):
            print(f"   {'':<4}{rot:<10}{m['iou']:>7.3f}{m['f1']:>7.3f}{m['precision']:>10.3f}"
                  f"{m['recall']:>8.3f}{m['omission']:>9.3f}{m['commission']:>10.3f}")
        print(f"\n   Matriz de confusão no limiar {thr:.2f} (em pixels do teste):")
        _matriz(t)
        tot_f = t["tp"] + t["fn"]
        print(f"\n   Em palavras: dos {tot_f:,} pixels que são fogo de verdade, o modelo achou "
              f"{100 * t['recall']:.0f}%; dos pixels que marcou como fogo, {100 * t['precision']:.0f}% "
              "eram fogo de verdade.\n")
    print("   Como ler:")
    print("   - Precisão = dos pixels marcados como fogo, quantos eram fogo. Comissão = 1 − precisão (alarmes falsos).")
    print("   - Recall = do fogo real, quanto foi encontrado. Omissão = 1 − recall (fogo perdido).")
    print("   - Precisão alta + recall baixo: o modelo é cauteloso e perde parte do fogo.")
    print("   - Precisão baixa + recall alto: marca fogo demais (comum com água escura, sombra e solo escuro).")
    print("   - O IoU é sempre menor que o F1 do mesmo modelo (F1 = 2·IoU / (1 + IoU)); ao comparar com a")
    print("     literatura, use a mesma métrica. As notas variam muito entre estudos (sensor, região, divisão).")
    print("   - Com poucos blocos de teste, estes números oscilam: use como indício, não como prova.")
    print(LINHA)
