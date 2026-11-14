"""
Resumos didaticos em portugues dos passos do notebook (chips, estatisticas, treino, teste).

Cada funcao so imprime texto; nenhuma calcula nada novo. O idioma fica concentrado
aqui (e em poucas mensagens de erro) para facilitar uma escolha de idioma depois.
"""

LINHA = "=" * 72

SPLIT_NOME = {"train": "treino", "val": "validacao", "test": "teste"}
SPLIT_PARA_QUE = {
    "train": "o modelo aprende com estes",
    "val": "escolhe quando parar o treino e o limiar",
    "test": "nota final, nunca vista no treino",
}


def _cab(titulo):
    print(LINHA)
    print(titulo)
    print(LINHA)


def resumo_chips(cfg, summary):
    """Passo 2: blocos, centros, chips por divisao e o que foi descartado."""
    if summary.get("skipped"):
        print(f"[CHIPS] {summary['period']}: ja existem {len(summary['shards'])} arquivo(s) de chips no GCS "
              "e nada foi refeito.")
        print("        Para refazer (por exemplo depois de mudar bandas, amostras ou regiao), "
              "use REFAZER_CHIPS = True.")
        return
    c = summary["chips"]
    b = summary["blocks"]
    cen = summary.get("centers_by_split", {})
    d = summary["dropped"]
    _cab(f"CHIPS  |  periodo {summary['period']}  |  chip {cfg['CHIP']}x{cfg['CHIP']} px")
    print("O que e feito: a regiao e dividida em blocos de "
          f"{cfg['BLOCK_KM']} km. Cada bloco com amostras vai INTEIRO para treino,")
    print("validacao ou teste (assim o teste nunca 'espia' lugares vistos no treino). Depois,")
    print("recortes (chips) sao tirados de dentro dos poligonos de cada divisao.\n")
    print(f"   {'divisao':<11}{'blocos':>7}{'centros fogo':>14}{'centros nao-fogo':>18}{'chips':>8}"
          f"{'px fogo':>10}{'px nao-fogo':>13}   para que serve")
    for s in ("train", "val", "test"):
        cc = cen.get(s, {})
        print(f"   {SPLIT_NOME[s]:<11}{b.get(s, 0):>7}{cc.get('1', 0):>14}{cc.get('0', 0):>18}"
              f"{c[s]['chips']:>8}{c[s]['fire_px']:>10,}{c[s]['notfire_px']:>13,}   {SPLIT_PARA_QUE[s]}")
    print(f"\n   Chips descartados: {d['cross_block']} cruzavam blocos de divisoes diferentes | "
          f"{d['few_labels']} com poucos pixels rotulados | {d['nodata']} com muito nodata")
    ov = summary["fire_notfire_overlaps"]
    if ov["chips"]:
        print(f"   Poligonos de fogo e nao-fogo se sobrepoem em {ov['chips']} chip(s) "
              f"({ov['pixels']} pixels): vale o fogo.")
    print("\nO QUE OBSERVAR")
    avisos = []
    for s in ("val", "test"):
        if c[s]["chips"] == 0:
            cc = cen.get(s, {})
            if b.get(s, 0) == 0:
                motivo = "nenhum bloco foi sorteado para esta divisao (poucos blocos com amostras)"
            elif not cc.get("1") or not cc.get("0"):
                motivo = "os blocos desta divisao tem so uma classe (so fogo ou so nao-fogo)"
            else:
                motivo = "todos os chips foram descartados (veja a linha de descartes)"
            efeito = ("O treino nao roda sem validacao." if s == "val"
                      else "Sem teste nao ha nota final honesta.")
            avisos.append(f"{SPLIT_NOME[s].upper()} SEM CHIPS: {motivo}. {efeito} "
                          "Opcoes: acrescentar meses em PERIODS, reduzir BLOCK_KM (ex.: 10) ou "
                          "conferir os poligonos.")
        elif c[s]["fire_px"] == 0 or c[s]["notfire_px"] == 0:
            avisos.append(f"{SPLIT_NOME[s]} tem so uma classe nos pixels rotulados: as metricas dessa "
                          "divisao nao sao confiaveis.")
    if c["train"]["chips"] < 100:
        avisos.append(f"So {c['train']['chips']} chips de treino: e muito pouco para uma U-Net; "
                      "espere resultado instavel. Mais meses ou amostras ajudam.")
    if not avisos:
        avisos.append("Divisoes com as duas classes e chips em todas: pode seguir para o passo 3.")
    for a in avisos:
        print(f"   - {a}")
    print("   - Confira a figura abaixo: vermelho = fogo, azul = nao-fogo. Cuidado com lagoas "
          "(agua escura parece cicatriz), nuvens e sombras.")
    print(LINHA)


def resumo_estatisticas(stats):
    """Passo 3: o que e a normalizacao e os valores por canal."""
    _cab("NORMALIZACAO  |  estatisticas calculadas so nos chips de treino")
    print("Para que serve: a rede aprende melhor quando todos os canais tem escala parecida.")
    print("Cada valor x vira  (x_cortado - media) / desvio.  'Cortado' = limitado entre o")
    print("percentil 1 e o 99, para que pixels extremos (nuvem, brilho) nao distorcam a escala.")
    print("Usar so o treino evita que informacao do teste vaze para o modelo.\n")
    print(f"   Pixels validos de treino lidos: {stats.get('train_pixels_seen', stats['n_pixels']):,} "
          f"(amostra usada: {stats['n_pixels']:,})")
    print("   Valores em reflectancia (0 a 1); 0.05 = 5% da luz refletida.\n")
    print(f"   {'canal':<8}{'corte min (p1)':>16}{'corte max (p99)':>17}{'media':>9}{'desvio':>9}")
    for n, lo, hi, m, s in zip(stats["channels"], stats["clip_low"], stats["clip_high"],
                               stats["mean"], stats["std"]):
        print(f"   {n:<8}{lo:>16.4f}{hi:>17.4f}{m:>9.4f}{s:>9.4f}")
    print("\n   Como ler: canal com desvio pequeno varia pouco entre pixels; swir2 costuma ter media")
    print("   maior que as bandas visiveis (solo e vegetacao seca refletem mais no infravermelho).")
    print("   Estes numeros ficam salvos e sao reaplicados na classificacao (stats/norm_stats.json).")
    print(LINHA)


def resumo_treino(cfg, meta):
    """Passo 4: epocas, melhor validacao e leitura das metricas."""
    h = meta["history"]
    ep = meta["epochs_run"]
    iou = h["val_iou_fire"]
    melhor = max(range(len(iou)), key=lambda i: iou[i])
    _cab(f"TREINO  |  modelo {meta['model']}  |  {len(meta['channels'])} canais")
    print(f"   Chips: {len(meta['train_shards'])} arquivo(s) de treino, {len(meta['val_shards'])} de validacao")
    print(f"   Epocas rodadas: {ep} (maximo {cfg['EPOCHS']}); parada antecipada apos {cfg['PATIENCE']} "
          "epocas sem melhorar")
    print(f"   Melhor epoca: {melhor + 1} | IoU do fogo na validacao: {iou[melhor]:.3f} | "
          f"F1: {h['val_f1_fire'][melhor]:.3f}" if "val_f1_fire" in h else
          f"   Melhor epoca: {melhor + 1} | IoU do fogo na validacao: {iou[melhor]:.3f}")
    print("\n   Como ler:")
    print("   - IoU = area acertada / area somada (previsto + real). 1.0 e perfeito; abaixo de 0.3 e fraco.")
    print("   - F1 = equilibrio entre precisao (o que marcou e fogo mesmo) e recall (fogo que achou).")
    print("   - Se a perda de treino cai mas o IoU de validacao nao sobe, o modelo esta decorando (overfitting).")
    print(LINHA)


def resumo_teste(unet_results):
    """Passo 5: metricas de teste em linguagem simples."""
    _cab("TESTE  |  nota final, em pixels que o modelo nunca viu")
    print("Limiar = probabilidade a partir da qual um pixel conta como fogo. Foi escolhido na")
    print("VALIDACAO (melhor F1), nunca no teste, para a nota ser honesta.\n")
    for per, res in unet_results.items():
        t = res["test_at_best"]
        t5 = res["test_at_0_5"]
        nome = "todos os periodos juntos" if per == "all" else per
        print(f"   {nome}  |  limiar escolhido: {res['val_best_threshold']:.2f}")
        print(f"   {'':<4}{'limiar':<12}{'IoU':>7}{'F1':>7}{'precisao':>10}{'recall':>8}")
        for rot, m in ((f"{res['val_best_threshold']:.2f}", t), ("0.50", t5)):
            print(f"   {'':<4}{rot:<12}{m['iou']:>7.3f}{m['f1']:>7.3f}"
                  f"{m.get('precision', float('nan')):>10.3f}{m.get('recall', float('nan')):>8.3f}")
        print()
    print("   Como ler:")
    print("   - precisao alta + recall baixo: o modelo e cauteloso, perde parte do fogo.")
    print("   - precisao baixa + recall alto: marca fogo demais (comum com agua escura e sombra).")
    print("   - com poucos blocos de teste, estes numeros variam muito: use como indicio, nao como prova.")
    print(LINHA)
