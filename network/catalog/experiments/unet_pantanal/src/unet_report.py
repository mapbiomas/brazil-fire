"""
U-Net experiment - test evaluation and comparison with the MLP (spec 5.11, 5.13).

All models are scored on the SAME labeled pixels of the test chips:
  A  MLP, bands only          (version_04 M4 topology, trained on train-chip pixels)
  B  MLP, bands + indices
  C  U-Net, bands + indices
The decision threshold is chosen on validation (best F1) and the test is
also reported at 0.5.
"""
import numpy as np

from unet_config import channel_names, paths
from unet_features import best_threshold, metrics_at, prob_histograms
from unet_io import _log, download_prefix, local_dir, read_json, write_json
from unet_model import chip_files, labeled_pixels, split_histograms

METRIC_KEYS = ("f1", "iou", "precision", "recall", "omission", "commission")


def _summary(h_val_fire, h_val_not, h_te_fire, h_te_not):
    best = best_threshold(h_val_fire, h_val_not)
    thr = best["threshold"]
    return {
        "val_best_threshold": thr,
        "val_at_best": best,
        "test_at_best": metrics_at(h_te_fire, h_te_not, thr),
        "test_at_0_5": metrics_at(h_te_fire, h_te_not, 0.5),
    }


def evaluate_unet(cfg, model, logger=None):
    """Masked test metrics per period and for all periods together."""
    P = paths(cfg)
    stats = read_json(P["stats"])
    files = download_prefix(P["chips"], local_dir("chips"), suffix=".tfrecord.gz")
    results = {}
    total = None
    for period in cfg["PERIODS"]:
        va, te = chip_files(files, [period], "val"), chip_files(files, [period], "test")
        if not va or not te:
            _log(logger, f"[TESTE][AVISO] {period}: faltam chips de validação ou teste", "warning")
            continue
        hv = split_histograms(model, va, stats, cfg)
        ht = split_histograms(model, te, stats, cfg)
        res = _summary(*hv, *ht)
        write_json(cfg, f"{P['reports']}/metrics_{period}.json", {"model": cfg["MODEL_NAME"], **res})
        results[period] = res
        total = [*hv, *ht] if total is None else [a + b for a, b in zip(total, [*hv, *ht])]
        t = res["test_at_best"]
        _log(logger, f"[TESTE] {period} U-Net no limiar {res['val_best_threshold']:.2f}: "
                     f"F1 {t['f1']:.3f} IoU {t['iou']:.3f} | no limiar 0.5 F1 {res['test_at_0_5']['f1']:.3f}")
    if total is not None:
        results["all"] = _summary(*total)
    return results


# ---------------------------------------------------------------------------
# MLP baseline (TF2 Keras copy of the version_04 M4 network)
# ---------------------------------------------------------------------------

def _mean_std(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8  # same as M4_data_extractor.compute_normalizer
    return mean, std


def train_mlp(X, Y, cfg, seed):
    """Dense ReLU layers (M4 default 7,14,7,14,7), sigmoid output, Adam, random
    batches of MLP_BATCH for MLP_ITERS steps, no dropout - as version_04 M4.

    Written in TF2 instead of importing M4 (tf.compat.v1 graph mode would
    switch off TF2 behavior for the whole runtime).
    """
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)
    L = tf.keras.layers
    init = tf.keras.initializers.GlorotUniform(seed=seed)
    inp = L.Input((X.shape[1],))
    t = inp
    for n in cfg["MLP_LAYERS"]:
        t = L.Dense(n, activation="relu", kernel_initializer=init)(t)
    out = L.Dense(1, activation="sigmoid", dtype="float32")(t)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(cfg["MLP_LR"]), loss="binary_crossentropy")
    ds = (tf.data.Dataset.from_tensor_slices((X, Y.astype(np.float32)))
          .shuffle(min(len(X), 200_000), seed=seed).repeat().batch(cfg["MLP_BATCH"]))
    model.fit(ds, steps_per_epoch=cfg["MLP_ITERS"], epochs=1, verbose=0)
    return model


def _predict(model, X):
    return np.asarray(model.predict(X, batch_size=65536, verbose=0))[:, 0]


def compare_with_mlp(cfg, unet_results, logger=None):
    """Train MLP A and B on the train-chip pixels; return the A x B x C table."""
    P = paths(cfg)
    files = download_prefix(P["chips"], local_dir("chips"), suffix=".tfrecord.gz")
    periods = cfg["PERIODS"]
    tr, va, te = (chip_files(files, periods, s) for s in ("train", "val", "test"))
    Xtr, Ytr = labeled_pixels(tr, cfg, cfg["MLP_MAX_PIXELS"], cfg["SEED"])
    Xva, Yva = labeled_pixels(va, cfg)
    Xte, Yte = labeled_pixels(te, cfg)
    _log(logger, f"[MLP] pixels rotulados: treino {len(Xtr)} | validação {len(Xva)} | teste {len(Xte)}")

    n_bands = len(cfg["BANDS"])
    variants = {"A_mlp_bands": slice(0, n_bands)}
    if cfg["INDICES"]:  # B only makes sense when there are indices
        variants["B_mlp_bands_indices"] = slice(0, len(channel_names(cfg)))
    rows = {}
    for name, cols in variants.items():
        mean, std = _mean_std(Xtr[:, cols])
        model = train_mlp((Xtr[:, cols] - mean) / std, Ytr, cfg, cfg["SEED"])
        pv = _predict(model, (Xva[:, cols] - mean) / std)
        pt = _predict(model, (Xte[:, cols] - mean) / std)
        rows[name] = _summary(*prob_histograms(pv, Yva), *prob_histograms(pt, Yte))
        t = rows[name]["test_at_best"]
        _log(logger, f"[MLP] {name}: teste F1 {t['f1']:.3f} IoU {t['iou']:.3f}")
    rows["C_unet"] = unet_results["all"]

    table = comparison_table(rows)
    write_json(cfg, f"{P['reports']}/comparison_{'-'.join(periods)}.json",
               {"periods": periods, "test_pixels": int(len(Xte)), "rows": rows, "markdown": table})
    return rows, table


def comparison_table(rows):
    """Markdown table: one line per model, test metrics at the val threshold and at 0.5."""
    head = ("| Modelo | Limiar (validação) | F1 | IoU | Omissão | Comissão | F1 @0.5 | IoU @0.5 |\n"
            "|---|---|---|---|---|---|---|---|")
    lines = [head]
    for name, r in rows.items():
        b, h = r["test_at_best"], r["test_at_0_5"]
        lines.append(f"| {name} | {r['val_best_threshold']:.2f} | {b['f1']:.3f} | {b['iou']:.3f} | "
                     f"{b['omission']:.3f} | {b['commission']:.3f} | {h['f1']:.3f} | {h['iou']:.3f} |")
    return "\n".join(lines)
