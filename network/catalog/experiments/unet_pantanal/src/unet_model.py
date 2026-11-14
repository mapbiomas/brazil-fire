"""
U-Net experiment - dataset, model, masked loss/metrics, training (spec 5.8-5.11).

TensorFlow 2 / Keras. Do not run this in the same Colab runtime as the
catalog M4 trainer: M4 calls `tf.compat.v1.disable_v2_behavior()`, which
breaks Keras 2 training. Use a fresh runtime for the U-Net notebook.
"""
import json
import os

import numpy as np

from unet_config import IGNORE, channel_names, paths
from unet_features import prob_histograms


def _tf():
    import tensorflow as tf
    return tf


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def chip_files(local_files, periods, split):
    """Shards of the given periods and split ('2026_10_train_000.tfrecord.gz')."""
    from unet_config import split_chip_name
    wanted = {str(p) for p in periods}
    out = []
    for f in local_files:
        try:
            period, s, _ = split_chip_name(f)
        except ValueError:
            continue
        if s == split and period in wanted:
            out.append(f)
    return sorted(out)


def _parse_fn(n_channels):
    tf = _tf()
    spec = {
        "x": tf.io.FixedLenFeature([], tf.string),
        "y": tf.io.FixedLenFeature([], tf.string),
        "h": tf.io.FixedLenFeature([], tf.int64),
        "w": tf.io.FixedLenFeature([], tf.int64),
    }

    def parse(record):
        ex = tf.io.parse_single_example(record, spec)
        h, w = tf.cast(ex["h"], tf.int32), tf.cast(ex["w"], tf.int32)
        x = tf.reshape(tf.io.decode_raw(ex["x"], tf.float32), [h, w, n_channels])
        y = tf.reshape(tf.io.decode_raw(ex["y"], tf.uint8), [h, w, 1])
        return x, tf.cast(y, tf.float32)

    return parse


def tf_normalize(x, stats):
    """TensorFlow twin of unet_features.normalize (same clip, mean/std, NaN -> 0)."""
    tf = _tf()
    lo = tf.constant(stats["clip_low"], tf.float32)
    hi = tf.constant(stats["clip_high"], tf.float32)
    mean = tf.constant(stats["mean"], tf.float32)
    std = tf.constant(stats["std"], tf.float32)
    finite = tf.math.is_finite(x)
    xs = tf.where(finite, x, tf.zeros_like(x))
    out = (tf.minimum(tf.maximum(xs, lo), hi) - mean) / std
    return tf.where(finite, out, tf.zeros_like(out))


def _augment(x, y):
    """Same random flips / 90-degree rotation on x and y."""
    tf = _tf()
    c = tf.shape(x)[-1]
    xy = tf.concat([x, y], axis=-1)
    xy = tf.image.random_flip_left_right(xy)
    xy = tf.image.random_flip_up_down(xy)
    xy = tf.image.rot90(xy, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
    return xy[..., :c], xy[..., c:]


def make_dataset(files, stats, cfg, training):
    tf = _tf()
    n_ch = len(channel_names(cfg))
    ds = tf.data.TFRecordDataset(files, compression_type="GZIP",
                                 num_parallel_reads=tf.data.AUTOTUNE)
    ds = ds.map(_parse_fn(n_ch), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (tf_normalize(x, stats), y), num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=cfg["SEED"], reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(cfg["BATCH"]).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_unet(n_channels, filters=(32, 64, 128, 256), bottom=512):
    tf = _tf()
    L = tf.keras.layers

    def block(t, f):
        for _ in range(2):
            t = L.Conv2D(f, 3, padding="same", use_bias=False)(t)
            t = L.BatchNormalization()(t)
            t = L.Activation("relu")(t)
        return t

    inp = L.Input((None, None, n_channels))
    skips, t = [], inp
    for f in filters:
        t = block(t, f)
        skips.append(t)
        t = L.MaxPooling2D(2)(t)
    t = block(t, bottom)
    for f, s in zip(reversed(filters), reversed(skips)):
        t = L.Conv2DTranspose(f, 2, strides=2, padding="same")(t)
        t = L.Concatenate()([t, s])
        t = block(t, f)
    out = L.Conv2D(1, 1, activation="sigmoid", dtype="float32", name="prob")(t)
    return tf.keras.Model(inp, out, name="unet_fire")


def masked_bce_dice(y_true, y_pred):
    """BCE + Dice over labeled pixels only (label 255 ignored)."""
    tf = _tf()
    y_pred = tf.cast(y_pred, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, IGNORE), tf.float32)
    yb = tf.where(mask > 0, y_true, tf.zeros_like(y_true))
    p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = -(yb * tf.math.log(p) + (1 - yb) * tf.math.log(1 - p))
    n = tf.reduce_sum(mask) + 1e-6
    bce = tf.reduce_sum(bce * mask) / n
    inter = tf.reduce_sum(p * yb * mask)
    dice = 1.0 - (2.0 * inter + 1.0) / (tf.reduce_sum(p * mask) + tf.reduce_sum(yb * mask) + 1.0)
    return bce + dice


def masked_metric_class():
    tf = _tf()

    class MaskedFireMetric(tf.keras.metrics.Metric):
        """IoU / F1 / precision / recall of the fire class, labeled pixels only."""

        def __init__(self, kind="iou", threshold=0.5, name=None, **kw):
            super().__init__(name=name or f"{kind}_fire", **kw)
            self.kind, self.threshold = kind, threshold
            self.tp = self.add_weight(name="tp", initializer="zeros")
            self.fp = self.add_weight(name="fp", initializer="zeros")
            self.fn = self.add_weight(name="fn", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            mask = tf.not_equal(y_true, IGNORE)
            t = tf.logical_and(mask, tf.equal(y_true, 1))
            f = tf.logical_and(mask, tf.equal(y_true, 0))
            p = tf.cast(y_pred, tf.float32) >= self.threshold
            self.tp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(t, p), tf.float32)))
            self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(f, p), tf.float32)))
            self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_and(t, tf.logical_not(p)), tf.float32)))

        def result(self):
            tp, fp, fn = self.tp, self.fp, self.fn
            prec = tf.math.divide_no_nan(tp, tp + fp)
            rec = tf.math.divide_no_nan(tp, tp + fn)
            if self.kind == "precision":
                return prec
            if self.kind == "recall":
                return rec
            if self.kind == "f1":
                return tf.math.divide_no_nan(2 * prec * rec, prec + rec)
            return tf.math.divide_no_nan(tp, tp + fp + fn)

        def reset_state(self):
            for v in (self.tp, self.fp, self.fn):
                v.assign(0.0)

        def get_config(self):
            return {**super().get_config(), "kind": self.kind, "threshold": self.threshold}

    return MaskedFireMetric


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _gpu_available():
    return bool(_tf().config.list_physical_devices("GPU"))


def train(cfg, logger=print):
    """Train the U-Net on the train chips, early-stop on val IoU, save to GCS."""
    tf = _tf()
    from unet_io import download_prefix, local_dir, read_json, upload, write_json

    P = paths(cfg)
    tf.keras.utils.set_random_seed(cfg["SEED"])
    if cfg["MIXED_PRECISION"] and _gpu_available():
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        logger("[TREINO] precisão mista (mixed_float16) na GPU")
    elif not _gpu_available():
        logger("[TREINO][AVISO] Sem GPU: Ambiente de execução > Alterar tipo > GPU.")

    stats = read_json(P["stats"])
    files = download_prefix(P["chips"], local_dir("chips"), suffix=".tfrecord.gz")
    tr = chip_files(files, cfg["PERIODS"], "train")
    va = chip_files(files, cfg["PERIODS"], "val")
    if not tr or not va:
        falta = "treino" if not tr else "validação"
        raise FileNotFoundError(
            f"Faltam chips de {falta} (arquivos: treino={len(tr)}, validação={len(va)}).\n"
            "Causa provável: na divisão por blocos de BLOCK_KM km, a validação (ou o treino) ficou sem "
            "blocos ou só com uma classe (só fogo ou só não-fogo), e seus chips não foram gerados.\n"
            "O que fazer: (1) volte ao passo 2 e veja a tabela de chips por divisão; (2) rode-o de novo "
            "com REFAZER_CHIPS = True (os chips já salvos são de uma execução anterior); (3) se ainda "
            "faltar, acrescente meses em PERIODS ou reduza BLOCK_KM (ex.: 10).")

    model = build_unet(len(channel_names(cfg)), cfg["FILTERS"], cfg["BOTTOM_FILTERS"])
    M = masked_metric_class()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["LR"]),
        loss=masked_bce_dice,
        metrics=[M("iou"), M("f1"), M("precision"), M("recall")],
    )
    out_dir = local_dir("models")
    best = os.path.join(out_dir, f"{cfg['MODEL_NAME']}.keras")
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_iou_fire", mode="max",
                                         patience=cfg["PATIENCE"], restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_iou_fire", mode="max",
                                             factor=0.5, patience=max(2, cfg["PATIENCE"] // 2)),
        tf.keras.callbacks.ModelCheckpoint(best, monitor="val_iou_fire", mode="max",
                                           save_best_only=True),
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, "history.csv")),
    ]
    hist = model.fit(make_dataset(tr, stats, cfg, True),
                     validation_data=make_dataset(va, stats, cfg, False),
                     epochs=cfg["EPOCHS"], callbacks=callbacks, verbose=2)
    model.save(best)  # best weights restored by EarlyStopping

    rel = f"{P['models']}/{cfg['MODEL_NAME']}.keras"
    upload(cfg, best, rel)
    meta = {
        "model": cfg["MODEL_NAME"], "architecture": "unet",
        "channels": channel_names(cfg), "bands": cfg["BANDS"], "indices": cfg["INDICES"],
        "scale": cfg["SCALE"], "valid_band": cfg["VALID_BAND"], "chip": cfg["CHIP"],
        "periods": cfg["PERIODS"], "region": cfg["REGION"], "stats": P["stats"],
        "train_shards": [os.path.basename(f) for f in tr],
        "val_shards": [os.path.basename(f) for f in va],
        "epochs_run": len(hist.history["loss"]),
        "history": {k: [float(v) for v in vals] for k, vals in hist.history.items()},
        "config": cfg,
    }
    write_json(cfg, f"{P['models']}/{cfg['MODEL_NAME']}_metadata.json", meta)
    upload(cfg, os.path.join(out_dir, "history.csv"), f"{P['models']}/{cfg['MODEL_NAME']}_history.csv")
    logger(f"[TREINO] melhor modelo salvo em gs://{P['bucket']}/{rel}")
    return model, meta


def load_model(cfg):
    """Load the trained U-Net from GCS (compile=False: inference only)."""
    tf = _tf()
    import unet_env as gcs
    from unet_io import local_dir
    P = paths(cfg)
    local = os.path.join(local_dir("models"), f"{cfg['MODEL_NAME']}.keras")
    if not os.path.exists(local):
        gcs.download(f"{P['models']}/{cfg['MODEL_NAME']}.keras", local)
    return tf.keras.models.load_model(local, compile=False)


def split_histograms(model, files, stats, cfg):
    """Probability histograms (fire / not fire) over the labeled pixels of a split."""
    h_fire = h_not = None
    for xb, yb in make_dataset(files, stats, cfg, False):
        prob = model.predict_on_batch(xb)
        hf, hn = prob_histograms(np.asarray(prob)[..., 0], yb.numpy()[..., 0])
        h_fire = hf if h_fire is None else h_fire + hf
        h_not = hn if h_not is None else h_not + hn
    return h_fire, h_not


def labeled_pixels(files, cfg, max_pixels=None, seed=42):
    """Raw features and labels of labeled, valid pixels (for the MLP baseline)."""
    tf = _tf()
    n_ch = len(channel_names(cfg))
    xs, ys = [], []
    for x, y in tf.data.TFRecordDataset(files, compression_type="GZIP").map(_parse_fn(n_ch)):
        x, y = x.numpy().reshape(-1, n_ch), y.numpy().reshape(-1)
        keep = (y != IGNORE) & np.all(np.isfinite(x), axis=1)
        xs.append(x[keep])
        ys.append(y[keep].astype(np.int64))
    X = np.concatenate(xs) if xs else np.empty((0, n_ch), np.float32)
    Y = np.concatenate(ys) if ys else np.empty((0,), np.int64)
    if max_pixels and len(X) > max_pixels:
        idx = np.random.default_rng(seed).choice(len(X), max_pixels, replace=False)
        X, Y = X[idx], Y[idx]
    return X, Y


def dump_json(obj):
    return json.dumps(obj, indent=2, default=str)
