"""
U-Net experiment - spectral indices, normalization and pixel metrics.

NumPy only. The same functions are used to build the training chips and
to run inference, so training and prediction see identical features.
"""
import json

import numpy as np

EPS = 1e-6

# Index formulas on reflectance 0-1. Each takes a dict {band: array}.
INDEX_FORMULAS = {
    "nbr": lambda b: (b["nir"] - b["swir2"]) / (b["nir"] + b["swir2"] + EPS),
    "nbr2": lambda b: (b["swir1"] - b["swir2"]) / (b["swir1"] + b["swir2"] + EPS),
    "mirbi": lambda b: 10.0 * b["swir2"] - 9.8 * b["swir1"] + 2.0,
    "ndvi": lambda b: (b["nir"] - b["red"]) / (b["nir"] + b["red"] + EPS),
    "bai": lambda b: 1.0 / ((0.1 - b["red"]) ** 2 + (0.06 - b["nir"]) ** 2 + EPS),
    "evi": lambda b: 2.5 * (b["nir"] - b["red"])
    / (b["nir"] + 6.0 * b["red"] - 7.5 * b["blue"] + 1.0),
}


def add_indices(x, bands, indices):
    """Append spectral indices to a band stack.

    Args:
        x: float array (..., len(bands)) with reflectance 0-1 (NaN = nodata).
        bands: band names in channel order (e.g. ['blue','red','nir','swir1','swir2']).
        indices: index names from INDEX_FORMULAS.

    Returns:
        float32 array (..., len(bands) + len(indices)).
    """
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] != len(bands):
        raise ValueError(f"Expected {len(bands)} channels, got {x.shape[-1]}.")
    if not indices:
        return x
    named = {b: x[..., i] for i, b in enumerate(bands)}
    out = [x]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for name in indices:
            if name not in INDEX_FORMULAS:
                raise ValueError(f"Unknown index '{name}'. Options: {sorted(INDEX_FORMULAS)}")
            val = INDEX_FORMULAS[name](named).astype(np.float32)
            val[~np.isfinite(val) & ~np.isnan(val)] = np.nan
            out.append(val[..., None])
    return np.concatenate(out, axis=-1)


def scale_bands(raw, scale, valid):
    """uint8/int raw bands (C, H, W) -> float32 (H, W, C) reflectance, NaN where invalid."""
    x = np.moveaxis(np.asarray(raw, dtype=np.float32) / float(scale), 0, -1)
    x[~valid] = np.nan
    return x


# ---------------------------------------------------------------------------
# Normalization (statistics from TRAIN chips only)
# ---------------------------------------------------------------------------

class PixelReservoir:
    """Uniform random sample of valid pixels, bounded in memory.

    Each pixel gets a random key; the `max_pixels` smallest keys are kept,
    which is an exact uniform sample of everything added.
    """

    def __init__(self, max_pixels, n_channels, seed=42):
        self.max_pixels = int(max_pixels)
        self.n_channels = int(n_channels)
        self.rng = np.random.default_rng(seed)
        self.data = np.empty((0, n_channels), dtype=np.float32)
        self.keys = np.empty((0,), dtype=np.float64)
        self.seen = 0

    def add(self, x):
        """x: (H, W, C) or (N, C); rows with any non-finite value are skipped."""
        px = np.asarray(x, dtype=np.float32).reshape(-1, self.n_channels)
        px = px[np.all(np.isfinite(px), axis=1)]
        if not len(px):
            return
        self.seen += len(px)
        self.data = np.concatenate([self.data, px])
        self.keys = np.concatenate([self.keys, self.rng.random(len(px))])
        if len(self.data) > self.max_pixels:
            keep = np.argpartition(self.keys, self.max_pixels)[: self.max_pixels]
            self.data = self.data[keep]
            self.keys = self.keys[keep]


def compute_norm_stats(pixels, channel_names, clip_percentiles=(1.0, 99.0)):
    """Clip limits (percentiles) and mean/std after clipping, per channel."""
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != len(channel_names):
        raise ValueError("pixels must be (N, C) with C = len(channel_names).")
    if len(pixels) == 0:
        raise ValueError("No valid train pixels to compute the statistics.")
    lo_p, hi_p = clip_percentiles
    lo = np.nanpercentile(pixels, lo_p, axis=0)
    hi = np.nanpercentile(pixels, hi_p, axis=0)
    clipped = np.clip(pixels, lo, hi)
    mean = np.nanmean(clipped, axis=0)
    std = np.nanstd(clipped, axis=0) + 1e-8
    return {
        "channels": list(channel_names),
        "clip_low": lo.tolist(),
        "clip_high": hi.tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "n_pixels": int(len(pixels)),
        "clip_percentiles": list(clip_percentiles),
    }


def normalize(x, stats):
    """Clip, standardize and set non-finite values (nodata) to 0. x: (..., C)."""
    lo = np.asarray(stats["clip_low"], dtype=np.float32)
    hi = np.asarray(stats["clip_high"], dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    finite = np.isfinite(x)
    out = (np.minimum(np.maximum(np.where(finite, x, 0), lo), hi) - mean) / std
    return np.where(finite, out, 0).astype(np.float32)


def stats_to_json(stats):
    return json.dumps(stats, indent=2)


# ---------------------------------------------------------------------------
# Pixel metrics for the fire class (labels 0/1, 255 ignored)
# ---------------------------------------------------------------------------

N_BINS = 1000


def prob_histograms(prob, label, ignore=255, n_bins=N_BINS):
    """Histogram of probabilities for fire and not-fire labeled pixels."""
    prob = np.asarray(prob, dtype=np.float64).ravel()
    label = np.asarray(label).ravel()
    keep = (label != ignore) & np.isfinite(prob)
    prob, label = prob[keep], label[keep]
    bins = np.clip((prob * n_bins).astype(int), 0, n_bins - 1)
    h_fire = np.bincount(bins[label == 1], minlength=n_bins)
    h_not = np.bincount(bins[label == 0], minlength=n_bins)
    return h_fire.astype(np.int64), h_not.astype(np.int64)


def metrics_at(h_fire, h_not, threshold, n_bins=N_BINS):
    """Fire-class metrics when pixels with prob >= threshold are fire."""
    k = int(np.clip(np.ceil(threshold * n_bins - 1e-9), 0, n_bins))
    tp = int(h_fire[k:].sum())
    fn = int(h_fire[:k].sum())
    fp = int(h_not[k:].sum())
    tn = int(h_not[:k].sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    total = tp + fp + fn + tn
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "omission": 1.0 - recall if tp + fn else 0.0,
        "commission": 1.0 - precision if tp + fp else 0.0,
        "accuracy": (tp + tn) / total if total else 0.0,
    }


def best_threshold(h_fire, h_not, n_bins=N_BINS):
    """Threshold (0.01..0.99) with the highest fire F1."""
    best = None
    for t in np.arange(0.01, 1.0, 0.01):
        m = metrics_at(h_fire, h_not, float(round(t, 2)), n_bins)
        if best is None or m["f1"] > best["f1"]:
            best = m
    return best
