"""
U-Net experiment - reading mosaics, samples and regions; GCS uploads.

Mosaics: the M2 COGs (one band per file) are read window by window, never
whole. Samples: the M3 CSVs (`.geo` GeoJSON + `fire`, `region`, `period`).
"""
import fnmatch
import json
import os

import numpy as np

import unet_env as gcs
from unet_config import (mosaic_band_paths, normalize_period, paths, period_from_file_name,
                         period_info, region_slug)


def _log(logger, msg, level="info"):
    if logger:
        logger(msg, level)
    else:
        print(msg)


def _is_colab():
    return "COLAB_RELEASE_TAG" in os.environ


# ---------------------------------------------------------------------------
# Mosaic reader
# ---------------------------------------------------------------------------

def _open_raster(rel_path):
    """Open a GCS raster: /vsigs/ streaming, gcsfs handle as fallback."""
    import rasterio
    try:
        return rasterio.open(f"/vsigs/{rel_path}")
    except Exception:
        return rasterio.open(gcs.open_file(rel_path, "rb"))


class MosaicReader:
    """Reads the same window from every band COG of one period.

    `read(window)` returns (raw, valid): raw is (len(BANDS), h, w) in the file
    dtype and valid is a bool (h, w) mask (VALID_BAND > 0, as in M5). Parts of
    the window outside the image are filled with 0 / invalid.
    """

    def __init__(self, cfg, period, logger=None):
        self.cfg = cfg
        self.period = period_info(period)[0]
        self.bands = list(cfg["BANDS"])
        self.valid_band = cfg.get("VALID_BAND") or None
        self.all_bands = self.bands + ([self.valid_band] if self.valid_band else [])
        self.paths = mosaic_band_paths(cfg, self.period)
        self.multiband = bool(cfg.get("MULTIBAND_FILE"))
        self.sources = {}
        try:
            if self.multiband:
                ds = _open_raster(self.paths[self.bands[0]])
                for b in self.all_bands:
                    self.sources[b] = ds
            else:
                for b in self.all_bands:
                    self.sources[b] = _open_raster(self.paths[b])
        except Exception as exc:
            self.close()
            raise FileNotFoundError(
                f"Nao foi possivel abrir o mosaico {self.period}. Confira se todas as bandas de BANDS "
                f"(e VALID_BAND) foram exportadas para esse periodo; a banda que nao existe aparece no erro. "
                f"Caminhos: {self.paths}. Erro: {exc}"
            ) from exc
        self._check_grid()
        ref = self.sources[self.bands[0]]
        self.crs = ref.crs
        self.transform = ref.transform
        self.width = ref.width
        self.height = ref.height
        self.profile = ref.profile.copy()
        _log(logger, f"[MOSAICO] {self.period}: {self.width}x{self.height} px | {self.crs} | pixel {ref.res}")

    def _check_grid(self):
        ref_b = self.bands[0]
        ref = self.sources[ref_b]
        bad = []
        for b, ds in self.sources.items():
            same = (ds.crs == ref.crs and ds.transform.almost_equals(ref.transform)
                    and ds.width == ref.width and ds.height == ref.height)
            if not same:
                bad.append(f"{b}: crs={ds.crs} transform={tuple(ds.transform)[:6]} size={ds.width}x{ds.height}")
        if bad:
            raise ValueError(
                "As bandas do mosaico nao estao na mesma grade de "
                f"'{ref_b}' (crs={ref.crs}, tamanho={ref.width}x{ref.height}):\n  " + "\n  ".join(bad)
            )

    def _band_index(self, b):
        if self.multiband:
            return self.all_bands.index(b) + 1
        return 1

    def read_band(self, b, window):
        from rasterio.windows import Window
        h, w = int(window.height), int(window.width)
        out = np.zeros((h, w), dtype=self.sources[b].dtypes[0])
        c0, r0 = int(window.col_off), int(window.row_off)
        c_a, r_a = max(c0, 0), max(r0, 0)
        c_b, r_b = min(c0 + w, self.width), min(r0 + h, self.height)
        if c_b <= c_a or r_b <= r_a:
            return out
        inner = Window(c_a, r_a, c_b - c_a, r_b - r_a)
        data = self.sources[b].read(self._band_index(b), window=inner)
        out[r_a - r0:r_b - r0, c_a - c0:c_b - c0] = data
        return out

    def read(self, window):
        raw = np.stack([self.read_band(b, window) for b in self.bands])
        if self.valid_band:
            valid = self.read_band(self.valid_band, window) > 0
        else:
            # No dayOfYear: outside the image / masked pixels are 0 in every band.
            valid = np.any(raw > 0, axis=0)
        for b, arr in zip(self.bands, raw):
            nodata = self.sources[b].nodata
            if nodata is not None:
                valid &= arr != nodata
        return raw, valid

    def read_doy(self, window):
        """dayOfYear of the window, or None when VALID_BAND is not set."""
        if not self.valid_band:
            return None
        return self.read_band(self.valid_band, window)

    def inventory(self):
        info = {"period": self.period, "crs": str(self.crs), "width": self.width,
                "height": self.height, "transform": list(self.transform)[:6],
                "scale_divisor": self.cfg["SCALE"], "bands": {}}
        info["valid_band"] = self.valid_band or "none (valid = bands not nodata and not all 0)"
        for b in self.all_bands:
            ds = self.sources[b]
            info["bands"][b] = {"path": f"gs://{self.paths[b]}", "dtype": ds.dtypes[0],
                                "nodata": ds.nodata, "res": list(ds.res)}
        return info

    def close(self):
        seen = set()
        for ds in self.sources.values():
            if id(ds) not in seen:
                seen.add(id(ds))
                try:
                    ds.close()
                except Exception:
                    pass
        self.sources = {}


# ---------------------------------------------------------------------------
# Samples (M3 CSV)
# ---------------------------------------------------------------------------

def list_sample_files(cfg, period):
    period = period_info(period)[0]
    P = paths(cfg)
    folder = f"{P['bucket']}/{P['samples']}"
    names = [n for n in gcs.ls(folder) if n.endswith(".csv")]
    if cfg.get("SAMPLE_FILES"):
        wanted = [n if n.endswith(".csv") else f"{n}.csv" for n in cfg["SAMPLE_FILES"]]
        missing = [n for n in wanted if n not in names]
        if missing:
            raise FileNotFoundError(f"Arquivos de amostra nao encontrados em gs://{folder}: {missing}")
        return wanted
    pattern = cfg["SAMPLE_GLOB"].format(period=period, region=cfg["REGION"])
    return sorted(n for n in names if fnmatch.fnmatch(n, pattern))


def sample_periods(df, file_name):
    """Period of each row, with the same rules as the catalog M4 extractor:
    the `period` column (normalized); empty -> date at the end of the file
    name; if the column has a year later than the file date, the file date
    wins. Returns (Series of keys, number of corrected rows)."""
    import pandas as pd
    file_date = period_from_file_name(file_name)
    if "period" in df.columns:
        per = df["period"].map(normalize_period)
    else:
        per = pd.Series([""] * len(df), index=df.index)
    corrected = 0
    if file_date:
        per = per.where(per != "", file_date)
        fy = int(file_date[:4])
        years = per.map(lambda p: int(p[:4]) if p[:4].isdigit() else 0)
        if (years > fy).any():
            corrected = int(len(per))
            per = pd.Series([file_date] * len(df), index=df.index)
    return per, corrected


def load_samples(cfg, period, crs, logger=None):
    """M3 polygons of one period as a GeoDataFrame in the mosaic CRS.

    Columns kept: class (0/1), source file, geometry. Invalid geometries are
    repaired (make_valid); only polygon parts are kept.
    """
    period = period_info(period)[0]
    import geopandas as gpd
    import pandas as pd
    from shapely import make_valid
    from shapely.geometry import MultiPolygon, Polygon, shape

    P = paths(cfg)
    files = list_sample_files(cfg, period)
    if not files:
        raise FileNotFoundError(
            f"Nenhum CSV de amostras casa com '{cfg['SAMPLE_GLOB']}' em gs://{P['bucket']}/{P['samples']}. "
            "O nome do arquivo precisa ter o periodo (ex.: ..._2026_10.csv) ou use SAMPLE_FILES=[...]."
        )
    field = cfg["CLASS_FIELD"]
    frames = []
    corrected = {}
    for name in files:
        with gcs.open_file(f"{P['bucket']}/{P['samples']}/{name}", "r") as f:
            df = pd.read_csv(f)
        if field not in df.columns or ".geo" not in df.columns:
            raise ValueError(f"{name}: precisa das colunas '{field}' e '.geo' (formato M3).")
        df["_source"] = name
        df["_period"], n_fix = sample_periods(df, name)
        if n_fix:
            corrected[name] = n_fix
            _log(logger, f"[AMOSTRAS][AVISO] {name}: a coluna 'period' tem data posterior a do nome do "
                         f"arquivo; vale a data do arquivo (regra do M4).", "warning")
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    n_read = len(df)
    periods_found = sorted(df["_period"].unique().tolist())
    df = df[df["_period"] == period]
    if df.empty:
        raise ValueError(f"Nenhuma amostra e do periodo {period}. Periodos encontrados em {files}: {periods_found}")
    sample_region = cfg.get("SAMPLE_REGION") or cfg.get("REGION")
    if cfg.get("FILTER_SAMPLES_BY_REGION") and sample_region and "region" in df.columns:
        regions = sorted(df["region"].astype(str).unique().tolist())
        df = df[df["region"].astype(str).str.lower() == str(sample_region).lower()]
        if df.empty:
            raise ValueError(
                f"Nenhuma amostra tem region == '{sample_region}'. Regioes encontradas: {regions}. "
                "Ajuste SAMPLE_REGION para uma delas, ou use FILTER_SAMPLES_BY_REGION=False."
            )

    df = df[pd.to_numeric(df[field], errors="coerce").isin([0, 1])].copy()
    df["class"] = df[field].astype(int)

    def _poly(geo):
        g = make_valid(shape(json.loads(geo)))
        if isinstance(g, (Polygon, MultiPolygon)):
            return g
        parts = [p for p in getattr(g, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
        if not parts:
            return None
        polys = []
        for p in parts:
            polys.extend(p.geoms if isinstance(p, MultiPolygon) else [p])
        return MultiPolygon(polys)

    df["geometry"] = df[".geo"].map(_poly)
    df = df[df["geometry"].notna()]
    gdf = gpd.GeoDataFrame(df[["class", "_source", "geometry"]], geometry="geometry", crs="EPSG:4326")
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf = gdf.to_crs(crs)

    area_ha = gdf.to_crs("EPSG:6933").area / 1e4
    report = {
        "files": files, "rows_read": n_read, "periods_found": periods_found,
        "period_corrected_rows": corrected, "polygons": int(len(gdf)),
        "by_class": {
            str(c): {"polygons": int((gdf["class"] == c).sum()),
                     "area_ha": float(area_ha[gdf["class"] == c].sum())}
            for c in (1, 0)
        },
    }
    _log(logger, f"[SAMPLES] {period}: {len(files)} file(s), {len(gdf)} polygons | "
                 f"fire {report['by_class']['1']['polygons']} ({report['by_class']['1']['area_ha']:.0f} ha) | "
                 f"not fire {report['by_class']['0']['polygons']} ({report['by_class']['0']['area_ha']:.0f} ha)")
    return gdf, report


def print_inventory(cfg, inv, sample_files, report):
    """Resumo didatico do inventario: mosaico (de onde vem a imagem) e amostras (o que o modelo aprende)."""
    deg_to_m = 111320.0
    res_deg = next(iter(inv["bands"].values()))["res"][0]
    geographic = str(inv["crs"]).upper() in ("EPSG:4326", "OGC:CRS84")
    res_m = res_deg * deg_to_m if geographic else res_deg
    line = "=" * 72
    print(line)
    print(f"INVENTARIO  |  periodo {inv['period']}  |  regiao {cfg['REGION']}  |  {cfg['SENSOR']} {cfg['MOSAIC']}")
    print(line)
    print("Para que serve: confere, sem treinar nada, se as imagens (mosaico) e as amostras")
    print("(poligonos de fogo / nao-fogo) existem, estao na mesma grade e sao suficientes.\n")

    print("1) MOSAICO (as imagens que entram no modelo)")
    print(f"   Grade: {inv['width']:,} x {inv['height']:,} pixels | CRS {inv['crs']} | pixel ~ {res_m:.1f} m")
    print("   (o mosaico cobre a America do Sul inteira; o codigo le so a janela da regiao)")
    print(f"   {'banda':<11}{'tipo':<8}{'nodata':<9}{'pixel (graus)':<16}arquivo")
    for name, b in inv["bands"].items():
        print(f"   {name:<11}{b['dtype']:<8}{str(b['nodata']):<9}{b['res'][0]:<16.3e}{b['path'].rsplit('/', 1)[-1]}")
    n_feat = len(cfg["BANDS"]) + len(cfg["INDICES"])
    print(f"   -> o modelo recebe {n_feat} canais: bandas {cfg['BANDS']}"
          + (f" + indices {cfg['INDICES']}" if cfg["INDICES"] else " (sem indices)"))
    vb = inv["valid_band"]
    if cfg.get("VALID_BAND"):
        print(f"   -> pixel valido: onde {vb} > 0; a banda 2 da saida traz o dia do ano da queimada")
    else:
        print("   -> sem dayOfYear: pixel valido = bandas nao-nodata e nao todas 0; banda 2 da saida fica vazia")
    res_ok = len({round(b["res"][0], 12) for b in inv["bands"].values()}) == 1
    print(f"   -> todas as bandas na mesma grade: {'SIM' if res_ok else 'NAO - verifique os COGs'}\n")

    print("2) AMOSTRAS (o que o modelo aprende)")
    print(f"   Arquivo(s): {', '.join(os.path.basename(str(f)) for f in sample_files) or 'nenhum'}")
    print(f"   Regiao filtrada: {cfg.get('SAMPLE_REGION') or cfg['REGION']} | periodos no arquivo: {report['periods_found']}")
    fixed = sum(report["period_corrected_rows"].values()) if isinstance(report["period_corrected_rows"], dict) \
        else report["period_corrected_rows"]
    print(f"   Linhas com periodo corrigido pela regra do M4: {fixed}")
    f, n = report["by_class"]["1"], report["by_class"]["0"]
    total = f["polygons"] + n["polygons"]
    print(f"   {'classe':<22}{'poligonos':>10}{'area (ha)':>13}{'media/poligono (ha)':>22}")
    for label, c in (("fogo (1)", f), ("nao-fogo (0)", n)):
        mean = c["area_ha"] / c["polygons"] if c["polygons"] else 0
        print(f"   {label:<22}{c['polygons']:>10}{c['area_ha']:>13,.0f}{mean:>22,.1f}")
    print(f"   {'total':<22}{total:>10}\n")

    print("3) O QUE OBSERVAR")
    notes = []
    if total < 100:
        notes.append(f"Poucos poligonos ({total}): com blocos de {cfg['BLOCK_KM']} km, validacao e teste "
                     "podem ficar com 1 ou 2 blocos. Acrescente meses em PERIODS se possivel.")
    if n["polygons"] < 10:
        notes.append(f"So {n['polygons']} poligonos de nao-fogo: o modelo pode confundir agua, sombra e "
                     "solo escuro com cicatriz.")
    if f["polygons"] and n["polygons"] and (f["area_ha"] / max(n["area_ha"], 1e-9) < 0.2
                                             or f["area_ha"] / max(n["area_ha"], 1e-9) > 5):
        notes.append("Areas de fogo e nao-fogo muito diferentes; os chips sao balanceados pelo numero, "
                     "nao pela area.")
    if not cfg["INDICES"]:
        notes.append("Sem indices espectrais (faltam nir/swir1): a comparacao com o MLP tera so as variantes A e C.")
    if not notes:
        notes.append("Nada de especial: pode seguir para a etapa de chips.")
    for t in notes:
        print(f"   - {t}")
    print(line)


# ---------------------------------------------------------------------------
# Region geometry (GEE regions asset, cached in GCS)
# ---------------------------------------------------------------------------

def regions_collection(cfg):
    """The regions FeatureCollection with the computed REGION_PROPERTY.

    With REGION_NAME_FIELDS = ["bioma", "id"] and REGION_NAME_JOIN = "_r", each
    feature gets region_name = bioma + "_r" + id.
    """
    import ee
    fc = ee.FeatureCollection(cfg["REGIONS_ASSET"])
    fields = cfg.get("REGION_NAME_FIELDS")
    if not fields:
        return fc
    join, prop = cfg.get("REGION_NAME_JOIN") or "", cfg["REGION_PROPERTY"]

    def add_name(f):
        parts = [ee.String(ee.Algorithms.String(f.get(k))) for k in fields]
        name = parts[0]
        for p in parts[1:]:
            name = name.cat(join).cat(p)
        return f.set(prop, name)

    return fc.map(add_name)


def region_geometry(cfg, crs, logger=None, simplify_m=100):
    """Shapely geometry of REGION in the mosaic CRS (None if REGION is None)."""
    import geopandas as gpd
    from shapely.geometry import shape

    region = cfg.get("REGION")
    if not region:
        return None
    P = paths(cfg)
    cache = f"{P['splits']}/region_{region_slug(cfg)}.geojson"
    if gcs.blob_exists(cache):
        gj = json.loads(gcs.read_text(cache))
    else:
        import ee
        asset, prop = cfg["REGIONS_ASSET"], cfg["REGION_PROPERTY"]
        all_regions = regions_collection(cfg)
        fc = all_regions.filter(ee.Filter.eq(prop, region))
        if fc.size().getInfo() == 0:
            names = sorted(set(map(str, all_regions.aggregate_array(prop).getInfo())))
            raise ValueError(f"A regiao '{region}' ({prop}) nao existe em {asset}. Opcoes: {names}")
        gj = fc.geometry().simplify(maxError=simplify_m).getInfo()
        write_text(cfg, cache, json.dumps(gj))
        _log(logger, f"[REGIAO] geometria de {region} salva em gs://{P['bucket']}/{cache}")
    geom = gpd.GeoSeries([shape(gj)], crs="EPSG:4326").to_crs(crs).iloc[0]
    return geom


# ---------------------------------------------------------------------------
# GCS helpers (only inside the experiment folder)
# ---------------------------------------------------------------------------

def _check_inside(cfg, rel_path):
    root = paths(cfg)["root"]
    if not rel_path.startswith(root + "/"):
        raise PermissionError(f"Recusado: so e permitido escrever dentro de {root} ({rel_path})")


def upload(cfg, local_path, rel_path):
    _check_inside(cfg, rel_path)
    gcs.upload(local_path, rel_path)


def write_text(cfg, rel_path, text):
    _check_inside(cfg, rel_path)
    gcs.write_text(rel_path, text)


def write_json(cfg, rel_path, data):
    write_text(cfg, rel_path, json.dumps(data, indent=2, default=str))


def delete_files(cfg, gs_paths):
    """Delete objects, only inside the experiment folder."""
    for p in gs_paths:
        _check_inside(cfg, gcs.gs_path(p)[5:].split("/", 1)[1])
    gcs.rm_files(gs_paths)


def read_json(rel_path):
    return json.loads(gcs.read_text(rel_path))


def download_prefix(rel_prefix, local_dir, suffix=None):
    """Copy every file under a GCS prefix to a local folder; returns local paths."""
    os.makedirs(local_dir, exist_ok=True)
    out = []
    for remote in gcs.list_files(rel_prefix, suffix=suffix):
        local = os.path.join(local_dir, remote.rstrip("/").split("/")[-1])
        if not os.path.exists(local):
            gcs.download(remote, local)
        out.append(local)
    return sorted(out)


def local_dir(*parts):
    return gcs.temp_dir(*parts)
