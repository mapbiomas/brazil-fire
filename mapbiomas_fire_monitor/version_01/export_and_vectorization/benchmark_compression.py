"""Benchmark de compressao do mosaico (offline, sob demanda).

Remonta 1 mes com diferentes codecs (-a_nodata 0) e imprime o tamanho de cada
COG, para escolher a compactacao mais eficiente.

Uso (a partir da pasta version_01, onde ha GDAL + acesso GCS, ex.: Colab):
    python -m export_and_vectorization.benchmark_compression \
        --country indonesia --year 2024 --month 7
    python -m export_and_vectorization.benchmark_compression \
        --country brazil --year 2024 --month 7 --codecs LZW DEFLATE9 --blocksize 512

Requer: gdalbuildvrt + gdal_translate (GDAL) e tiles ja exportados no GCS.
"""

import argparse
import os
import shutil
import subprocess
import time

from . import config
from .mosaic import list_tiles
from .state import _get_fs

VARIANTS = {
    "LZW":      ["-co", "COMPRESS=LZW", "-co", "PREDICTOR=2"],
    "DEFLATE9": ["-co", "COMPRESS=DEFLATE", "-co", "ZLEVEL=9", "-co", "PREDICTOR=2"],
    "ZSTD15":   ["-co", "COMPRESS=ZSTD", "-co", "ZSTD_LEVEL=15", "-co", "PREDICTOR=2"],
}


def run(country, year, month, codecs=None, blocksize=None, work=None):
    if country:
        config.COUNTRY = country
    if codecs:
        codecs = [c for c in codecs if c in VARIANTS]
    else:
        codecs = list(VARIANTS)

    fs = _get_fs()
    tiles = list_tiles(year, month)
    if not tiles:
        print(f"No tiles for {config.COUNTRY} {year}_{month:02d}. Run the Export step first.")
        return

    work = work or f"/content/temp/bench_{year}_{month:02d}_{int(time.time())}"
    os.makedirs(work, exist_ok=True)
    inputs = os.path.join(work, "inputs.txt")
    vrt = os.path.join(work, "bench.vrt")

    with open(inputs, "w") as f:
        for t in tiles:
            f.write(f"/vsigs/{t}\n")

    r = subprocess.run(["gdalbuildvrt", "-input_file_list", inputs, vrt], capture_output=True, text=True)
    if r.returncode != 0:
        print("gdalbuildvrt failed:", (r.stderr or "").strip())
        return

    base = [
        "gdal_translate", "-of", "GTiff", "-ot", "Byte", "-a_nodata", "0",
        "-co", "TILED=YES", "-co", "BIGTIFF=YES", "-co", "NUM_THREADS=ALL_CPUS",
    ]
    if blocksize:
        base += ["-co", f"BLOCKSIZE={blocksize}"]

    print(f"Comparing compression for {config.COUNTRY} {year}_{month:02d} "
          f"({len(tiles)} tiles) [{'BLOCKSIZE=' + str(blocksize) if blocksize else 'default tiles'}]:")
    results = {}
    for name in codecs:
        out = os.path.join(work, f"{name.replace('+', '_')}.tif")
        cmd = base + VARIANTS[name] + [vrt, out]
        start = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        elapsed = time.time() - start
        if r.returncode != 0:
            print(f"{name:10s}  ERROR: {(r.stderr or '').strip()[-200:]}")
            continue
        size = os.path.getsize(out) / 1e6
        results[name] = size
        print(f"{name:10s}  {size:8.1f} MB  ({elapsed:.1f}s)")

    shutil.rmtree(work, ignore_errors=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default=None, help="country code (ex.: indonesia)")
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--codecs", nargs="+", default=None,
                        help="subset of codecs (default: all)")
    parser.add_argument("--blocksize", type=int, default=None,
                        help="optional COG block size (ex.: 512)")
    parser.add_argument("--work", default=None, help="work dir (default: /content/temp/...)")
    args = parser.parse_args()
    run(args.country, args.year, args.month, args.codecs, args.blocksize, args.work)


if __name__ == "__main__":
    main()
