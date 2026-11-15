# M2 mask benchmark: national disk admission

The national mode is experimental and best effort. After downloading the source,
`prepare` estimates additional peak disk usage from the source TIFF size, raster
dimensions and data type. It requires free space to be at least that estimate
plus the configured reserve. This is a **minimum admission threshold**, not a
worst-case bound or a guarantee that the run will complete.

The heuristic adds one byte per pixel for the country mask, an estimated masked
TIFF of 1.5 times the source size plus one bit per pixel, and twice that masked
estimate for COG output and temporary files. The source file's observed size
reflects its own compression and may include masks or overviews; the outputs use
LZW and can compress differently. TIFF metadata, temporary files, concurrent
writers and other factors may exceed the estimate.

The country mask is a tiled LZW GTiff with `SPARSE_OK=YES`, so unwritten zero
blocks can remain sparse; its eventual size depends on the rasterized geometry
and compression. The byte-per-pixel term is a reference allowance, not a
guaranteed maximum. The disk guard checks the reserve during stages and
supervises subprocesses, but sampling and concurrent writes cannot guarantee
that the reserve is never crossed.

The local `disk_estimate.json` and `report.json` record the estimate and free
space; subprocess logs include numerical disk diagnostics when available.
Optional checkpoints record progress and failure status, not resumable raster
intermediates. Failed runs clean up their temporary raster files.

For example, with 28,766,214,368 pixels, a two-byte data type and an
8,143,833,141-byte source, the estimate is 76,200,793,892 additional bytes.
With a 10 GiB reserve, admission requires 86,938,212,132 free bytes after
download. Both 107,381,436,416 and 234,209,320,960 free bytes meet that
threshold; neither implies that national processing will finish.

Offline tests: `python -m unittest discover -s network/catalog/development/version_04/tests/m2_mask_benchmark -p "test*.py"`.
The GDAL rasterize integration test additionally requires `gdal_rasterize` on
`PATH`; it is skipped otherwise. Cloud credentials are not needed for these
synthetic tests.
