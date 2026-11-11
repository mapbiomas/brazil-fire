# Fire legend snapshot

These CSVs are copied unchanged from `utils/fire_legends` in
https://github.com/mapbiomas/brazil-fire at commit
`ad70d566e5676402c597339d10907e1fdf2163f9`. They include `registry.csv` and
the legend CSVs. Run `python build_vera_fire_legends.py` from the package root
to reproduce the packaged JSON.

The generator indexes active records with explicit country, collection,
product, and asset. Only consecutive integer classes with at least two colors
are represented; single or sparse classes and unmatched records return no
automatic style. Numeric collection aliases are normalized. This snapshot
does not validate access, bands, masks, or interpretation in Earth Engine.
