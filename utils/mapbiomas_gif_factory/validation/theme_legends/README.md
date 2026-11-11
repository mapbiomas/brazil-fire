# Provisional registry of non-fire styles

This local inventory reproduces selected styles from `visualization.source.yaml`, a pinned copy of the earlier Factory's `config/visualization.yaml`. It is not an official legend and does not validate whether the styles are suitable for any data. Links to country, collection, asset, and processor, as well as validation, **have not been established**. Names in the YAML are source metadata only, not evidence of those links.

The explicit selection is in `registry.py`; it includes land cover, fragmentation, secondary vegetation, soil, deforestation, and Landsat styles. It excludes fire styles, burned cover, and `nbr_min`, which is associated with the fire workflow. There is no automatic mapping to products. The source's `includes` is not expanded: included files are not part of this inventory.

`styles.csv` contains one row per ID: the source path and SHA-256 of its bytes, the literal YAML style fragment, provisional status, and explicit fields for unverified links and validation. The fragment preserves all fields and their order, YAML types, quotes, lists, colors, empty labels, and special values without interpretation; it does not turn positions into class codes or resolve `auto_bg`. Internal line breaks are contained in quoted CSV cells. The loader rejects duplicate keys, and the generator requires each isolated fragment to reproduce the interpreted source metadata; otherwise it fails rather than inferring values.

From the `utils/mapbiomas_gif_factory` directory, with Python and PyYAML installed:

```powershell
python validation/theme_legends/registry.py
python -m unittest discover -s validation/theme_legends -p "test_*.py" -v
```

Regenerating changes the registry to reflect the current source. The selection is deliberate and should be reviewed manually if the source changes; the tests detect differences between the CSV and the source, but do not establish semantics, links, or legend correctness.
