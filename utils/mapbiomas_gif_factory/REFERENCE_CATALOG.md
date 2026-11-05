# Reference catalog

The Products panel exposes the packaged reference inventory by initiative,
theme and collection. Reference records are not automatically validated.
The existing presets remain available without changes.

To add a reference layer for an experimental run, select it, enter minimum,
maximum, comma-separated six-digit hex colors and one label per color, then
confirm that the asset, bands, image type and legend have been checked. The
added layer is marked experimental and persists only for the current UI
session. The adapter accepts multiband images and simple temporal image
collections. For collections containing multiple multiband variables, use
the option-discovery button and select a variable. Soil depth is detected for
known collection and multiband layouts. Collections with repeated periods,
more than 500 images, products requiring a processor, and legends whose pixel
values are not represented correctly still require adaptation.

Some inventory records have no published asset ID and cannot be added this
way. The inventory does not establish asset access, palette correctness or a
successful GIF. Use an authorized Earth Engine project and verify each
candidate before sharing a result. Territories for initiatives without presets
must be supplied separately via a custom FeatureCollection.
