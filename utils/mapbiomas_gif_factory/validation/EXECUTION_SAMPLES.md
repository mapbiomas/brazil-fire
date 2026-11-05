# GEE execution samples

These are small technical smoke tests, not full product or visual validation.
They use `mapbiomas-brazil`, Petrolina from FAO GAUL 2025 and 300-pixel output.
No Earth Engine export tasks or assets were created.

| Product | Data structure | Frames | Result | Caveat |
|---|---|---:|---|
| Brazil Fire 5.1 `monthly_burned` | Multiband image, annual bands | 1 (2024) | GIF produced | Visual inspection revealed an incorrect 0-based legend prefix; preset changed to suppress prefixes, fix still needs rendering at normal size |
| Brazil Soil 3 `soil_clay_fraction` | ImageCollection indexed by depth | 2 depth intervals | GIF produced | Test palette is illustrative, not an official legend |
| Brazil Soil 2 `granulometry_clay_percentage` | Multiband image indexed by depth | 2 depth intervals | GIF produced | Test palette is illustrative, not an official legend |
| Brazil Atmosphere 1 `air_temperature_annual` | ImageCollection with multiband variables | 2 years, maximum | GIF produced | Test range and palette are illustrative, not an official legend |

Results only demonstrate that the tested asset, bands, territory and pipeline
executed in this environment. They do not prove coverage, correct masking,
scientific interpretation, correct colors, or support for other territories.
