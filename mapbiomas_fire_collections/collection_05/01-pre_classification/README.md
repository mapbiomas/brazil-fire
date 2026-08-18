# Collection 05 - 01. Pre-classification

This directory contains the scripts and supporting tools used to prepare the datasets required for the initial burned-area mapping workflow.

## 🛠️ Main Script

- **`01-toolkit_for_collection_samples_and_export_mosaics_to_google_cloud.js`**: Main interface for navigating and inspecting Landsat and Sentinel mosaics, collecting training samples, and exporting data for subsequent processing.

## 📁 Auxiliary Directory (`./auxiliar/`)

Contains supporting modules used by the Toolkit or independently during the pre-classification stage:

- **`module-blockList.js`**: Defines scenes or images that should be excluded from processing due to quality issues or noise.
- **`toolkit-investigate-scenes.js`**: Tool for detailed inspection of individual scenes and associated metadata.

## 📖 How to Use

1. Configure the biome and year parameters in the `Toolkit`.
2. Use the visualization layers to inspect mosaics and identify burned areas.
3. Export mosaics or training samples to Google Cloud Storage or Google Earth Engine Assets for use in the classification stage.
