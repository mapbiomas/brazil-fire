# 01. Pre-Classification

This directory is responsible for preparing the datasets and essential tools required to initiate the mapping of burned area scars.

## 🛠️ Main Scripts
- **01-toolkit_for_collection_samples_and_export_mosaics_to_google_cloud.js**: The primary interface for data exploration, inspection of Landsat/Sentinel mosaics, and exporting training samples.

## 📁 Auxiliary Folder (`./auxiliar/`)
Contains supporting modules called by the Toolkit or used independently:
- **module-blockList.js**: A list of scenes/images to be excluded from the processing pipeline due to quality issues or artifacts.
- **toolkit-investigate-scenes.js**: A tool for detailed inspection of individual satellite scenes and their metadata.

## 📖 Usage
1. Configure the biome and year parameters within the `Toolkit` script.
2. Utilize the visualization layers to identify burned areas.
3. Export the mosaics or training samples to Google Cloud Storage or GEE Assets to proceed to the classification stage.
