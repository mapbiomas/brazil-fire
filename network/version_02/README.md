# MapBiomas Fire Network - Version 02 [ACTIVE]
Developed by [Amazon Environmental Research Institute - IPAM](https://ipam.org.br/pt/)

## About
**This is the active directory for the MapBiomas Fire Network** scripts. This directory contains the latest algorithms (associated with Collection 2 onwards) for mapping burned areas across partner countries in South America.

The methodology utilizes an improved Deep Neural Network (DNN) structure, with better integration, logging, and streamlined execution capabilities.

## Quick Start: Google Colab
The most efficient way to utilize these scripts without complex local environment setup is via our official notebook:

1. Open the [MapBiomas Fire Landsat Classification Notebook (v1)](https://colab.research.google.com/github/mapbiomas/brazil-fire/blob/main/network/fire_landsat_30m/version_02/mapbiomas_fire_classification_v1.ipynb) on Google Colab.
2. In the notebook, clone the repository exactly as indicated:
    ```bash
    !git clone https://github.com/mapbiomas/brazil-fire.git
    ```
3. Follow the sequence of cells for parameterization, training, and algorithmic classification.

## Methodology Overview

### 1. **Pre-Processing**
#### Step 01: Export Annual Landsat Mosaics
- Use Google Earth Engine (GEE) to cleanly export the regions to Google Cloud Storage.

#### Step 02: Collect & Export Training Samples (A_1)
- Advanced routine for collecting fire and non-fire training samples using a graphic Earth Engine interface. 

### 2. **Classification**
#### Step 03: Train the Model (A_2)
- Fast-track DNN training interface using TensorFlow, with regions processed dynamically.

#### Step 04: Classify Burned Areas (A_3)
- Use local models or GCS models to apply classification dynamically over defined extents and validate results with source mosaics.

## Folder Structure
- **`classification_algorithms/`**: Holds all Python scripts (`A_0`, `A_1`, `A_2`, `A_3`) that power the Colab workflow locally and in the cloud.
- **`final_products_steps/`**: Post-classification JavaScript codes for masking, frequency analysis, and exporting definitive LULC products.

## Contact
For clarifications or to report issues/bugs, please contact:  
[Vera Arruda](mailto:vera.arruda@ipam.org.br)  
[Wallace Silva](mailto:wallace.silva@ipam.org.br)
