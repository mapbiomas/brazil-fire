# MapBiomas Fire Network - Version 00 [LEGACY]
Developed by [Amazon Environmental Research Institute - IPAM](https://ipam.org.br/pt/)

> **⚠️ LEGACY VERSION**: This directory contains the original Phase 1/Collection 1 classification algorithms. It is kept for historical reference. For the most recent algorithms, please utilize `network/catalog/development/version_01/`.
>
> **Location:** moved from `network/version_01` to `network/catalog/development/version_00`. It is kept for compatibility; removal is planned after a public notice.
>
> **Reference for the legacy root mirror:** `network/version_01/`.

## About
This directory contains the legacy scripts for mapping burned areas in South American countries. Utilizing a Deep Neural Network (DNN) algorithm for model training and classification, these scripts formed the basis of our early international expansions.

## Methodology Overview (Collection 1)
The methodology encompasses data preparation, model training, and final classification.

### 1. **Pre-Processing**
#### Step 01: Export Annual Landsat Mosaics
- Generation of annual Landsat mosaics via Google Earth Engine (GEE).
- Export processes to Google Cloud Storage.

#### Step 02: Export Training Samples
- Collection and export of training samples (fire and non-fire).

### 2. **Classification**
#### Step 03: Train the Model
- Training of the DNN model utilizing the derived training samples.

#### Step 04: Classify Burned Areas
- Application of the trained model to classify burned areas.
- Basic validation using Landsat mosaics.

## Repository Structure
- **`classification_algorithms/`**: The main legacy code structure corresponding to the routines described above.

## Contact
[Vera Arruda](mailto:vera.arruda@ipam.org.br)  
[Wallace Silva](mailto:wallace.silva@ipam.org.br)
