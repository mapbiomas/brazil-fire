# MapBiomas Fire Network
Developed by [Amazon Environmental Research Institute - IPAM](https://ipam.org.br/pt/)

## About
This repository contains scripts for mapping burned areas in South American countries as part of the MapBiomas Fire Network initiative. 
The methodology is designed to be replicable across different regions of the continent, using a Deep Neural Network (DNN) algorithm for model training and classification.

## Methodology Overview
The methodology involves several key steps from data preparation to model training and classification, which can be adapted to various regions within South American countries:

### 1. **Pre-Processing**
#### Step 01: Export Annual Landsat Mosaics
- Use Google Earth Engine (GEE) to generate annual Landsat mosaics for the region of interest.
- Export these mosaics to Google Cloud Storage for further processing.

#### Step 02: Export Training Samples
- Collect and export training samples, distinguishing between fire and non-fire areas for each specific region of the country.
- Ensure that the samples cover a diverse range of ecosystems within the country's boundaries to improve model accuracy.

### 2. **Classification**
#### Step 03: Train the Model
- Train a Deep Neural Network (DNN) model using the collected training samples for the specified region.

#### Step 04: Classify Burned Areas
- Apply the trained model to classify burned areas for the respective region oof the country.
- Validate the results using Landat mosaics to ensure accuracy.

## Repository Structure
- **Scripts**: Contains all the code and algorithms used for mapping burned areas, including model training and classification scripts tailored to different regions and countries.
- **Adapt the Input Data**: Modify the Landsat mosaic generation script to work with the region of interest in your country.

## Contact
For clarifications or to report issues/bugs, please contact:  
[Vera Arruda](mailto:vera.arruda@ipam.org.br)  
[Wallace Silva](mailto:wallace.silva@ipam.org.br)
