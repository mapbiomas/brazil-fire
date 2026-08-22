# Collection 04 - MapBiomas Fire

This directory contains the complete pipeline of scripts and processing workflows for **Collection 04** and its subsequent update, **Collection 04.1**, of the MapBiomas Fire project.

## 📌 Overview
Collection 04 delivers the historical mapping of burned area scars across Brazil using Landsat satellite imagery. Version 4.1 introduces reclassification refinements and spatial adjustments based on the most recent Land Use and Land Cover (LULC) data.

For detailed information about the data and modeling approach, please refer to the [MapBiomas Fire Algorithm Theoretical Basis Document (ATBD)](https://brasil.mapbiomas.org/metodo-mapbiomas-fogo/).

## 📂 Directory Structure
- **[01-pre_classification](./01-pre_classification/)**: Tools for mosaic generation, training sample collection, and scene screening.
- **[02-classification_algorithms](./02-classification_algorithms/)**: Classification algorithms and machine learning models (e.g., Random Forest, DNN).
- **[03-post_classification](./03-post_classification/)**: Temporal filters, exclusion mask applications, and generation of active fire/burned area subproducts.
- **[04-statistics](./04-statistics/)**: Scripts for area calculation and exporting tabular statistics to Google Drive.

## 🚀 Workflow Steps
1. **Pre-processing**: Data preparation, image composite adjustments, and supporting tools.
2. **Classification**: Execution of the primary burned area mapping models.
3. **Post-processing**: Refinement of raw classification outputs and annual fire mask generation.
4. **Statistics**: Data consolidation for dashboards, reports, and final matrix outputs.

## ✉️ Contact
For clarifications or to report issues/bugs, please contact [contato@mapbiomas.org](mailto:contato@mapbiomas.org).

---
**MapBiomas Fire Team**
