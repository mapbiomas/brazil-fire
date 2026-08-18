# Collection 05 - MapBiomas Fire

This directory contains the scripts and processing workflows for **Collection 05** of the MapBiomas Fire project.

## 📌 Overview

Collection 05 contains the workflows used for burned-area mapping in Brazil, including data preparation, classification, post-classification processing, and the generation of statistics and derived products.

## 📂 Directory Structure

- **[01-pre_classification](./01-pre_classification/)**: Tools for mosaic generation, sample collection, and satellite scene analysis.
- **[02-classification_algorithms](./02-classification_algorithms/)**: Classification algorithms and models, including Random Forest.
- **[03-post_classification](./03-post_classification/)**: Temporal filters, exclusion masks, and generation of derived products.
- **[04-statistics](./04-statistics/)**: Scripts for area calculation and export of tabular statistics to Google Drive.

## 🚀 Workflow

1. **Pre-classification**: Prepare input data, mosaics, samples, and supporting datasets.
2. **Classification**: Run the primary burned-area classification workflows.
3. **Post-classification**: Refine classification results, apply filters and masks, and generate annual products.
4. **Statistics**: Calculate and consolidate statistics for analysis, reporting, and visualization.
