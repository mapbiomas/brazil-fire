#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MapBiomas Fire - Data Processing Environment
--------------------------------------------

This script initializes the Python environment used for processing
MapBiomas Fire datasets and interacting with Google Earth Engine.

It loads the required scientific, geospatial, and machine learning
libraries and initializes the Earth Engine API.

Main capabilities supported by this environment:
- Google Earth Engine data access
- Geospatial raster processing
- Scientific computation and visualization
- Machine learning (TensorFlow 1.x compatibility)
- Geometry manipulation and coordinate transformations

Notes
-----
Earth Engine authentication may be required before execution.

Example authentication commands:

    earthengine authenticate --auth_mode=notebook
    earthengine authenticate --auth_mode=gcloud --quiet
"""

# ======================================================================
# Core Libraries
# ======================================================================

import os
import glob
import math
import time
import string
import datetime
import tempfile
import zipfile
import urllib.request as urllib

# ======================================================================
# Scientific Computing
# ======================================================================

import numpy as np
import pandas as pd
from scipy import ndimage

# ======================================================================
# Visualization
# ======================================================================

import seaborn as sns
from matplotlib import pyplot as plt
from IPython import display

# ======================================================================
# Geospatial Processing
# ======================================================================

import rasterio
from rasterio.mask import mask
from osgeo import gdal

from shapely.geometry import shape, mapping, box
from shapely.ops import transform

import pyproj

# ======================================================================
# Machine Learning (TensorFlow 1.x Compatibility)
# ======================================================================

# TensorFlow 1.x API is required for legacy models used in the workflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# ======================================================================
# Utility Libraries
# ======================================================================

from termcolor import colored

# ======================================================================
# Google Earth Engine
# ======================================================================

import ee

# Uncomment if authentication is required
# ee.Authenticate()

# Initialize Earth Engine
ee.Initialize()

# ======================================================================
# Temporary Data Directory
# ======================================================================

# Create a temporary directory for intermediate files if needed
# if not os.path.exists("tmp"):
#     os.makedirs("tmp")