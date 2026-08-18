#!/usr/bin/env python3
#-*- coding: utf-8 -*-

# ## import libraries
#earthengine authenticate --auth_mode=notebook
#earthengine authenticate --auth_mode=gcloud --quiet

import ee
from IPython import display
import math
from matplotlib import pyplot as plt # para mostrar imagens ou gráficos
import numpy as np # para computação numérica menos intensiva
import pandas as pd
import seaborn as sns
from scipy import ndimage
import rasterio
from rasterio.mask import mask
import tempfile
#import tensorflow as tf # para redes neurais
import tensorflow.compat.v1 as tf # tensorflow para versão 1.x
tf.disable_v2_behavior() # desabilita a versão 2 do tensorflow
import urllib.request as urllib
import zipfile
import datetime
import time
import os
import string
from termcolor import colored
import glob
from osgeo import gdal
from shapely.geometry import shape, mapping, box
from shapely.ops import transform
import pyproj

#ee.Authenticate()

ee.Initialize()

# cria uma pasta para colocar os dados
# if not os.path.exists('tmp'):
#     os.makedirs('tmp')
