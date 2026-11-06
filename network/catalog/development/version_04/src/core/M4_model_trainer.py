"""
Fachada de compatibilidade para cadernos legados.
Os modulos foram divididos em M4_data_extractor, M4_hub_manager, M4_algorithms_dnn, M4_analytics, M4_ui.
"""

from M4_ui import run_ui, start_training, ModelTrainerUI
from M4_hub_manager import list_trained_models
from M4_data_extractor import extract_pixels_from_gcs, list_sample_collections_gcs
from M4_algorithms_dnn import ModelTrainer
