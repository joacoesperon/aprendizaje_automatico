"""
Configuración global del proyecto
"""
import os
from pathlib import Path

# Directorios del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Crear directorios si no existen
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuración de datos
METADATA_FILE = RAW_DATA_DIR / "metadata.csv"
IMAGE_SIZE = (224, 224)  # Tamaño para redimensionar las imágenes
BATCH_SIZE = 32
NUM_WORKERS = 4

# Configuración de entrenamiento
RANDOM_SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10

# Configuración de dispositivo (GPU/CPU)
DEVICE = "cuda"  # cambiar a "cpu" si no hay GPU disponible

# Clases
CLASS_NAMES = ["sin_nodulo", "con_nodulo"]
NUM_CLASSES = 2
