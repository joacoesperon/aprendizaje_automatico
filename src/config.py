"""
Configuración global del proyecto
"""
import os
from pathlib import Path
import torch

# Directorios del proyecto
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset_node21" / "cxr_images" / "proccessed_data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"

# Crear directorios si no existen
for dir_path in [MODELS_DIR, FIGURES_DIR, METRICS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Configuración de datos
METADATA_FILE = DATA_DIR / "metadata.csv"
IMAGE_DIR = DATA_DIR / "images"  # Directorio donde están las imágenes .mha
IMAGE_SIZE = (224, 224)  # Tamaño para redimensionar las imágenes
BATCH_SIZE = 32
NUM_WORKERS = 4

# Configuración para Google Colab (se detectará automáticamente)
try:
    import google.colab
    IN_COLAB = True
    # En Colab ajustamos algunos parámetros
    NUM_WORKERS = 2  # Colab tiene limitaciones de CPU
except:
    IN_COLAB = False

# Configuración de entrenamiento
# RANDOM_SEED: Usar SOLO para splits train/test y K-Fold CV (reproducibilidad)
RANDOM_SEED = 42  # NO cambiar: afecta reproducibilidad de splits
TRAIN_RATIO = 0.80  # 80% train, 20% test (sin validation set)
TEST_RATIO = 0.20
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 10
K_FOLD_SPLITS = 5  # Para K-Fold CV dentro del train set

# Configuración de dispositivo (detectar automáticamente)
if torch.cuda.is_available():
    DEVICE = "cuda"
    print("GPU disponible - Usando CUDA")
else:
    DEVICE = "cpu"
    print("GPU no disponible - Usando CPU")
    if not IN_COLAB:
        print("Se recomienda usar Google Colab con GPU habilitada para entrenar los modelos")

# Clases
CLASS_NAMES = ["sin_nodulo", "con_nodulo"]
NUM_CLASSES = 2

# Orden de entrenamiento de modelos (SimpleCNN primero)
MODELS_ORDER = [
    "SimpleCNN",      # 1. Modelo desde cero (primer modelo a entrenar)
    "ResNet50",       # 2. Transfer Learning
    "DenseNet121",    # 3. Transfer Learning
    "EfficientNetB0"  # 4. Transfer Learning
]
