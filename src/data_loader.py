"""
Utilidades para cargar y procesar imágenes médicas en formato .mha
"""
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from typing import Tuple, Optional
import torch
from torch.utils.data import Dataset
from PIL import Image

from src.config import DATA_DIR, METADATA_FILE, IMAGE_SIZE, IMAGE_DIR


def load_mha_image(image_path: str) -> np.ndarray:
    """
    Carga una imagen en formato .mha usando SimpleITK
    
    Args:
        image_path: Ruta al archivo .mha
        
    Returns:
        Array numpy con la imagen
    """
    try:
        image = sitk.ReadImage(str(image_path))
        image_array = sitk.GetArrayFromImage(image)
        
        # Las imágenes médicas suelen tener una dimensión extra, reducirla si es necesario
        if image_array.ndim == 3:
            image_array = image_array.squeeze()
            
        return image_array
    except Exception as e:
        print(f"Error cargando imagen {image_path}: {e}")
        return None


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza la imagen a rango [0, 1]
    
    Args:
        image: Array numpy con la imagen
        
    Returns:
        Imagen normalizada
    """
    if image is None:
        return None
    
    # Normalización min-max
    image_min = image.min()
    image_max = image.max()
    
    if image_max - image_min > 0:
        normalized = (image - image_min) / (image_max - image_min)
    else:
        normalized = image - image_min
        
    return normalized


def load_metadata(metadata_path: Optional[str] = None) -> pd.DataFrame:
    """
    Carga el archivo metadata.csv con las anotaciones
    
    Args:
        metadata_path: Ruta al archivo metadata.csv (opcional)
        
    Returns:
        DataFrame con los metadatos
    """
    if metadata_path is None:
        metadata_path = METADATA_FILE
        
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"No se encontró el archivo de metadatos en {metadata_path}")
        
    df = pd.read_csv(metadata_path)
    return df


class NODE21Dataset(Dataset):
    """
    Dataset personalizado para cargar imágenes NODE21
    """
    
    def __init__(
        self, 
        image_dir: Path,
        image_names: list,
        labels: list,
        transform=None,
        image_size: Tuple[int, int] = IMAGE_SIZE
    ):
        """
        Args:
            image_dir: Directorio con las imágenes .mha
            image_names: Lista de nombres de archivos
            labels: Lista de etiquetas correspondientes
            transform: Transformaciones a aplicar (opcional)
            image_size: Tamaño al que redimensionar las imágenes
        """
        self.image_dir = Path(image_dir)
        self.image_names = image_names
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        # Construir ruta de la imagen
        image_path = self.image_dir / self.image_names[idx]
        
        # Cargar imagen
        image = load_mha_image(image_path)
        
        if image is None:
            # Retornar imagen en blanco si falla la carga
            image = np.zeros(self.image_size)
        
        # Normalizar
        image = normalize_image(image)
        
        # Convertir a PIL Image para aplicar transformaciones
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image).convert('RGB')
        
        # Redimensionar
        image = image.resize(self.image_size, Image.BILINEAR)
        
        # Aplicar transformaciones si existen
        if self.transform:
            image = self.transform(image)
        
        # Obtener etiqueta
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return image, label


def get_class_distribution(metadata_df: pd.DataFrame) -> dict:
    """
    Calcula la distribución de clases en el dataset
    
    Args:
        metadata_df: DataFrame con columna 'label'
        
    Returns:
        Diccionario con el conteo de cada clase
    """
    distribution = metadata_df['label'].value_counts().to_dict()
    return distribution


def calculate_class_weights(metadata_df: pd.DataFrame) -> torch.Tensor:
    """
    Calcula pesos para balancear clases desbalanceadas
    
    Args:
        metadata_df: DataFrame con columna 'label'
        
    Returns:
        Tensor con los pesos de clase
    """
    distribution = get_class_distribution(metadata_df)
    total_samples = len(metadata_df)
    
    weights = []
    for class_idx in sorted(distribution.keys()):
        weight = total_samples / (len(distribution) * distribution[class_idx])
        weights.append(weight)
    
    return torch.tensor(weights, dtype=torch.float32)


def get_train_transforms():
    """Transformaciones para entrenamiento (con augmentation)"""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_val_test_transforms():
    """Transformaciones para validación/test (sin augmentation)"""
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

