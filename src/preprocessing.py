"""
Transformaciones y augmentación de datos
"""
import torch
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=(224, 224)):
    """
    Transformaciones para entrenamiento (con augmentación)
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_val_test_transforms(image_size=(224, 224)):
    """
    Transformaciones para validación y test (sin augmentación)
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def get_albumentations_train_transforms(image_size=(224, 224)):
    """
    Transformaciones más avanzadas usando Albumentations
    (Opcional, para mejorar resultados)
    """
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.OneOf([
            A.GaussNoise(p=1.0),
            A.GaussianBlur(p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=1.0),
            A.RandomGamma(p=1.0),
        ], p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
