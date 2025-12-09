"""
Definición de modelos de clasificación
"""
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, DenseNet121_Weights, EfficientNet_B0_Weights


class ResNet50Transfer(nn.Module):
    """
    Modelo 1: ResNet50 con Transfer Learning
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet50Transfer, self).__init__()
        
        # Cargar modelo pre-entrenado
        if pretrained:
            self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.model = models.resnet50(weights=None)
        
        # Congelar capas iniciales (opcional, descongelar después)
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Reemplazar la última capa fully connected
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_layers(self, num_layers=10):
        """Descongelar últimas capas para fine-tuning"""
        layers = list(self.model.children())
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True


class DenseNet121Transfer(nn.Module):
    """
    Modelo 2: DenseNet121 con Transfer Learning
    Especialmente bueno para imágenes médicas
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(DenseNet121Transfer, self).__init__()
        
        if pretrained:
            self.model = models.densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.model = models.densenet121(weights=None)
        
        # Congelar capas
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modificar clasificador
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_layers(self):
        """Descongelar todas las capas"""
        for param in self.model.parameters():
            param.requires_grad = True


class EfficientNetTransfer(nn.Module):
    """
    Modelo 3: EfficientNet-B0 con Transfer Learning
    Arquitectura moderna y eficiente
    """
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetTransfer, self).__init__()
        
        if pretrained:
            self.model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        else:
            self.model = models.efficientnet_b0(weights=None)
        
        # Congelar capas
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Modificar clasificador
        num_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)
    
    def unfreeze_layers(self):
        """Descongelar todas las capas"""
        for param in self.model.parameters():
            param.requires_grad = True


class SimpleCNN(nn.Module):
    """
    Modelo 4: CNN simple entrenada desde cero
    Arquitectura custom para comparación con transfer learning
    """
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Bloque 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Bloque 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
            
            # Bloque 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_model(model_name: str, num_classes: int = 2, pretrained: bool = True):
    """
    Factory function para obtener un modelo por nombre
    
    Args:
        model_name: Nombre del modelo ('resnet50', 'densenet121', 'efficientnet', 'simplecnn')
        num_classes: Número de clases
        pretrained: Si usar pesos pre-entrenados
        
    Returns:
        Modelo instanciado
    """
    models_dict = {
        'resnet50': ResNet50Transfer,
        'densenet121': DenseNet121Transfer,
        'efficientnet': EfficientNetTransfer,
        'simplecnn': SimpleCNN,
    }
    
    model_name = model_name.lower()
    if model_name not in models_dict:
        raise ValueError(f"Modelo {model_name} no reconocido. Opciones: {list(models_dict.keys())}")
    
    if model_name == 'simplecnn':
        return models_dict[model_name](num_classes=num_classes)
    else:
        return models_dict[model_name](num_classes=num_classes, pretrained=pretrained)
