"""
Evaluación de modelos y cálculo de métricas
"""
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def evaluate_model(model, dataloader, device='cuda'):
    """
    Evalúa un modelo y retorna predicciones y etiquetas verdaderas
    
    Args:
        model: Modelo a evaluar
        dataloader: DataLoader con datos de test
        device: Dispositivo (cuda/cpu)
        
    Returns:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        y_proba: Probabilidades de clase positiva
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    y_true = []
    y_pred = []
    y_proba = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            outputs = model(images)
            
            # Probabilidades
            probs = torch.softmax(outputs, dim=1)
            
            # Predicciones
            _, predicted = outputs.max(1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_proba.extend(probs[:, 1].cpu().numpy())  # Probabilidad de clase positiva
    
    return np.array(y_true), np.array(y_pred), np.array(y_proba)


def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calcula métricas de clasificación
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        y_proba: Probabilidades (opcional, para AUC)
        
    Returns:
        Diccionario con métricas
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary'),
        'recall': recall_score(y_true, y_pred, average='binary'),
        'f1_score': f1_score(y_true, y_pred, average='binary'),
    }
    
    if y_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
    
    return metrics


def print_metrics(metrics, model_name="Model"):
    """Imprime métricas de forma legible"""
    print(f"\n{'='*50}")
    print(f"Métricas de {model_name}")
    print(f"{'='*50}")
    for metric_name, value in metrics.items():
        print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
    print(f"{'='*50}\n")


def plot_confusion_matrix(y_true, y_pred, class_names=['Sin nódulo', 'Con nódulo'], 
                         save_path=None, model_name="Model"):
    """
    Plotea matriz de confusión
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        class_names: Nombres de las clases
        save_path: Ruta para guardar la figura
        model_name: Nombre del modelo
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusión - {model_name}')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Predicción')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Matriz de confusión guardada en {save_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_proba, save_path=None, model_name="Model"):
    """
    Plotea curva ROC
    
    Args:
        y_true: Etiquetas verdaderas
        y_proba: Probabilidades de clase positiva
        save_path: Ruta para guardar la figura
        model_name: Nombre del modelo
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Curva ROC - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Curva ROC guardada en {save_path}")
    
    plt.show()


def plot_training_history(history, save_path=None, model_name="Model"):
    """
    Plotea historial de entrenamiento
    
    Args:
        history: Diccionario con historial de train/val loss y accuracy
        save_path: Ruta para guardar la figura
        model_name: Nombre del modelo
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'Training/Validation Loss - {model_name}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'Training/Validation Accuracy - {model_name}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Historial de entrenamiento guardado en {save_path}")
    
    plt.show()


def generate_classification_report(y_true, y_pred, class_names=['Sin nódulo', 'Con nódulo']):
    """
    Genera reporte de clasificación detallado
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        class_names: Nombres de las clases
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nReporte de Clasificación:")
    print(report)
    return report
