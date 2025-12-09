"""
Script de entrenamiento de modelos
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import json
from datetime import datetime

from src.config import DEVICE, NUM_EPOCHS, LEARNING_RATE, EARLY_STOPPING_PATIENCE


class EarlyStopping:
    """Early stopping para detener el entrenamiento cuando no mejora"""
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict().copy()
            self.counter = 0


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Entrena el modelo por una época"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Estadísticas
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Actualizar barra de progreso
        pbar.set_postfix({
            'loss': loss.item(),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, dataloader, criterion, device):
    """Valida el modelo"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=NUM_EPOCHS,
    device=DEVICE,
    save_path=None,
    early_stopping_patience=EARLY_STOPPING_PATIENCE
):
    """
    Entrena un modelo completo
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        criterion: Función de pérdida
        optimizer: Optimizador
        scheduler: Learning rate scheduler (opcional)
        num_epochs: Número de épocas
        device: Dispositivo (cuda/cpu)
        save_path: Ruta para guardar el mejor modelo
        early_stopping_patience: Paciencia para early stopping
        
    Returns:
        Diccionario con historial de entrenamiento
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=early_stopping_patience, verbose=True)
    
    # Historial
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    
    print(f"\nEntrenando en {device}")
    print(f"Épocas: {num_epochs}")
    print("-" * 50)
    
    for epoch in range(num_epochs):
        print(f'\nÉpoca {epoch+1}/{num_epochs}')
        
        # Entrenar
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validar
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Imprimir resultados
        print(f'\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f'Mejor modelo guardado con val_acc: {val_acc:.2f}%')
        
        # Scheduler
        if scheduler:
            scheduler.step(val_loss)
        
        # Early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("\nEarly stopping triggered!")
            # Restaurar mejor modelo
            model.load_state_dict(early_stopping.best_model_state)
            break
    
    # Cargar mejor modelo si existe
    if save_path and Path(save_path).exists():
        model.load_state_dict(torch.load(save_path))
        print(f"\nMejor modelo cargado desde {save_path}")
    
    return history


def save_training_history(history, save_path):
    """Guarda el historial de entrenamiento en JSON"""
    with open(save_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Historial guardado en {save_path}")
