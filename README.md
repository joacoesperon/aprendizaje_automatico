# NODE21 - Lung Nodule Classification

Proyecto de clasificacion binaria de nodulos pulmonares en radiografias de torax (CXR) usando el dataset NODE21.

## Estructura del Proyecto

```
├── src/
│   ├── config.py                 # Configuracion global
│   ├── models.py                 # SimpleCNN, ResNet50, DenseNet121, EfficientNetB0
│   ├── data_loader.py            # Dataset y transformaciones
│   ├── data_splits.py            # Generacion de splits train/test
│   └── evaluate.py               # Metricas
├── notebooks/
│   ├── 01_eda.ipynb              # Analisis exploratorio (COMPLETADO)
│   ├── 02_train_SimpleCNN.ipynb  # Entrenar SimpleCNN
│   ├── 03_train_ResNet50.ipynb   # Entrenar ResNet50
│   ├── 04_train_DenseNet121.ipynb # Entrenar DenseNet121
│   ├── 05_train_EfficientNetB0.ipynb # Entrenar EfficientNetB0
│   └── 06_comparison.ipynb       # Comparar modelos
├── dataset_node21/
│   └── cxr_images/proccessed_data/
│       ├── images/               # Imagenes .mha
│       └── metadata.csv
└── models/                       # Modelos entrenados (auto-generado)
```

## Modelos

1. **SimpleCNN** - Red convolucional desde cero (4 bloques)
2. **ResNet50** - Transfer Learning con ImageNet
3. **DenseNet121** - Transfer Learning con ImageNet
4. **EfficientNet-B0** - Transfer Learning con ImageNet

## Uso en Google Colab

### 1. Subir proyecto a Google Drive

Subir toda la carpeta `aprendizaje_automatico/` a tu Google Drive.

### 2. Ejecutar notebooks en orden

Cada notebook esta preparado para Google Colab con GPU:

1. **01_eda.ipynb** - Analisis exploratorio (ya completado)
2. **02_train_SimpleCNN.ipynb** - Entrenar primer modelo
3. **03_train_ResNet50.ipynb** - Transfer Learning
4. **04_train_DenseNet121.ipynb** - Transfer Learning
5. **05_train_EfficientNetB0.ipynb** - Transfer Learning
6. **06_comparison.ipynb** - Comparar resultados

### 3. Configuracion de rutas

En cada notebook, ajustar esta linea segun donde este tu proyecto en Drive:

```python
PROJECT_PATH = Path('/content/drive/MyDrive/aprendizaje_automatico')
```

## Dataset

- Total imagenes: 4,882
- Positivas (con nodulos): 1,134 (23%)
- Negativas (sin nodulos): 3,748 (77%)
- Formato: .mha (Medical Imaging)
- Resolucion: 1024x1024
- Split: 80% train, 20% test (estratificado)

## Entrenamiento

- K-Fold Cross-Validation: 5 folds en train
- Early Stopping: Patience de 10 epochs
- Batch size: 32
- Learning rate: 0.001
- Random seed: 42 (reproducibilidad)

## Metricas

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC

## Salida

Cada modelo genera:

```
models/
└── [ModelName]/
    ├── best_model.pth        # Modelo entrenado
    └── metrics.json          # Metricas en test
```

## Workflow

1. Analisis exploratorio (01_eda.ipynb) - COMPLETADO
2. Entrenar 4 modelos (notebooks 02-05) - Uno por uno en Colab
3. Comparar resultados (06_comparison.ipynb)

## Reproducibilidad

Todos los splits usan random_state=42:
- Split train/test: random_state=42
- K-Fold CV: random_state=42
- torch.manual_seed: 42

Los splits se generan identicamente en cada notebook, garantizando que todos los modelos usen exactamente el mismo train/test set.

## Notas

- Cada notebook es independiente (genera sus propios splits)
- Los splits son identicos porque usan el mismo random_seed
- El test set nunca se usa durante entrenamiento
- K-Fold CV se aplica solo en train para early stopping
- Entrenar un modelo a la vez en Colab (GPU limitada)
