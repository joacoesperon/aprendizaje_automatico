# NODE21 - Clasificación de Nódulos Pulmonares

Este proyecto implementa clasificación binaria de radiografías de pecho para detectar nódulos pulmonares usando el dataset NODE21.

## Estructura del Proyecto

```
aprendizaje_automatico/
├── data/
│   ├── raw/                    # Coloca aquí los datos descargados de NODE21
│   │   ├── metadata.csv        # Archivo de anotaciones
│   │   └── images/             # Carpeta con imágenes .mha
│   └── processed/              # Datos preprocesados
├── notebooks/
│   ├── 01_exploracion_datos.ipynb
│   ├── 02_entrenamiento_modelos.ipynb
│   └── 03_evaluacion_resultados.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuración global
│   ├── data_loader.py          # Carga de imágenes .mha
│   ├── preprocessing.py        # Transformaciones y augmentación
│   ├── models.py               # Definición de los 4 modelos
│   ├── train.py                # Entrenamiento
│   └── evaluate.py             # Evaluación y métricas
├── models/                     # Modelos entrenados (.pth)
├── results/
│   ├── figures/                # Gráficos y visualizaciones
│   └── metrics/                # Métricas en JSON/CSV
├── requirements.txt
└── README.md
```

## Instalación

El entorno ya está configurado con todas las dependencias necesarias.

## Modelos Implementados

### 1. ResNet50 (Transfer Learning)
- Pre-entrenado en ImageNet
- Fine-tuning de últimas capas

### 2. DenseNet121 (Transfer Learning)
- Pre-entrenado en ImageNet
- Popular en aplicaciones médicas

### 3. EfficientNet-B0 (Transfer Learning)
- Arquitectura moderna y eficiente

### 4. SimpleCNN (Desde cero)
- Arquitectura custom para comparación

## Siguiente Paso: Descargar Datos

1. Ve a: https://node21.grand-challenge.org
2. Descarga el dataset de entrenamiento
3. Extrae los archivos en `data/raw/`

## Flujo de Trabajo

1. **Exploración de Datos** - Visualizar y analizar
2. **Entrenamiento** - Entrenar los 4 modelos
3. **Evaluación** - Comparar resultados

## Métricas de Evaluación

- Accuracy
- Precision
- Recall
- F1-Score
- AUC-ROC
