# Entrenar localmente los modelos NODE21 (Windows)

Este proyecto entrena clasificadores binarios (nódulo vs no nódulo) sobre el dataset NODE21 utilizando PyTorch.

## Requisitos
- Python 3.10+
- Windows 10/11
- (Opcional) GPU NVIDIA con CUDA/cuDNN instalados para aceleración

## Instalación
En PowerShell (ajusta la ruta del proyecto según tu ordenador):
```powershell
# Seleccionar directorio del proyecto (ejemplo, ajusta según tu PC)
cd <seleccionar-directorio-del-proyecto>

# Crear y activar entorno virtual
python -m venv .venv
.venv\Scripts\Activate.ps1

# Instalar dependencias
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Estructura esperada del dataset
- `dataset_node21/cxr_images/proccessed_data/metadata.csv`
- `dataset_node21/cxr_images/proccessed_data/images/` (imágenes PNG/JPG)

## Ejecutar los notebooks localmente
1. Abrir VS Code y el workspace del proyecto.
2. Seleccionar el kernel de Python del entorno `.venv` (Menú superior del notebook → Selector de kernel → elige el intérprete de `.venv`).
3. Abrir uno de los notebooks en `notebooks/`:
   - `02_train_SimpleCNN.ipynb`
   - `03_train_ResNet50.ipynb`
   - `04_train_DenseNet121.ipynb`
   - `05_train_EfficientNetB0.ipynb`
4. Ejecutar las celdas de arriba hacia abajo.

Cada notebook:
- Configura rutas locales (`PROJECT_PATH` y `DATASET_PATH`).
- Carga `metadata.csv` y genera el split 80/20 estratificado (`random_state=42`).
- Entrena por hasta 20 épocas con `lr=0.01` y early stopping según pérdida de entrenamiento (paciencia=3).
- Evalúa en el test set y guarda `best_model.pth` y `metrics.json` en `models/<Modelo>/`.

## Verificar aceleración (CUDA)
En la primera celda de cada notebook se imprime si hay GPU disponible (`torch.cuda.is_available()`). En Windows, para usar GPU NVIDIA necesitas instalar PyTorch con CUDA compatible con tu driver:

1. Verifica la versión de CUDA soportada por tu GPU/driver.
2. Instala PyTorch con CUDA siguiendo la guía oficial: https://pytorch.org/get-started/locally/

Ejemplo (PowerShell) para CUDA 12.1:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Si `torch.cuda.is_available()` es `False`, el entrenamiento se ejecutará en CPU.

## Problemas comunes
- Rutas: Asegura que `dataset_node21` existe en la raíz del proyecto.
- Activación del entorno: En PowerShell usa `.venv\Scripts\Activate.ps1`.
- CUDA: Instala el build correcto de PyTorch con CUDA que coincida con tu driver.
 - Directorio del proyecto: Usa `cd <seleccionar-directorio-del-proyecto>` y ajusta la ruta al folder donde está el repositorio en tu PC.
