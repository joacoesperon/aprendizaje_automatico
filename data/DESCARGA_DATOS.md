# Instrucciones para Descargar el Dataset NODE21

## Opción 1: Descarga Manual desde Grand Challenge

1. **Ve al sitio oficial:**
   - URL: https://node21.grand-challenge.org/
   - Crea una cuenta si no la tienes

2. **Descarga el dataset:**
   - Ve a la sección "Data" o "Download"
   - Descarga el dataset de entrenamiento (Training set)
   - El archivo será grande (~varios GB)

3. **Extrae los archivos:**
   - Descomprime el archivo descargado
   - Busca el archivo `metadata.csv`
   - Busca la carpeta con las imágenes `.mha`

4. **Organiza en tu proyecto:**
   ```
   data/raw/
   ├── metadata.csv
   └── images/          # o el nombre de la carpeta con imágenes
       ├── image1.mha
       ├── image2.mha
       └── ...
   ```

## Opción 2: Usar OpenCXR (si está disponible)

OpenCXR puede descargar algunos datasets automáticamente:

```python
import opencxr

# Verificar si NODE21 está disponible
# (nota: puede que necesites acceso especial)
# opencxr.download_node21('data/raw/')
```

## Estructura Esperada del Metadata

El archivo `metadata.csv` debe contener al menos estas columnas:

- `image_id` o `filename`: Nombre del archivo de imagen
- `label`: 0 (sin nódulo) o 1 (con nódulo)
- Opcionalmente: `x`, `y`, `width`, `height` (bounding boxes)

Ejemplo:
```csv
image_id,label,x,y,width,height
image_001.mha,0,,,
image_002.mha,1,256,512,64,64
image_003.mha,1,128,256,48,48
...
```

## Verificar la Descarga

Después de descargar, ejecuta:

```bash
python src/verify_setup.py
```

Esto verificará que los archivos estén en el lugar correcto.

## Tamaño del Dataset

- **Total de imágenes:** ~4,882
- **Con nódulos:** ~1,134 imágenes (23%)
- **Sin nódulos:** ~3,748 imágenes (77%)
- **Tamaño aproximado:** 3-5 GB

## Problemas Comunes

### No puedo acceder a Grand Challenge
- Necesitas crear una cuenta gratuita
- Acepta los términos de uso del dataset
- Puede tardar unos minutos en activarse tu cuenta

### El archivo está corrupto
- Verifica que la descarga se completó
- Prueba descargar nuevamente
- Verifica el checksum si está disponible

### No encuentro metadata.csv
- Puede estar dentro de una subcarpeta
- Busca archivos con nombres similares: `labels.csv`, `annotations.csv`
- Si no existe, tendrás que crear uno basado en la estructura del dataset

## Alternativas si no puedes descargar NODE21

Si tienes problemas con NODE21, puedes probar con otros datasets de radiografías:

1. **NIH Chest X-ray Dataset:**
   - URL: https://www.kaggle.com/datasets/nih-chest-xrays/data
   - Más accesible, en Kaggle
   - Similar al problema de NODE21

2. **CheXpert:**
   - URL: https://stanfordmlgroup.github.io/competitions/chexpert/
   - Dataset de Stanford
   - Requiere registro

3. **MIMIC-CXR:**
   - Dataset muy completo
   - Requiere credenciales de PhysioNet

## Siguiente Paso

Una vez descargados los datos, continúa con:
```bash
# Verificar instalación
python src/verify_setup.py

# Abrir notebook de exploración
jupyter notebook notebooks/01_exploracion_datos.ipynb
```
