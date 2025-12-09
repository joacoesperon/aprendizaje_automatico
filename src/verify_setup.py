"""
Script para verificar que el entorno está correctamente configurado
"""
import sys
from pathlib import Path

def check_imports():
    """Verifica que todas las librerías necesarias estén instaladas"""
    print("Verificando imports...")
    
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'sklearn': 'scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'PIL': 'Pillow',
        'SimpleITK': 'SimpleITK',
        'opencxr': 'OpenCXR',
        'albumentations': 'Albumentations',
    }
    
    failed = []
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NO INSTALADO")
            failed.append(name)
    
    if failed:
        print(f"\n⚠️  Paquetes faltantes: {', '.join(failed)}")
        return False
    else:
        print("\n✓ Todos los paquetes están instalados correctamente")
        return True


def check_cuda():
    """Verifica disponibilidad de CUDA/GPU"""
    print("\nVerificando GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA disponible")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Versión CUDA: {torch.version.cuda}")
        else:
            print("⚠️  CUDA no disponible - Se usará CPU")
            print("  Considera usar Google Colab o Kaggle para GPU gratuita")
    except Exception as e:
        print(f"✗ Error verificando CUDA: {e}")


def check_directory_structure():
    """Verifica que exista la estructura de directorios"""
    print("\nVerificando estructura de directorios...")
    
    project_root = Path(__file__).parent.parent
    required_dirs = [
        'data/raw',
        'data/processed',
        'notebooks',
        'src',
        'models',
        'results/figures',
        'results/metrics',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - NO EXISTE")
            all_exist = False
    
    return all_exist


def check_data_files():
    """Verifica que existan los archivos de datos"""
    print("\nVerificando archivos de datos...")
    
    project_root = Path(__file__).parent.parent
    metadata_file = project_root / 'data' / 'raw' / 'metadata.csv'
    
    if metadata_file.exists():
        print(f"✓ metadata.csv encontrado")
        try:
            import pandas as pd
            df = pd.read_csv(metadata_file)
            print(f"  Total de imágenes: {len(df)}")
            if 'label' in df.columns:
                print(f"  Distribución de clases:")
                print(df['label'].value_counts().to_dict())
        except Exception as e:
            print(f"⚠️  Error leyendo metadata.csv: {e}")
    else:
        print(f"⚠️  metadata.csv NO encontrado en data/raw/")
        print(f"  Descarga los datos de NODE21 y colócalos en data/raw/")


def main():
    """Ejecuta todas las verificaciones"""
    print("="*60)
    print("VERIFICACIÓN DEL ENTORNO - NODE21")
    print("="*60)
    
    # Verificar imports
    imports_ok = check_imports()
    
    # Verificar CUDA
    check_cuda()
    
    # Verificar estructura
    structure_ok = check_directory_structure()
    
    # Verificar datos
    check_data_files()
    
    # Resumen
    print("\n" + "="*60)
    if imports_ok and structure_ok:
        print("✓ ENTORNO CONFIGURADO CORRECTAMENTE")
        print("\nSiguientes pasos:")
        print("1. Descarga el dataset NODE21 si aún no lo has hecho")
        print("2. Coloca metadata.csv y las imágenes en data/raw/")
        print("3. Ejecuta los notebooks en orden: 01, 02, 03")
    else:
        print("⚠️  CONFIGURACIÓN INCOMPLETA")
        print("Revisa los errores anteriores")
    print("="*60)


if __name__ == "__main__":
    main()
