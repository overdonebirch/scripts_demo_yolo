#!/usr/bin/env python3
"""
Pipeline para automatizar el análisis de imágenes 360° con YOLO
Ejecuta secuencialmente: convert_images.py -> analyze_faces.py -> extract_full_trees.py
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔄 {description}...")
    print(f"Ejecutando: {' '.join(command)}")
    
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print(f"✅ {description} completado exitosamente")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}")
        print(f"Error: {e.stderr}")
        return False

def create_directory(path):
    """Crea un directorio si no existe"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"📁 Directorio creado/verificado: {path}")

def get_image_name_without_extension(image_path):
    """Obtiene el nombre de la imagen sin extensión"""
    return Path(image_path).stem

def main():
    parser = argparse.ArgumentParser(description="Pipeline de análisis de imágenes 360°")
    parser.add_argument("-i", "--image", required=True, help="Ruta a la imagen equirectangular 360°")
    parser.add_argument("-m", "--model", required=True, help="Ruta al modelo YOLO (.pt)")
    parser.add_argument("-r", "--results-dir", default="imagenes_resultados", 
                       help="Directorio base de resultados (por defecto: imagenes_resultados)")
    
    args = parser.parse_args()
    
    # Verificar que la imagen existe
    if not os.path.exists(args.image):
        print(f"❌ Error: La imagen {args.image} no existe")
        sys.exit(1)
    
    # Verificar que el modelo existe
    if not os.path.exists(args.model):
        print(f"❌ Error: El modelo {args.model} no existe")
        sys.exit(1)
    
    # Obtener el directorio del script actual
    script_dir = Path(__file__).parent
    scripts_dir = script_dir / "scripts"
    
    # Verificar que los scripts existen
    convert_script = scripts_dir / "convert_images.py"
    analyze_script = scripts_dir / "analyze_faces.py"
    extract_script = scripts_dir / "extract_full_trees.py"
    
    for script in [convert_script, analyze_script, extract_script]:
        if not script.exists():
            print(f"❌ Error: El script {script} no existe")
            sys.exit(1)
    
    # Configurar rutas
    image_name = get_image_name_without_extension(args.image)
    results_base = Path(args.results_dir)
    
    # Crear estructura de directorios
    main_dir = results_base / image_name
    faces_dir = main_dir / f"{image_name}_faces"
    detections_dir = main_dir / f"{image_name}_detections"
    full_trees_dir = main_dir / f"{image_name}_full_trees"
    
    print(f"🚀 Iniciando pipeline para imagen: {args.image}")
    print(f"📂 Directorio principal: {main_dir}")
    
    # Crear directorios
    create_directory(main_dir)
    create_directory(faces_dir)
    create_directory(detections_dir)
    create_directory(full_trees_dir)
    
    # Paso 1: Convertir imagen 360° en caras
    print(f"\n{'='*60}")
    print("PASO 1: Convertir imagen 360° en caras")
    print(f"{'='*60}")
    
    convert_cmd = [
        "python", str(convert_script),
        "-i", args.image,
        "-o", str(faces_dir)
    ]
    
    if not run_command(convert_cmd, "Conversión de imagen 360°"):
        print("❌ Pipeline abortado en el paso 1")
        sys.exit(1)
    
    # Paso 2: Analizar caras con YOLO
    print(f"\n{'='*60}")
    print("PASO 2: Analizar caras con YOLO")
    print(f"{'='*60}")
    
    analyze_cmd = [
        "python", str(analyze_script),
        "-f", str(faces_dir),
        "-m", args.model,
        "-o", str(detections_dir)
    ]
    
    if not run_command(analyze_cmd, "Análisis de caras con YOLO"):
        print("❌ Pipeline abortado en el paso 2")
        sys.exit(1)
    
    # Paso 3: Extraer árboles completos
    print(f"\n{'='*60}")
    print("PASO 3: Extraer árboles completos")
    print(f"{'='*60}")
    
    detections_json = detections_dir / "detections.json"
    
    # Verificar que el archivo de detecciones existe
    if not detections_json.exists():
        print(f"❌ Error: No se encontró el archivo {detections_json}")
        sys.exit(1)
    
    extract_cmd = [
        "python", str(extract_script),
        "-e", args.image,
        "-d", str(detections_json),
        "-o", str(full_trees_dir)
    ]
    
    if not run_command(extract_cmd, "Extracción de árboles completos"):
        print("❌ Pipeline abortado en el paso 3")
        sys.exit(1)
    
    # Pipeline completado
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"📂 Resultados guardados en: {main_dir}")
    print(f"   ├── {faces_dir.name}/ (caras de la imagen 360°)")
    print(f"   ├── {detections_dir.name}/ (detecciones YOLO)")
    print(f"   └── {full_trees_dir.name}/ (árboles extraídos)")

if __name__ == "__main__":
    main()
