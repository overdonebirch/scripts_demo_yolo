#!/usr/bin/env python3
"""
Pipeline para automatizar el análisis de imágenes 360° con YOLO
Ejecuta secuencialmente: convert_images.py -> analyze_faces.py -> extract_full_trees.py
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"\n🔄 {description}...")
    print(f"Ejecutando: {' '.join(command)}")
    
    try:
        # Usar UTF-8 para evitar problemas de codificación en Windows
        result = subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')
        print(f"✅ {description} completado exitosamente")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error en {description}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def create_directory(path):
    """Crea un directorio si no existe"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"📁 Directorio creado/verificado: {path}")

def copy_original_image(source_image, destination_dir):
    """Copia la imagen original al directorio de destino con nombre fijo"""
    destination_path = Path(destination_dir) / "imagen_360_original.jpg"
    try:
        shutil.copy2(source_image, destination_path)
        print(f"📸 Imagen original copiada a: {destination_path}")
        return True
    except Exception as e:
        print(f"❌ Error al copiar imagen original: {e}")
        return False

def get_image_name_without_extension(image_path):
    """Obtiene el nombre de la imagen sin extensión"""
    return Path(image_path).stem

def main():
    parser = argparse.ArgumentParser(description="Pipeline de análisis de imágenes 360°")
    parser.add_argument("-i", "--image", required=True, help="Ruta a la imagen equirectangular 360°")
    parser.add_argument("-m", "--model", required=True, help="Ruta al modelo YOLO (.pt)")
    parser.add_argument("-r", "--results-dir", default="imagenes_resultados", 
                       help="Directorio base de resultados (por defecto: imagenes_resultados)")
    parser.add_argument("--api-key", required=True, help="API key de Gemini")
    
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
    
    # Copiar imagen original
    print(f"\n{'='*60}")
    print("PASO 0: Copiar imagen 360° original")
    print(f"{'='*60}")
    
    if not copy_original_image(args.image, main_dir):
        print("⚠️ Advertencia: No se pudo copiar la imagen original, pero el pipeline continuará")
    
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
        
    # Paso 4: Analizar con IA
    print(f"\n{'='*60}")
    print("PASO 4: Analizar recortes con IA")
    print(f"{'='*60}")

    # Crear directorio de resultados
    results_dir = main_dir / "results"
    create_directory(results_dir)

    # Verificar si existen subdirectorios y analizarlos
    trees_dir = full_trees_dir / "trees"
    planters_dir = full_trees_dir / "planters"

    # Analizar árboles si existen
    if trees_dir.exists() and any(trees_dir.iterdir()):
        trees_output = results_dir / "arboles_results.json"
        analyze_trees_cmd = [
            "python", str(script_dir / "analizador_arboles.py"),
            str(trees_dir),
            "--api-key", args.api_key,
            "--tipo", "arboles",
            "--output", str(trees_output),
            "--resumen"
        ]
        
        if run_command(analyze_trees_cmd, "Análisis de árboles con IA"):
            print(f"📄 Resultados de árboles guardados en: {trees_output}")
        else:
            print(f"❌ Error analizando árboles")
    else:
        print(f"⚠️  No se encontraron árboles para analizar")

    # Analizar alcorques si existen
    if planters_dir.exists() and any(planters_dir.iterdir()):
        planters_output = results_dir / "alcorques_results.json"
        analyze_planters_cmd = [
            "python", str(script_dir / "analizador_arboles.py"),
            str(planters_dir),
            "--api-key", args.api_key,
            "--tipo", "alcorques",
            "--output", str(planters_output),
            "--resumen"
        ]
        
        if run_command(analyze_planters_cmd, "Análisis de alcorques con IA"):
            print(f"📄 Resultados de alcorques guardados en: {planters_output}")
        else:
            print(f"❌ Error analizando alcorques")
    else:
        print(f"⚠️  No se encontraron alcorques para analizar")

    # Mostrar resumen de archivos generados
    print(f"\n📋 Archivos de resultados generados:")
    if (results_dir / "arboles_results.json").exists():
        print(f"   ✅ {results_dir / 'arboles_results.json'}")
    if (results_dir / "alcorques_results.json").exists():
        print(f"   ✅ {results_dir / 'alcorques_results.json'}")

    if not any((results_dir / f).exists() for f in ["arboles_results.json", "alcorques_results.json"]):
        print(f"   ⚠️  No se generaron archivos de resultados (posiblemente no había imágenes para analizar)")
    # Pipeline completado
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"{'='*60}")
    print(f"📂 Resultados guardados en: {main_dir}")
    print(f"   ├── imagen_360_original.jpg (imagen 360° original)")
    print(f"   ├── {faces_dir.name}/ (caras de la imagen 360°)")
    print(f"   ├── {detections_dir.name}/ (detecciones YOLO)")
    print(f"   └── {full_trees_dir.name}/ (árboles extraídos)")

if __name__ == "__main__":
    main()
