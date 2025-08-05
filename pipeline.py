#!/usr/bin/env python3
"""
Pipeline para automatizar el análisis de imágenes 360° con YOLO
Ejecuta secuencialmente: convert_images.py -> analyze_faces.py -> extract_full_trees.py
Las imágenes no-360° se procesan directamente sin conversión de cubemap

Flujo del pipeline:
1. Detecta si la imagen es 360° (ratio 2:1) o normal
2. Para 360°: convierte a caras de cubemap / Para normal: copia directamente
3. Analiza con YOLO para detectar objetos
4. Extrae recortes de árboles completos
5. Analiza recortes con IA (Gemini) para clasificación detallada
"""

import os
import sys
import subprocess
import argparse
import shutil
import glob
from pathlib import Path

# Configuración de codificación UTF-8 para Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def run_command(command, description):
    """
    Ejecuta un comando del sistema y maneja errores de forma elegante
    
    Args:
        command: Lista con el comando y argumentos
        description: Descripción humana del proceso
    
    Returns:
        bool: True si exitoso, False si hay error
    """
    print(f"\n🔄 {description}...")
    print(f"Ejecutando: {' '.join(command)}")
    
    try:
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
    """Crea directorio con estructura jerárquica si no existe"""
    Path(path).mkdir(parents=True, exist_ok=True)
    print(f"📁 Directorio creado/verificado: {path}")

def copy_original_image(source_image, destination_dir):
    """Guarda copia de la imagen original para referencia"""
    destination_path = Path(destination_dir) / "imagen_360_original.jpg"
    try:
        shutil.copy2(source_image, destination_path)
        print(f"📸 Imagen original copiada a: {destination_path}")
        return True
    except Exception as e:
        print(f"❌ Error al copiar imagen original: {e}")
        return False

def copy_image_as_face(source_image, destination_dir, face_name="front.jpg"):
    """
    Para imágenes normales: copia como 'cara' para que el pipeline YOLO funcione
    El sistema espera imágenes en el directorio de caras, así que simulamos una cara
    """
    destination_path = Path(destination_dir) / face_name
    try:
        shutil.copy2(source_image, destination_path)
        print(f"📸 Imagen copiada como cara: {destination_path}")
        return True
    except Exception as e:
        print(f"❌ Error al copiar imagen como cara: {e}")
        return False

def get_image_files(image_path):
    """
    Obtiene lista de archivos de imagen desde ruta (archivo individual o directorio)
    Soporta: .jpg, .jpeg, .png, .bmp, .tiff, .tif
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    if os.path.isfile(image_path):
        # Archivo individual
        if Path(image_path).suffix.lower() in image_extensions:
            return [image_path]
        else:
            print(f"❌ El archivo {image_path} no es una imagen válida")
            return []
    
    elif os.path.isdir(image_path):
        # Directorio: buscar todas las imágenes
        image_files = []
        for ext in image_extensions:
            pattern = os.path.join(image_path, f"*{ext}")
            image_files.extend(glob.glob(pattern))
            # También mayúsculas
            pattern = os.path.join(image_path, f"*{ext.upper()}")
            image_files.extend(glob.glob(pattern))
        
        return sorted(image_files)
    
    else:
        print(f"❌ La ruta {image_path} no existe")
        return []

def validate_360_image(image_path):
    """
    Detecta automáticamente si una imagen es 360° equirectangular
    
    Criterio: Las imágenes 360° tienen ratio de aspecto 2:1 (ancho:alto)
    Ejemplo: 4000x2000px, 8000x4000px, etc.
    
    Returns:
        bool: True si es 360°, False si es imagen normal
    """
    try:
        from PIL import Image
        
        with Image.open(image_path) as img:
            width, height = img.size
            aspect_ratio = width / height
            
            # Ratio esperado 2:1 con tolerancia del 10%
            expected_ratio = 2.0
            tolerance = 0.1
            
            if abs(aspect_ratio - expected_ratio) <= tolerance:
                print(f"✅ Imagen 360° válida (ratio: {aspect_ratio:.2f})")
                return True
            else:
                print(f"⚠️ Imagen no es 360° (ratio: {aspect_ratio:.2f}, esperado: ~2.0) - se procesará como imagen normal")
                return False
                
    except ImportError:
        print("⚠️ PIL no disponible, asumiendo imagen normal")
        return False
    except Exception as e:
        print(f"⚠️ Error validando imagen 360°: {e} - se procesará como imagen normal")
        return False

def get_image_name_without_extension(image_path):
    """Extrae nombre base del archivo sin extensión para nombrar directorios"""
    return Path(image_path).stem

def main():
    # Configuración de argumentos de línea de comandos
    parser = argparse.ArgumentParser(description="Pipeline de análisis de imágenes 360° y normales con validación automática")
    parser.add_argument("-i", "--image", required=True, help="Ruta a imagen individual, directorio con imágenes, o patrón (ej: *.jpg)")
    parser.add_argument("-m", "--model", required=True, help="Ruta al modelo YOLO (.pt)")
    parser.add_argument("-r", "--results-dir", default="imagenes_resultados", 
                       help="Directorio base de resultados (por defecto: imagenes_resultados)")
    parser.add_argument("--api-key", required=True, help="API key de Gemini")
    parser.add_argument("--force-360", action="store_true",
                       help="Forzar procesamiento como 360° sin validación")
    parser.add_argument("--force-normal", action="store_true",
                       help="Forzar procesamiento como imagen normal sin validación")
    
    args = parser.parse_args()
    
    # Validación inicial: obtener imágenes y verificar modelo
    image_files = get_image_files(args.image)
    if not image_files:
        print(f"❌ No se encontraron imágenes en: {args.image}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"❌ Error: El modelo {args.model} no existe")
        sys.exit(1)
    
    # Configuración de rutas de scripts auxiliares
    script_dir = Path(__file__).parent
    scripts_dir = script_dir / "scripts"
    
    # Scripts del pipeline: convert -> analyze -> extract
    convert_script = scripts_dir / "convert_images.py"    # Convierte 360° a caras
    analyze_script = scripts_dir / "analyze_faces.py"     # YOLO para detectar objetos
    extract_script = scripts_dir / "extract_full_trees.py" # Extrae recortes de árboles
    
    # Verificar que todos los scripts existen
    for script in [convert_script, analyze_script, extract_script]:
        if not script.exists():
            print(f"❌ Error: El script {script} no existe")
            sys.exit(1)
    
    # Contadores para estadísticas finales
    processed_360 = 0
    processed_normal = 0
    errors = 0
    
    # BUCLE PRINCIPAL: procesar cada imagen
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*60}")
        print(f"Procesando imagen {i}/{len(image_files)}: {Path(image_path).name}")
        print(f"{'='*60}")
        
        # DECISIÓN: ¿Es imagen 360° o normal?
        is_360 = False
        if args.force_360:
            is_360 = True
            print("🔧 Forzando procesamiento como imagen 360°")
        elif args.force_normal:
            is_360 = False
            print("🔧 Forzando procesamiento como imagen normal")
        else:
            # Detección automática basada en ratio de aspecto
            is_360 = validate_360_image(image_path)
        
        try:
            # Configuración de estructura de directorios para esta imagen
            image_name = get_image_name_without_extension(image_path)
            results_base = Path(args.results_dir)
            
            # Estructura: imagen_name/
            #   ├── imagen_name_faces/          (caras de cubemap o imagen original)
            #   ├── imagen_name_detections/     (resultados YOLO)
            #   ├── imagen_name_full_trees/     (recortes extraídos)
            #   └── results/                    (análisis IA final)
            main_dir = results_base / image_name
            faces_dir = main_dir / f"{image_name}_faces"
            detections_dir = main_dir / f"{image_name}_detections"
            full_trees_dir = main_dir / f"{image_name}_full_trees"
            
            if is_360:
                print(f"🌐 Procesando como imagen 360°: {image_path}")
            else:
                print(f"🖼️ Procesando como imagen normal: {image_path}")
            
            print(f"📂 Directorio principal: {main_dir}")
            
            # Crear estructura de directorios
            create_directory(main_dir)
            create_directory(faces_dir)
            create_directory(detections_dir)
            create_directory(full_trees_dir)
            
            # PASO 0: Guardar imagen original para referencia
            print(f"\n{'='*60}")
            print("PASO 0: Copiar imagen original")
            print(f"{'='*60}")
            
            if not copy_original_image(image_path, main_dir):
                print("⚠️ Advertencia: No se pudo copiar la imagen original, pero el pipeline continuará")
            
            # PASO 1: Preparar imagen para análisis YOLO
            print(f"\n{'='*60}")
            if is_360:
                print("PASO 1: Convertir imagen 360° en caras")
            else:
                print("PASO 1: Preparar imagen normal para análisis")
            print(f"{'='*60}")
            
            if is_360:
                # Imágenes 360°: convertir a 6 caras de cubemap (front, back, left, right, up, down)
                convert_cmd = [
                    "python", str(convert_script),
                    "-i", image_path,
                    "-o", str(faces_dir)
                ]
                
                if not run_command(convert_cmd, "Conversión de imagen 360°"):
                    print("❌ Pipeline abortado en el paso 1")
                    errors += 1
                    continue
            else:
                # Imágenes normales: copiar directamente como si fuera una "cara"
                print("📋 Copiando imagen normal para análisis directo...")
                if not copy_image_as_face(image_path, faces_dir, "original.jpg"):
                    print("❌ Error copiando imagen para análisis")
                    errors += 1
                    continue
                print("✅ Imagen preparada para análisis directo")
            
            # PASO 2: Detectar objetos con YOLO
            print(f"\n{'='*60}")
            if is_360:
                print("PASO 2: Analizar caras con YOLO")
            else:
                print("PASO 2: Analizar imagen con YOLO")
            print(f"{'='*60}")
            
            analyze_cmd = [
                "python", str(analyze_script),
                "-f", str(faces_dir),      # Directorio con caras o imagen
                "-m", args.model,          # Modelo YOLO entrenado
                "-o", str(detections_dir)  # Salida: detections.json
            ]
            
            analysis_desc = "Análisis de caras con YOLO" if is_360 else "Análisis de imagen con YOLO"
            if not run_command(analyze_cmd, analysis_desc):
                print("❌ Pipeline abortado en el paso 2")
                errors += 1
                continue
            
            # PASO 3: Extraer recortes de árboles completos basado en detecciones YOLO
            print(f"\n{'='*60}")
            print("PASO 3: Extraer árboles completos")
            print(f"{'='*60}")
            
            detections_json = detections_dir / "detections.json"
            
            # Verificar que YOLO generó detecciones
            if not detections_json.exists():
                print(f"❌ Error: No se encontró el archivo {detections_json}")
                errors += 1
                continue
            
            extract_cmd = [
                "python", str(extract_script),
                "-e", image_path,              # Imagen original (para recortar)
                "-d", str(detections_json),    # Detecciones YOLO
                "-o", str(full_trees_dir)      # Salida: trees/ y planters/
            ]
            
            if not run_command(extract_cmd, "Extracción de árboles completos"):
                print("❌ Pipeline abortado en el paso 3")
                errors += 1
                continue
                
            # PASO 4: Análisis inteligente con IA (Gemini)
            print(f"\n{'='*60}")
            print("PASO 4: Analizar recortes con IA")
            print(f"{'='*60}")

            # Crear directorio para resultados finales de IA
            results_dir = main_dir / "results"
            create_directory(results_dir)

            # El extractor genera dos subdirectorios:
            trees_dir = full_trees_dir / "trees"      # Recortes de árboles
            planters_dir = full_trees_dir / "planters" # Recortes de alcorques

            # Analizar árboles con IA si se encontraron
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
                    print(f"⚠️ Error analizando árboles")
            else:
                print(f"⚠️ No se encontraron árboles para analizar")

            # Analizar alcorques con IA si se encontraron
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
                    print(f"⚠️ Error analizando alcorques")
            else:
                print(f"⚠️ No se encontraron alcorques para analizar")

            # Actualizar contadores según tipo procesado
            if is_360:
                processed_360 += 1
                print(f"\n✅ Imagen 360° procesada exitosamente: {image_name}")
            else:
                processed_normal += 1
                print(f"\n✅ Imagen normal procesada exitosamente: {image_name}")
            
        except Exception as e:
            print(f"\n❌ Error procesando imagen: {str(e)}")
            errors += 1
            continue
    
    # RESUMEN FINAL
    print(f"\n{'='*60}")
    print("🎉 PIPELINE COMPLETADO")
    print(f"{'='*60}")
    print(f"Imágenes 360° procesadas: {processed_360}")
    print(f"Imágenes normales procesadas: {processed_normal}")
    print(f"Total procesadas exitosamente: {processed_360 + processed_normal}")
    print(f"Errores: {errors}")
    print(f"Resultados guardados en: {args.results_dir}")

if __name__ == "__main__":
    main()