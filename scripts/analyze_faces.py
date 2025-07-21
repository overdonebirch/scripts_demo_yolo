#!/usr/bin/env python3
"""
analyze_faces.py
Script para ejecutar detección YOLO en cualquier conjunto de imágenes.
Versión flexible que no depende de nombres específicos de caras.
"""
import os
import json
import argparse
import glob
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

def get_image_files(directory):
    """
    Obtiene todas las imágenes .jpg y .png del directorio
    """
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    
    for extension in image_extensions:
        pattern = os.path.join(directory, extension)
        image_files.extend(glob.glob(pattern))
    
    # Ordenar para tener un orden consistente
    image_files.sort()
    return image_files

def get_base_name(file_path):
    """
    Extrae el nombre base del archivo sin extensión
    """
    return os.path.splitext(os.path.basename(file_path))[0]

def main():
    parser = argparse.ArgumentParser(description="YOLO detection on any set of images")
    parser.add_argument("-f", "--faces-dir", required=True, help="Directorio con imágenes (.jpg, .png)")
    parser.add_argument("-m", "--model", default="best.pt", help="Ruta al modelo YOLO (.pt)")
    parser.add_argument("-o", "--output-dir", default=None, help="Directorio de salida (por defecto faces-dir)")
    parser.add_argument("--confidence", type=float, default=0.25, help="Umbral de confianza mínimo")
    parser.add_argument("--save-crops", action="store_true", help="Guardar recortes de detecciones")
    args = parser.parse_args()

    faces_dir = args.faces_dir
    model_path = args.model
    output_dir = args.output_dir or faces_dir
    confidence_threshold = args.confidence
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear subdirectorio para crops si se solicita
    if args.save_crops:
        crops_dir = os.path.join(output_dir, "crops")
        os.makedirs(crops_dir, exist_ok=True)

    # Colores para diferentes clases
    colors = [
        (255, 0, 0),    # Rojo
        (0, 255, 0),    # Verde
        (0, 0, 255),    # Azul
        (255, 255, 0),  # Amarillo
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Cian
        (255, 128, 0),  # Naranja
        (128, 0, 255),  # Violeta
        (0, 255, 128),  # Verde claro
        (255, 0, 128),  # Rosa
    ]

    # Obtener todas las imágenes del directorio
    image_files = get_image_files(faces_dir)
    
    if not image_files:
        print(f"No se encontraron imágenes en {faces_dir}")
        return

    print(f"Encontradas {len(image_files)} imágenes en {faces_dir}")
    print(f"Iniciando detección YOLO con modelo: {model_path}")
    print(f"Umbral de confianza: {confidence_threshold}")
    
    # Cargar modelo YOLO
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return

    detections = {}
    total_detections = 0

    for idx, image_path in enumerate(image_files):
        base_name = get_base_name(image_path)
        print(f"Procesando imagen {idx + 1}/{len(image_files)}: {base_name}")
        
        try:
            # Ejecutar detección
            results = model.predict(
                source=image_path, 
                save=False, 
                save_txt=False, 
                verbose=False,
                conf=confidence_threshold
            )
            
            # Procesar resultados
            if results and hasattr(results[0], "boxes") and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
            else:
                boxes = np.array([])
                scores = np.array([])
                classes = np.array([])

            # Crear lista de detecciones con información completa
            boxes_with_data = []
            for i in range(len(boxes)):
                box_data = {
                    "coordinates": boxes[i].tolist(),
                    "score": float(scores[i]),
                    "class": int(classes[i])
                }
                boxes_with_data.append(box_data)
            
            detections[base_name] = {
                "image_path": image_path,
                "boxes": boxes_with_data,
                "num_detections": int(len(boxes))
            }
            
            total_detections += len(boxes_with_data)
            print(f"  -> {len(boxes_with_data)} detecciones encontradas")

            # Dibujar y guardar imagen con detecciones
            if len(boxes_with_data) > 0:
                img = Image.open(image_path)
                draw = ImageDraw.Draw(img)
                
                # Intentar cargar una fuente mejor
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
                except:
                    try:
                        font = ImageFont.truetype("arial.ttf", 16)
                    except:
                        font = ImageFont.load_default()
                
                for box_idx, box_data in enumerate(boxes_with_data):
                    x1, y1, x2, y2 = box_data["coordinates"]
                    cls = box_data["class"]
                    score = box_data["score"]
                    color = colors[cls % len(colors)]
                    
                    # Dibujar rectángulo de detección
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                    
                    # Texto con clase y confianza
                    text = f"Class {cls}: {score:.2f}"
                    
                    # Calcular posición del texto
                    text_bbox = draw.textbbox((0, 0), text, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    text_x = x1
                    text_y = max(0, y1 - text_height - 2)
                    
                    # Fondo para el texto (mejor legibilidad)
                    bg_coords = [text_x, text_y, text_x + text_width, text_y + text_height]
                    draw.rectangle(bg_coords, fill=(0, 0, 0, 180))
                    
                    # Dibujar texto
                    draw.text((text_x, text_y), text, fill=color, font=font)
                    
                    # Guardar crop si se solicita
                    if args.save_crops:
                        crop = img.crop((x1, y1, x2, y2))
                        crop_name = f"{base_name}_crop_{box_idx}_class{cls}_{score:.2f}.jpg"
                        crop_path = os.path.join(crops_dir, crop_name)
                        crop.save(crop_path, quality=95)
                
                # Guardar imagen con detecciones
                output_name = f"{base_name}_with_detections.jpg"
                output_path = os.path.join(output_dir, output_name)
                img.save(output_path, quality=95)
                print(f"  -> Guardado: {output_name}")
                
                if args.save_crops:
                    print(f"  -> Crops guardados en: crops/")
                    
        except Exception as e:
            print(f"Error procesando {base_name}: {e}")
            detections[base_name] = {
                "image_path": image_path,
                "boxes": [],
                "num_detections": 0,
                "error": str(e)
            }

    # Guardar JSON con todas las detecciones
    json_path = os.path.join(output_dir, "detections.json")
    with open(json_path, "w") as f:
        json.dump(detections, f, indent=2)
    
    # Resumen final
    print(f"\n=== RESUMEN ===")
    print(f"Imágenes procesadas: {len(image_files)}")
    print(f"Total detecciones: {total_detections}")
    print(f"Detecciones por imagen: {total_detections / len(image_files):.1f}")
    print(f"Resultados guardados en: {output_dir}")
    print(f"Detecciones JSON: {json_path}")
    
    # Estadísticas por imagen
    images_with_detections = sum(1 for det in detections.values() if det["num_detections"] > 0)
    print(f"Imágenes con detecciones: {images_with_detections}/{len(image_files)}")

if __name__ == '__main__':
    main()