#!/usr/bin/env python3
"""
extract_crops.py
Script para extraer crops de elementos detectados desde las caras del cubemap
"""
import os
import json
import argparse
from PIL import Image
import math

class CubemapCropExtractor:
    def __init__(self, faces_dir, detections_json_path, output_dir="crops"):
        """
        Extractor de crops desde caras del cubemap
        
        Args:
            faces_dir: Directorio con las caras del cubemap
            detections_json_path: Ruta al archivo detections.json
            output_dir: Directorio donde guardar los crops
        """
        self.faces_dir = faces_dir
        self.detections_json_path = detections_json_path
        self.output_dir = output_dir
        
        # Crear directorio de salida
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar detecciones
        with open(detections_json_path, 'r') as f:
            self.detections = json.load(f)
        
        # Nombres de las caras
        self.face_names = ["front", "right", "back", "left", "up", "down"]
        
        # Mapeo de clases (puedes personalizarlo)
        self.class_names = {
            0: "arbusto",
            1: "clase1", 
            2: "roca",
            3: "arbol",
            4: "clase4",
            5: "clase5"
        }
    
    def expand_bbox(self, bbox, expansion_factor=0.1, min_expansion=10):
        """
        Expande un bounding box para incluir más contexto
        
        Args:
            bbox: [x1, y1, x2, y2] coordenadas originales
            expansion_factor: Factor de expansión (0.1 = 10% más grande)
            min_expansion: Expansión mínima en píxeles
            
        Returns:
            bbox expandido: [x1, y1, x2, y2]
        """
        x1, y1, x2, y2 = bbox
        
        # Calcular dimensiones actuales
        width = x2 - x1
        height = y2 - y1
        
        # Calcular expansión
        expand_x = max(width * expansion_factor, min_expansion)
        expand_y = max(height * expansion_factor, min_expansion)
        
        # Aplicar expansión
        new_x1 = max(0, x1 - expand_x)
        new_y1 = max(0, y1 - expand_y)
        new_x2 = x2 + expand_x
        new_y2 = y2 + expand_y
        
        return [new_x1, new_y1, new_x2, new_y2]
    
    def extract_crop_from_face(self, face_name, bbox, crop_id, class_id, score, expand=True):
        """
        Extrae un crop de una cara específica del cubemap
        
        Args:
            face_name: Nombre de la cara (front, right, etc.)
            bbox: [x1, y1, x2, y2] coordenadas del bounding box
            crop_id: ID único para el crop
            class_id: ID de la clase detectada
            score: Score de confianza
            expand: Si expandir el bounding box para más contexto
            
        Returns:
            Ruta del archivo guardado o None si hay error
        """
        # Buscar archivo de la cara
        face_file = os.path.join(self.faces_dir, f"{face_name}.jpg")
        if not os.path.exists(face_file):
            print(f"⚠ No se encontró la cara: {face_file}")
            return None
        
        try:
            # Cargar imagen de la cara
            face_image = Image.open(face_file)
            face_width, face_height = face_image.size
            
            # Expandir bounding box si se solicita
            if expand:
                expanded_bbox = self.expand_bbox(bbox)
            else:
                expanded_bbox = bbox
            
            x1, y1, x2, y2 = expanded_bbox
            
            # Asegurar que las coordenadas estén dentro de los límites
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(face_width, int(x2))
            y2 = min(face_height, int(y2))
            
            # Verificar que el bounding box sea válido
            if x2 <= x1 or y2 <= y1:
                print(f"⚠ Bounding box inválido para {crop_id}: [{x1}, {y1}, {x2}, {y2}]")
                return None
            
            # Extraer crop
            crop = face_image.crop((x1, y1, x2, y2))
            
            # Generar nombre de archivo
            class_name = self.class_names.get(class_id, f"clase{class_id}")
            filename = f"{crop_id:04d}_{face_name}_{class_name}_score{score:.2f}.jpg"
            output_path = os.path.join(self.output_dir, filename)
            
            # Guardar crop
            crop.save(output_path, quality=95)
            
            # Información del crop
            crop_info = {
                "width": x2 - x1,
                "height": y2 - y1,
                "original_bbox": bbox,
                "expanded_bbox": [x1, y1, x2, y2] if expand else bbox
            }
            
            return output_path, crop_info
            
        except Exception as e:
            print(f"✗ Error extrayendo crop {crop_id}: {e}")
            return None
    
    def extract_all_crops(self, expand_bbox=True, min_score=0.0, filter_classes=None):
        """
        Extrae todos los crops de las detecciones
        
        Args:
            expand_bbox: Si expandir los bounding boxes
            min_score: Score mínimo para extraer (filtro de calidad)
            filter_classes: Lista de clases a extraer (None = todas)
            
        Returns:
            Información de todos los crops extraídos
        """
        print("=== Extractor de Crops desde Cubemap ===")
        print(f"Directorio de caras: {self.faces_dir}")
        print(f"Archivo de detecciones: {self.detections_json_path}")
        print(f"Directorio de salida: {self.output_dir}")
        print(f"Expandir bounding boxes: {expand_bbox}")
        print(f"Score mínimo: {min_score}")
        print(f"Filtro de clases: {filter_classes}")
        print()
        
        crop_id = 1
        extracted_crops = []
        total_detections = 0
        skipped_detections = 0
        
        # Procesar cada cara
        for face_name in self.face_names:
            if face_name not in self.detections:
                continue
            
            face_data = self.detections[face_name]
            boxes = face_data.get("boxes", [])
            
            print(f"Procesando cara '{face_name}': {len(boxes)} detecciones")
            
            # Procesar cada detección en esta cara
            for box_data in boxes:
                total_detections += 1
                
                coordinates = box_data["coordinates"]
                score = box_data["score"]
                class_id = box_data["class"]
                
                # Aplicar filtros
                if score < min_score:
                    skipped_detections += 1
                    print(f"  - Saltando detección {crop_id} (score {score:.2f} < {min_score})")
                    continue
                
                if filter_classes is not None and class_id not in filter_classes:
                    skipped_detections += 1
                    print(f"  - Saltando detección {crop_id} (clase {class_id} no en filtro)")
                    continue
                
                # Extraer crop
                result = self.extract_crop_from_face(
                    face_name, coordinates, crop_id, class_id, score, expand_bbox
                )
                
                if result:
                    output_path, crop_info = result
                    crop_info.update({
                        "crop_id": crop_id,
                        "face_name": face_name,
                        "class_id": class_id,
                        "score": score,
                        "output_path": output_path
                    })
                    extracted_crops.append(crop_info)
                    print(f"  ✓ Crop {crop_id:04d}: {os.path.basename(output_path)}")
                else:
                    skipped_detections += 1
                
                crop_id += 1
        
        # Guardar información de los crops
        crops_info_path = os.path.join(self.output_dir, "crops_info.json")
        with open(crops_info_path, 'w') as f:
            json.dump(extracted_crops, f, indent=2)
        
        # Resumen
        print(f"\n=== Resumen ===")
        print(f"Total detecciones procesadas: {total_detections}")
        print(f"Crops extraídos exitosamente: {len(extracted_crops)}")
        print(f"Detecciones saltadas: {skipped_detections}")
        print(f"Información guardada en: {crops_info_path}")
        print(f"Crops guardados en: {self.output_dir}")
        
        # Estadísticas por clase
        if extracted_crops:
            print(f"\n=== Estadísticas por clase ===")
            class_counts = {}
            for crop in extracted_crops:
                class_id = crop["class_id"]
                class_name = self.class_names.get(class_id, f"clase{class_id}")
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            for class_name, count in sorted(class_counts.items()):
                print(f"  {class_name}: {count} crops")
        
        return extracted_crops

def main():
    parser = argparse.ArgumentParser(description="Extraer crops de elementos detectados desde caras del cubemap")
    parser.add_argument("-f", "--faces-dir", required=True, help="Directorio con caras del cubemap")
    parser.add_argument("-d", "--detections", required=True, help="Ruta al archivo detections.json")
    parser.add_argument("-o", "--output-dir", default="crops", help="Directorio de salida para crops")
    parser.add_argument("--no-expand", action="store_true", help="No expandir bounding boxes")
    parser.add_argument("--min-score", type=float, default=0.0, help="Score mínimo para extraer")
    parser.add_argument("--classes", type=int, nargs="+", help="Clases específicas a extraer (ej: 0 3)")
    parser.add_argument("--preview", action="store_true", help="Solo mostrar estadísticas sin extraer")
    
    args = parser.parse_args()
    
    # Verificar que los directorios/archivos existen
    if not os.path.exists(args.faces_dir):
        print(f"Error: No se encontró el directorio {args.faces_dir}")
        return
    
    if not os.path.exists(args.detections):
        print(f"Error: No se encontró el archivo {args.detections}")
        return
    
    # Crear extractor
    extractor = CubemapCropExtractor(args.faces_dir, args.detections, args.output_dir)
    
    if args.preview:
        # Modo preview: solo mostrar estadísticas
        print("=== MODO PREVIEW ===")
        total_detections = 0
        class_counts = {}
        
        for face_name, face_data in extractor.detections.items():
            boxes = face_data.get("boxes", [])
            for box_data in boxes:
                if box_data["score"] >= args.min_score:
                    if args.classes is None or box_data["class"] in args.classes:
                        total_detections += 1
                        class_id = box_data["class"]
                        class_name = extractor.class_names.get(class_id, f"clase{class_id}")
                        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        print(f"Se extraerían {total_detections} crops:")
        for class_name, count in sorted(class_counts.items()):
            print(f"  {class_name}: {count} crops")
    else:
        # Extraer crops
        extracted_crops = extractor.extract_all_crops(
            expand_bbox=not args.no_expand,
            min_score=args.min_score,
            filter_classes=args.classes
        )

if __name__ == "__main__":
    main()