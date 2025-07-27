#!/usr/bin/env python3
"""
extract_full_trees.py
Script para extraer árboles completos desde la imagen 360° original
basándose en las detecciones de YOLO en las caras del cubemap.
"""
import cv2
import numpy as np
import json
import os
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import math
from collections import defaultdict

@dataclass
class TreeDetection:
    """Clase para almacenar información de detección de árbol"""
    face_name: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    yaw: float  # ángulo horizontal en la imagen 360
    pitch: float  # ángulo vertical en la imagen 360
    bounds: Dict  # límites esféricos del objeto
    
class TreeExtractor:
    def __init__(self, equirect_path: str, detections_json: str, cube_size: int = 4096):
        """
        Inicializa el extractor de árboles
        
        Args:
            equirect_path: Ruta a la imagen equirectangular original
            detections_json: Ruta al archivo JSON con las detecciones
            cube_size: Tamaño de las caras del cubemap usado
        """
        self.equirect_path = equirect_path
        self.detections_json = detections_json
        self.cube_size = cube_size
        
        # Cargar imagen equirectangular
        self.equirect_img = cv2.imread(equirect_path)
        if self.equirect_img is None:
            raise ValueError(f"No se pudo cargar la imagen: {equirect_path}")
        
        self.equirect_height, self.equirect_width = self.equirect_img.shape[:2]
        
        # Cargar detecciones
        with open(detections_json, 'r') as f:
            self.detections = json.load(f)
            
        # Mapeo de nombres de caras a orientaciones
        self.face_orientations = {
            'front': {'yaw': 0, 'pitch': 0},
            'right': {'yaw': 90, 'pitch': 0},
            'back': {'yaw': 180, 'pitch': 0},
            'left': {'yaw': 270, 'pitch': 0},
            'zenith': {'yaw': 0, 'pitch': 90},
            'nadir': {'yaw': 0, 'pitch': -90}
        }
        
        # Aspect ratios típicos por clase
        self.typical_aspect_ratios = {
            0: 1.0,   # Alcorques suelen ser más o menos cuadrados
            3: 1.5,   # Árboles suelen ser más altos que anchos (ratio altura/ancho)
            2: 1.2    # Clase 2 (ajustar según necesidad)
        }
        
    def face_coords_to_spherical(self, x: float, y: float, face_name: str) -> Tuple[float, float]:
        """
        Convierte coordenadas de píxel en una cara a coordenadas esféricas
        
        Args:
            x, y: Coordenadas en la cara del cubemap
            face_name: Nombre de la cara
            
        Returns:
            yaw, pitch: Ángulos en grados
        """
        # Normalizar coordenadas al rango [-1, 1]
        normalized_x = (2.0 * x / self.cube_size) - 1.0
        normalized_y = 1.0 - (2.0 * y / self.cube_size)
        
        # Obtener orientación base de la cara
        base_orientation = self.face_orientations.get(face_name, {'yaw': 0, 'pitch': 0})
        base_yaw = np.radians(base_orientation['yaw'])
        base_pitch = np.radians(base_orientation['pitch'])
        
        # Coordenadas 3D en el espacio de la cara
        if face_name in ['front', 'back', 'left', 'right']:
            # Caras laterales
            if face_name == 'front':
                x3d, y3d, z3d = normalized_x, normalized_y, 1.0
            elif face_name == 'back':
                x3d, y3d, z3d = -normalized_x, normalized_y, -1.0
            elif face_name == 'right':
                x3d, y3d, z3d = 1.0, normalized_y, -normalized_x
            elif face_name == 'left':
                x3d, y3d, z3d = -1.0, normalized_y, normalized_x
        elif face_name == 'zenith':
            x3d, y3d, z3d = normalized_x, 1.0, -normalized_y
        elif face_name == 'nadir':
            x3d, y3d, z3d = normalized_x, -1.0, normalized_y
        else:
            # Cara personalizada - usar la orientación base
            x3d, y3d, z3d = normalized_x, normalized_y, 1.0
            
        # Normalizar vector
        length = np.sqrt(x3d*x3d + y3d*y3d + z3d*z3d)
        x3d, y3d, z3d = x3d/length, y3d/length, z3d/length
        
        # Convertir a coordenadas esféricas
        pitch = np.arcsin(np.clip(y3d, -1.0, 1.0))  # Clip para evitar errores de dominio
        yaw = np.arctan2(x3d, z3d)
        
        return np.degrees(yaw), np.degrees(pitch)
    
    def spherical_to_equirect(self, yaw: float, pitch: float) -> Tuple[int, int]:
        """
        Convierte coordenadas esféricas a píxeles en la imagen equirectangular
        
        Args:
            yaw: Ángulo horizontal en grados
            pitch: Ángulo vertical en grados
            
        Returns:
            x, y: Coordenadas de píxel en la imagen equirectangular
        """
        # Convertir a radianes
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        
        # Mapear a coordenadas de imagen
        x = (yaw_rad / np.pi + 1.0) * 0.5 * self.equirect_width
        y = (0.5 - pitch_rad / np.pi) * self.equirect_height
        
        # Asegurar que estén dentro de los límites
        x = int(np.clip(x, 0, self.equirect_width - 1))
        y = int(np.clip(y, 0, self.equirect_height - 1))
        
        return x, y
    
    def estimate_tree_bounds(self, center_yaw: float, center_pitch: float, 
                           initial_width: float, initial_height: float,
                           class_id: int = 3) -> Dict:
        """
        Estima los límites completos del árbol en la imagen equirectangular
        
        Args:
            center_yaw: Yaw del centro de la detección
            center_pitch: Pitch del centro de la detección
            initial_width: Ancho inicial del bbox en grados
            initial_height: Alto inicial del bbox en grados
            class_id: ID de la clase del objeto detectado
            
        Returns:
            Diccionario con los límites del árbol
        """
        # Obtener aspect ratio típico para la clase
        typical_ratio = self.typical_aspect_ratios.get(class_id, 1.3)
        current_ratio = initial_height / max(initial_width, 0.1)  # Evitar división por cero
        
        # Lógica diferente según la clase
        if class_id == 0:  # Alcorque (elemento en el suelo)
            # Para alcorques, expansión mínima ya que están bien definidos en el suelo
            expand_bottom = 1.1
            expand_top = 1.1
            expand_sides = 1.2
        elif class_id == 2:  # Otra clase (ajustar según necesidad)
            expand_bottom = 1.3
            expand_top = 1.3
            expand_sides = 1.3
        else:  # Clase 3 (árboles) y otras
            # Para árboles, lógica original basada en la posición vertical
            if center_pitch > 30:  # Mirando hacia arriba
                expand_bottom = 3.0  # Expandir mucho hacia abajo para capturar el tronco
                expand_top = 1.2
            elif center_pitch < -30:  # Mirando hacia abajo
                expand_bottom = 1.2
                expand_top = 3.0  # Expandir hacia arriba para capturar la copa
            else:  # Vista horizontal
                expand_bottom = 2.0  # Expandir moderadamente en ambas direcciones
                expand_top = 1.5
            
            expand_sides = 1.3  # Expansión lateral para capturar ramas
            
            # Ajuste adaptativo basado en aspect ratio
            if current_ratio < typical_ratio * 0.7:
                # El bbox detectado es más ancho de lo esperado, expandir más verticalmente
                expand_bottom *= 1.3
                expand_top *= 1.3
            elif current_ratio > typical_ratio * 1.3:
                # El bbox detectado es más alto de lo esperado, expandir más horizontalmente
                expand_sides *= 1.3
        
        # Calcular nuevos límites
        tree_bounds = {
            'center_yaw': center_yaw,
            'center_pitch': center_pitch,
            'yaw_min': center_yaw - (initial_width * expand_sides / 2),
            'yaw_max': center_yaw + (initial_width * expand_sides / 2),
            'pitch_min': center_pitch - (initial_height * expand_bottom / 2),
            'pitch_max': center_pitch + (initial_height * expand_top / 2),
        }
        
        # Limitar pitch a valores válidos
        tree_bounds['pitch_min'] = max(-90, tree_bounds['pitch_min'])
        tree_bounds['pitch_max'] = min(90, tree_bounds['pitch_max'])
        
        return tree_bounds
    
    def extract_tree_from_equirect(self, tree_bounds: Dict, padding: float = 0.1) -> np.ndarray:
        """
        Extrae la región del árbol de la imagen equirectangular
        
        Args:
            tree_bounds: Límites del árbol en coordenadas esféricas
            padding: Padding adicional como fracción del tamaño
            
        Returns:
            Imagen recortada del árbol
        """
        # Convertir límites esféricos a píxeles
        x_min, y_max = self.spherical_to_equirect(tree_bounds['yaw_min'], tree_bounds['pitch_max'])
        x_max, y_min = self.spherical_to_equirect(tree_bounds['yaw_max'], tree_bounds['pitch_min'])
        
        # Asegurar que x_min < x_max y y_min < y_max
        if x_min > x_max:
            x_min, x_max = x_max, x_min
        if y_min > y_max:
            y_min, y_max = y_max, y_min
        
        # Aplicar padding
        width = x_max - x_min
        height = y_max - y_min
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        x_min = max(0, x_min - pad_x)
        x_max = min(self.equirect_width - 1, x_max + pad_x)
        y_min = max(0, y_min - pad_y)
        y_max = min(self.equirect_height - 1, y_max + pad_y)
        
        # Verificar dimensiones válidas
        if x_max <= x_min or y_max <= y_min:
            print(f"  ADVERTENCIA: Dimensiones inválidas: x({x_min},{x_max}) y({y_min},{y_max})")
            return np.array([])  # Retornar array vacío
        
        # Manejar el caso cuando el árbol cruza el meridiano 180°
        if tree_bounds['yaw_min'] < -170 and tree_bounds['yaw_max'] > 170:
            # El árbol cruza el borde de la imagen
            # Extraer dos partes y unirlas
            x_min_left = int(self.equirect_width * 0.9)  # Parte izquierda de la imagen
            x_max_right = int(self.equirect_width * 0.1)  # Parte derecha
            
            left_part = self.equirect_img[y_min:y_max, x_min_left:]
            right_part = self.equirect_img[y_min:y_max, :x_max_right]
            
            if left_part.size > 0 and right_part.size > 0:
                tree_crop = np.concatenate([left_part, right_part], axis=1)
            else:
                tree_crop = np.array([])
        else:
            # Extracción normal
            tree_crop = self.equirect_img[y_min:y_max, x_min:x_max]
        
        return tree_crop
    
    def check_overlap(self, bounds1: Dict, bounds2: Dict, class1: int, class2: int) -> bool:
        """
        Verifica si dos detecciones se solapan significativamente
        
        Args:
            bounds1, bounds2: Límites esféricos de las detecciones
            class1, class2: Clases de las detecciones
            
        Returns:
            True si las detecciones se solapan significativamente
        """
        # Solo combinar detecciones de la misma clase
        if class1 != class2:
            return False
            
        # Calcular solapamiento en yaw (considerando wrap-around en 180°)
        yaw_diff = abs(bounds1['center_yaw'] - bounds2['center_yaw'])
        if yaw_diff > 180:
            yaw_diff = 360 - yaw_diff
            
        # Calcular solapamiento en pitch
        pitch_diff = abs(bounds1['center_pitch'] - bounds2['center_pitch'])
        
        # Umbrales de solapamiento según la clase
        if class1 == 0:  # Alcorques
            yaw_threshold = 15  # Grados
            pitch_threshold = 10
        else:  # Árboles y otros
            yaw_threshold = 25  # Los árboles pueden ser más grandes
            pitch_threshold = 20
            
        return yaw_diff < yaw_threshold and pitch_diff < pitch_threshold
    
    def merge_detections(self, detections: List[TreeDetection]) -> List[TreeDetection]:
        """
        Combina detecciones que corresponden al mismo objeto visto desde diferentes caras
        
        Args:
            detections: Lista de todas las detecciones
            
        Returns:
            Lista de detecciones combinadas
        """
        if len(detections) <= 1:
            return detections
            
        # Agrupar detecciones por proximidad
        merged = []
        used = set()
        
        for i, det1 in enumerate(detections):
            if i in used:
                continue
                
            # Buscar detecciones que se solapen con esta
            overlapping = [det1]
            used.add(i)
            
            for j, det2 in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                if self.check_overlap(det1.bounds, det2.bounds, det1.class_id, det2.class_id):
                    overlapping.append(det2)
                    used.add(j)
            
            # Combinar las detecciones solapadas
            if len(overlapping) > 1:
                # Usar la detección con mayor confianza como base
                best_det = max(overlapping, key=lambda d: d.confidence)
                
                # Expandir los límites para incluir todas las detecciones
                min_yaw = min(d.bounds['yaw_min'] for d in overlapping)
                max_yaw = max(d.bounds['yaw_max'] for d in overlapping)
                min_pitch = min(d.bounds['pitch_min'] for d in overlapping)
                max_pitch = max(d.bounds['pitch_max'] for d in overlapping)
                
                # Crear detección combinada
                combined_bounds = {
                    'center_yaw': (min_yaw + max_yaw) / 2,
                    'center_pitch': (min_pitch + max_pitch) / 2,
                    'yaw_min': min_yaw,
                    'yaw_max': max_yaw,
                    'pitch_min': min_pitch,
                    'pitch_max': max_pitch,
                }
                
                combined_det = TreeDetection(
                    face_name=f"combined_{len(overlapping)}_faces",
                    bbox=best_det.bbox,
                    confidence=max(d.confidence for d in overlapping),
                    class_id=best_det.class_id,
                    yaw=combined_bounds['center_yaw'],
                    pitch=combined_bounds['center_pitch'],
                    bounds=combined_bounds
                )
                
                merged.append(combined_det)
                print(f"  Combinadas {len(overlapping)} detecciones del mismo objeto")
            else:
                merged.append(det1)
                
        return merged
    
    def process_detections(self, output_dir: str = "extracted_trees", 
                          confidence_threshold: float = 0.3,
                          target_classes: List[int] = [0, 3]):  # Incluir alcorques (0) y árboles (3)
        """
        Procesa todas las detecciones y extrae árboles completos
        
        Args:
            output_dir: Directorio donde guardar los árboles extraídos
            confidence_threshold: Umbral mínimo de confianza
            target_classes: Lista de IDs de clases a extraer (árboles)
        """
        trees_dir = os.path.join(output_dir, "trees")
        planters_dir = os.path.join(output_dir, "planters") 
        os.makedirs(trees_dir, exist_ok=True)
        os.makedirs(planters_dir, exist_ok=True)
        
        # Crear subdirectorio para visualizaciones
        viz_dir = os.path.join(output_dir, "viz")
        os.makedirs(viz_dir, exist_ok=True)
        
        tree_count = 0
        extracted_trees = []
        all_detections = []
        
        # Primero, recopilar todas las detecciones con sus coordenadas esféricas
        print("=== FASE 1: Recopilando detecciones ===")
        for face_name, face_data in self.detections.items():
            if face_data['num_detections'] == 0:
                continue
                
            print(f"\nProcesando cara: {face_name}")
            
            for idx, detection in enumerate(face_data['boxes']):
                # Filtrar por confianza y clase
                if detection['score'] < confidence_threshold:
                    continue
                if detection['class'] not in target_classes:
                    continue
                
                x1, y1, x2, y2 = detection['coordinates']
                
                # Calcular centro del bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Convertir a coordenadas esféricas
                center_yaw, center_pitch = self.face_coords_to_spherical(center_x, center_y, face_name)
                
                # Calcular dimensiones angulares aproximadas del bbox
                # Usar el campo de visión de la cara del cubemap (90 grados)
                fov = 90.0  # Campo de visión de cada cara
                
                # Calcular dimensiones en grados basándose en el tamaño del bbox
                bbox_width_pixels = x2 - x1
                bbox_height_pixels = y2 - y1
                
                # Aproximación: el tamaño angular es proporcional al tamaño en píxeles
                bbox_width_deg = (bbox_width_pixels / self.cube_size) * fov
                bbox_height_deg = (bbox_height_pixels / self.cube_size) * fov
                
                print(f"  Detección {idx}: Centro en yaw={center_yaw:.1f}°, pitch={center_pitch:.1f}°")
                print(f"  Tamaño inicial: {bbox_width_deg:.1f}° x {bbox_height_deg:.1f}°")
                
                # Estimar límites completos del árbol
                tree_bounds = self.estimate_tree_bounds(center_yaw, center_pitch, 
                                                      bbox_width_deg, bbox_height_deg,
                                                      detection['class'])
                
                # Crear objeto TreeDetection
                tree_det = TreeDetection(
                    face_name=face_name,
                    bbox=detection['coordinates'],
                    confidence=detection['score'],
                    class_id=detection['class'],
                    yaw=center_yaw,
                    pitch=center_pitch,
                    bounds=tree_bounds
                )
                
                all_detections.append(tree_det)
        
        # Combinar detecciones del mismo objeto
        print("\n=== FASE 2: Combinando detecciones solapadas ===")
        merged_detections = self.merge_detections(all_detections)
        print(f"Detecciones originales: {len(all_detections)}")
        print(f"Detecciones después de combinar: {len(merged_detections)}")
        
        # Procesar las detecciones combinadas
        print("\n=== FASE 3: Extrayendo objetos ===")
        for det in merged_detections:
            # Extraer árbol de la imagen equirectangular
            tree_crop = self.extract_tree_from_equirect(det.bounds)
            
            # Verificar que el crop no esté vacío
            if tree_crop.size == 0 or tree_crop.shape[0] == 0 or tree_crop.shape[1] == 0:
                print(f"  ADVERTENCIA: Crop vacío, saltando esta detección")
                continue
    
            # Generar nombre de archivo
            class_names = {0: "alcorque", 1: "luz_trafico", 2: "señal_trafico", 3: "arbol"}
            class_name = class_names.get(det.class_id, f"class{det.class_id}")
            filename = f"{class_name}_{tree_count:03d}_{det.face_name}_conf{det.confidence:.2f}.jpg"
    
            # Determinar directorio según clase
            if det.class_id == 0:  # alcorques
                target_dir = planters_dir
            elif det.class_id == 3:  # árboles
                target_dir = trees_dir
            else:
                target_dir = output_dir  # otros van al directorio base
    
            # Guardar imagen en el directorio correcto
            filepath = os.path.join(target_dir, filename)
            cv2.imwrite(filepath, tree_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
            print(f"  Guardado: {filename} en {os.path.basename(target_dir)}/ (tamaño: {tree_crop.shape[1]}x{tree_crop.shape[0]})")
    
            # Guardar metadatos (con ruta relativa al subdirectorio)
            relative_filepath = os.path.join(os.path.basename(target_dir), filename)
            tree_info = {
                'id': tree_count,
                'filename': filename,
                'filepath': relative_filepath,  # Ruta relativa para facilitar acceso
                'directory': os.path.basename(target_dir),  # trees/planters
                'source_face': det.face_name,
                'confidence': det.confidence,
                'class': det.class_id,
                'original_bbox': det.bbox,
                'spherical_center': {'yaw': det.yaw, 'pitch': det.pitch},
                'extracted_bounds': det.bounds,
                'crop_size': {'width': tree_crop.shape[1], 'height': tree_crop.shape[0]}
            }
            extracted_trees.append(tree_info)
    
            tree_count += 1
    
            # Opcionalmente, crear una visualización con el bbox original y expandido
            if tree_count <= 5:  # Solo para los primeros árboles
                viz_filename = f"viz_{tree_count:03d}_{class_name}_{det.face_name}.jpg"
                viz_path = os.path.join(viz_dir, viz_filename)
                # Crear diccionario de detección para compatibilidad
                detection_dict = {
                    'coordinates': det.bbox,
                    'score': det.confidence,
                    'class': det.class_id
                }
                self.create_visualization(det.face_name, detection_dict, det.bounds, viz_path)
        
        # Guardar metadatos de todos los árboles extraídos
        metadata_path = os.path.join(output_dir, "extracted_trees_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(extracted_trees, f, indent=2)
        
        print(f"\n=== RESUMEN ===")
        print(f"Total de árboles extraídos: {tree_count}")
        print(f"Imágenes guardadas en: {output_dir}")
        print(f"Metadatos en: {metadata_path}")
        
        return extracted_trees
    
    def create_visualization(self, face_name: str, detection: Dict, 
                           tree_bounds: Dict, output_path: str):
        """
        Crea una visualización mostrando el bbox original y el área expandida
        """
        # Crear copia de la imagen equirectangular
        viz_img = self.equirect_img.copy()
        
        # Solo dibujar el bbox original si no es una detección combinada
        if not face_name.startswith("combined_"):
            # Dibujar el bbox original proyectado
            x1, y1, x2, y2 = detection['coordinates']
            corners = [
                (x1, y1), (x2, y1), (x2, y2), (x1, y2)
            ]
            
            # Proyectar esquinas a la imagen equirectangular
            prev_point = None
            for corner in corners + [corners[0]]:  # Cerrar el polígono
                yaw, pitch = self.face_coords_to_spherical(corner[0], corner[1], face_name)
                x, y = self.spherical_to_equirect(yaw, pitch)
                
                if prev_point is not None:
                    cv2.line(viz_img, prev_point, (x, y), (0, 0, 255), 3)  # Rojo para bbox original
                prev_point = (x, y)
        
        # Dibujar el área expandida
        expanded_corners = [
            (tree_bounds['yaw_min'], tree_bounds['pitch_max']),
            (tree_bounds['yaw_max'], tree_bounds['pitch_max']),
            (tree_bounds['yaw_max'], tree_bounds['pitch_min']),
            (tree_bounds['yaw_min'], tree_bounds['pitch_min'])
        ]
        
        prev_point = None
        for yaw, pitch in expanded_corners + [expanded_corners[0]]:
            x, y = self.spherical_to_equirect(yaw, pitch)
            
            if prev_point is not None:
                # Color diferente para detecciones combinadas
                color = (0, 255, 255) if face_name.startswith("combined_") else (0, 255, 0)
                cv2.line(viz_img, prev_point, (x, y), color, 3)
            prev_point = (x, y)
        
        # Marcar el centro
        center_x, center_y = self.spherical_to_equirect(tree_bounds['center_yaw'], 
                                                       tree_bounds['center_pitch'])
        cv2.circle(viz_img, (center_x, center_y), 10, (255, 255, 0), -1)  # Amarillo para centro
        
        # Redimensionar para visualización (la imagen 360 puede ser muy grande)
        max_viz_width = 2000
        if viz_img.shape[1] > max_viz_width:
            scale = max_viz_width / viz_img.shape[1]
            new_size = (max_viz_width, int(viz_img.shape[0] * scale))
            viz_img = cv2.resize(viz_img, new_size, interpolation=cv2.INTER_AREA)
        
        cv2.imwrite(output_path, viz_img, [cv2.IMWRITE_JPEG_QUALITY, 90])


def main():
    parser = argparse.ArgumentParser(description="Extrae árboles completos desde imagen 360°")
    parser.add_argument("-e", "--equirect", required=True, 
                       help="Ruta a la imagen equirectangular original")
    parser.add_argument("-d", "--detections", required=True, 
                       help="Ruta al archivo JSON con las detecciones")
    parser.add_argument("-o", "--output-dir", default="extracted_trees", 
                       help="Directorio de salida")
    parser.add_argument("-c", "--cube-size", type=int, default=4096, 
                       help="Tamaño de las caras del cubemap")
    parser.add_argument("--confidence", type=float, default=0.3, 
                       help="Umbral mínimo de confianza")
    parser.add_argument("--classes", type=int, nargs='+', default=[0, 3], 
                       help="IDs de clases a extraer (por defecto: 0=alcorques, 3=árboles)")
    
    args = parser.parse_args()
    
    print("=== Extractor de Árboles Completos desde Imagen 360° ===\n")
    
    try:
        extractor = TreeExtractor(args.equirect, args.detections, args.cube_size)
        extracted_trees = extractor.process_detections(
            output_dir=args.output_dir,
            confidence_threshold=args.confidence,
            target_classes=args.classes
        )
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())