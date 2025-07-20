#!/usr/bin/env python3
"""
visualize_360_detections.py
Script para visualizar las detecciones YOLO en la imagen 360° original
usando el archivo detections.json generado por analyze_faces.py
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import math
import json
import argparse
import os

class CubemapTo360Visualizer:
    def __init__(self, original_image_path, detections_json_path, cube_size=None):
        """
        Visualizador de detecciones en imagen 360°
        
        Args:
            original_image_path: Ruta a la imagen 360° original
            detections_json_path: Ruta al archivo detections.json
            cube_size: Tamaño de cada cara del cubemap (debe coincidir con la conversión original)
        """
        self.original_image_path = original_image_path
        self.detections_json_path = detections_json_path
        self.cube_size = cube_size
        
        # Cargar imagen original
        self.image = Image.open(original_image_path)
        self.width, self.height = self.image.size
        
        if self.cube_size is None:
            self.cube_size = self.width // 4
        
        # Cargar detecciones
        with open(detections_json_path, 'r') as f:
            self.detections = json.load(f)
        
        # Mapeo de nombres de caras a índices
        self.face_name_to_index = {
            "front": 0,
            "right": 1, 
            "back": 2,
            "left": 3,
            "up": 4,
            "down": 5
        }
        
        # Colores para diferentes clases
        self.colors = [
            (255, 0, 0),    # Clase 0: Rojo
            (0, 255, 0),    # Clase 1: Verde
            (0, 0, 255),    # Clase 2: Azul
            (255, 255, 0),  # Clase 3: Amarillo
            (255, 0, 255),  # Clase 4: Magenta
            (0, 255, 255),  # Clase 5: Cian
        ]
    
    def cubemap_to_equirectangular_coord(self, face_index, cube_x, cube_y):
        """
        Convierte coordenadas de una cara del cubo a coordenadas equirectangulares
        
        Args:
            face_index: Índice de la cara (0-5)
            cube_x, cube_y: Coordenadas en la cara del cubo
            
        Returns:
            eq_x, eq_y: Coordenadas en imagen equirectangular
        """
        # Normalizar coordenadas del cubo a rango [-1, 1]
        a = 2.0 * cube_x / self.cube_size - 1.0
        b = 1.0 - 2.0 * cube_y / self.cube_size
        
        # Convertir a coordenadas 3D según la cara
        if face_index == 0:  # Frente (+Z)
            x, y, z = a, b, 1.0
        elif face_index == 1:  # Derecha (+X)
            x, y, z = 1.0, b, -a
        elif face_index == 2:  # Atrás (-Z)
            x, y, z = -a, b, -1.0
        elif face_index == 3:  # Izquierda (-X)
            x, y, z = -1.0, b, a
        elif face_index == 4:  # Arriba (+Y)
            x, y, z = a, 1.0, -b
        elif face_index == 5:  # Abajo (-Y)
            x, y, z = a, -1.0, b
        
        # Convertir a coordenadas esféricas
        theta = math.atan2(y, math.sqrt(x*x + z*z))
        phi = math.atan2(x, z)
        
        # Convertir a coordenadas de imagen equirectangular
        eq_x = (phi / math.pi + 1.0) * 0.5 * self.width
        eq_y = (0.5 - theta / math.pi) * self.height
        
        # Asegurar que las coordenadas estén dentro de los límites
        eq_x = max(0, min(self.width - 1, eq_x))
        eq_y = max(0, min(self.height - 1, eq_y))
        
        return int(eq_x), int(eq_y)
    
    def transform_bbox_to_equirectangular(self, face_index, bbox):
        """
        Transforma un bounding box de una cara del cubo a coordenadas equirectangulares
        
        Args:
            face_index: Índice de la cara del cubo
            bbox: [x1, y1, x2, y2] en coordenadas de la cara del cubo
            
        Returns:
            Lista de puntos [(x, y)] que forman el contorno en coordenadas equirectangulares
        """
        x1, y1, x2, y2 = bbox
        
        # Crear puntos del perímetro del bounding box
        perimeter_points = []
        
        # Número de puntos para samplear cada lado
        num_points = 20
        
        # Borde superior (y1)
        for i in range(num_points + 1):
            x = x1 + (x2 - x1) * i / num_points
            eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, x, y1)
            perimeter_points.append((eq_x, eq_y))
        
        # Borde derecho (x2)
        for i in range(1, num_points + 1):
            y = y1 + (y2 - y1) * i / num_points
            eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, x2, y)
            perimeter_points.append((eq_x, eq_y))
        
        # Borde inferior (y2)
        for i in range(1, num_points + 1):
            x = x2 - (x2 - x1) * i / num_points
            eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, x, y2)
            perimeter_points.append((eq_x, eq_y))
        
        # Borde izquierdo (x1)
        for i in range(1, num_points):
            y = y2 - (y2 - y1) * i / num_points
            eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, x1, y)
            perimeter_points.append((eq_x, eq_y))
        
        return perimeter_points
    
    def get_bbox_center_in_equirectangular(self, face_index, bbox):
        """
        Calcula el centro de un bounding box y lo convierte a coordenadas equirectangulares
        
        Args:
            face_index: Índice de la cara del cubo
            bbox: [x1, y1, x2, y2] en coordenadas de la cara del cubo
            
        Returns:
            (center_x, center_y): Centro en coordenadas equirectangulares
        """
        x1, y1, x2, y2 = bbox
        
        # Calcular centro del bounding box en coordenadas del cubemap
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Convertir a coordenadas equirectangulares
        eq_x, eq_y = self.cubemap_to_equirectangular_coord(face_index, center_x, center_y)
        
        return eq_x, eq_y
    
    def draw_detection_point(self, draw, center, color, radius=8, label_text="", font=None):
        """
        Dibuja un punto circular para marcar una detección
        
        Args:
            draw: Objeto ImageDraw
            center: (x, y) coordenadas del centro
            color: Color del punto
            radius: Radio del círculo
            label_text: Texto de la etiqueta
            font: Fuente para el texto
        """
        x, y = center
        
        # Dibujar círculo exterior (borde negro)
        draw.ellipse([x-radius-1, y-radius-1, x+radius+1, y+radius+1], fill=(0, 0, 0))
        
        # Dibujar círculo interior (color de la clase)
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color)
        
        # Dibujar punto central más pequeño para mejor visibilidad
        draw.ellipse([x-2, y-2, x+2, y+2], fill=(255, 255, 255))
        
        # Añadir etiqueta si se proporciona
        if label_text and font:
            # Calcular posición de la etiqueta (ligeramente desplazada)
            label_x = x + radius + 5
            label_y = y - 10
            
            # Asegurar que la etiqueta no se salga de la imagen
            if label_x + len(label_text) * 6 > self.width:
                label_x = x - radius - len(label_text) * 6 - 5
            if label_y < 0:
                label_y = y + radius + 5
            
            # Fondo semi-transparente para la etiqueta
            text_bbox = draw.textbbox((label_x, label_y), label_text, font=font)
            padding = 2
            bg_bbox = [text_bbox[0]-padding, text_bbox[1]-padding, 
                      text_bbox[2]+padding, text_bbox[3]+padding]
            draw.rectangle(bg_bbox, fill=(0, 0, 0, 180))
            
            # Texto de la etiqueta
            draw.text((label_x, label_y), label_text, fill=color, font=font)
    
    def visualize_detections(self, output_path, use_points=True, point_radius=8):
        """
        Visualiza todas las detecciones en la imagen 360° original
        
        Args:
            output_path: Ruta donde guardar la imagen con las detecciones
            use_points: Si True, dibuja puntos en lugar de bounding boxes completos
            point_radius: Radio de los puntos de detección
        """
        # Crear una copia de la imagen original para dibujar
        result_image = self.image.copy()
        draw = ImageDraw.Draw(result_image)
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        total_detections = 0
        
        # Procesar cada cara
        for face_name, face_data in self.detections.items():
            if face_name not in self.face_name_to_index:
                continue
                
            face_index = self.face_name_to_index[face_name]
            boxes = face_data.get("boxes", [])
            
            print(f"Procesando cara '{face_name}' (índice {face_index}): {len(boxes)} detecciones")
            
            # Procesar cada detección en esta cara
            for box_data in boxes:
                coordinates = box_data["coordinates"]
                score = box_data["score"]
                class_id = box_data["class"]
                
                # Obtener color para esta clase
                color = self.colors[class_id % len(self.colors)]
                
                if use_points:
                    # Calcular centro del bounding box y convertir a coordenadas equirectangulares
                    center_x, center_y = self.get_bbox_center_in_equirectangular(face_index, coordinates)
                    
                    # Crear etiqueta
                    label_text = f"C{class_id}: {score:.2f}"
                    
                    # Dibujar punto de detección
                    self.draw_detection_point(draw, (center_x, center_y), color, 
                                            radius=point_radius, label_text=label_text, font=font)
                else:
                    # Modo original: dibujar bounding box completo transformado
                    eq_points = self.transform_bbox_to_equirectangular(face_index, coordinates)
                    self.draw_polygon_outline(draw, eq_points, color, width=3)
                    
                    # Agregar etiqueta
                    if eq_points:
                        label_x, label_y = eq_points[0]
                        label_text = f"C{class_id}: {score:.2f}"
                        text_bbox = draw.textbbox((label_x, label_y), label_text, font=font)
                        draw.rectangle(text_bbox, fill=(0, 0, 0, 128))
                        draw.text((label_x, label_y), label_text, fill=color, font=font)
                
                total_detections += 1
        
        # Guardar imagen resultado
        result_image.save(output_path, quality=95)
        
        mode_text = "puntos" if use_points else "bounding boxes"
        print(f"\n¡Visualización completada usando {mode_text}!")
        print(f"Total de detecciones procesadas: {total_detections}")
        print(f"Imagen guardada en: {output_path}")
        
        return result_image
    
    def draw_polygon_outline(self, draw, points, color, width=3):
        """
        Dibuja el contorno de un polígono punto por punto
        """
        for i in range(len(points)):
            start_point = points[i]
            end_point = points[(i + 1) % len(points)]
            
            # Dibujar línea entre puntos consecutivos
            draw.line([start_point, end_point], fill=color, width=width)

def main():
    parser = argparse.ArgumentParser(description="Visualizar detecciones YOLO en imagen 360° original")
    parser.add_argument("-i", "--image", required=True, help="Ruta a imagen 360° original")
    parser.add_argument("-d", "--detections", required=True, help="Ruta al archivo detections.json")
    parser.add_argument("-o", "--output", default="360_with_detections.jpg", help="Ruta de salida")
    parser.add_argument("-c", "--cube-size", type=int, default=None, help="Tamaño de cara del cubemap")
    parser.add_argument("--points", action="store_true", default=True, help="Usar puntos en lugar de bounding boxes")
    parser.add_argument("--boxes", action="store_true", help="Usar bounding boxes completos en lugar de puntos")
    parser.add_argument("--radius", type=int, default=8, help="Radio de los puntos de detección")
    
    args = parser.parse_args()
    
    # Verificar que los archivos existen
    if not os.path.exists(args.image):
        print(f"Error: No se encontró la imagen {args.image}")
        return
    
    if not os.path.exists(args.detections):
        print(f"Error: No se encontró el archivo {args.detections}")
        return
    
    # Determinar modo de visualización
    use_points = not args.boxes  # Por defecto usar puntos, a menos que se especifique --boxes
    
    mode_text = "puntos" if use_points else "bounding boxes"
    print("=== Visualizador de Detecciones 360° ===")
    print(f"Imagen original: {args.image}")
    print(f"Detecciones: {args.detections}")
    print(f"Salida: {args.output}")
    print(f"Modo: {mode_text}")
    if use_points:
        print(f"Radio de puntos: {args.radius}")
    
    # Crear visualizador
    visualizer = CubemapTo360Visualizer(args.image, args.detections, args.cube_size)
    
    # Generar visualización
    visualizer.visualize_detections(args.output, use_points=use_points, point_radius=args.radius)

if __name__ == "__main__":
    main()