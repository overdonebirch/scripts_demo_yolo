import cv2
import numpy as np
import math
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import time

class FlexibleCubemapConverter:
    def __init__(self, input_image_path, output_dir="cubemap_output", cube_size=None):
        """
        Conversor de cubemap con ángulos personalizables
        """
        self.input_path = input_image_path
        self.output_dir = output_dir
        self.image = None
        self.width = 0
        self.height = 0
        self.cube_size = cube_size
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_image(self):
        """Carga la imagen 360° usando OpenCV"""
        try:
            self.image = cv2.imread(self.input_path)
            if self.image is None:
                raise ValueError("No se pudo cargar la imagen")
            
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.height, self.width = self.image.shape[:2]
            
            if self.cube_size is None:
                self.cube_size = self.width // 4
                
            print(f"Imagen cargada: {self.width}x{self.height}")
            print(f"Tamaño de cara del cubo: {self.cube_size}x{self.cube_size}")
            
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return False
        return True
    
    def create_custom_face_mapping(self, yaw, pitch, roll=0):
        """
        Crea mapeo para una cara con orientación personalizada
        
        Args:
            yaw: Rotación horizontal (0-360°) - 0° = Norte, 90° = Este
            pitch: Elevación (-90° a +90°) - 0° = horizonte, +90° = directamente arriba
            roll: Rotación de la cámara (generalmente 0°)
        """
        # Convertir grados a radianes
        yaw_rad = np.radians(yaw)
        pitch_rad = np.radians(pitch)
        roll_rad = np.radians(roll)
        
        # Crear grilla de coordenadas
        i, j = np.meshgrid(np.arange(self.cube_size), np.arange(self.cube_size))
        
        # Normalizar coordenadas de la cara (-1 a +1)
        a = 2.0 * i / self.cube_size - 1.0
        b = 1.0 - 2.0 * j / self.cube_size
        
        # Coordenadas iniciales de la cara (mirando hacia +Z)
        x = a
        y = b
        z = np.ones_like(a)
        
        # Aplicar rotaciones
        # Rotación pitch (alrededor del eje X)
        y_rot = y * np.cos(pitch_rad) - z * np.sin(pitch_rad)
        z_rot = y * np.sin(pitch_rad) + z * np.cos(pitch_rad)
        y = y_rot
        z = z_rot
        
        # Rotación yaw (alrededor del eje Y)
        x_rot = x * np.cos(yaw_rad) + z * np.sin(yaw_rad)
        z_rot = -x * np.sin(yaw_rad) + z * np.cos(yaw_rad)
        x = x_rot
        z = z_rot
        
        # Rotación roll (alrededor del eje Z) - opcional
        if roll != 0:
            x_rot = x * np.cos(roll_rad) - y * np.sin(roll_rad)
            y_rot = x * np.sin(roll_rad) + y * np.cos(roll_rad)
            x = x_rot
            y = y_rot
        
        # Convertir a coordenadas esféricas
        theta = np.arctan2(y, np.sqrt(x*x + z*z))
        phi = np.arctan2(x, z)
        
        # Mapear a coordenadas de imagen equirectangular
        img_x = (phi / np.pi + 1.0) * 0.5 * self.width
        img_y = (0.5 - theta / np.pi) * self.height
        
        # Asegurar límites
        img_x = np.clip(img_x, 0, self.width - 1)
        img_y = np.clip(img_y, 0, self.height - 1)
        
        return img_x.astype(np.float32), img_y.astype(np.float32)
    
    def extract_custom_face(self, yaw, pitch, roll=0):
        """Extrae una cara con orientación personalizada"""
        map_x, map_y = self.create_custom_face_mapping(yaw, pitch, roll)
        face_image = cv2.remap(self.image, map_x, map_y, cv2.INTER_LINEAR)
        return face_image
    
    def convert_tree_optimized_views(self, elevation_angle=30):
        """
        Genera vistas optimizadas para capturar árboles
        
        Args:
            elevation_angle: Ángulo de elevación en grados (recomendado: 15-45°)
        """
        if not self.load_image():
            return False
        
        print(f"Generando vistas optimizadas para árboles (elevación: {elevation_angle}°)...")
        start_time = time.time()
        
        # Configuración de vistas para capturar árboles
        views = [
            # Vistas elevadas en 4 direcciones principales
            {"name": "north_elevated", "yaw": 0, "pitch": elevation_angle},
            {"name": "east_elevated", "yaw": 90, "pitch": elevation_angle},
            {"name": "south_elevated", "yaw": 180, "pitch": elevation_angle},
            {"name": "west_elevated", "yaw": 270, "pitch": elevation_angle},
            
            # Vistas adicionales en diagonales
            {"name": "northeast_elevated", "yaw": 45, "pitch": elevation_angle},
            {"name": "southeast_elevated", "yaw": 135, "pitch": elevation_angle},
            {"name": "southwest_elevated", "yaw": 225, "pitch": elevation_angle},
            {"name": "northwest_elevated", "yaw": 315, "pitch": elevation_angle},
            
            # Vista directamente hacia arriba para referencia
            {"name": "zenith", "yaw": 0, "pitch": 90},
            
            # Vistas horizontales tradicionales para comparación
            {"name": "north_horizon", "yaw": 0, "pitch": 0},
            {"name": "east_horizon", "yaw": 90, "pitch": 0},
            {"name": "south_horizon", "yaw": 180, "pitch": 0},
            {"name": "west_horizon", "yaw": 270, "pitch": 0},
        ]
        
        face_paths = []
        
        for view in views:
            print(f"Procesando vista: {view['name']} (yaw: {view['yaw']}°, pitch: {view['pitch']}°)")
            
            face_image = self.extract_custom_face(view['yaw'], view['pitch'])
            filename = f"{view['name']}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, face_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            face_paths.append(filepath)
            
            print(f"Guardado: {filepath}")
        
        end_time = time.time()
        print(f"¡Conversión completada en {end_time - start_time:.2f} segundos!")
        return face_paths
    
    def convert_custom_angles(self, angle_configs):
        """
        Convierte usando configuraciones de ángulos personalizadas
        
        Args:
            angle_configs: Lista de diccionarios con 'name', 'yaw', 'pitch', 'roll' (opcional)
        """
        if not self.load_image():
            return False
        
        print("Generando vistas con ángulos personalizados...")
        start_time = time.time()
        
        face_paths = []
        
        for config in angle_configs:
            name = config['name']
            yaw = config['yaw']
            pitch = config['pitch']
            roll = config.get('roll', 0)
            
            print(f"Procesando: {name} (yaw: {yaw}°, pitch: {pitch}°, roll: {roll}°)")
            
            face_image = self.extract_custom_face(yaw, pitch, roll)
            filename = f"{name}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, face_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            face_paths.append(filepath)
            
            print(f"Guardado: {filepath}")
        
        end_time = time.time()
        print(f"¡Conversión completada en {end_time - start_time:.2f} segundos!")
        return face_paths
    
    def convert_multi_elevation_survey(self, yaw_angles=None, pitch_angles=None):
        """
        Genera un survey completo con múltiples elevaciones
        Ideal para análisis detallado de vegetación
        """
        if yaw_angles is None:
            yaw_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Cada 45°
        if pitch_angles is None:
            pitch_angles = [0, 15, 30, 45, 60, 75, 90]  # Desde horizonte hasta cenital
        
        if not self.load_image():
            return False
        
        print(f"Generando survey completo ({len(yaw_angles)} direcciones × {len(pitch_angles)} elevaciones)...")
        start_time = time.time()
        
        # Crear subdirectorio para el survey
        survey_dir = os.path.join(self.output_dir, "multi_elevation_survey")
        os.makedirs(survey_dir, exist_ok=True)
        
        face_paths = []
        
        for pitch in pitch_angles:
            for yaw in yaw_angles:
                name = f"survey_yaw{yaw:03d}_pitch{pitch:02d}"
                print(f"Procesando: {name}")
                
                face_image = self.extract_custom_face(yaw, pitch)
                filename = f"{name}.jpg"
                filepath = os.path.join(survey_dir, filename)
                
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, face_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                face_paths.append(filepath)
        
        end_time = time.time()
        print(f"¡Survey completado en {end_time - start_time:.2f} segundos!")
        print(f"Generadas {len(face_paths)} imágenes en: {survey_dir}")
        return face_paths


def main():
    """Función principal mejorada"""
    parser = argparse.ArgumentParser(description="Convertidor de cubemap con ángulos personalizables")
    parser.add_argument("-i", "--image", required=True, help="Ruta a imagen equirectangular 360°")
    parser.add_argument("-o", "--output-dir", default="cubemap_output", help="Directorio de salida")
    parser.add_argument("-c", "--cube-size", type=int, default=None, help="Tamaño de cada cara del cubemap")
    parser.add_argument("-m", "--method", choices=["trees", "custom", "survey"], 
                       default="trees", help="Método de conversión")
    parser.add_argument("-e", "--elevation", type=int, default=-45, 
                       help="Ángulo de elevación para vista de árboles (grados)")
    
    args = parser.parse_args()
    
    print(f"=== Convertidor de Cubemap Flexible (método: {args.method}) ===\n")
    
    converter = FlexibleCubemapConverter(args.image, args.output_dir, args.cube_size)
    
    if args.method == "trees":
        # Optimizado para capturar árboles
        face_paths = converter.convert_tree_optimized_views(args.elevation)
    
    elif args.method == "custom":
        # Ejemplo de configuración personalizada
        custom_angles = [
            {"name": "trees_north", "yaw": 0, "pitch": 25},
            {"name": "trees_south", "yaw": 180, "pitch": 25},
            {"name": "canopy_overview", "yaw": 0, "pitch": 60},
            {"name": "trunk_detail", "yaw": 90, "pitch": -10},
        ]
        face_paths = converter.convert_custom_angles(custom_angles)
    
    elif args.method == "survey":
        # Survey completo multi-elevación
        face_paths = converter.convert_multi_elevation_survey()
    
    if face_paths:
        print(f"\n¡Conversión exitosa! Archivos guardados en: {args.output_dir}")
        print(f"Total de imágenes generadas: {len(face_paths)}")
    else:
        print("Error en la conversión")


if __name__ == "__main__":
    main()