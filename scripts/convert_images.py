import cv2
import numpy as np
import math
import os
import argparse
from concurrent.futures import ThreadPoolExecutor
import time

class OptimizedCubemapConverter:
    def __init__(self, input_image_path, output_dir="cubemap_output", cube_size=None):
        """
        Conversor de cubemap optimizado con OpenCV y NumPy
        
        Args:
            input_image_path: Ruta de la imagen 360° equirectangular
            output_dir: Directorio donde guardar las caras del cubo
            cube_size: Tamaño de cada cara del cubo
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
            # Cargar en RGB directamente
            self.image = cv2.imread(self.input_path)
            if self.image is None:
                raise ValueError("No se pudo cargar la imagen")
            
            # Convertir de BGR a RGB inmediatamente después de cargar
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
    
    def create_face_mapping(self, face_index):
        """
        Crea el mapeo de coordenadas para una cara usando operaciones vectorizadas
        Mucho más rápido que el procesamiento pixel por pixel
        """
        # Crear grillas de coordenadas
        i, j = np.meshgrid(np.arange(self.cube_size), np.arange(self.cube_size))
        
        # Normalizar coordenadas
        a = 2.0 * i / self.cube_size - 1.0
        b = 1.0 - 2.0 * j / self.cube_size
        
        # Mapear según la cara
        if face_index == 0:  # Frente (+Z)
            x, y, z = a, b, np.ones_like(a)
        elif face_index == 1:  # Derecha (+X)
            x, y, z = np.ones_like(a), b, -a
        elif face_index == 2:  # Atrás (-Z)
            x, y, z = -a, b, -np.ones_like(a)
        elif face_index == 3:  # Izquierda (-X)
            x, y, z = -np.ones_like(a), b, a
        elif face_index == 4:  # Arriba (+Y)
            x, y, z = a, np.ones_like(a), -b
        elif face_index == 5:  # Abajo (-Y)
            x, y, z = a, -np.ones_like(a), b
        
        # Convertir a coordenadas esféricas
        theta = np.arctan2(y, np.sqrt(x*x + z*z))
        phi = np.arctan2(x, z)
        
        # Mapear a coordenadas de imagen
        img_x = (phi / np.pi + 1.0) * 0.5 * self.width
        img_y = (0.5 - theta / np.pi) * self.height
        
        # Asegurar que están dentro de los límites
        img_x = np.clip(img_x, 0, self.width - 1)
        img_y = np.clip(img_y, 0, self.height - 1)
        
        return img_x.astype(np.float32), img_y.astype(np.float32)
    
    def extract_face_vectorized(self, face_index):
        """Extrae una cara usando operaciones vectorizadas de OpenCV"""
        map_x, map_y = self.create_face_mapping(face_index)
        
        # Usar remap de OpenCV para sampling eficiente
        face_image = cv2.remap(self.image, map_x, map_y, cv2.INTER_LINEAR)
        
        return face_image
    
    def extract_face_parallel(self, face_index):
        """Wrapper para procesamiento paralelo"""
        return face_index, self.extract_face_vectorized(face_index)
    
    def convert_to_cubemap_parallel(self):
        """Convierte usando procesamiento paralelo"""
        if not self.load_image():
            return False
        
        face_names = ["front", "right", "back", "left", "up", "down"]
        face_paths = []
        
        print("Iniciando conversión paralela a cubemap...")
        start_time = time.time()
        
        # Procesamiento paralelo de todas las caras
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(self.extract_face_parallel, i) for i in range(6)]
            
            for future in futures:
                face_index, face_image = future.result()
                
                filename = f"{face_names[face_index]}.jpg"
                filepath = os.path.join(self.output_dir, filename)
                
                # Guardar con cv2.imwrite (que espera BGR, así que convertimos RGB->BGR)
                face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, face_image_bgr)
                face_paths.append(filepath)
                
                print(f"Guardado: {filepath}")
        
        end_time = time.time()
        print(f"¡Conversión completada en {end_time - start_time:.2f} segundos!")
        return face_paths
    
    def convert_to_cubemap_sequential(self):
        """Versión secuencial optimizada"""
        if not self.load_image():
            return False
        
        face_names = ["front", "right", "back", "left", "up", "down"]
        face_paths = []
        
        print("Iniciando conversión secuencial optimizada...")
        start_time = time.time()
        
        for face_index in range(6):
            print(f"Procesando cara {face_index + 1}/6: {face_names[face_index]}")
            
            face_image = self.extract_face_vectorized(face_index)
            filename = f"{face_names[face_index]}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            # Guardar con cv2.imwrite (convertir RGB->BGR)
            face_image_bgr = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, face_image_bgr)
            face_paths.append(filepath)
            
            print(f"Guardado: {filepath}")
        
        end_time = time.time()
        print(f"¡Conversión completada en {end_time - start_time:.2f} segundos!")
        return face_paths


# Alternativa usando py360convert (biblioteca especializada)
def convert_with_py360convert(input_path, output_dir, cube_size=None):
    """
    Alternativa usando la biblioteca py360convert
    Requiere: pip install py360convert
    """
    try:
        import py360convert
        
        # Cargar imagen
        equirectangular = cv2.imread(input_path)
        if equirectangular is None:
            print("Error al cargar la imagen")
            return False
        
        h, w = equirectangular.shape[:2]
        if cube_size is None:
            cube_size = w // 4
        
        print(f"Convirtiendo con py360convert (tamaño: {cube_size}x{cube_size})...")
        start_time = time.time()
        
        # Convertir a cubemap
        cubemap = py360convert.e2c(equirectangular, face_w=cube_size, mode='bilinear')
        
        # Guardar caras
        face_names = ["front", "right", "back", "left", "up", "down"]
        face_paths = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i, face_name in enumerate(face_names):
            face_image = cubemap[:, :, i*3:(i+1)*3]  # Extraer cara
            filename = f"{face_name}.jpg"
            filepath = os.path.join(output_dir, filename)
            
            cv2.imwrite(filepath, face_image)
            face_paths.append(filepath)
            print(f"Guardado: {filepath}")
        
        end_time = time.time()
        print(f"¡Conversión con py360convert completada en {end_time - start_time:.2f} segundos!")
        return face_paths
        
    except ImportError:
        print("py360convert no está instalado. Instálalo con: pip install py360convert")
        return False


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Convertidor de cubemap optimizado")
    parser.add_argument("-i", "--image", required=True, help="Ruta a imagen equirectangular 360°")
    parser.add_argument("-o", "--output-dir", default="cubemap_output", help="Directorio de salida")
    parser.add_argument("-c", "--cube-size", type=int, default=None, help="Tamaño de cada cara del cubemap")
    parser.add_argument("-m", "--method", choices=["opencv", "parallel", "py360convert"], 
                       default="parallel", help="Método de conversión")
    
    args = parser.parse_args()
    
    print(f"=== Convertidor de Cubemap Optimizado (método: {args.method}) ===\n")
    
    if args.method == "py360convert":
        face_paths = convert_with_py360convert(args.image, args.output_dir, args.cube_size)
    else:
        converter = OptimizedCubemapConverter(args.image, args.output_dir, args.cube_size)
        
        if args.method == "parallel":
            face_paths = converter.convert_to_cubemap_parallel()
        else:  # opencv
            face_paths = converter.convert_to_cubemap_sequential()
    
    if face_paths:
        print(f"\n¡Conversión exitosa! Archivos guardados en: {args.output_dir}")
    else:
        print("Error en la conversión")


if __name__ == "__main__":
    main()