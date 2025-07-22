import cv2
import numpy as np
import math
import os
import argparse

class TreeViewExtractor:
    def __init__(self, input_image_path, output_dir="tree_views"):
        """
        Extractor de vistas específicas de árboles desde imágenes 360°
        """
        self.input_path = input_image_path
        self.output_dir = output_dir
        self.image = None
        self.width = 0
        self.height = 0
        
        # # Coordenadas hardcodeadas de la cámara y el árbol
        # self.camera_lat = 40.4281686
        # self.camera_lon = -3.6933211
        # self.camera_heading = 196.84  # Orientación inicial de la imagen
        
        self.tree_lat = 40.4281550
        self.tree_lon = -3.6932950
        
        os.makedirs(output_dir, exist_ok=True)
        
    def load_image(self):
        """Carga la imagen 360° usando OpenCV"""
        try:
            self.image = cv2.imread(self.input_path)
            if self.image is None:
                raise ValueError("No se pudo cargar la imagen")
            
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.height, self.width = self.image.shape[:2]
            
            print(f"Imagen cargada: {self.width}x{self.height}")
            return True
            
        except Exception as e:
            print(f"Error al cargar la imagen: {e}")
            return False
    
    def calculate_bearing(self, lat1, lon1, lat2, lon2):
        """
        Calcula el bearing (ángulo) desde punto 1 hacia punto 2
        """
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        lon1_rad = math.radians(lon1)
        lon2_rad = math.radians(lon2)
        
        dlon = lon2_rad - lon1_rad
        
        x = math.sin(dlon) * math.cos(lat2_rad)
        y = math.cos(lat1_rad) * math.sin(lat2_rad) - math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)
        
        bearing = math.atan2(x, y)
        bearing_degrees = (math.degrees(bearing) + 360) % 360
        
        return bearing_degrees
    
    def extract_view_at_bearing(self, target_bearing, fov=90, pitch=0, output_size=(1024, 1024)):
        """
        Extrae una vista de la imagen 360° mirando hacia un bearing específico
        
        Args:
            target_bearing: Dirección hacia donde mirar (0-360°)
            fov: Campo de visión en grados (por defecto 90°)
            pitch: Inclinación vertical en grados (0 = horizonte, + = arriba, - = abajo)
            output_size: Tamaño de la imagen de salida (ancho, alto)
        """
        width_out, height_out = output_size
        
        # Ajustar el bearing considerando la orientación inicial de la cámara
        # La imagen 360° tiene su "frente" apuntando hacia camera_heading
        adjusted_bearing = (target_bearing - self.camera_heading) % 360
        
        # Convertir a radianes
        yaw_rad = np.radians(adjusted_bearing)
        pitch_rad = np.radians(pitch)
        fov_rad = np.radians(fov)
        
        # Crear grilla de coordenadas para la imagen de salida
        i, j = np.meshgrid(np.arange(width_out), np.arange(height_out))
        
        # Normalizar coordenadas (-1 a +1)
        x_norm = 2.0 * i / width_out - 1.0
        y_norm = 1.0 - 2.0 * j / height_out
        
        # Aplicar FOV
        x_norm *= np.tan(fov_rad / 2)
        y_norm *= np.tan(fov_rad / 2)
        
        # Coordenadas 3D de la vista
        x = x_norm
        y = y_norm
        z = np.ones_like(x_norm)
        
        # Aplicar rotación pitch
        y_rot = y * np.cos(pitch_rad) - z * np.sin(pitch_rad)
        z_rot = y * np.sin(pitch_rad) + z * np.cos(pitch_rad)
        y = y_rot
        z = z_rot
        
        # Aplicar rotación yaw
        x_rot = x * np.cos(yaw_rad) + z * np.sin(yaw_rad)
        z_rot = -x * np.sin(yaw_rad) + z * np.cos(yaw_rad)
        x = x_rot
        z = z_rot
        
        # Convertir a coordenadas esféricas
        theta = np.arctan2(y, np.sqrt(x*x + z*z))
        phi = np.arctan2(x, z)
        
        # Mapear a coordenadas de imagen equirectangular
        img_x = (phi / np.pi + 1.0) * 0.5 * self.width
        img_y = (0.5 - theta / np.pi) * self.height
        
        # Asegurar límites
        img_x = np.clip(img_x, 0, self.width - 1)
        img_y = np.clip(img_y, 0, self.height - 1)
        
        # Remapear la imagen
        map_x = img_x.astype(np.float32)
        map_y = img_y.astype(np.float32)
        
        extracted_view = cv2.remap(self.image, map_x, map_y, cv2.INTER_LINEAR)
        
        return extracted_view
    
    def extract_tree_views(self):
        """
        Extrae vistas del árbol desde diferentes configuraciones
        """
        if not self.load_image():
            return False
        
        # Calcular bearing desde la cámara hacia el árbol
        tree_bearing = self.calculate_bearing(
            self.camera_lat, self.camera_lon,
            self.tree_lat, self.tree_lon
        )
        
        print(f"\n=== Información de extracción ===")
        print(f"Posición cámara: {self.camera_lat}, {self.camera_lon}")
        print(f"Heading cámara: {self.camera_heading}°")
        print(f"Posición árbol: {self.tree_lat}, {self.tree_lon}")
        print(f"Bearing hacia el árbol: {tree_bearing:.2f}°")
        print(f"Diferencia con heading: {(tree_bearing - self.camera_heading):.2f}°")
        
        # Configuraciones de vistas a extraer
        views = [
            # Vista principal centrada en el árbol
            {
                "name": "tree_centered_fov90",
                "bearing": tree_bearing,
                "fov": 90,
                "pitch": 0,
                "size": (1024, 1024)
            },
            # Vista con FOV más estrecho para más detalle
            {
                "name": "tree_zoom_fov60",
                "bearing": tree_bearing,
                "fov": 60,
                "pitch": 0,
                "size": (1024, 1024)
            },
            # Vista mirando ligeramente hacia arriba (para capturar copa)
            {
                "name": "tree_elevated_pitch15",
                "bearing": tree_bearing,
                "fov": 90,
                "pitch": 15,
                "size": (1024, 1024)
            },
            # Vista mirando hacia abajo (para capturar base/alcorque)
            {
                "name": "tree_base_pitch-15",
                "bearing": tree_bearing,
                "fov": 90,
                "pitch": -15,
                "size": (1024, 1024)
            },
            # Vista panorámica más amplia
            {
                "name": "tree_wide_fov120",
                "bearing": tree_bearing,
                "fov": 120,
                "pitch": 0,
                "size": (1280, 720)
            }
        ]
        
        saved_files = []
        
        print(f"\nExtrayendo {len(views)} vistas del árbol...")
        
        for view in views:
            print(f"\nProcesando: {view['name']}")
            print(f"  - FOV: {view['fov']}°")
            print(f"  - Pitch: {view['pitch']}°")
            print(f"  - Tamaño: {view['size']}")
            
            extracted_view = self.extract_view_at_bearing(
                view['bearing'],
                view['fov'],
                view['pitch'],
                view['size']
            )
            
            # Guardar imagen
            filename = f"{view['name']}.jpg"
            filepath = os.path.join(self.output_dir, filename)
            
            extracted_view_bgr = cv2.cvtColor(extracted_view, cv2.COLOR_RGB2BGR)
            cv2.imwrite(filepath, extracted_view_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            saved_files.append(filepath)
            print(f"  ✓ Guardado: {filepath}")
        
        # Crear una imagen de referencia mostrando la dirección
        self._create_reference_image(tree_bearing)
        
        return saved_files
    
    def _create_reference_image(self, tree_bearing):
        """
        Crea una imagen de referencia mostrando la ubicación del árbol en la imagen 360°
        """
        # Crear copia de la imagen original
        reference = self.image.copy()
        height, width = reference.shape[:2]
        
        # Calcular posición X del árbol en la imagen 360°
        adjusted_bearing = (tree_bearing - self.camera_heading) % 360
        x_position = int((adjusted_bearing / 360.0) * width)
        
        # Dibujar línea vertical donde está el árbol
        cv2.line(reference, (x_position, 0), (x_position, height), (0, 255, 0), 3)
        
        # Añadir texto
        cv2.putText(reference, f"Tree bearing: {tree_bearing:.1f}°", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Guardar imagen de referencia
        reference_bgr = cv2.cvtColor(reference, cv2.COLOR_RGB2BGR)
        reference_path = os.path.join(self.output_dir, "reference_360.jpg")
        cv2.imwrite(reference_path, reference_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"\n✓ Imagen de referencia guardada: {reference_path}")


def main():
    parser = argparse.ArgumentParser(description="Extractor de vistas de árboles desde imágenes 360°")
    parser.add_argument("-i", "--image", required=True, help="Ruta a imagen 360° de Street View")
    parser.add_argument("-o", "--output-dir", default="tree_views", help="Directorio de salida")
    
    # Opción para usar coordenadas personalizadas
    parser.add_argument("--camera-lat", type=float, help="Latitud de la cámara (opcional)")
    parser.add_argument("--camera-lon", type=float, help="Longitud de la cámara (opcional)")
    parser.add_argument("--camera-heading", type=float, help="Heading de la cámara (opcional)")
    parser.add_argument("--tree-lat", type=float, help="Latitud del árbol (opcional)")
    parser.add_argument("--tree-lon", type=float, help="Longitud del árbol (opcional)")
    
    args = parser.parse_args()
    
    print("=== Extractor de Vista de Árbol ===\n")
    
    extractor = TreeViewExtractor(args.image, args.output_dir)
    
    # Sobrescribir coordenadas si se proporcionan
    if args.camera_lat:
        extractor.camera_lat = args.camera_lat
    if args.camera_lon:
        extractor.camera_lon = args.camera_lon
    if args.camera_heading:
        extractor.camera_heading = args.camera_heading
    if args.tree_lat:
        extractor.tree_lat = args.tree_lat
    if args.tree_lon:
        extractor.tree_lon = args.tree_lon
    
    # Extraer vistas
    saved_files = extractor.extract_tree_views()
    
    if saved_files:
        print(f"\n¡Extracción completada!")
        print(f"Se guardaron {len(saved_files)} vistas en: {args.output_dir}")
    else:
        print("\nError en la extracción")


if __name__ == "__main__":
    main()