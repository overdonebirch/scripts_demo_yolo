#!/usr/bin/env python3
"""
image_validator.py
Validador de imágenes 360° equirectangulares - versión permisiva
"""

import os
import sys
from PIL import Image
from PIL.ExifTags import TAGS
import argparse
from pathlib import Path

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

class ImageValidator:
    def __init__(self):
        pass
    
    def extract_exif_data(self, image_path):
        """Extrae datos EXIF básicos"""
        try:
            with Image.open(image_path) as image:
                exif_data = image.getexif()
                if not exif_data:
                    return None
                
                exif_dict = {}
                for tag_id, value in exif_data.items():
                    tag = TAGS.get(tag_id, tag_id)
                    exif_dict[tag] = value
                
                return exif_dict
        except:
            return None
    
    def check_aspect_ratio(self, image_path):
        """Verifica aspect ratio 2:1"""
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                ratio = width / height
                return abs(ratio - 2.0) < 0.2, ratio  # Más tolerancia
        except:
            return False, 0
    
    def is_360_image(self, image_path, verbose=False):
        """
        Validación más permisiva de imágenes 360°
        """
        exif_data = self.extract_exif_data(image_path)
        
        # Criterio 1: Software conocido (más flexible)
        if exif_data:
            software = exif_data.get('Software', '').lower()
            if any(term in software for term in ['google', 'street view', 'ricoh', 'theta', 'gear', '360', 'samsung']):
                if verbose:
                    print(f"✅ Software 360°: {software}")
                return True
        
        # Criterio 2: Cámaras conocidas (más flexible)
        if exif_data:
            make = exif_data.get('Make', '').lower()
            model = exif_data.get('Model', '').lower()
            
            if any(term in f"{make} {model}" for term in ['ricoh', 'theta', 'samsung', 'gear', 'lg', 'garmin', 'kodak', 'insta360']):
                if verbose:
                    print(f"✅ Cámara 360°: {make} {model}")
                return True
        
        # Criterio 3: Aspect ratio 2:1 (muy común en 360°)
        is_2_1, ratio = self.check_aspect_ratio(image_path)
        if is_2_1:
            if verbose:
                print(f"✅ Aspect ratio 2:1: {ratio:.2f}")
            return True
        
        # Criterio 4: Nombre de archivo sugerente + dimensiones grandes
        filename = Path(image_path).name.lower()
        if any(term in filename for term in ['360', 'equirect', 'pano', 'spherical', 'street']):
            try:
                with Image.open(image_path) as image:
                    width, height = image.size
                    if width >= 3000:  # Imágenes 360° suelen ser grandes
                        if verbose:
                            print(f"✅ Nombre + tamaño: {filename}, {width}x{height}")
                        return True
            except:
                pass
        
        if verbose:
            print(f"❌ No cumple criterios 360°")
            if exif_data:
                print(f"   Software: {exif_data.get('Software', 'N/A')}")
                print(f"   Cámara: {exif_data.get('Make', 'N/A')} {exif_data.get('Model', 'N/A')}")
            print(f"   Ratio: {self.check_aspect_ratio(image_path)[1]:.2f}")
        
        return False
    
    def validate_image(self, image_path, verbose=False):
        """Compatibilidad con pipeline"""
        is_360 = self.is_360_image(image_path, verbose)
        return is_360, 100 if is_360 else 0, {"is_360": is_360}

def main():
    parser = argparse.ArgumentParser(description="Validador de imágenes 360°")
    parser.add_argument("images", nargs="+", help="Imágenes para validar")
    parser.add_argument("-v", "--verbose", action="store_true", help="Modo detallado")
    
    args = parser.parse_args()
    
    validator = ImageValidator()
    valid_count = 0
    
    for image_path in args.images:
        if os.path.isdir(image_path):
            continue
            
        if not os.path.exists(image_path):
            print(f"⚠️ No encontrado: {image_path}")
            continue
            
        is_360 = validator.is_360_image(str(image_path), args.verbose)
        
        if is_360:
            valid_count += 1
            print(f"✅ {Path(image_path).name}")
        else:
            print(f"❌ {Path(image_path).name}")
    
    print(f"\nImágenes 360°: {valid_count}/{len(args.images)}")
    return 0

if __name__ == "__main__":
   main()