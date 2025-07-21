#!/usr/bin/env python3
import google.generativeai as genai
from PIL import Image
import json
import os
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime

class AnalizadorArboles:
    def __init__(self, api_key):
        """Inicializar el analizador con la API key de Gemini"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def analizar_arbol(self, imagen_path):
        """Analiza una imagen individual de árbol usando Gemini"""
        try:
            imagen = Image.open(imagen_path)
            
            prompt = """Como experto arboricultor, analiza esta imagen y responde las siguientes preguntas:

1. ¿El árbol está interfiriendo con la visibilidad de señales o semáforos? Indica si la obstrucción es parcial o total.
2. ¿Se observan ramas o follaje tocando o muy próximas a fachadas, farolas u otros elementos construidos? Clasifica la severidad: leve, moderada o severa.
3. ¿Presenta el árbol síntomas de mal estado (coloración anómala, baja densidad foliar, grietas, ramas rotas, inclinación peligrosa)? Enumera los síntomas detectados.
4. ¿Hay ramas que proyecten sobre la calzada por debajo de 4,5 m o sobre la acera por debajo de 2,2 m? Señala ubicación y grado de severidad.
5. ¿Se detectan bolsones sedosos característicos de la procesionaria? Indica número aproximado y proximidad a zonas sensibles (colegios, parques).
6. ¿Aparecen estructuras de gran tamaño (> 40 cm) en la copa, como nidos de cotorras? Describe su diámetro aproximado y posición en el árbol.

Responde SOLO en formato JSON válido con la siguiente estructura:
{
    "interferencia_senales": "...",
    "ramas_follaje": "...",
    "sintomas_mal_estado": "...",
    "ramas_circulacion": "...",
    "bolsones_procesionaria": "...",
    "estructuras_copa": "...",
    "descripcion": "descripción breve"
}
"""
            
            response = self.model.generate_content([prompt, imagen])
            return response.text
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "hay_arbol": False,
                "estado_general": "error",
                "riesgo_nivel": 0,
                "descripcion": f"Error procesando imagen: {str(e)}"
            })
    
    def procesar_imagen_individual(self, imagen_path):
        """Procesa una imagen individual"""
        print(f"🌳 Analizando: {Path(imagen_path).name}")
        
        try:
            analisis_texto = self.analizar_arbol(imagen_path)
            
            # Intentar parsear JSON
            try:
                analisis = json.loads(analisis_texto)
            except json.JSONDecodeError:
                # Si no es JSON válido, crear estructura básica
                analisis = {
                    "hay_arbol": "árbol" in analisis_texto.lower(),
                    "estado_general": "indeterminado",
                    "riesgo_nivel": 5,
                    "descripcion": "Análisis no estructurado",
                    "analisis_texto": analisis_texto
                }
            
            # Mostrar resumen en consola
            if analisis.get('hay_arbol', False):
                estado = analisis.get('estado_general', 'indeterminado')
                riesgo = analisis.get('riesgo_nivel', 0)
                descripcion = analisis.get('descripcion', '')
                print(f"   ✅ Árbol detectado")
                print(f"   📊 Estado: {estado}")
                print(f"   ⚠️  Riesgo: {riesgo}/10")
                if descripcion:
                    print(f"   📝 {descripcion}")
                
                problemas = analisis.get('problemas', [])
                if problemas:
                    print(f"   🚨 Problemas: {', '.join(problemas)}")
                    
                obstrucciones = analisis.get('obstrucciones', [])
                if obstrucciones:
                    print(f"   🚧 Obstrucciones: {', '.join(obstrucciones)}")
            else:
                print(f"   ⚪ No se detectó árbol en esta imagen")
            
            return analisis
            
        except Exception as e:
            print(f"   ❌ Error analizando imagen: {e}")
            return {
                "error": str(e),
                "hay_arbol": False,
                "estado_general": "error",
                "riesgo_nivel": 0
            }
    
    def procesar_directorio(self, directorio):
        """Procesa todas las imágenes de un directorio"""
        extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        imagenes = []
        
        for ext in extensiones:
            imagenes.extend(glob.glob(os.path.join(directorio, ext)))
            imagenes.extend(glob.glob(os.path.join(directorio, ext.upper())))
        
        if not imagenes:
            print(f"❌ No se encontraron imágenes en {directorio}")
            return []
        
        print(f"📁 Procesando {len(imagenes)} imágenes del directorio: {directorio}")
        
        resultados = []
        for i, imagen_path in enumerate(imagenes, 1):
            print(f"\n[{i}/{len(imagenes)}]", end=" ")
            
            try:
                analisis = self.procesar_imagen_individual(imagen_path)
                resultado = {
                    'imagen': imagen_path,
                    'nombre': Path(imagen_path).name,
                    'analisis': analisis
                }
                resultados.append(resultado)
                
            except Exception as e:
                print(f"❌ Error procesando {Path(imagen_path).name}: {e}")
                resultados.append({
                    'imagen': imagen_path,
                    'nombre': Path(imagen_path).name,
                    'error': str(e)
                })
        
        return resultados
    
    def guardar_resultados(self, resultados, output_file):
        """Guarda los resultados en un archivo JSON"""
        try:
            # Crear directorio de salida si no existe
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados guardados en: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
            return False
    
    def generar_resumen(self, resultados):
        """Genera un resumen de los resultados"""
        if not resultados:
            return
        
        total = len(resultados)
        con_arboles = sum(1 for r in resultados if r.get('analisis', {}).get('hay_arbol', False))
        errores = sum(1 for r in resultados if 'error' in r)
        
        # Calcular estadísticas de riesgo
        riesgos = []
        estados = {}
        
        for resultado in resultados:
            analisis = resultado.get('analisis', {})
            if isinstance(analisis, dict) and analisis.get('hay_arbol', False):
                riesgo = analisis.get('riesgo_nivel', 0)
                if isinstance(riesgo, (int, float)):
                    riesgos.append(riesgo)
                
                estado = analisis.get('estado_general', 'desconocido')
                estados[estado] = estados.get(estado, 0) + 1
        
        print(f"\n📊 RESUMEN DEL ANÁLISIS")
        print(f"{'='*40}")
        print(f"Total imágenes procesadas: {total}")
        print(f"Imágenes con árboles: {con_arboles}")
        print(f"Imágenes sin árboles: {total - con_arboles - errores}")
        print(f"Errores: {errores}")
        
        if riesgos:
            riesgo_promedio = sum(riesgos) / len(riesgos)
            riesgo_maximo = max(riesgos)
            print(f"\nRiesgo promedio: {riesgo_promedio:.1f}/10")
            print(f"Riesgo máximo: {riesgo_maximo}/10")
            
        if estados:
            print(f"\nEstados encontrados:")
            for estado, cantidad in estados.items():
                print(f"  {estado}: {cantidad}")

class AnalizadorAlcorques:
    def __init__(self, api_key):
        """Inicializar el analizador de alcorques con la API key de Gemini"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def analizar_alcorque(self, imagen_path):
        """Analiza una imagen individual de alcorque usando Gemini"""
        try:
            imagen = Image.open(imagen_path)
            
            prompt = """Como experto en gestión de alcorques, analiza esta imagen y evalúa:

1. ¿Se observan levantamientos, grietas o desplazamientos en el pavimento junto al alcorque? ¿Hay raíces superficiales conectando el daño con el árbol?
2. ¿El alcorque está completamente vacío (sin árbol ni tocón)? Confirma asimismo la ausencia de plantones o tutores recientes.
3. ¿Hay basura, escombros u otros objetos no autorizados dentro del alcorque? Describe su naturaleza y volumen aproximado.
4. ¿Se aprecia un tocón en el alcorque? Indica su altura aproximada y posición relativa al borde.
5. ¿La cobertura de malas hierbas supera el 50 % de la superficie o su altura excede los 15 cm? Indica densidad y especies dominantes.
6. ¿Hay presencia de charcos o encharcamientos en el alcorque o acera adyacente? Estima su extensión y si el agua parece estancada o reciente.

Responde SOLO en formato JSON válido con la siguiente estructura:
{
    "levantamientos": "...",
    "alcorque_vacio": true/false,
    "basura": "...",
    "tocon": "...",
    "malas_hierbas": "...",
    "charcos": "...",
    "descripcion": "descripción breve"
}
"""
            
            response = self.model.generate_content([prompt, imagen])
            return response.text
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "descripcion": f"Error procesando imagen de alcorque: {str(e)}"
            })

    def procesar_imagen_individual_alcorque(self, imagen_path):
        """Procesa una imagen individual de alcorque"""
        print(f"🛠️ Analizando alcorque: {Path(imagen_path).name}")
        
        try:
            analisis_texto = self.analizar_alcorque(imagen_path)
            try:
                analisis = json.loads(analisis_texto)
            except json.JSONDecodeError:
                analisis = {
                    "analisis_texto": analisis_texto
                }
            return analisis
        except Exception as e:
            print(f"   ❌ Error analizando alcorque: {e}")
            return {
                "error": str(e)
            }

    def procesar_directorio_alcorque(self, directorio):
        """Procesa todas las imágenes de un directorio de alcorques"""
        extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        imagenes = []
        for ext in extensiones:
            imagenes.extend(glob.glob(os.path.join(directorio, ext)))
            imagenes.extend(glob.glob(os.path.join(directorio, ext.upper())))
        if not imagenes:
            print(f"❌ No se encontraron imágenes de alcorques en {directorio}")
            return []
        print(f"📁 Procesando {len(imagenes)} imágenes de alcorques en el directorio: {directorio}")
        resultados = []
        for i, imagen_path in enumerate(imagenes, 1):
            print(f"\n[{i}/{len(imagenes)}]", end=" ")
            analisis = self.procesar_imagen_individual_alcorque(imagen_path)
            resultados.append({'imagen': imagen_path, 'nombre': Path(imagen_path).name, 'analisis': analisis})
        return resultados

    def guardar_resultados_alcorque(self, resultados, output_file):
        """Guarda los resultados de alcorques en un archivo JSON"""
        try:
            # Crear directorio de salida si no existe
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados de alcorques guardados en: {output_file}")
            return True
        except Exception as e:
            print(f"❌ Error guardando resultados de alcorques: {e}")
            return False

    def generar_resumen_alcorque(self, resultados):
        """Genera resumen de resultados de alcorques"""
        if not resultados:
            return
        total = len(resultados)
        errores = sum(1 for r in resultados if 'error' in r.get('analisis', {}))
        print(f"\n📊 RESUMEN ANÁLISIS ALCORQUES")
        print(f"{'='*40}")
        print(f"Total imágenes procesadas: {total}")
        print(f"Errores: {errores}")


def inferir_tipo_desde_nombre(nombre_archivo):
    """Infiere el tipo de análisis desde el nombre del archivo"""
    nombre_lower = nombre_archivo.lower()
    if "_tree_" in nombre_lower or nombre_lower.endswith("_tree") or "_tree." in nombre_lower:
        return 'arboles'
    elif "_planter_" in nombre_lower or "_alcorque_" in nombre_lower or "_planter." in nombre_lower:
        return 'alcorques'
    else:
        return None

def procesar_directorio_mixto(directorio, analizador_arboles, analizador_alcorques):
    """Procesa un directorio con imágenes mixtas (árboles y alcorques)"""
    extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    imagenes = []
    
    for ext in extensiones:
        imagenes.extend(glob.glob(os.path.join(directorio, ext)))
        imagenes.extend(glob.glob(os.path.join(directorio, ext.upper())))
    
    if not imagenes:
        print(f"❌ No se encontraron imágenes en {directorio}")
        return {'arboles': [], 'alcorques': []}
    
    print(f"📁 Procesando {len(imagenes)} imágenes del directorio: {directorio}")
    
    resultados_arboles = []
    resultados_alcorques = []
    sin_clasificar = []
    
    for i, imagen_path in enumerate(imagenes, 1):
        nombre_archivo = Path(imagen_path).name
        tipo = inferir_tipo_desde_nombre(nombre_archivo)
        
        print(f"\n[{i}/{len(imagenes)}] {nombre_archivo}", end=" ")
        
        if tipo == 'arboles':
            print("(🌳)", end=" ")
            try:
                analisis = analizador_arboles.procesar_imagen_individual(imagen_path)
                resultado = {
                    'imagen': imagen_path,
                    'nombre': nombre_archivo,
                    'analisis': analisis
                }
                resultados_arboles.append(resultado)
            except Exception as e:
                print(f"❌ Error: {e}")
                resultados_arboles.append({
                    'imagen': imagen_path,
                    'nombre': nombre_archivo,
                    'error': str(e)
                })
                
        elif tipo == 'alcorques':
            print("(🛠️)", end=" ")
            try:
                analisis = analizador_alcorques.procesar_imagen_individual_alcorque(imagen_path)
                resultado = {
                    'imagen': imagen_path,
                    'nombre': nombre_archivo,
                    'analisis': analisis
                }
                resultados_alcorques.append(resultado)
            except Exception as e:
                print(f"❌ Error: {e}")
                resultados_alcorques.append({
                    'imagen': imagen_path,
                    'nombre': nombre_archivo,
                    'error': str(e)
                })
        else:
            print("(❓ No clasificado)")
            sin_clasificar.append(nombre_archivo)
    
    if sin_clasificar:
        print(f"\n⚠️  Archivos no clasificados (no contienen 'tree' ni 'planter'): {len(sin_clasificar)}")
        for archivo in sin_clasificar[:5]:  # Mostrar solo los primeros 5
            print(f"   - {archivo}")
        if len(sin_clasificar) > 5:
            print(f"   ... y {len(sin_clasificar) - 5} más")
    
    return {
        'arboles': resultados_arboles,
        'alcorques': resultados_alcorques,
        'sin_clasificar': sin_clasificar
    }

def crear_ruta_output(entrada, tipo=None):
    """Crea la ruta de output en la carpeta resultados/"""
    # Obtener directorio raíz del script
    script_dir = Path(__file__).parent
    resultados_dir = script_dir / "resultados"
    
    # Crear directorio resultados si no existe
    resultados_dir.mkdir(exist_ok=True)
    
    if os.path.isfile(entrada):
        # Para imagen individual: resultados/nombre_imagen_tipo.json
        nombre_base = Path(entrada).stem
        if tipo:
            output_file = resultados_dir / f"{nombre_base}_{tipo}.json"
        else:
            output_file = resultados_dir / f"{nombre_base}.json"
    else:
        # Para directorio: resultados/nombre_directorio_mixto_timestamp.json
        nombre_directorio = Path(entrada).name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if tipo:
            output_file = resultados_dir / f"{nombre_directorio}_{tipo}_{timestamp}.json"
        else:
            output_file = resultados_dir / f"{nombre_directorio}_mixto_{timestamp}.json"
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Analizador de imágenes de árboles y alcorques con IA')
    parser.add_argument('entrada', help='Ruta a imagen individual o directorio con imágenes')
    parser.add_argument('--api-key', required=True, help='API Key de Google Gemini')
    parser.add_argument('--output', '-o', help='Archivo para guardar resultados JSON (por defecto en carpeta resultados/)')
    parser.add_argument('--resumen', action='store_true', help='Mostrar resumen al final')
    parser.add_argument('--tipo', choices=['arboles','alcorques'], help='Forzar tipo de agente (por defecto se infiere del nombre de archivo)')
    
    args = parser.parse_args()
    
    # Verificar entrada
    if not os.path.exists(args.entrada):
        print(f"❌ Error: No se encontró {args.entrada}")
        sys.exit(1)
    
    # Modo directorio o archivo individual
    es_directorio = os.path.isdir(args.entrada)
    
    if es_directorio and not args.tipo:
        # Modo directorio mixto - procesar cada imagen según su nombre
        print("📁 Modo: Directorio mixto (detectando tipo por nombre de archivo)")
        
        # Inicializar ambos analizadores
        try:
            analizador_arboles = AnalizadorArboles(args.api_key)
            analizador_alcorques = AnalizadorAlcorques(args.api_key)
            print("✅ Analizadores inicializados correctamente")
        except Exception as e:
            print(f"❌ Error inicializando analizadores: {e}")
            sys.exit(1)
        
        # Procesar directorio mixto
        resultados_mixtos = procesar_directorio_mixto(args.entrada, analizador_arboles, analizador_alcorques)
        
        # Crear salida si no se especificó
        if not args.output:
            args.output = crear_ruta_output(args.entrada)
            print(f"📁 Resultados se guardarán en: {args.output}")
        
        # Guardar resultados mixtos
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(resultados_mixtos, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados guardados en: {args.output}")
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
        
        # Mostrar resumen si se solicita
        if args.resumen:
            print(f"\n📊 RESUMEN DEL ANÁLISIS MIXTO")
            print(f"{'='*40}")
            print(f"Imágenes de árboles procesadas: {len(resultados_mixtos['arboles'])}")
            print(f"Imágenes de alcorques procesadas: {len(resultados_mixtos['alcorques'])}")
            print(f"Imágenes sin clasificar: {len(resultados_mixtos['sin_clasificar'])}")
            
            # Resumen detallado para árboles si hay
            if resultados_mixtos['arboles']:
                analizador_arboles.generar_resumen(resultados_mixtos['arboles'])
            
            # Resumen detallado para alcorques si hay
            if resultados_mixtos['alcorques']:
                analizador_alcorques.generar_resumen_alcorque(resultados_mixtos['alcorques'])
        
    else:
        # Modo tradicional - un solo tipo
        if os.path.isfile(args.entrada):
            # Derivar tipo para imagen individual si no se especificó
            if not args.tipo:
                args.tipo = inferir_tipo_desde_nombre(Path(args.entrada).name)
                if not args.tipo:
                    print("⚠️  No pude derivar el tipo de agente del nombre de la imagen.")
                    print("    Usa --tipo arboles o --tipo alcorques para especificarlo.")
                    sys.exit(1)
                print(f"🔍 Tipo detectado: {args.tipo}")
        
        # Crear ruta de output si no se especificó
        if not args.output:
            args.output = crear_ruta_output(args.entrada, args.tipo)
            print(f"📁 Resultados se guardarán en: {args.output}")

        # Inicializar analizador específico
        try:
            if args.tipo == 'alcorques':
                analizador = AnalizadorAlcorques(args.api_key)
            else:
                analizador = AnalizadorArboles(args.api_key)
            print("✅ Analizador inicializado correctamente")
        except Exception as e:
            print(f"❌ Error inicializando analizador: {e}")
            sys.exit(1)
        
        # Procesar según tipo
        if os.path.isfile(args.entrada):
            # Procesar imagen individual
            print(f"\n📸 Modo: Imagen individual")
            if args.tipo == 'alcorques':
                analisis = analizador.procesar_imagen_individual_alcorque(args.entrada)
            else:
                analisis = analizador.procesar_imagen_individual(args.entrada)
            
            resultados = [{
                'imagen': args.entrada,
                'nombre': Path(args.entrada).name,
                'analisis': analisis
            }]
            
        elif os.path.isdir(args.entrada):
            # Procesar directorio de un solo tipo
            print(f"\n📁 Modo: Directorio de {args.tipo}")
            if args.tipo == 'alcorques':
                resultados = analizador.procesar_directorio_alcorque(args.entrada)
            else:
                resultados = analizador.procesar_directorio(args.entrada)
        else:
            print(f"❌ Error: {args.entrada} no es un archivo ni directorio válido")
            sys.exit(1)
        
        # Guardar resultados
        if args.tipo == 'alcorques':
            analizador.guardar_resultados_alcorque(resultados, args.output)
        else:
            analizador.guardar_resultados(resultados, args.output)
        
        # Mostrar resumen si se solicita
        if args.resumen:
            if args.tipo == 'alcorques':
                analizador.generar_resumen_alcorque(resultados)
            else:
                analizador.generar_resumen(resultados)
    
    print(f"\n🎯 Análisis completado")

if __name__ == "__main__":
    main()