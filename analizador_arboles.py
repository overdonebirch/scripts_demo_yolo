#!/usr/bin/env python3
import google.generativeai as genai
from PIL import Image
import json
import csv
import os
import sys
import argparse
import glob
from pathlib import Path
from datetime import datetime

# Fix para codificación en Windows - DEBE IR ANTES DE CUALQUIER PRINT CON EMOJIS
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

def leer_prompt_desde_archivo(nombre_archivo):
    """Lee el contenido de un archivo de prompt y lo devuelve como string"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        ruta_prompt = os.path.join(script_dir, nombre_archivo)
        
        with open(ruta_prompt, 'r', encoding='utf-8') as archivo:
            return archivo.read().strip()
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo de prompt '{nombre_archivo}'")
        print(f"   Asegúrate de que el archivo existe en el mismo directorio que el script.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error al leer el archivo de prompt '{nombre_archivo}': {e}")
        sys.exit(1)

def convertir_json_a_csv(resultados, output_file, tipo_analisis):
    """Convierte los resultados JSON a formato CSV"""
    try:
        csv_path = output_file.replace('.json', '.csv')
        
        # Campos CSV según la estructura definida
        campos_csv = [
            'id_imagen', 'nombre_archivo', 'ruta_imagen', 'tipo_analisis', 
            'descripcion_incidencia', 'requiere_intervencion', 'confianza_modelo', 
            'estado_general', 'error'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=campos_csv)
            
            # Escribir cabecera
            writer.writeheader()
            
            # Escribir datos
            for resultado in resultados:
                analisis = resultado.get('analisis', {})
                
                # Si hay error, usar datos básicos
                if 'error' in resultado:
                    row = {
                        'id_imagen': resultado['nombre'][:7] if len(resultado['nombre']) >= 7 else resultado['nombre'],
                        'nombre_archivo': resultado['nombre'],
                        'ruta_imagen': resultado['imagen'],
                        'tipo_analisis': tipo_analisis,
                        'descripcion_incidencia': 'Error de procesamiento',
                        'requiere_intervencion': False,
                        'confianza_modelo': '0',
                        'estado_general': 'error',
                        'error': resultado['error']
                    }
                else:
                    # Extraer datos del análisis JSON
                    row = {
                        'id_imagen': resultado['nombre'][:7] if len(resultado['nombre']) >= 7 else resultado['nombre'],
                        'nombre_archivo': resultado['nombre'],
                        'ruta_imagen': resultado['imagen'],
                        'tipo_analisis': tipo_analisis,
                        'descripcion_incidencia': analisis.get('descripcion', analisis.get('descripcion_incidencia', 'sin descripción')),
                        'requiere_intervencion': analisis.get('requiere_intervencion', False),
                        'confianza_modelo': str(analisis.get('confianza_modelo', analisis.get('riesgo_nivel', '50'))),
                        'estado_general': analisis.get('estado_general', 'indeterminado'),
                        'error': analisis.get('error', '')
                    }
                
                writer.writerow(row)
        
        print(f"📊 Archivo CSV generado: {csv_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error generando CSV: {e}")
        return False

class AnalizadorArboles:
    def __init__(self, api_key):
        """Inicializar el analizador con la API key de Gemini"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def analizar_arbol(self, imagen_path):
        """Analiza una imagen individual de árbol usando Gemini"""
        try:
            imagen = Image.open(imagen_path)
            
            # Leer el prompt desde el archivo externo
            prompt = leer_prompt_desde_archivo('prompt_arbol.txt')
            
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
                # Limpiar markdown si está presente
                texto_limpio = analisis_texto.strip()
                if texto_limpio.startswith('```json'):
                    texto_limpio = texto_limpio[7:]  # Remover ```json
                if texto_limpio.endswith('```'):
                    texto_limpio = texto_limpio[:-3]  # Remover ```
                texto_limpio = texto_limpio.strip()
                
                analisis = json.loads(texto_limpio)
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
        
        # Eliminar duplicados (importante en Windows donde los nombres no son case-sensitive)
        imagenes = list(set(imagenes))
        
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
        """Guarda los resultados en formato JSON y CSV"""
        try:
            # Crear directorio de salida si no existe
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados JSON guardados en: {output_file}")
            
            # Generar CSV
            convertir_json_a_csv(resultados, output_file, 'arbol')
            
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
            
            # Leer el prompt desde el archivo externo
            prompt = leer_prompt_desde_archivo('prompt_alcorque.txt')
            
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
                # Limpiar markdown si está presente
                texto_limpio = analisis_texto.strip()
                if texto_limpio.startswith('```json'):
                    texto_limpio = texto_limpio[7:]  # Remover ```json
                if texto_limpio.endswith('```'):
                    texto_limpio = texto_limpio[:-3]  # Remover ```
                texto_limpio = texto_limpio.strip()
                
                analisis = json.loads(texto_limpio)
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
        
        # Eliminar duplicados (importante en Windows donde los nombres no son case-sensitive)
        imagenes = list(set(imagenes))
        
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
        """Guarda los resultados de alcorques en formato JSON y CSV"""
        try:
            # Crear directorio de salida si no existe
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados JSON de alcorques guardados en: {output_file}")
            
            # Generar CSV
            convertir_json_a_csv(resultados, output_file, 'alcorque')
            
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


class AnalizadorLimpieza:
    def __init__(self, api_key):
        """Inicializar el analizador de limpieza con la API key de Gemini"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
    def analizar_limpieza(self, imagen_path):
        """Analiza una imagen específicamente para problemas de limpieza"""
        try:
            imagen = Image.open(imagen_path)
            
            # Leer el prompt desde el archivo externo
            prompt = leer_prompt_desde_archivo('prompt_limpieza.txt')
            
            response = self.model.generate_content([prompt, imagen])
            return response.text
            
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "estado_general": "error",
                "requiere_intervencion": False,
                "prioridad": "baja",
                "descripcion": f"Error procesando imagen de limpieza: {str(e)}"
            })

    def procesar_imagen_individual_limpieza(self, imagen_path):
        """Procesa una imagen individual para análisis de limpieza"""
        print(f"🧹 Analizando limpieza: {Path(imagen_path).name}")
        
        try:
            analisis_texto = self.analizar_limpieza(imagen_path)
            try:
                # Limpiar markdown si está presente
                texto_limpio = analisis_texto.strip()
                if texto_limpio.startswith('```json'):
                    texto_limpio = texto_limpio[7:]  # Remover ```json
                if texto_limpio.endswith('```'):
                    texto_limpio = texto_limpio[:-3]  # Remover ```
                texto_limpio = texto_limpio.strip()
                
                analisis = json.loads(texto_limpio)
            except json.JSONDecodeError:
                analisis = {
                    "estado_general": "indeterminado",
                    "requiere_intervencion": False,
                    "prioridad": "baja",
                    "analisis_texto": analisis_texto
                }
            
            # Mostrar resumen en consola
            estado = analisis.get('estado_general', 'indeterminado')
            requiere = analisis.get('requiere_intervencion', False)
            prioridad = analisis.get('prioridad', 'baja')
            descripcion = analisis.get('descripcion', '')
            
            print(f"   🧹 Estado limpieza: {estado}")
            print(f"   🚨 Requiere intervención: {'Sí' if requiere else 'No'}")
            print(f"   📊 Prioridad: {prioridad}")
            if descripcion:
                print(f"   📝 {descripcion}")
            
            # Mostrar problemas específicos detectados
            problemas = []
            if analisis.get('basura_alcorque', 'no detectada') != 'no detectada':
                problemas.append(f"Basura en alcorque: {analisis['basura_alcorque']}")
            if analisis.get('residuos_ramas', 'no detectados') != 'no detectados':
                problemas.append(f"Residuos en ramas: {analisis['residuos_ramas']}")
            if analisis.get('papeleras_desbordadas', 'no visible') not in ['no visible', 'no']:
                problemas.append(f"Papeleras: {analisis['papeleras_desbordadas']}")
            if analisis.get('acumulacion_acera', 'no detectada') != 'no detectada':
                problemas.append(f"Acera: {analisis['acumulacion_acera']}")
            if analisis.get('excrementos', 'no detectados') != 'no detectados':
                problemas.append(f"Excrementos: {analisis['excrementos']}")
            
            if problemas:
                print(f"   🗑️ Problemas detectados:")
                for problema in problemas[:3]:  # Mostrar máximo 3 para no saturar
                    print(f"      - {problema}")
                if len(problemas) > 3:
                    print(f"      ... y {len(problemas) - 3} más")
            
            return analisis
            
        except Exception as e:
            print(f"   ❌ Error analizando limpieza: {e}")
            return {
                "error": str(e),
                "estado_general": "error",
                "requiere_intervencion": False,
                "prioridad": "baja"
            }

    def procesar_directorio_limpieza(self, directorio):
        """Procesa todas las imágenes de un directorio para análisis de limpieza"""
        extensiones = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        imagenes = []
        for ext in extensiones:
            imagenes.extend(glob.glob(os.path.join(directorio, ext)))
            imagenes.extend(glob.glob(os.path.join(directorio, ext.upper())))
        
        if not imagenes:
            print(f"❌ No se encontraron imágenes de limpieza en {directorio}")
            return []
        
        print(f"📁 Procesando {len(imagenes)} imágenes para análisis de limpieza en: {directorio}")
        
        resultados = []
        for i, imagen_path in enumerate(imagenes, 1):
            print(f"\n[{i}/{len(imagenes)}]", end=" ")
            analisis = self.procesar_imagen_individual_limpieza(imagen_path)
            resultados.append({
                'imagen': imagen_path, 
                'nombre': Path(imagen_path).name, 
                'analisis': analisis
            })
        
        return resultados

    def guardar_resultados_limpieza(self, resultados, output_file):
        """Guarda los resultados de limpieza en formato JSON y CSV"""
        try:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Guardar JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(resultados, f, indent=2, ensure_ascii=False)
            print(f"💾 Resultados JSON de limpieza guardados en: {output_file}")
            
            # Generar CSV
            convertir_json_a_csv(resultados, output_file, 'limpieza')
            
            return True
        except Exception as e:
            print(f"❌ Error guardando resultados de limpieza: {e}")
            return False

    def generar_resumen_limpieza(self, resultados):
        """Genera resumen de resultados de limpieza"""
        if not resultados:
            return
        
        total = len(resultados)
        errores = sum(1 for r in resultados if 'error' in r.get('analisis', {}))
        requieren_intervencion = sum(1 for r in resultados 
                                   if r.get('analisis', {}).get('requiere_intervencion', False))
        
        # Contar por estado general
        estados = {}
        prioridades = {}
        problemas_frecuentes = {
            'basura_alcorque': 0,
            'residuos_ramas': 0,
            'papeleras_desbordadas': 0,
            'acumulacion_acera': 0,
            'excrementos': 0
        }
        
        for resultado in resultados:
            analisis = resultado.get('analisis', {})
            if isinstance(analisis, dict) and 'error' not in analisis:
                estado = analisis.get('estado_general', 'desconocido')
                estados[estado] = estados.get(estado, 0) + 1
                
                prioridad = analisis.get('prioridad', 'baja')
                prioridades[prioridad] = prioridades.get(prioridad, 0) + 1
                
                # Contar problemas específicos
                for problema in problemas_frecuentes.keys():
                    valor = analisis.get(problema, '')
                    if valor and valor not in ['no detectada', 'no detectados', 'no visible', 'no']:
                        problemas_frecuentes[problema] += 1
        
        print(f"\n📊 RESUMEN ANÁLISIS DE LIMPIEZA")
        print(f"{'='*40}")
        print(f"Total imágenes procesadas: {total}")
        print(f"Imágenes que requieren intervención: {requieren_intervencion}")
        print(f"Imágenes limpias: {total - requieren_intervencion - errores}")
        print(f"Errores: {errores}")
        
        if estados:
            print(f"\nEstados de limpieza detectados:")
            for estado, cantidad in sorted(estados.items()):
                print(f"  {estado}: {cantidad}")
        
        if prioridades:
            print(f"\nPrioridades asignadas:")
            for prioridad, cantidad in sorted(prioridades.items()):
                print(f"  {prioridad}: {cantidad}")
        
        if any(count > 0 for count in problemas_frecuentes.values()):
            print(f"\nProblemas más frecuentes:")
            problemas_ordenados = sorted(problemas_frecuentes.items(), 
                                       key=lambda x: x[1], reverse=True)
            for problema, cantidad in problemas_ordenados:
                if cantidad > 0:
                    nombre_problema = problema.replace('_', ' ').title()
                    print(f"  {nombre_problema}: {cantidad} casos")


def crear_ruta_output(entrada, tipo):
    """Crea la ruta de output en la carpeta resultados/"""
    # Obtener directorio raíz del script
    script_dir = Path(__file__).parent
    resultados_dir = script_dir / "resultados"
    
    # Crear directorio resultados si no existe
    resultados_dir.mkdir(exist_ok=True)
    
    if os.path.isfile(entrada):
        # Para imagen individual: resultados/nombre_imagen_tipo.json
        nombre_base = Path(entrada).stem
        output_file = resultados_dir / f"{nombre_base}_{tipo}.json"
    else:
        # Para directorio: resultados/nombre_directorio_tipo_timestamp.json
        nombre_directorio = Path(entrada).name
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = resultados_dir / f"{nombre_directorio}_{tipo}_{timestamp}.json"
    
    return str(output_file)


def main():
    parser = argparse.ArgumentParser(description='Analizador de imágenes de árboles, alcorques y limpieza con IA')
    parser.add_argument('entrada', help='Ruta a imagen individual o directorio con imágenes')
    parser.add_argument('--api-key', required=True, help='API Key de Google Gemini')
    parser.add_argument('--output', '-o', help='Archivo para guardar resultados JSON (por defecto en carpeta resultados/)')
    parser.add_argument('--resumen', action='store_true', help='Mostrar resumen al final')
    parser.add_argument('--tipo', choices=['arboles','alcorques','limpieza'], required=True, help='Tipo de análisis a realizar')
    
    args = parser.parse_args()
    
    # Verificar entrada
    if not os.path.exists(args.entrada):
        print(f"❌ Error: No se encontró {args.entrada}")
        sys.exit(1)
    
    # Crear ruta de output si no se especificó
    if not args.output:
        args.output = crear_ruta_output(args.entrada, args.tipo)
        print(f"📁 Resultados se guardarán en: {args.output}")

    # Inicializar analizador específico
    try:
        if args.tipo == 'alcorques':
            analizador = AnalizadorAlcorques(args.api_key)
        elif args.tipo == 'limpieza':
            analizador = AnalizadorLimpieza(args.api_key)
        else:
            analizador = AnalizadorArboles(args.api_key)
        print("✅ Analizador inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando analizador: {e}")
        sys.exit(1)
    
    # Procesar según tipo
    if os.path.isfile(args.entrada):
        # Procesar imagen individual
        print(f"\n📸 Modo: Imagen individual - {args.tipo}")
        if args.tipo == 'alcorques':
            analisis = analizador.procesar_imagen_individual_alcorque(args.entrada)
        elif args.tipo == 'limpieza':
            analisis = analizador.procesar_imagen_individual_limpieza(args.entrada)
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
        elif args.tipo == 'limpieza':
            resultados = analizador.procesar_directorio_limpieza(args.entrada)
        else:
            resultados = analizador.procesar_directorio(args.entrada)
    else:
        print(f"❌ Error: {args.entrada} no es un archivo ni directorio válido")
        sys.exit(1)
    
    # Guardar resultados (JSON + CSV)
    if args.tipo == 'alcorques':
        analizador.guardar_resultados_alcorque(resultados, args.output)
    elif args.tipo == 'limpieza':
        analizador.guardar_resultados_limpieza(resultados, args.output)
    else:
        analizador.guardar_resultados(resultados, args.output)
    
    # Mostrar resumen si se solicita
    if args.resumen:
        if args.tipo == 'alcorques':
            analizador.generar_resumen_alcorque(resultados)
        elif args.tipo == 'limpieza':
            analizador.generar_resumen_limpieza(resultados)
        else:
            analizador.generar_resumen(resultados)
    
    print(f"\n🎯 Análisis completado")
    print(f"📄 Se generaron archivos JSON y CSV")

if __name__ == "__main__":
    main()