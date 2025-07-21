#!/usr/bin/env python3
import google.generativeai as genai
from PIL import Image
import json
import os
import sys
import argparse
import glob
from pathlib import Path

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


def main():
    parser = argparse.ArgumentParser(description='Analizador de imágenes de árboles y alcorques con IA')
    parser.add_argument('entrada', help='Ruta a imagen individual o directorio con imágenes')
    parser.add_argument('--api-key', required=True, help='API Key de Google Gemini')
    parser.add_argument('--output', '-o', help='Archivo para guardar resultados JSON')
    parser.add_argument('--resumen', action='store_true', help='Mostrar resumen al final')
    parser.add_argument('--tipo', choices=['arboles','alcorques'], default='arboles', help='Tipo de agente: arboles o alcorques')
    
    args = parser.parse_args()
    
    # Verificar entrada
    if not os.path.exists(args.entrada):
        print(f"❌ Error: No se encontró {args.entrada}")
        sys.exit(1)
    
    # Inicializar analizador
    try:
        if args.tipo == 'alcorques':
            analizador = AnalizadorAlcorques(args.api_key)
        else:
            analizador = AnalizadorArboles(args.api_key)
        print("✅ Analizador inicializado correctamente")
    except Exception as e:
        print(f"❌ Error inicializando analizador: {e}")
        sys.exit(1)
    
    # Determinar si es archivo o directorio
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
        # Procesar directorio
        print(f"\n📁 Modo: Directorio completo")
        if args.tipo == 'alcorques':
            resultados = analizador.procesar_directorio_alcorque(args.entrada)
        else:
            resultados = analizador.procesar_directorio(args.entrada)
    else:
        print(f"❌ Error: {args.entrada} no es un archivo ni directorio válido")
        sys.exit(1)
    
    # Guardar resultados si se especifica
    if args.output:
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