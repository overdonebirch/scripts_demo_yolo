## 1 Dividir Imagen en caras : 

python .\scripts_yolo\convert_images.py -i .\imagenes\c_fernando_el_santo.jpg

## 2 Analizar Cada Cara : 

python .\scripts_yolo\analyze_faces.py -f .\nombre_carpeta_output_con_caras\ -m .\nombre_modelo.pt


## 3 Insertar puntos en la imagen original :

python .scripts_yolo\bbox_to_360.py -i .\imagenes\c_fernando_el_santo.jpg -d .nombre_carpeta_output_con_caras\detections.json --radius 30 -o nombre_resultado.jpg

## 4 Extraer imagenes de arboles completos : 

python .\scripts\extract_full_trees.py -e imagen_360_original.jpg -d .\nombre_carpeta_output_con_caras\detections.json -o nombre_carpeta_output

## 5 Analizar imagenes : 

python analizador_arboles.py carpeta_output/ --api-key TU_API_KEY --resumen

(Los resultados se guardan en la carpeta resultados en la raiz)

## Ejecutar Pipeline : 

python pipeline.py -i .\imagenes\c_fernando_el_santo.jpg -m .\nombre_modelo.pt -r .\imagenes_resultados