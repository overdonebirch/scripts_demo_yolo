## 1 Dividir Imagen en caras : 

python .\scripts_yolo\convert_images.py -i .\imagenes\c_fernando_el_santo.jpg

## 2 Analizar Cada Cara : 

python .\scripts_yolo\analyze_faces.py -f .\nombre_carpeta_output_con_caras\ -m .\nombre_modelo.pt


## 3 Insertar puntos en la imagen original :

python .scripts_yolo\bbox_to_360.py -i .\imagenes\c_fernando_el_santo.jpg -d .nombre_carpeta_output_con_caras\detections.json --radius 30 -o nombre_resultado.jpg

## 4 Extraer crops : 

python .\scripts\extract_crops.py -f .\nombre_carpeta_output_con_caras\ -d .\nombre_carpeta_output_con_caras\detections.json -o crops/

## 5 Analizar Crops : 

python analizador_arboles.py crops/ --api-key TU_API_KEY --resumen

(Los resultados se guardan en la carpeta resultados en la raiz)
