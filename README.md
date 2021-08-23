# Hackathon RIIAA 2021 "JusticIA para los desaparecidos"
# Reto 1 y Reto 2

# Nombre del equipo
Pista-Latente-ML

# Integrantes

* Andrea Berenice Ek Hobak
* Gabriela Marali Mundo Cortes
* Mario Xavier Canche Uc
* Myrna Citlali Castillo Silva
* Ramón Sidonio Aparicio García 

# Descripión

En este repositorio se encontrara el código fuente de los algoritmos desarrollados para el reto 1 y 2, durante el Hackathon RIIAA 2021 "JusticIA para los desaparecidos".

# Pipeline
## Reto 1
1. Lectura de la imagen.
2. Detección y registro de la ficha (eliminacion de ruido, umbralizacion,limpieza morfologica,deteccion de perfiles, recorte de la ficha a color).
3. Eliminación de ruido de la imagen a color.
4. Umbralización adaptativa.
5. Análisis de componentes conectados.
6. Eliminación de regiones pequeñas.
7. Extracción de características con una red neuronal VGG16.
8. Eliminación de columnas constantes de cero.
9. Proyección PCA.
10. Clusterización con K-Means.
11. Identificación de clusters de firmas, sellos, líneas, texto.
12. Identificar rostros en las imágenes.

## Reto 2
### Segmentación de Texto
1. Lectura de la imagen.
2. Detección y registro de la ficha (eliminacion de ruido, umbralizacion,limpieza morfologica,deteccion de perfiles, recorte de la ficha a color).
3. Eliminación de ruido de la imagen a color.
4. Umbralización adaptativa.
5. Análisis de componentes conectados.
6. Eliminación de regiones pequeñas.
7. Detección de texto con Tesseract.
8. Guardar el texto detectado en un csv de salida.
9. Procesar la siguiente imagen y agregar resultados al csv.
### Procesar texto
10. Separar el texto en palabras.
11. Etiquetar palabras según corpus de vocabulario espanol.
12. Las palabras que no son etiquetadas dentro del vocabulario, como sustantivos, compararlas mediante expresiones regulares con las posibles entidades (enjuiciados, servidores publicos, lugares, organizaciones) proporcionadas por los organizadores, además
13. Una vez etiquetadas todas las palabras, utilizar estructuras gramaticales para encontrar y determinar si son expresiones de nombres completos, lugares u organizaciones.

# Instrucciones de ejecución:
* Para generar el entregable del reto 1, ejecutar primero el notebook Reto1/Genera_Entregable_Reto1.ipynb y luego Reto1/Genera_Entregable_Reto1_FaceRecog.ipynb. La salida lo encontraran en la carpeta Reto1/output.
* Para generar el entregable del reto 2, ejecutar el notebook Reto2/Genera_Entregable_Reto2A.ipynb y luego el Reto2/Notebook_ExtraccionDeEntidades.ipynb.
* Para ver paso a paso la clusterización del reto 1, ejecutar el notebook Reto1/StepByStep_Reto1.ipynb.
* Para ver paso a paso el algoritmo de extracción de texto del reto 2, ejecutar el notebook Reto2/StepByStep_Reto2A.ipynb.

