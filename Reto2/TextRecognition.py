# -*- coding: utf-8 -*-
"""
# Hackaton RIIAA 2021 Reto 2
# Segmentacion y Reconocimiento de Texto en un Documento
# Equipo: Pista latente ML
# Integrantes: 
#       - Andrea Berenice Ek Hobak
#       - Gabriela Marali Mundo Cortes
#       - Mario Xavier Canche Uc
#       - Myrna Citlali Castillo Silva
#       - Ramon Sidonio Aparicio Garcia

#---------- Requerimientos:
# Instalamos Tesseract
!sudo apt install tesseract-ocr

# Instalamos tesseract para python
!pip install pytesseract

# Instalamos pylsd, it  is the python bindings for LSD - Line Segment Detector
!pip install ocrd-fork-pylsd

#---------- Descargar el mejor modelo para tesseract
# Commented out IPython magic to ensure Python compatibility.
# Descargar el mejor modelo del github de tesseract
# https://github.com/tesseract-ocr/tessdata_best

# Eliminamos el modelo default
# %rm /usr/share/tesseract-ocr/4.00/tessdata/eng.traineddata

# Copiamos el modelo descargado
# %cp eng.traineddata /usr/share/tesseract-ocr/4.00/tessdata/

"""

# Cargamos las librerias
import PIL.Image

import numpy as np
import matplotlib.pyplot as plt
import cv2

import pytesseract

from scipy import ndimage
import re

#-----------------------------------------------------------------------
def get_text(archivo_scr, view=False):
    """ Funcion para reconocer texto en una imagen """

    # Cargamos la imagen
    image = PIL.Image.open(archivo_scr).convert("RGB")
    image = np.array(image)

    #-----------------
    # convert the warped image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.medianBlur(gray, 5)
    # sharpen image
    sharpen = cv2.GaussianBlur(gray, (7,7), 0)
    
    #Get the best value for T with otsu thresholding
    (T, threshInv) = cv2.threshold(sharpen, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    threshInv = cv2.erode(threshInv, None, iterations=2)
    threshInv = cv2.dilate(threshInv, None, iterations=2)
    
    # Detectamos los perfiles del documento segmentado
    profile_h = threshInv.sum(0)
    profile_v = threshInv.sum(1)
    
    # Umbralizamos de acuerdo a la densidad de pixeles del documento
    eje_x = np.where(profile_h > 200000)[0]
    eje_y = np.where(profile_v > 200000)[0]
    
    # Encontramos los vertices del documento
    coor_x0 = eje_x[0]
    coor_xf = eje_x[len(eje_x)-1]
    coor_y0 = eje_y[0]
    coor_yf = eje_y[len(eje_y)-1]
    
    # Recortamos la imagen principal
    image = image[coor_y0:coor_yf,coor_x0:coor_xf,:]

    #-----------------
    # Detectamos la orientacion con tesseract
    if image.shape[1]<4000: 
        osd = pytesseract.image_to_osd(image)
        angle = 360-int(re.search('(?<=Rotate: )\d+', osd).group(0))
    else:
        angle = 0
        
    # Rotamos la imagen en caso de ser necesario
    if angle!=0 and angle!=360:
        rotated = ndimage.rotate(image, float(angle))

        del image, osd, angle
        image = rotated
    
    #-----------------
    ## Iniciamos la limpieza de la imagen
    # convert the warped image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # sharpen image
    sharpen = cv2.GaussianBlur(gray, (0,0), 3)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

    # apply adaptive threshold to get black and white effect
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, 21)

    thresh = cv2.medianBlur(thresh,3)

    del sharpen, gray

    ## Analysis of connected components
    # Etiquetamos cada region
    label_im, nb_labels = ndimage.label(thresh)
    # Calculamos el area de cada region
    sizes = ndimage.sum(thresh, label_im, range(nb_labels + 1))
    
    # Eliminamos las regiones con area menor al umbral (chicas)
    mask_size = sizes < 2000
    remove_pixel = mask_size[label_im]
    remove_pixel.shape
    label_im[remove_pixel] = 0
    
    del remove_pixel, mask_size, sizes, nb_labels, thresh

    # Usamos Tesseract para reconocer el texto
    output = pytesseract.image_to_string(label_im>0, lang="eng")
    
    if view:
        # Visualizamos la segmentacion del texto
        plt.figure(figsize=(15,10))
        plt.imshow(label_im>0, cmap="gray")
        plt.show()
        
        print(output)

    return(output)

#-----------------------------------------------------------------------

if __name__ == "__main__":

    # Ruta de la imagen
    #archivo_scr = "../pruebas/images_texto/DFS_Exp._009-011-014,_L-1-5-.JPG.png"
    #archivo_scr = "../pruebas/images_texto/DFS,_Exp._012-011-025,_L-1-80-06-03_a_85-12-01-7-.JPG.png"
    archivo_scr = "../pruebas/images_texto/DFS_Exp._009-011-014,_L-1-79-07-25_a_83-10-16-259-.JPG.png"
    #archivo_scr = "../pruebas/images_texto/DFS_012-028-002,_L-2-79-10-11_a_80-01-25-317-.JPG.png"

    output1 = get_text(archivo_scr)
    print(output1)
