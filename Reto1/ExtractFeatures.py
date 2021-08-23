# -*- coding: utf-8 -*-
"""clustering_vgg16_2.ipynb
"""

"""## Cargamos las librerias"""

# Cargamos las librerias
from PIL import Image
import PIL.Image

import numpy as np
import matplotlib.pyplot as plt
import cv2

from scipy import ndimage
from skimage.measure import regionprops
import re

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

# Cargamos el modelo con los pesos del imagenet
model = VGG16(weights='imagenet', include_top=False)

#-----------------------------------------------
def extract_features_vgg16(archivo_scr):
    # Cargamos la imagen
    image = PIL.Image.open(archivo_scr).convert("RGB")
    image = np.array(image)


    """## Registramos (recortamos/orientamos) el documento"""
    # convert the warped image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:,:,0] # cuadro chico
    #gray = cv2.equalizeHist(gray) # segunda vuelta
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

    eje_x = np.where(profile_h > 200000)[0]
    eje_y = np.where(profile_v > 200000)[0]

    if eje_x.size == 0:
        coor_x0 = 0
        coor_xf = len(profile_h)
    else:
        coor_x0 = eje_x[0]
        coor_xf = eje_x[len(eje_x)-1]

    if eje_y.size == 0:
        coor_y0 = 0
        coor_yf = len(profile_v)
    else:
        coor_y0 = eje_y[0]
        coor_yf = eje_y[len(eje_y)-1]

    image = image[coor_y0:coor_yf,coor_x0:coor_xf,:]


    """## Detectamos la orientacion del documento"""
    """## Limpieza de la Imagen"""
    # convert the warped image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # sharpen image
    sharpen = cv2.GaussianBlur(gray, (0,0), 3)
    #sharpen = cv2.medianBlur(gray,5)
    sharpen = cv2.addWeighted(gray, 1.5, sharpen, -0.5, 0)

    # apply adaptive threshold to get black and white effect
    thresh = cv2.adaptiveThreshold(sharpen, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 71, 21) # buena 2

    """## Analysis of connected components"""

    label_im, nb_labels = ndimage.label(thresh)
    nb_labels # how many regions?

    sizes = ndimage.sum(thresh, label_im, range(nb_labels + 1))

    mask_size = sizes < 2000

    remove_pixel = mask_size[label_im]
    remove_pixel.shape
    label_im[remove_pixel] = 0

    """element = np.ones((5,5))
    dilate = cv2.dilate(np.array(label_im>0,dtype=np.uint8),element)
    dilate = cv2.erode(dilate,element)

    # Visualizamos la imagen
    plt.figure(figsize=(15,10))
    plt.imshow(dilate)
    plt.show()

    ## Calculamos los bounding box
    """

    #label_im, nb_labels = ndimage.label(dilate)

    # Calculamos las Measure region properties de cada region
    props = regionprops(label_im)

    """
    image2 = image.copy()
    label = np.zeros(image.shape[:2], dtype=np.uint8)

    # Recorremos cada region segmentada en la imagen
    cont = 1
    for prop in props:
        # Boundig Box
        minr, minc, maxr, maxc = prop['bbox']

        # region de interes
        roi = label[minr:maxr,minc:maxc]

        # asignamos label
        if roi.sum() == 0:
            label[minr:maxr,minc:maxc] = cont
        else:
            uni = np.unique(roi)
            # si comparte bounding box con otra region le asignamos su etiqueta
            if uni[0] != 0:
                label[minr:maxr,minc:maxc] = uni[0]
            else:
                label[minr:maxr,minc:maxc] = uni[1]

        cont += 1
        #cv2.rectangle(image2, (minc0, np.min([minr0, minr]) ), (maxc, np.max([maxr0,maxr]) ), (0, 255, 0), 2)

    plt.figure(figsize=(15,10))
    plt.imshow(label)
    plt.show()
    """

    #label_im2, nb_labels2 = ndimage.label(label>0)
    #props = regionprops(label_im2)

    """## Calculamos las caracterÃ­sticas con un VGG16"""

    data = []
    # Recorremos cada region segmentada en la imagen
    for prop in props:
        # Boundig Box
        minr, minc, maxr, maxc = prop['bbox']
        if prop['bbox_area'] < 100:
            continue

        a = 0 if minr-5<0 else minr-5
        b = maxr+5
        c = 0 if minc-5<0 else minc-5
        d = maxc+5

        # region de interes
        roi = image[a:b,c:d,:]

        #roi = cv2.resize(roi, (93,93), interpolation = cv2.INTER_AREA)
        roi = cv2.resize(roi, (61,61), interpolation = cv2.INTER_AREA)

        # Preprocesamos la imagen
        x = np.expand_dims(roi, axis=0)
        x = preprocess_input(x)

        # Calculamos las features
        features = model.predict(x)

        # Guardamos en modo one-hot las caracteristicas
        data.append([archivo_scr]+[a,b,c,d]+features.flatten().tolist())
        
    return(data)

#-----------------------------------------------

if __name__ == "__main__":
    """## Analizamos las caracteristicas"""

    archivo_scr = "../../Datos - Hackathon JusticIA/Expedientes/AGN_DFS_C.18_2_de_2_Cabrera_Arenas_Emma_pg-279.jpg"
    data = extract_features_vgg16(archivo_scr)
