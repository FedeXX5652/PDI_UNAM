import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure

# Función para eliminar las zonas con información nula (cero)
def eliminar_zonas_nulas(imagen):
    #Obteniendo el numero de columnas en la imagen
    height, width = imagen.shape
    # buscando la primera columna donde la información sea nula para realizar el corte vertical.
    top_left = -1
    top_right = -1
    # Se determinan los limites de la zona nula en la primera columna.
    top_left_is_set = False
    for i in range(width):
        if not top_left_is_set and imagen[0][i] == 0:
            top_left = i
            top_left_is_set = True
        if imagen[0][i] == 0:
            top_right = i
    # Se determinan los limites de la zona nula en la última columna.
    bottom_left = - 1
    bottom_right = - 1
    bottom_left_is_set = False
    for i in range(width):
        if not bottom_left_is_set and imagen[height - 1][i] == 0:
            bottom_left = i
            bottom_left_is_set = True
        if imagen[height - 1][i] == 0:
            bottom_right = i
    # Se realiza el corte vertical según si la zona nula está orientada a la izquierda o a la derecha
    if bottom_left == 0 or top_left == 0:
        return imagen[0:height, max(bottom_right, top_right):width]
    return imagen[0:height, 0:min(bottom_left, top_left)]

# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
def colorQuant(Z, K, criteria):

   ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
   
   # Now convert back into uint8, and make original image
   center = np.uint8(center)
   res = center[label.flatten()]
   res2 = res.reshape((Z.shape))
   return res2

# Cargar las imágenes satelitales en escala de grises
zona_A = cv2.imread('zonaA_b5.tif', cv2.IMREAD_GRAYSCALE)
zona_B = cv2.imread('zonaB_b5.tif', cv2.IMREAD_GRAYSCALE)
# Verificar si las imágenes se cargaron correctamente
if zona_A is None or zona_B is None:
    raise Exception("No se pudieron cargar las imágenes satelitales")

# Imagen de la zona A sin nulidad
zona_A_sin_nulidad = eliminar_zonas_nulas(zona_A)
# Imagen de la zona b sin nulidad
zona_B_sin_nulidad = eliminar_zonas_nulas(zona_B)
# Histograma de la Zona A
hist_zona_A_sin_nulidad = cv2.calcHist([zona_A_sin_nulidad], [0], None, [256], [0, 256])
# Histograma de la Zona B
hist_zona_B_sin_nulidad = cv2.calcHist([zona_B_sin_nulidad], [0], None, [256], [0, 256])
# Concatenación horizontal de las imágenes de las zonas A y B.
mosaico = np.hstack((zona_A_sin_nulidad, zona_B_sin_nulidad))
# Histograma del Mosaico
hist_mosaico = cv2.calcHist([mosaico], [0], None, [256], [0, 256])
# Especificando Zona A según Mosaico
zona_A_especificada = exposure.match_histograms(zona_A_sin_nulidad, mosaico)
zona_A_especificada = zona_A_especificada.astype(np.uint8) # float -> int
# Especificando Zona B según Mosaico
zona_B_especificada = exposure.match_histograms(zona_B_sin_nulidad, mosaico)
zona_B_especificada = zona_B_especificada.astype(np.uint8) # float -> int
# Histrograma de la Zona A especificada
hist_zona_A_especificada = cv2.calcHist([zona_A_especificada], [0], None, [256], [0, 256])
# Histrograma de la Zona B especificada
hist_zona_B_especificada = cv2.calcHist([zona_B_especificada], [0], None, [256], [0, 256])
# Mosaico resultante de la unión de ambas zonas especificadas
mosaico_especificado = np.hstack((zona_A_especificada, zona_B_especificada))
# Histograma del Mosaico especificado 
hist_mosaico_especificado = cv2.calcHist([mosaico_especificado], [0], None, [256], [0, 256])
# Reduciendo los niveles de cuantización del mosaico a 128

# Mostrar las imágenes originales y la imagen completada
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(mosaico_especificado, cmap='gray')
plt.title('')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia


plt.subplot(2, 2, 2)
plt.plot(hist_mosaico_especificado)
plt.title('')

plt.tight_layout()
plt.show()

mosaico_128 = mosaico // 2
# Histograma del mozaico con 128 niveles de cuantización
hist_mosaico_128 = cv2.calcHist([mosaico_128], [0], None, [128], [0, 128])
# Especificando Zona A según el Mosaico de 128 niveles.
zona_A_especificada_128 = exposure.match_histograms(zona_A_sin_nulidad, mosaico_128)
zona_A_especificada_128 = zona_A_especificada_128.astype(np.uint8) # float -> int
# Histograma de la Zona A especificada con 128 niveles.
hist_zona_A_especificada_128 = cv2.calcHist([zona_A_especificada_128], [0], None, [128], [0, 128])
# Especificando Zona B según el Mosaico de 128 niveles.
zona_B_especificada_128 = exposure.match_histograms(zona_B_sin_nulidad, mosaico_128)
zona_B_especificada_128 = zona_B_especificada_128.astype(np.uint8) # float -> int
# Histograma de la Zona B especificada con 128 niveles.
hist_zona_B_especificada_128 = cv2.calcHist([zona_B_especificada_128], [0], None, [128], [0, 128])
# Mosaico especificado con 128 niveles de cuantización
mosaico_especificado_128 = np.hstack((zona_A_especificada_128, zona_B_especificada_128))
# Histograma del mosaico especificado con 128 niveles de cuantización
hist_mosaico_especificado_128 = cv2.calcHist([mosaico_especificado_128], [0], None, [128], [0, 128])

# Reduciendo los niveles de cuantización del mosaico a 64
mosaico_64 = mosaico // 4
# Histograma del mozaico con 64 niveles de cuantización
hist_mosaico_64 = cv2.calcHist([mosaico_64], [0], None, [64], [0, 64])
# Especificando Zona A según el Mosaico de 64 niveles.
zona_A_especificada_64 = exposure.match_histograms(zona_A_sin_nulidad, mosaico_64)
zona_A_especificada_64 = zona_A_especificada_64.astype(np.uint8) # float -> int
# Histograma de la Zona A especificada con 64 niveles.
hist_zona_A_especificada_64 = cv2.calcHist([zona_A_especificada_64], [0], None, [64], [0, 64])
# Especificando Zona B según el Mosaico de 64 niveles.
zona_B_especificada_64 = exposure.match_histograms(zona_B_sin_nulidad, mosaico_64)
zona_B_especificada_64 = zona_B_especificada_64.astype(np.uint8) # float -> int
# Histograma de la Zona B especificada con 64 niveles.
hist_zona_B_especificada_64 = cv2.calcHist([zona_B_especificada_64], [0], None, [64], [0, 64])
# Mosaico especificado con 64 niveles de cuantización
mosaico_especificado_64 = np.hstack((zona_A_especificada_64, zona_B_especificada_64))
# Histograma del mosaico especificado con 64 niveles de cuantización
hist_mosaico_especificado_64 = cv2.calcHist([mosaico_especificado_64], [0], None, [64], [0, 64])

# Reduciendo los niveles de cuantización del mosaico a 32
mosaico_32 = mosaico // 8
# Histograma del mozaico con 32 niveles de cuantización
hist_mosaico_32 = cv2.calcHist([mosaico_32], [0], None, [32], [0, 32])
# Especificando Zona A según el Mosaico de 32 niveles.
zona_A_especificada_32 = exposure.match_histograms(zona_A_sin_nulidad, mosaico_32)
zona_A_especificada_32 = zona_A_especificada_32.astype(np.uint8) # float -> int
# Histograma de la Zona A especificada con 32 niveles.
hist_zona_A_especificada_32 = cv2.calcHist([zona_A_especificada_32], [0], None, [32], [0, 32])
# Especificando Zona B según el Mosaico de 32 niveles.
zona_B_especificada_32 = exposure.match_histograms(zona_B_sin_nulidad, mosaico_32)
zona_B_especificada_32 = zona_B_especificada_32.astype(np.uint8) # float -> int
# Histograma de la Zona B especificada con 32 niveles.
hist_zona_B_especificada_32 = cv2.calcHist([zona_B_especificada_32], [0], None, [32], [0, 32])
# Mosaico especificado con 32 niveles de cuantización
mosaico_especificado_32 = np.hstack((zona_A_especificada_32, zona_B_especificada_32))
# Histograma del mosaico especificado con 32 niveles de cuantización
hist_mosaico_especificado_32 = cv2.calcHist([mosaico_especificado_32], [0], None, [32], [0, 32])

# Reduciendo los niveles de cuantización del mosaico a 16
mosaico_16 = mosaico // 16
# Histograma del mozaico con 16 niveles de cuantización
hist_mosaico_16 = cv2.calcHist([mosaico_16], [0], None, [16], [0, 16])
# Especificando Zona A según el Mosaico de 16 niveles.
zona_A_especificada_16 = exposure.match_histograms(zona_A_sin_nulidad, mosaico_16)
zona_A_especificada_16 = zona_A_especificada_16.astype(np.uint8) # float -> int
# Histograma de la Zona A especificada con 16 niveles.
hist_zona_A_especificada_16 = cv2.calcHist([zona_A_especificada_16], [0], None, [16], [0, 16])
# Especificando Zona B según el Mosaico de 16 niveles.
zona_B_especificada_16 = exposure.match_histograms(zona_B_sin_nulidad, mosaico_16)
zona_B_especificada_16 = zona_B_especificada_16.astype(np.uint8) # float -> int
# Histograma de la Zona B especificada con 16 niveles.
hist_zona_B_especificada_16 = cv2.calcHist([zona_B_especificada_16], [0], None, [16], [0, 16])
# Mosaico especificado con 16 niveles de cuantización
mosaico_especificado_16 = np.hstack((zona_A_especificada_16, zona_B_especificada_16))
# Histograma del mosaico especificado con 16 niveles de cuantización
hist_mosaico_especificado_16 = cv2.calcHist([mosaico_especificado_16], [0], None, [16], [0, 16])

# Reduciendo los niveles de cuantización del mosaico a 8
mosaico_8 = mosaico // 32
# Histograma del mozaico con 8 niveles de cuantización
hist_mosaico_8 = cv2.calcHist([mosaico_8], [0], None, [8], [0, 8])
# Especificando Zona A según el Mosaico de 8 niveles.
zona_A_especificada_8 = exposure.match_histograms(zona_A_sin_nulidad, mosaico_8)
zona_A_especificada_8 = zona_A_especificada_8.astype(np.uint8) # float -> int
# Histograma de la Zona A especificada con 8 niveles.
hist_zona_A_especificada_8 = cv2.calcHist([zona_A_especificada_8], [0], None, [8], [0, 8])
# Especificando Zona B según el Mosaico de 8 niveles.
zona_B_especificada_8 = exposure.match_histograms(zona_B_sin_nulidad, mosaico_8)
zona_B_especificada_8 = zona_B_especificada_8.astype(np.uint8) # float -> int
# Histograma de la Zona B especificada con 8 niveles.
hist_zona_B_especificada_8 = cv2.calcHist([zona_B_especificada_8], [0], None, [8], [0, 8])
# Mosaico especificado con 8 niveles de cuantización
mosaico_especificado_8 = np.hstack((zona_A_especificada_8, zona_B_especificada_8))
# Histograma del mosaico especificado con 8 niveles de cuantización
hist_mosaico_especificado_8 = cv2.calcHist([mosaico_especificado_8], [0], None, [8], [0, 8])

# Reduciendo los niveles de cuantización del mosaico a 2
mosaico_2 = mosaico // 128
# Histograma del mozaico con 2 niveles de cuantización
hist_mosaico_2 = cv2.calcHist([mosaico_2], [0], None, [2], [0, 2])
# Especificando Zona A según el Mosaico de 2 niveles.
zona_A_especificada_2 = exposure.match_histograms(zona_A_sin_nulidad, mosaico_2)
zona_A_especificada_2 = zona_A_especificada_2.astype(np.uint8) # float -> int
# Histograma de la Zona A especificada con 2 niveles.
hist_zona_A_especificada_2 = cv2.calcHist([zona_A_especificada_2], [0], None, [2], [0, 2])
# Especificando Zona B según el Mosaico de 2 niveles.
zona_B_especificada_2 = exposure.match_histograms(zona_B_sin_nulidad, mosaico_2)
zona_B_especificada_2 = zona_B_especificada_2.astype(np.uint8) # float -> int
# Histograma de la Zona B especificada con 2 niveles.
hist_zona_B_especificada_2 = cv2.calcHist([zona_B_especificada_2], [0], None, [2], [0, 2])
# Mosaico especificado con 2 niveles de cuantización
mosaico_especificado_2 = np.hstack((zona_A_especificada_2, zona_B_especificada_2))
# Histograma del mosaico especificado con 2 niveles de cuantización
hist_mosaico_especificado_2 = cv2.calcHist([mosaico_especificado_2], [0], None, [2], [0, 2])

# Mostrar las imágenes originales y la imagen completada
plt.figure(figsize=(12, 12))
plt.subplot(2, 2, 1)
plt.imshow(mosaico_especificado_2, cmap='gray')
plt.title('')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia


plt.subplot(2, 2, 2)
plt.plot(hist_mosaico_especificado_2)
plt.title('')

plt.tight_layout()
plt.show()