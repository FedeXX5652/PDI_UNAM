import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def cuantizacion(imagen, niveles):
    imagen_cuantizada = np.floor(imagen / (256 / niveles)) * (256 / niveles) # Se dividen en 256 niveles
    imagen_cuantizada = np.uint8(imagen_cuantizada) #Vital, pasarlo a formato uint8
    imagen_ecualizada = cv2.equalizeHist(imagen_cuantizada) #Funcion cuantizacion
    return imagen_ecualizada

directorio = os.path.dirname(os.path.abspath(__file__)) #Obtener directorio
#archivo = 'pinos512.tif'
archivo = 'chestrayosX.tif'
ruta_imagen = os.path.join(directorio, archivo)
imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE) #Se usa escala de grises

if imagen is None:
    print(f"La imagen '{archivo}' no se pudo cargar.")
else:
    imagen = np.uint8(imagen) # Cambiar imagen a uint8
    plt.figure(figsize=(12, 6)) # Imagen original
    plt.subplot(2, 2, 1)
    plt.imshow(imagen, cmap='gray')
    plt.title('Original')

    plt.subplot(2, 2, 2) #Histograma
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    plt.plot(histograma)
    plt.title('Histograma')

    imagen_ecualizada = cv2.equalizeHist(imagen) #Ecualizar imagen y mostrarla

    plt.subplot(2, 2, 3)
    plt.imshow(imagen_ecualizada, cmap='gray')
    plt.title('Imagen Ecualizada')

    plt.subplot(2, 2, 4) # Histograma de imagen ecualizada
    hist_ecualizado = cv2.calcHist([imagen_ecualizada], [0], None, [256], [0, 256])
    plt.plot(hist_ecualizado)
    plt.title('Histograma Ecualizado')

    plt.show()

    niveles_cuant = [128, 64, 32, 16, 8, 2] # Niveles

    plt.figure(figsize=(18, 12))

    for i, niveles in enumerate(niveles_cuant):
        imagen_ecucuant = cuantizacion(imagen, niveles)

        plt.subplot(2, len(niveles_cuant), i + 1) # Mostrar grafica
        plt.imshow(imagen_ecucuant, cmap='gray')
        plt.title(f'{niveles} niveles')

        # Histogramas actualizados
        plt.subplot(2, len(niveles_cuant), i + 1 + len(niveles_cuant))
        hist_ecucuant = cv2.calcHist([imagen_ecucuant], [0], None, [256], [0, 256])
        plt.plot(hist_ecucuant)
        plt.title(f'Histograma {niveles} niveles')

    plt.tight_layout()
    plt.show()
