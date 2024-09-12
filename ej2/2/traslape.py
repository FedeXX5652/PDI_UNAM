import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función para eliminar las zonas con información nula (cero)
def eliminar_zonas_nulas(imagen):
    # Crear una máscara donde los píxeles no son cero
    mascara = imagen > 0
    return mascara

# Función para ajustar la intensidad de una imagen para que coincida con un brillo objetivo
def ajustar_brillo(imagen, brillo_objetivo):
    brillo_actual = np.mean(imagen[imagen > 0])
    if brillo_actual == 0:
        return imagen
    factor = brillo_objetivo / brillo_actual
    imagen_ajustada = np.clip(imagen * factor, 0, 256)
    return imagen_ajustada.astype(np.uint8)  # Asegurarse de que el tipo de datos sea np.uint8

# Función para completar la imagen con valores de la otra imagen donde hay ceros
def completar_imagenes(imagen1, imagen2):
    # Crear la máscara para ambas imágenes
    mascara1 = eliminar_zonas_nulas(imagen1)
    mascara2 = eliminar_zonas_nulas(imagen2)
    
    # Calcular el brillo promedio en áreas no nulas
    brillo1 = np.mean(imagen1[mascara1])
    brillo2 = np.mean(imagen2[mascara2])
    
    # Ajustar el brillo de las imágenes
    imagen1_ajustada = ajustar_brillo(imagen1, brillo2)
    imagen2_ajustada = ajustar_brillo(imagen2, brillo1)
    
    # Crear la imagen completada usando la máscara
    imagen_completada = np.copy(imagen1_ajustada)
    imagen_completada[~mascara1] = imagen2_ajustada[~mascara1]  # Llenar donde imagen1 tiene ceros con los valores de imagen2

    return imagen_completada

# Función para ajustar el histograma de una imagen usando un histograma objetivo
def especificar_histograma(imagen, histograma_objetivo):
    # Normalizar el histograma objetivo
    histograma_objetivo = histograma_objetivo / histograma_objetivo.sum()
    cdf_objetivo = histograma_objetivo.cumsum()
    
    # Calcular el histograma de la imagen
    hist_imagen = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    hist_imagen = hist_imagen / hist_imagen.sum()
    cdf_imagen = hist_imagen.cumsum()
    
    # Mapear los valores de pixel según la CDF
    mapeo = np.interp(cdf_imagen, cdf_objetivo, np.arange(256))
    
    # Aplicar el mapeo a la imagen
    imagen_ajustada = mapeo[imagen]
    imagen_ajustada = np.uint8(imagen_ajustada)
    
    return imagen_ajustada

# Cargar las imágenes satelitales en escala de grises
imagen1 = cv2.imread('zonaA_b5.tif', cv2.IMREAD_GRAYSCALE)
imagen2 = cv2.imread('zonaB_b5.tif', cv2.IMREAD_GRAYSCALE)

# Verificar si las imágenes se cargaron correctamente
if imagen1 is None or imagen2 is None:
    raise Exception("No se pudieron cargar las imágenes satelitales")

# Completar la imagen 1 con los valores de imagen 2 donde hay ceros
imagen_completada = completar_imagenes(imagen1, imagen2)

# Mostrar las imágenes originales y la imagen completada
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
plt.imshow(imagen1, cmap='gray')
plt.title('Imagen Satelital 1')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia

plt.subplot(2, 2, 2)
plt.imshow(imagen2, cmap='gray')
plt.title('Imagen Satelital 2')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia

plt.subplot(2, 2, 3)
plt.imshow(imagen_completada, cmap='gray')
plt.title('Imagen Completada')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia

plt.tight_layout()
plt.show()

# Calcular y mostrar el histograma de las imágenes
plt.figure(figsize=(12, 12))

plt.subplot(2, 2, 1)
hist1 = cv2.calcHist([imagen1.astype(np.uint8)], [0], None, [256], [0, 256])
plt.plot(hist1)
plt.title('Histograma Imagen 1')

plt.subplot(2, 2, 2)
hist2 = cv2.calcHist([imagen2.astype(np.uint8)], [0], None, [256], [0, 256])
plt.plot(hist2)
plt.title('Histograma Imagen 2')

# Mostrar el histograma de la imagen completada
plt.subplot(2, 2, 3)
hist_completada = cv2.calcHist([imagen_completada.astype(np.uint8)], [0], None, [256], [0, 256])
plt.plot(hist_completada)
plt.title('Histograma de la Imagen Completada')

plt.tight_layout()
plt.show()

# Extraer y mostrar la zona de traslape
# Aquí se asume que el traslape es en la parte central de la imagen
altura, ancho = imagen1.shape
traslape = imagen_completada[:, ancho//2:]

plt.figure(figsize=(6, 6))
plt.imshow(traslape, cmap='gray')
plt.title('Zona de Traslape')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia
plt.show()

# Calcular y mostrar el histograma de la zona de traslape
hist_traslape = cv2.calcHist([traslape.astype(np.uint8)], [0], None, [256], [0, 256])

plt.figure(figsize=(6, 6))
plt.plot(hist_traslape)
plt.title('Histograma Zona de Traslape')
plt.show()

# Ajustar las imágenes usando el histograma de la zona de traslape
imagen1_ajustada = especificar_histograma(imagen1, hist_traslape)
imagen2_ajustada = especificar_histograma(imagen2, hist_traslape)

# Mostrar las imágenes ajustadas
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(imagen1_ajustada, cmap='gray')
plt.title('Imagen 1 Ajustada')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia

plt.subplot(1, 2, 2)
plt.imshow(imagen2_ajustada, cmap='gray')
plt.title('Imagen 2 Ajustada')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia

plt.tight_layout()
plt.show()

# Mostrar el histograma de las imágenes ajustadas
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
hist1_ajustada = cv2.calcHist([imagen1_ajustada.astype(np.uint8)], [0], None, [256], [0, 256])
plt.plot(hist1_ajustada)
plt.title('Histograma Imagen 1 Ajustada')

plt.subplot(1, 2, 2)
hist2_ajustada = cv2.calcHist([imagen2_ajustada.astype(np.uint8)], [0], None, [256], [0, 256])
plt.plot(hist2_ajustada)
plt.title('Histograma Imagen 2 Ajustada')

plt.tight_layout()
plt.show()

# Unir las imágenes ajustadas para formar el mosaico
mosaico_ajustado = np.hstack((imagen1_ajustada, imagen2_ajustada))

# Mostrar el mosaico ajustado
plt.figure(figsize=(12, 6))
plt.imshow(mosaico_ajustado, cmap='gray')
plt.title('Mosaico con Especificación de Histograma')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia
plt.show()

# Mostrar el histograma del mosaico final
hist_mosaico = cv2.calcHist([mosaico_ajustado.astype(np.uint8)], [0], None, [256], [0, 256])

plt.figure(figsize=(6, 6))
plt.plot(hist_mosaico)
plt.title('Histograma del Mosaico Ajustado')
plt.show()

# Superponer las dos imágenes ajustadas
# Para evitar problemas de dimensiones, redimensionamos las imágenes si es necesario
if imagen1_ajustada.shape != imagen2_ajustada.shape:
    alto, ancho = min(imagen1_ajustada.shape[0], imagen2_ajustada.shape[0]), min(imagen1_ajustada.shape[1], imagen2_ajustada.shape[1])
    imagen1_ajustada = cv2.resize(imagen1_ajustada, (ancho, alto))
    imagen2_ajustada = cv2.resize(imagen2_ajustada, (ancho, alto))

# Crear una imagen superpuesta (mezcla de las dos imágenes ajustadas)
superposicion = cv2.addWeighted(imagen1_ajustada, 0.5, imagen2_ajustada, 0.5, 0)

# Mostrar la imagen superpuesta
plt.figure(figsize=(6, 6))
plt.imshow(superposicion, cmap='gray')
plt.title('Imágenes Superpuestas')
plt.axis('off')  # Desactivar los ejes para una visualización más limpia
plt.show()
