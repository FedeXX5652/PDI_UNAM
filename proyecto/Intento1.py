import cv2
import numpy as np
import os

# Ruta relativa de la imagen
img_path = os.path.join('Proy_PDI', 'img1.png')

# Cargar la imagen en escala de grises
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Validar si la imagen se cargó correctamente
if img is None:
    print(f"Error: No se pudo cargar la imagen en {img_path}. Verifica la ruta o el archivo.")
    exit()

# Mejorar contraste con ecualización de histograma
img_eq = cv2.equalizeHist(img)

# Aplicar un filtro Gaussiano para reducir ruido
img_filtered = cv2.GaussianBlur(img_eq, (5, 5), 0)

# Aplicar umbralización (Otsu) para segmentar los mastocitos
_, img_thresh = cv2.threshold(img_filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Detectar bordes con Canny
edges = cv2.Canny(img_thresh, 100, 200)

# Encontrar contornos de las regiones detectadas
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filtrar contornos por área y circularidad
mastocitos = []
for c in contours:
    area = cv2.contourArea(c)
    if 50 < area < 500:  # Ajustar estos valores según el tamaño esperado de los mastocitos
        perimeter = cv2.arcLength(c, True)
        circularity = (4 * np.pi * area) / (perimeter ** 2)
        if circularity > 0.7:  # Filtrar por formas más redondeadas
            mastocitos.append(c)

# Dibujar los contornos detectados en la imagen original (en color)
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(img_color, mastocitos, -1, (0, 255, 0), 2)

# Mostrar la imagen original con los mastocitos detectados
cv2.imshow('Mastocitos detectados', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Ruta relativa para guardar la imagen con los contornos detectados
output_path = os.path.join('Proy_PDI', 'mastocitos_detectados.png')
cv2.imwrite(output_path, img_color)
print(f"Imagen guardada en: {output_path}")
