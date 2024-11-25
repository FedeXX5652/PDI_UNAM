import cv2
import numpy as np
import os

# Construir la ruta de la imagen relativa al directorio actual
img_path = os.path.join('Proy_PDI', 'img1.png')

# 1. Cargar la imagen en escala de grises
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Validar si la imagen se cargó correctamente
if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta o el archivo.")
    exit()

# 2. Mejorar contraste con ecualización de histograma
img_eq = cv2.equalizeHist(img)

# 3. Reducir ruido con filtro Gaussiano
img_filtered = cv2.GaussianBlur(img_eq, (5, 5), 0)

# 4. Umbralización adaptativa para ajustar a diferentes intensidades
adaptive_thresh = cv2.adaptiveThreshold(
    img_filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
)

# 5. Operaciones morfológicas para limpiar ruido y conectar regiones
kernel = np.ones((3, 3), np.uint8)
morph_open = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# 6. Detectar regiones de fondo y objeto
# Fondo seguro (dilatación)
sure_bg = cv2.dilate(morph_open, kernel, iterations=3)

# Objeto seguro (distancia transformada)
dist_transform = cv2.distanceTransform(morph_open, cv2.DIST_L2, 5)
_, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)

# Regiones desconocidas
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

# 7. Crear marcadores para Watershed
_, markers = cv2.connectedComponents(sure_fg)
markers = markers + 1  # Incrementar los marcadores para que el fondo sea 1
markers[unknown == 255] = 0  # Áreas desconocidas se marcan como 0

# 8. Aplicar Watershed
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para dibujar en color
markers = cv2.watershed(img_color, markers)

# Dibujar los contornos detectados
img_color[markers == -1] = [0, 0, 255]  # Contornos en rojo

# Opcional: Resaltar regiones detectadas como objetos (mastocitos)
for marker_id in range(2, np.max(markers) + 1):
    mask = np.uint8(markers == marker_id)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(img_color, [contour], -1, (0, 255, 0), 2)

# Construir la ruta relativa para guardar el resultado
output_path = os.path.join('Proy_PDI', 'mastocitos_mejorados.png')
cv2.imwrite(output_path, img_color)  # Guardar la imagen procesada
print(f"Imagen guardada en: {output_path}")

# 9. Mostrar el resultado
cv2.imshow('Segmentación mejorada con Watershed', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
