import cv2
import numpy as np
import os

# Ruta relativa de la imagen
img_path = os.path.join('Proy_PDI', 'img1.png')

# 1. Cargar la imagen en escala de grises
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Validar si la imagen se cargó correctamente
if img is None:
    print(f"Error: No se pudo cargar la imagen en {img_path}. Verifica la ruta o el archivo.")
    exit()

# 2. Mejorar contraste con ecualización de histograma
img_eq = cv2.equalizeHist(img)

# 3. Reducir ruido con filtro Gaussiano
img_filtered = cv2.GaussianBlur(img_eq, (5, 5), 0)

# 4. Detectar bordes con Canny (ajustar parámetros según la imagen)
edges = cv2.Canny(img_filtered, 100, 200)

# 5. Usar la transformada de Hough para detectar círculos
circles = cv2.HoughCircles(
    img_filtered,
    cv2.HOUGH_GRADIENT,
    dp=1.5,          # Resolución inversa del acumulador
    minDist=20,      # Distancia mínima entre los centros de los círculos
    param1=50,       # Umbral superior para Canny
    param2=30,       # Umbral para la acumulación (ajusta para filtrar falsos positivos)
    minRadius=10,    # Radio mínimo de los círculos detectados
    maxRadius=50     # Radio máximo de los círculos detectados
)

# 6. Dibujar los círculos detectados en la imagen original
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Convertir a BGR para dibujar en color
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Dibujar el contorno del círculo
        cv2.circle(img_color, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Contorno en verde
        # Dibujar el centro del círculo
        cv2.circle(img_color, (i[0], i[1]), 2, (0, 0, 255), 3)  # Centro en rojo
else:
    print("No se detectaron círculos. Intenta ajustar los parámetros.")

# 7. Guardar el resultado en una ruta relativa
output_path = os.path.join('Proy_PDI', 'mastocitos_hough.png')
cv2.imwrite(output_path, img_color)  # Guardar la imagen procesada
print(f"Imagen guardada en: {output_path}")

# 8. Mostrar el resultado
cv2.imshow('Círculos detectados (Transformada de Hough)', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()
