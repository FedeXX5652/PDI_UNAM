import openslide
from PIL import Image
import os

# Cargar la imagen SVS
svs_path = os.path.join(os.getcwd(), "P23-5346A.svs")
output_dir = os.path.join(os.getcwd(), "mosaicos")
os.makedirs(output_dir, exist_ok=True)

# Abrir la imagen SVS
slide = openslide.OpenSlide(svs_path)

# Dimensiones del mosaico
tile_size = 4000  # Tamaño del mosaico
max_tiles = 200  # Número máximo de mosaicos
tile_count = 0

# Dimensiones de la imagen completa
width, height = slide.dimensions

# Calcular el centro de la imagen
center_x, center_y = width // 2, height // 2

# Generar un rango para desplazarse desde el centro
offsets = list(range(0, max(width, height), tile_size))

# Generar mosaicos desde el centro hacia los alrededores
for offset_y in offsets:
    for offset_x in offsets:
        # Generar coordenadas alrededor del centro
        for dx, dy in [(-offset_x, -offset_y), (offset_x, -offset_y),
                       (-offset_x, offset_y), (offset_x, offset_y)]:
            x, y = center_x + dx, center_y + dy
            
            # Verificar que los límites no excedan el tamaño de la imagen
            if x < 0 or y < 0 or x + tile_size > width or y + tile_size > height:
                continue
            
            # Leer y guardar el mosaico
            tile = slide.read_region((x, y), 0, (tile_size, tile_size)).convert("RGB")
            tile.save(os.path.join(output_dir, f"tile_{x}_{y}.png"))
            tile_count += 1

            # Detenerse cuando se alcanzan 200 mosaicos
            if tile_count >= max_tiles:
                break
        if tile_count >= max_tiles:
            break
    if tile_count >= max_tiles:
        break

print(f"Se han generado {tile_count} mosaicos en la carpeta: {output_dir}")