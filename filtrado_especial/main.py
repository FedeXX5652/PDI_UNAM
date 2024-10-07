from filter import Filter
import os
import numpy as np

IMG = os.path.join(os.path.dirname(__file__), "Aerial04.jpg")

# Cargar imagen en color
image = Filter.load_image(IMG)

# Crear versión con ruido
noisy_image = Filter.add_noise(image)

# Mostrar imágenes originales
Filter.show_image("Imagen original", image, save=True)
Filter.show_image("Imagen con ruido", noisy_image, save=True)

#2
block_sizes = [3, 7, 9, 11]
block_filters = [Filter.block_lowpass_filter(size) for size in block_sizes]
block_results = Filter.apply_filters(image, noisy_image, 
                                     [lambda img, f=f: Filter.apply_filter_to_image(img, f) for f in block_filters],
                                     [f"Paso bajas de bloque {size}x{size}" for size in block_sizes])
Filter.display_results(block_results)

#3
binomial_sizes = [3, 7, 9, 11]
binomial_filters = [Filter.binomial_lowpass_filter(size) for size in binomial_sizes]
binomial_results = Filter.apply_filters(image, noisy_image, 
                                        [lambda img, f=f: Filter.apply_filter_to_image(img, f) for f in binomial_filters],
                                        [f"Paso bajas binomial {size}x{size}" for size in binomial_sizes])
Filter.display_results(binomial_results)

#4
# a) Filtro de bloque [1 -1]
block_edge = np.array([[1, -1]])
block_edge_results = Filter.apply_filters(image, noisy_image, 
                                          [lambda img: Filter.apply_filter_to_image(img, block_edge)],
                                          ["Filtro de borde [1 -1]"])

# b) Prewitt
prewitt_x, prewitt_y = Filter.prewitt_filter()
prewitt_results = Filter.apply_filters(image, noisy_image, 
                                       [lambda img: Filter.apply_filter_to_image(img, prewitt_x),
                                        lambda img: Filter.apply_filter_to_image(img, prewitt_y)],
                                       ["Prewitt X", "Prewitt Y"])

# c) Sobel
sobel_x, sobel_y = Filter.sobel_filter()
sobel_results = Filter.apply_filters(image, noisy_image, 
                                     [lambda img: Filter.apply_filter_to_image(img, sobel_x),
                                      lambda img: Filter.apply_filter_to_image(img, sobel_y)],
                                     ["Sobel X", "Sobel Y"])

# d) Primera derivada de Gaussiana
gauss_sizes = [5, 7, 11]
gauss_filters = [Filter.gaussian_derivative_filter(size) for size in gauss_sizes]
gauss_results = []
for size, (gx, gy) in zip(gauss_sizes, gauss_filters):
    gauss_results.extend(Filter.apply_filters(image, noisy_image, 
                                              [lambda img: Filter.apply_filter_to_image(img, gx),
                                               lambda img: Filter.apply_filter_to_image(img, gy)],
                                              [f"Gaussiana {size}x{size} X", f"Gaussiana {size}x{size} Y"]))

Filter.display_results(block_edge_results + prewitt_results + sobel_results + gauss_results)

#5
# a) Laplaciano 3x3
laplacian = Filter.laplacian_filter()
laplacian_results = Filter.apply_filters(image, noisy_image, 
                                         [lambda img: Filter.apply_filter_to_image(img, laplacian)],
                                         ["Laplaciano 3x3"])

# b) Laplacianos basados en la segunda derivada de Gaussiana
log_sizes = [5, 7, 11]
log_filters = [Filter.laplacian_of_gaussian_filter(size) for size in log_sizes]
log_results = Filter.apply_filters(image, noisy_image, 
                                   [lambda img, f=f: Filter.apply_filter_to_image(img, f) for f in log_filters],
                                   [f"LoG {size}x{size}" for size in log_sizes])

Filter.display_results(laplacian_results + log_results)

#6
# Difuminar las imágenes
blur_filter = Filter.block_lowpass_filter(5)
blurred = Filter.process_image(image, lambda img: Filter.apply_filter_to_image(img, blur_filter))
noisy_blurred = Filter.process_image(noisy_image, lambda img: Filter.apply_filter_to_image(img, blur_filter))

Filter.show_image("Imagen difuminada sin ruido", blurred, save=True)
Filter.show_image("Imagen difuminada con ruido", noisy_blurred, save=True)

# Aplicar unsharp masking
unsharp_filters = [Filter.block_lowpass_filter(3), Filter.block_lowpass_filter(7),
                   Filter.binomial_lowpass_filter(3), Filter.binomial_lowpass_filter(7)]
unsharp_names = ["Unsharp Block 3x3", "Unsharp Block 7x7", "Unsharp Binomial 3x3", "Unsharp Binomial 7x7"]

unsharp_results = Filter.apply_filters(image, noisy_image, 
                                       [lambda img, f=f: Filter.unsharp_masking(img, f) for f in unsharp_filters],
                                       unsharp_names)
Filter.display_results(unsharp_results)