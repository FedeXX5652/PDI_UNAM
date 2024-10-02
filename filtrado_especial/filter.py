import numpy as np
import cv2
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

class Filter:
    @staticmethod
    def conv2(x, y, mode='same'):
        # Realiza la convolución 2D de dos matrices, x e y.
        # Se usa la rotación de las matrices para ajustar la convolución.
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

    @staticmethod
    def load_image_as_grayscale(image_path):
        # Carga una imagen en escala de grises desde una ruta especificada.
        image = cv2.imread(image_path, 0)
        if image is None:
            raise ValueError("Image could not be loaded. Check the image path.")
        return image

    @staticmethod
    def apply_filter_to_image(image_matrix, filter_matrix):
        # Aplica un filtro a una imagen usando la función de convolución.
        return Filter.conv2(image_matrix, filter_matrix)

    @staticmethod
    def show_image(name, img, save=False):
        # Muestra una imagen en una ventana y opcionalmente la guarda en un archivo.
        cv2.imshow(name, np.uint8(img))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        if save:
            Filter.save_image(name + ".jpg", img)

    @staticmethod
    def save_image(path, img):
        # Guarda la imagen en el disco en la ruta especificada.
        if "." not in path:
            path = path + ".jpg"
        success = cv2.imwrite(path, np.uint8(img))
        if success:
            print(f"Image successfully saved at {path}")
        else:
            raise ValueError(f"Failed to save image at {path}")

    @staticmethod
    def block_lowpass_filter(size):
        # Crea un filtro de paso bajo uniforme (bloque) de tamaño dado.
        return np.ones((size, size)) / (size * size)

    @staticmethod
    def binomial_lowpass_filter(size):
        # Crea un filtro de paso bajo binomial de tamaño impar especificado.
        if size % 2 == 0:
            raise ValueError("Size must be odd")
        
        pascal_row = [1]
        for _ in range(size - 1):
            # Genera la fila de Pascal para el filtro binomial.
            pascal_row = [1] + [pascal_row[i] + pascal_row[i+1] for i in range(len(pascal_row)-1)] + [1]
        
        filter_1d = np.array(pascal_row)
        filter_2d = np.outer(filter_1d, filter_1d)
        return filter_2d / np.sum(filter_2d)

    @staticmethod
    def prewitt_filter():
        # Define los filtros de Prewitt para la detección de bordes.
        return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    @staticmethod
    def sobel_filter():
        # Define los filtros de Sobel para la detección de bordes.
        return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    @staticmethod
    def gaussian_derivative_filter(size, sigma=1.0):
        # Crea filtros de derivada gaussiana de tamaño dado y desviación estándar especificada.
        x, y = np.meshgrid(np.arange(-(size//2), size//2 + 1), np.arange(-(size//2), size//2 + 1))
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        gx = -x / (sigma**2) * g
        gy = -y / (sigma**2) * g
        return gx / np.sum(np.abs(gx)), gy / np.sum(np.abs(gy))

    @staticmethod
    def laplacian_filter():
        # Define el filtro laplaciano, utilizado para la detección de bordes.
        return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    @staticmethod
    def laplacian_of_gaussian_filter(size, sigma=1.0):
        # Crea un filtro que combina Laplaciano y Gaussiano.
        x, y = np.meshgrid(np.arange(-(size//2), size//2 + 1), np.arange(-(size//2), size//2 + 1))
        g = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        log = ((x**2 + y**2 - 2*sigma**2) / sigma**4) * g
        return log - np.mean(log)

    @staticmethod
    def unsharp_masking(image, lowpass_filter, alpha=1.5):
        # Aplica la técnica de máscara de enfoque para mejorar la nitidez de la imagen.
        blurred = Filter.apply_filter_to_image(image, lowpass_filter)
        return image + alpha * (image - blurred)

    @staticmethod
    def add_noise(image, mean=0, std=25):
        # Añade ruido gaussiano a la imagen.
        noise = np.random.normal(mean, std, image.shape).astype(np.float64)
        noisy_image = image.astype(np.float64) + noise
        return np.clip(noisy_image, 0, 255).astype(np.uint8)

    @staticmethod
    def process_image(image, filter_function, *args, **kwargs):
        # Aplica un filtro a la imagen y asegura que los valores estén en el rango válido.
        filtered = filter_function(image, *args, **kwargs)
        return np.clip(filtered, 0, 255).astype(np.uint8)

    @staticmethod
    def apply_filters(image, noisy_image, filters, filter_names):
        # Aplica varios filtros a una imagen y a una versión ruidosa de ella.
        results = []
        for filter_func, name in zip(filters, filter_names):
            filtered = Filter.process_image(image, filter_func)
            noisy_filtered = Filter.process_image(noisy_image, filter_func)
            results.append((filtered, noisy_filtered, name))
        return results

    @staticmethod
    def display_results(results):
        # Muestra y guarda los resultados de los filtros aplicados.
        for filtered, noisy_filtered, name in results:
            Filter.show_image(f"{name} - Sin ruido", filtered, save=True)
            Filter.show_image(f"{name} - Con ruido", noisy_filtered, save=True)