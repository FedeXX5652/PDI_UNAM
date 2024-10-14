import numpy as np
import cv2
from scipy.signal import convolve2d

class Filter:
    @staticmethod
    def conv2(x, y, mode='same'):
        # Realiza la convolución 2D de dos matrices, x e y.
        # Se usa la rotación de las matrices para ajustar la convolución.
        return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)

    @staticmethod
    def load_image(image_path, bw=False):
        # Carga una imagen en escala de grises
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE if bw else cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Image could not be loaded. Check the image path.")
        return image

    @staticmethod
    def apply_filter_to_image(image_matrix, filter_matrix):
        # Aplica el filtro a cada canal si la imagen es en color (más de un canal).
        if len(image_matrix.shape) == 3:  # Si la imagen tiene 3 canales (color)
            channels = cv2.split(image_matrix)  # Divide en canales R, G, B
            filtered_channels = [Filter.conv2(c, filter_matrix) for c in channels]  # Aplica filtro a cada canal
            return cv2.merge(filtered_channels)  # Une los canales filtrados
        else:
            # Para imágenes en escala de grises
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
    def add_noise(image, mean=0, std=25, seed=None):
        if seed is not None:
            np.random.seed(seed)  # Set the seed for reproducibility
        
        if len(image.shape) == 3:  # Color image
            noisy_channels = []
            for c in cv2.split(image):
                noise = np.random.normal(mean, std, c.shape).astype(np.float64)
                noisy_channel = c.astype(np.float64) + noise
                noisy_channels.append(np.clip(noisy_channel, 0, 255).astype(np.uint8))
            return cv2.merge(noisy_channels)
        else:
            # For grayscale images
            noise = np.random.normal(mean, std, image.shape).astype(np.float64)
            noisy_image = image.astype(np.float64) + noise
            return np.clip(noisy_image, 0, 255).astype(np.uint8)
        
    @staticmethod
    def zero_pad_img(image, filter_size):
        h, w = image.shape[:2]
        fh, fw = filter_size
        pad_h = h + fh - 1
        pad_w = w + fw - 1

        padded = np.zeros((pad_h, pad_w) + image.shape[2:], dtype=image.dtype)
        padded[:h, :w] = image
        return padded

    
    @staticmethod
    def zero_pad_filter(filter_matrix, image_shape):
        """
        Zero-pads a filter to match the size of the image.
        :param filter_matrix: The filter to be padded (a 2D array).
        :param image_shape: The shape of the target image (a tuple with height and width).
        :return: The zero-padded filter of the same size as the image.
        """
        # Get dimensions of the filter and the image
        filter_rows, filter_cols = filter_matrix.shape
        image_rows, image_cols = image_shape[:2]  # Ignore color channels for now

        # Create a new matrix full of zeros with the same size as the image
        padded_filter = np.zeros((image_rows, image_cols), dtype=np.float32)

        # Calculate the starting position to place the filter at the center of the padded image
        start_row = (image_rows - filter_rows)
        start_col = (image_cols - filter_cols)

        # Insert the filter into the padded matrix
        padded_filter[start_row:start_row + filter_rows, start_col:start_col + filter_cols] = filter_matrix

        return padded_filter


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

    @staticmethod
    def image2dft(image_matrix):
        if len(image_matrix.shape) == 3:
            image_matrix = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2GRAY)
        return cv2.dft(np.float32(image_matrix), flags=cv2.DFT_COMPLEX_OUTPUT)  # DFT_COMPLEX_OUTPUT: packed array into a real array of the same size as input
    
    @staticmethod
    def filter2dft(filter):
        return cv2.dft(np.float32(filter), flags=cv2.DFT_COMPLEX_OUTPUT)    # DFT_COMPLEX_OUTPUT: packed array into a real array of the same size as input

    @staticmethod
    def apply_dft_filter(dft_image, dft_filter):
        r = cv2.mulSpectrums(dft_image, dft_filter, 0)
        return r

    @staticmethod
    def dft2image(dft, original_size=None):
        # Step 1: Apply inverse DFT to convert from frequency domain to spatial domain
        spatial = cv2.idft(dft, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)  # DFT_SCALE: scales the result, DFT_REAL_OUTPUT: returns only real values
        
        # Step 2: Crop the spatial result if the original size is provided
        if original_size:
            h, w = original_size  # Original image dimensions (height, width)
            spatial = spatial[:h, :w]  # Crop the result to its original size
        
        return spatial


    @staticmethod
    def show_dft(name, dft, save=False):
        """
        Imprime el DFT con shift y amplificacion logaritmica
        """
        fourier_shift = np.fft.fftshift(dft)
        magnitude = cv2.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1])
        
        # Apply logarithmic scaling for better visibility
        magnitude = 20 * np.log(magnitude + 1)  # +1 to avoid log(0)
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)    # CV_8UC1: grayscale (8b, 1 channel), NORM_MINMAX: normalizes between 0-255
        
        # Display the magnitude spectrum instead of the DFT itself
        Filter.show_image(name, magnitude, save)
