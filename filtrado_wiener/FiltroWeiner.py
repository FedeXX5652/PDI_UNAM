import numpy as np
import cv2
from scipy import fftpack
import matplotlib.pyplot as plt
import os

def add_gaussian_noise(image, mean=0, variance=0.01):
    """Agregar ruido gaussiano a una imagen."""
    row, col = image.shape
    sigma = np.sqrt(variance)
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss * 255
    return np.clip(noisy, 0, 255).astype(np.uint8)

def create_lowpass_filter(size=9):
    """Crear un filtro paso bajas normalizado de tamaño size x size."""
    kernel = np.ones((size, size)) / (size * size)
    return kernel

def degradation_function(image, kernel):
    """Aplicar la función de degradación en el dominio de la frecuencia con shift correcto."""
    image_float = image.astype(float) / 255.0
    kernel_padded = np.zeros_like(image_float)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    kernel_padded = np.roll(kernel_padded, -(kh//2), axis=0)
    kernel_padded = np.roll(kernel_padded, -(kw//2), axis=1)
    image_fft = fftpack.fft2(image_float)
    kernel_fft = fftpack.fft2(kernel_padded)
    degraded_fft = image_fft * kernel_fft
    degraded = np.real(fftpack.ifft2(degraded_fft))
    return np.clip(degraded * 255, 0, 255).astype(np.uint8)

def wiener_filter_restore(degraded_image, kernel, K=0.02):
    """Restauración usando el filtro de Wiener con manejo correcto del shift."""
    image_float = degraded_image.astype(float) / 255.0
    kernel_padded = np.zeros_like(image_float)
    kh, kw = kernel.shape
    kernel_padded[:kh, :kw] = kernel
    kernel_padded = np.roll(kernel_padded, -(kh//2), axis=0)
    kernel_padded = np.roll(kernel_padded, -(kw//2), axis=1)
    degraded_fft = fftpack.fft2(image_float)
    kernel_fft = fftpack.fft2(kernel_padded)
    kernel_power_spectrum = np.abs(kernel_fft) ** 2
    wiener = np.conj(kernel_fft) / (kernel_power_spectrum + K)
    restored_fft = degraded_fft * wiener
    restored = np.real(fftpack.ifft2(restored_fft))
    return np.clip(restored * 255, 0, 255).astype(np.uint8), kernel_power_spectrum

# Cargar imagen
image = cv2.imread('Glaciar512.jpg', cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("No se pudo cargar la imagen")

# Parámetros
NOISE_MEAN = 0
NOISE_VAR = 0.01
K_CASE3 = 0.005
K_CASE4 = 0.03
LOWPASS_KERNEL_SIZE_CASE3 = 9
LOWPASS_KERNEL_SIZE_CASE4 = 9

# Crear filtros paso bajas específicos para cada caso
lowpass_kernel_case3 = create_lowpass_filter(LOWPASS_KERNEL_SIZE_CASE3)
lowpass_kernel_case4 = create_lowpass_filter(LOWPASS_KERNEL_SIZE_CASE4)

# Caso 3: Ruido gaussiano + pérdida de nitidez
print("Procesando Caso 3: Ruido gaussiano + pérdida de nitidez")
noisy_image = add_gaussian_noise(image, NOISE_MEAN, NOISE_VAR)
degraded_case3 = degradation_function(noisy_image, lowpass_kernel_case3)
restored_case3, kernel_power_spectrum_case3 = wiener_filter_restore(degraded_case3, lowpass_kernel_case3, K_CASE3)

# Caso 4: Pérdida de nitidez + ruido gaussiano (con ajustes adicionales)
print("Procesando Caso 4: Pérdida de nitidez + ruido gaussiano con ajustes")
blurred_image = degradation_function(image, lowpass_kernel_case4)
degraded_case4 = add_gaussian_noise(blurred_image, NOISE_MEAN, NOISE_VAR)
restored_case4, kernel_power_spectrum_case4 = wiener_filter_restore(degraded_case4, lowpass_kernel_case4, K_CASE4)

# Guardar imágenes
output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

cv2.imwrite(os.path.join(output_dir, 'original_image.jpg'), image)
cv2.imwrite(os.path.join(output_dir, 'case3_degraded.jpg'), degraded_case3)
cv2.imwrite(os.path.join(output_dir, 'case3_restored.jpg'), restored_case3)
cv2.imwrite(os.path.join(output_dir, 'case4_degraded.jpg'), degraded_case4)
cv2.imwrite(os.path.join(output_dir, 'case4_restored.jpg'), restored_case4)

# Guardar FFT de imágenes como imágenes JPG
def save_fft_as_image(fft_data, filename):
    """Guardar el espectro FFT como imagen JPG."""
    # Calcular la magnitud y escalar
    fft_magnitude = np.abs(fft_data)
    # Escalar a 0-255 para guardar como imagen
    fft_scaled = np.log1p(fft_magnitude)  # Usar log para mejorar la visualización
    fft_scaled = (fft_scaled / np.max(fft_scaled) * 255).astype(np.uint8)
    plt.imsave(filename, fft_scaled, cmap='gray')

# Guardar FFT de las imágenes y kernels como imágenes JPG
save_fft_as_image(fftpack.fft2(degraded_case3), os.path.join(output_dir, 'case3_degraded_fft.jpg'))
save_fft_as_image(fftpack.fft2(restored_case3), os.path.join(output_dir, 'case3_restored_fft.jpg'))
save_fft_as_image(fftpack.fft2(degraded_case4), os.path.join(output_dir, 'case4_degraded_fft.jpg'))
save_fft_as_image(fftpack.fft2(restored_case4), os.path.join(output_dir, 'case4_restored_fft.jpg'))
save_fft_as_image(fftpack.fft2(lowpass_kernel_case3), os.path.join(output_dir, 'case3_kernel_fft.jpg'))
save_fft_as_image(fftpack.fft2(lowpass_kernel_case4), os.path.join(output_dir, 'case4_kernel_fft.jpg'))

# Visualización
plt.figure(figsize=(15, 10))

plt.subplot(231)
plt.imshow(image, cmap='gray')
plt.title('Imagen Original')

plt.subplot(232)
plt.imshow(degraded_case3, cmap='gray')
plt.title('Caso 3: Degradada\n(Ruido + Desenfoque)')

plt.subplot(233)
plt.imshow(restored_case3, cmap='gray')
plt.title('Caso 3: Restaurada')

plt.subplot(235)
plt.imshow(degraded_case4, cmap='gray')
plt.title('Caso 4: Degradada\n(Desenfoque + Ruido)')

plt.subplot(236)
plt.imshow(restored_case4, cmap='gray')
plt.title('Caso 4: Restaurada con ajustes')

plt.tight_layout()
plt.show()
