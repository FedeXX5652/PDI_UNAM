import os
from binomial import Binomial
import numpy as np
import cv2

def generate_square_filter(coefficients):
    filter_2d = np.outer(coefficients, coefficients)
    filter_2d = filter_2d / np.sum(filter_2d)
    return filter_2d

def apply_filter(image_path, filter_2d):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    filtered_image = cv2.filter2D(image, -1, filter_2d)
    
    return filtered_image

if __name__ == "__main__":
    binom_small = Binomial(5)
    print("Triángulo de Pascal para N=5:")
    print(binom_small.coefficients)

    binom_medium = Binomial(11)
    print("\nTriángulo de Pascal para N=11:")
    print(binom_medium.coefficients)

    binom_large = Binomial(31)
    print("\nTriángulo de Pascal para N=31:")
    print(binom_large.coefficients)

    filter_2d = generate_square_filter(binom_small.coefficients)
    print("\nFiltro cuadrado normalizado:")
    print(filter_2d)

    image_path = os.path.join(os.path.dirname(__file__), "img1.png")
    filtered_image = apply_filter(image_path, filter_2d)

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("Imagen Original", original_image)
    cv2.imshow("Imagen Filtrada", filtered_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()