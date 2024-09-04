import sys
import numpy as np
import time
import os
from tee import Tee

def print_matrices(output, current_i, current_j):
    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            if i == current_i and j == current_j:
                print("{:4d}".format(int(output[i, j])), end="")  # Red for current position
            else:
                print("{:4d}".format(int(output[i, j])), end="")
        print()
    print(f"\nPosicion: [{current_i}, {current_j}]")

def convolve2d_animated(image, kernel):
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    o_height = i_height + k_height - 1
    o_width = i_width + k_width - 1
    output = np.zeros((o_height, o_width))
    
    kernel_inverted = np.flip(kernel)
    
    for i in range(o_height - 1, -1, -1):
        for j in range(o_width - 1, -1, -1):
            print_matrices(output, i, j)
            
            sum = 0
            for m in range(k_height):
                for n in range(k_width):
                    if i-m >= 0 and j-n >= 0 and i-m < i_height and j-n < i_width:
                        product = image[i-m, j-n] * kernel_inverted[m, n]
                        sum += product
                        print(f"  f[{i-m},{j-n}] * h[{m},{n}] = {image[i-m, j-n]} * {kernel_inverted[m, n]} = {product}")
            
            output[i, j] = sum
            print(f"Sumatoria para [{i},{j}] = {sum}\n")
    
    return output

# Example usage
image = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

kernel = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 8]
])

with open("output.txt", "w", encoding='utf-8') as f:
    tee = Tee(sys.stdout, f)
    sys.stdout = tee

    result = convolve2d_animated(image, kernel)

    print("\nMatriz final:")
    print(result.astype(int))
    print("\nDimension final:", result.shape)
sys.stdout = sys.__stdout__