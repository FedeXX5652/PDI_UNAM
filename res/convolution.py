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

def convolve2d(image, kernel):
    i_height, i_width = image.shape
    k_height, k_width = kernel.shape
    o_height = i_height + k_height - 1
    o_width = i_width + k_width - 1
    output = np.zeros((o_height, o_width))
    
    # Iterate from top-left to bottom-right
    for i in range(o_height):
        for j in range(o_width):
            print_matrices(output, i, j)
            
            sum = 0
            for m in range(k_height):
                for n in range(k_width):
                    ii = i - m
                    jj = j - n
                    if 0 <= ii < i_height and 0 <= jj < i_width:
                        product = image[ii, jj] * kernel[m, n]
                        sum += product
                        print(f"  f[{ii},{jj}] * h[{m},{n}] = {image[ii, jj]} * {kernel[m, n]} = {product}")
            
            output[i, j] = sum
    
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

    result = convolve2d(image, kernel)

    print("\nMatriz final:")
    print(result.astype(int))
    print("\nDimension final:", result.shape)
sys.stdout = sys.__stdout__