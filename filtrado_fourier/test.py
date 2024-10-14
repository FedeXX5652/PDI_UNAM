from filter import Filter
import os
import numpy as np

IMG = os.path.join(os.path.dirname(__file__), "img.jpg")
FILTER = 11

image = Filter.load_image(image_path=IMG, bw=True)
original_height, original_width = image.shape
Filter.show_image("Img", image, save=True)
img_dft = Filter.image2dft(image)
Filter.show_dft("Img DFT", img_dft, True)

print(f"img: {image.shape}")
print(f"dft: {img_dft.shape}")

block_filter = Filter.block_lowpass_filter(FILTER)
dft_filter = Filter.filter2dft(Filter.zero_pad_filter(block_filter, image.shape))
Filter.show_dft(f"TEST dft filter {FILTER}x{FILTER}", dft_filter)
print(f"filter: {block_filter.shape}")
print(f"dft filter: {dft_filter.shape}")

"""
NON-PADDED IMG
"""
filtered_dft = Filter.apply_dft_filter(img_dft, dft_filter)
filtered_img = Filter.dft2image(filtered_dft)
Filter.show_image(f"TEST Filtered {FILTER}x{FILTER} original (non-padded)", filtered_img, True)
Filter.show_dft(f"TEST DFT {FILTER}x{FILTER} x original (non-padded)", filtered_dft, True)
print(f"filtered img: {filtered_img.shape}")
print(f"filtered dft: {filtered_dft.shape}")

"""
PADDED IMG
"""
padded_img = Filter.zero_pad_img(image, (FILTER, FILTER))
print(f"padded img: {padded_img.shape}")
padded_img_dft = Filter.image2dft(padded_img)
print(f"padded img dft: {padded_img.shape}")
repadded_filter = Filter.zero_pad_filter(block_filter, padded_img.shape)
print(f"re-padded filter: {repadded_filter.shape}")
repadded_filter_dft = Filter.filter2dft(repadded_filter)
Filter.show_dft(f"re-padded filter dft {FILTER}x{FILTER}", repadded_filter_dft)
print(f"re-padded filter dft: {repadded_filter_dft.shape}")

filtered_dft = Filter.apply_dft_filter(padded_img_dft, repadded_filter_dft)
print(f"filtered padded dft: {filtered_dft.shape}")
filtered_img = Filter.dft2image(filtered_dft, image.shape)
print(f"filtered padded img: {filtered_img.shape}")
Filter.show_image(f"TEST Filtered {FILTER}x{FILTER} original (padded)", filtered_img, True)
Filter.show_dft(f"TEST DFT {FILTER}x{FILTER} x original (padded)", filtered_dft, True)
