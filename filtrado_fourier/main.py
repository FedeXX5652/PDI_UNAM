from filter import Filter
import os
import numpy as np

IMG = os.path.join(os.path.dirname(__file__), "img.jpg")

# Load and process the image
image = Filter.load_image(image_path=IMG, bw=True)
# noisy_image = Filter.add_noise(image, seed=25565)

# Get height and width of the image
original_height, original_width = image.shape

# Show original and noisy images
Filter.show_image("Original", image, save=True)
# Filter.show_image("Con ruido", noisy_image, save=True)

block_sizes = [11, 15]
block_filters = [Filter.block_lowpass_filter(size) for size in block_sizes]

# Non-padded version
dft_original = Filter.image2dft(image)
# dft_noisy = Filter.image2dft(noisy_image)
Filter.show_dft("Original DFT", dft_original, True)

# Compute the DFT of the filters for both versions
dft_filters = [Filter.filter2dft(Filter.zero_pad_filter(f, image.shape)) for f in block_filters]

# Apply filters and show results
for i, size in enumerate(block_sizes):
    """
    NON-PADDED
    """
    filtered_dft_original = Filter.apply_dft_filter(dft_original, dft_filters[i])
    # filtered_dft_noisy = Filter.apply_dft_filter(dft_noisy, dft_filters[i])
    
    filtered_original = Filter.dft2image(filtered_dft_original)
    # filtered_noisy = Filter.dft2image(filtered_dft_noisy)
    
    Filter.show_image(f"Filtered {size}x{size} original (non-padded)", filtered_original, True)
    # Filter.show_image(f"Filtered {size}x{size} noisy (non-padded)", filtered_noisy, True)
    Filter.show_dft(f"DFT {size}x{size} x original (non-padded)", filtered_dft_original, True)
    # Filter.show_dft(f"DFT {size}x{size} x noisy (non-padded)", filtered_dft_noisy, True)

    print(f"img size: {image.shape}")
    print(f"dft size: {dft_original.shape}")
    print(f"filter size: {filtered_original.shape}")
    print(f"dft filter size: {filtered_dft_original.shape}")

    """
    PADDED
    """
    # Pad the image based on the filter size
    filter_size = (size, size)  # Because we're using square filters
    image_padded = Filter.zero_pad_img(image, filter_size)
    Filter.show_image(f"Original Padded {size}x{size}", image_padded, True)
    
    dft_original_padded = Filter.image2dft(image_padded)
    Filter.show_dft(f"Original DFT Padded {size}x{size}", dft_original_padded, True)

    # The filter should now be the same size as the padded image
    padded_filter = Filter.zero_pad_filter(block_filters[i], image_padded.shape)
    dft_filter_padded = Filter.filter2dft(padded_filter)

    print(f"padded img size: {image_padded.shape}")
    print(f"padded dft size: {dft_original_padded.shape}")
    print(f"padded filter size: {padded_filter.shape}")
    print(f"padded dft filter size: {dft_filter_padded.shape}")

    filtered_dft_original_padded = Filter.apply_dft_filter(dft_original_padded, dft_filter_padded)
    
    filtered_original_padded = Filter.dft2image(filtered_dft_original_padded, original_size=image.shape)
    
    Filter.show_image(f"Filtered {size}x{size} original (padded)", filtered_original_padded, True)
    Filter.show_dft(f"DFT {size}x{size} x original (padded)", filtered_dft_original_padded, True)

    print(f"padded filtered img: {filtered_original_padded.shape}")
    print(f"padded filtered img dft: {filtered_dft_original_padded.shape}")