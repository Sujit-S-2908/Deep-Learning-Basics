import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, misc
from skimage import data, color

# Load an example image
image = data.camera()
#image = color.rgb2gray(image)

# Define the horizontal Sobel filter
sobel_horizontal = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Convolve the image with the horizontal Sobel filter
filtered_image = ndimage.convolve(image, sobel_horizontal)

# Plot the original and filtered images
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Filtered Image (Horizontal Sobel)')
plt.imshow(filtered_image, cmap='gray')
plt.axis('off')

plt.show()
