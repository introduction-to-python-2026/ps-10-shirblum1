import numpy as np
import matplotlib.pyplot as plt

from image_utils import load_image, edge_detection


# Load image
image = load_image("image.png")

# Edge detection
edges = edge_detection(image)

# Binary conversion
threshold = np.mean(edges)
binary_edges = edges > threshold

# Display results
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Edge Magnitude")
plt.imshow(edges, cmap='gray')
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Binary Edges")
plt.imshow(binary_edges, cmap='gray')
plt.axis("off")

plt.tight_layout()
plt.show()

# Save result
plt.imsave("edges.png", binary_edges, cmap='gray')
