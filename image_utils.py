from PIL import Image
import numpy as np
from scipy.signal import convolve2d


def load_image(path):
    """
    Loads an image from the given path and returns it as a numpy array.
    """
    img = Image.open(path)
    img_array = np.array(img)
    return img_array


def edge_detection(image):
    """
    Performs edge detection using Sobel filters.
    """

    # Convert to grayscale if image is RGB
    if image.ndim == 3:
        image = np.mean(image, axis=2)

    # Sobel kernels
    kernelY = np.array([[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]])

    kernelX = np.array([[ 1,  0, -1],
                        [ 2,  0, -2],
                        [ 1,  0, -1]])

    # Convolution
    edgeY = convolve2d(image, kernelY, mode='same',
                       boundary='fill', fillvalue=0)

    edgeX = convolve2d(image, kernelX, mode='same',
                       boundary='fill', fillvalue=0)

    # Edge magnitude
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)

    return edgeMAG
