from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    load_image = mpimg.imread('my_image.jpeg')
    plt.imshow(load_image)
    plt.imshow(load_image)
    plt.axis('off')
    plt.show()

def edge_detection(image):
    from scipy import signal
    grayscale_image = np.mean(mpimg.imread('/content/my_image.jpeg'), axis=2)
      kernelY = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
        ])
      kernelX = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
        ])
      edgeY = signal.convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
      edgeX = signal.convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)
      edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
      return edgeMAG

    edges = edge_detection('/content/my_image.jpeg')
    plt.imshow(edges, cmap='gray')
