import numpy as np
import matplotlib.image as mpimg
from scipy.signal import convolve2d

#LOAD_IMAGE
def load_image(filename):
  img = mpimg.imread(filename)
  return img
  
#EDGE_DETECTION
def edge_detection(image_array):
  grayscale_image = np.mean(image_array, axis=2)
  kernelY = np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]])
  kernelX = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
  edgeY = convolve2d(grayscale_image, kernelY, mode='same', boundary='fill', fillvalue=0)
  edgeX = convolve2d(grayscale_image, kernelX, mode='same', boundary='fill', fillvalue=0)
  edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
  return edgeMAG
