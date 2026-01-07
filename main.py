from image_utils import load_image, edge_detection
from PIL import Image
from skimage.filters import median
from skimage.morphology import ball
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy import signal

original_image = load_image('my_image.jpeg')
clean_image = median(original_image, ball(3))
edge_detection = edge_detection(clean_image)
plt.imshow(edge_detection, cmap='gray')
plt.axis('off')
plt.show()
