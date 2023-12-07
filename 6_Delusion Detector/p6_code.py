import cv2
import numpy as np
from skimage.feature import peak_local_max

def Helpr(image):
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, mask_10 = cv2.threshold(grayscale, 240, 255, cv2.THRESH_BINARY_INV)
    image_0 = np.zeros_like(grayscale)
    image_0[mask_10 != 0] = grayscale[mask_10 != 0]
    hist = np.sum(image_0, axis=0)
    peaks = peak_local_max(hist, min_distance=12)
    return hist, sorted(peaks)

def solution(image_path):
    image = cv2.imread(image_path)
    hist, heads = Helpr(image)
    head_4th= heads[3][0]
    head_6th= heads[6][0]
    if (len(heads) != 10):
        return 'fake'
    elif (hist[head_4th] < hist[head_6th]):
        return 'fake'
    return 'real'