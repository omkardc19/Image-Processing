import cv2
import numpy as np
import os
from skimage import transform


def add_padding(image, top_padding, bottom_padding, left_padding, right_padding, color=(255, 255, 255)):
    # Get the original image dimensions
    height, width = image.shape[:2]

    # Create a new larger image with padding
    new_height = height + top_padding + bottom_padding
    new_width = width + left_padding + right_padding

    # Initialize the new image with the specified color
    padded_image = np.full((new_height, new_width, 3), color, dtype=np.uint8)

    # Copy the original image into the center of the padded image
    padded_image[top_padding:top_padding + height, left_padding:left_padding + width] = image

    return padded_image

def helpr(img):

    # Add padding
    img = add_padding(img, top_padding=100, bottom_padding=100, left_padding=100, right_padding=100)
  
    I = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = I.shape

    # Resize if needed
    if w > 640:
        I = cv2.resize(I, (640, int((h / w) * 640)))

    # Demean
    I = I - np.mean(I)

    # Do the radon transform
    sinogram = transform.radon(I)

    # Find the RMS value of each row and find the "busiest" rotation
    r = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    rotation = np.argmax(r)
    

    if (90 - rotation) >= 0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 90 - rotation, 1)
    elif (90 - rotation) < 0:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), -90 - (rotation), 1)

    # Get the size of the resulting rotated image
    rotated_height, rotated_width = cv2.warpAffine(img, M, (img.shape[1], img.shape[0])).shape[:2]

    # Create a white canvas of the same size
    white_canvas = np.ones((rotated_height, rotated_width, 3), dtype=np.uint8) * 255

    # Rotate and place the skewed image onto the white canvas
    dst = cv2.warpAffine(img, M, (rotated_width, rotated_height), borderValue=(255, 255, 255))


    return dst

def orientation(img): 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _,img = cv2.threshold(img,200,255,cv2.THRESH_BINARY_INV)
    sum_along_rows = np.sum(img, axis=1)
    rows = [i for i, value in enumerate(sum_along_rows) if value > 0]
    row_1 = rows[0]
    row_last = rows[-1]
    half = (rows[0] + rows[-1])//2
    tot_sum=0
    for i in sum_along_rows:
        tot_sum+= i
    sum_dash=0
    for i in range(rows[0],half+1):
        sum_dash+= sum_along_rows[i]
    
    if sum_dash < tot_sum/2:
        return False
        # inv
    return True  
    # strt


def solution(image_path):
    img = cv2.imread(image_path)
  
    img=helpr(img)
    rotated_image = cv2.rotate(img, cv2.ROTATE_180)
    b=orientation(img)
    if (b==False) :
        img= rotated_image
 
    filename_without_extension = os.path.splitext(os.path.basename(image_path))[0]
    output_path = f"{filename_without_extension}_output.jpg"
    cv2.imwrite(output_path, img)
    return img