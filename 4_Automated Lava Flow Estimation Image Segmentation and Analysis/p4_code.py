import cv2
import numpy as np

def lava_segmentation(input_image):
    bgr_image = input_image
    lava_color = np.array([0, 165, 255])  # Orangish color
    distance = np.linalg.norm(bgr_image - lava_color, axis=-1)
    threshold = 150
    lava_mask = (distance < threshold).astype(np.uint8) * 255
    return lava_mask

def refine_segmentation(segmentation_mask, blur_kernel_size=5):  #5 initially
    refined_mask = cv2.medianBlur(segmentation_mask, blur_kernel_size)

    return refined_mask


def post_process_mask(segmentation_mask, min_contour_area=20000):
    contours, _ = cv2.findContours(segmentation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            cv2.drawContours(segmentation_mask, [contour], -1, 0, thickness=cv2.FILLED)

    return segmentation_mask

def fill_internal_holes(mask, closing_kernel_size=15, dilation_kernel_size=10, min_hole_size=5):
    
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((closing_kernel_size, closing_kernel_size), np.uint8))

    dilated_mask = cv2.dilate(closed_mask, np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8), iterations=1)

    contours, _ = cv2.findContours(dilated_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_hole_size:
            cv2.drawContours(dilated_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return dilated_mask

def remove_thin_connections(mask, closing_kernel_size=15):
 
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((closing_kernel_size, closing_kernel_size), np.uint8))

    return closed_mask

def fill_and_remove(mask, closing_kernel_size=15, opening_kernel_size=25, min_hole_size=5):

    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((closing_kernel_size, closing_kernel_size), np.uint8))

  
    opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, np.ones((opening_kernel_size, opening_kernel_size), np.uint8))

    contours, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_hole_size:
            cv2.drawContours(opened_mask, [contour], -1, 255, thickness=cv2.FILLED)

    return opened_mask



def solution(image_path):
    input_image = cv2.imread(image_path)
    lava_mask = lava_segmentation(input_image)
    refined_mask = refine_segmentation(lava_mask)
    processed_mask = post_process_mask(refined_mask)
    noholes = fill_internal_holes(processed_mask)
    edgesremoved=remove_thin_connections(noholes)
    final=fill_internal_holes(edgesremoved)
    final1= fill_and_remove(final)
    o1= np.stack([final1] * 3, axis=-1)
    return o1