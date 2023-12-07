import cv2
import numpy as np



def pad_image(img, spatial_kern):
    rows, cols, channels = img.shape
    padded_img = np.pad(img, ((spatial_kern, spatial_kern), (spatial_kern, spatial_kern), (0, 0)), mode='symmetric')
    padded_img[:, :spatial_kern, :] = np.flip(padded_img[:, spatial_kern:2*spatial_kern, :], axis=1)
    padded_img[:, cols+spatial_kern:, :] = np.flip(padded_img[:, cols-2*spatial_kern:cols-spatial_kern, :], axis=1)

    return padded_img


def gaussian(img, spatial_sigma, range_sigma):
    spatial_gaussian = 1 / np.sqrt(2 * np.pi * (spatial_sigma**2))
    range_gaussian = 1 / np.sqrt(2 * np.pi * (range_sigma**2))

    intensity_values = np.arange(256)
    matrix = np.exp(-intensity_values * intensity_values * range_gaussian)

    spatial_range = np.arange(2 * spatial_sigma + 1)
    xx, yy = np.meshgrid(-spatial_sigma + spatial_range, -spatial_sigma + spatial_range)

    spatial_kernel = spatial_gaussian * np.exp(-(xx**2 + yy**2) / (2 * (spatial_gaussian**2)))

    return matrix, spatial_kernel




def joint_bilateral_filter(no_flash_img, flash_img, spatial_sigma, range_sigma):
    height, width, channels = no_flash_img.shape
    padded_no_flash_img = pad_image(no_flash_img, spatial_sigma)
    padded_flash_img = pad_image(flash_img, spatial_sigma)
    matrix, spatial_kernel = gaussian(no_flash_img, spatial_sigma, range_sigma)

    filtered_img = np.zeros((height, width, channels), dtype=np.uint8)

    for x in range(spatial_sigma, spatial_sigma + height):
        for y in range(spatial_sigma, spatial_sigma + width):
            for c in range(channels):
                neighborhood = padded_flash_img[x-spatial_sigma:x+spatial_sigma+1, y-spatial_sigma:y+spatial_sigma+1, c]
                central = padded_flash_img[x, y, c]
                weights = matrix[np.abs(neighborhood - central)]
                normalization = np.sum(weights)
                filtered_img[x-spatial_sigma, y-spatial_sigma, c] = np.sum(weights * padded_no_flash_img[x-spatial_sigma:x+spatial_sigma+1, y-spatial_sigma:y+spatial_sigma+1, c]) / normalization

    return filtered_img



def enhance_edges(img, strength=0.45):
    kernel = np.array([[0, -1, 0],
                       [-1, strength +1 , -1],
                       [0, -1, 0]])
    enhanced_edges = np.zeros_like(img)
    for i in range(img.shape[2]):
        enhanced_edges[:, :, i] = np.clip(np.convolve(img[:, :, i].ravel(), kernel.ravel(), mode='same').reshape(img.shape[:2]), 0, 255)

    enhanced_edges = enhanced_edges.astype(np.uint8)
    result_image = np.clip(img + enhanced_edges, 0, 255).astype(np.uint8)
    return result_image

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    image1 = cv2.imread(image_path_a)  #non-flash high ISO image
    image2 = cv2.imread(image_path_b)  #flash low ISO image

    spatial_Kernel = 24 # 24 found better
    range_Kernel =10  # 10 found better
    image= joint_bilateral_filter(image1,image2,spatial_Kernel,range_Kernel)
    processed_img=enhance_edges(image)
    return processed_img
