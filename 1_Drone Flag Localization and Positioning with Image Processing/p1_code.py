import cv2
import numpy as np

 
def color_distance(color1, color2):
    # Calculate the Euclidean distance between two colors
    return np.linalg.norm(color1 - color2)   
    

                
#     return image
def sharpen_image(image):
    # Define the sharpening kernel
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
   
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image

def sort_points_clockwise(points):
    center = np.mean(points, axis=0)
    return np.array(sorted(points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0])))

def trans(image):

    original=image.copy()
    original=cv2.resize(original,(600,600))
 
    border_size = 10

    border_color = (0, 0, 0)

    # Add a black border to the image
    image = cv2.copyMakeBorder(image, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=border_color)

    # Resize the image for processing
    image = cv2.resize(image, (600, 600))

    # Define the dimensions of the output image (warped)
    output_size = (600, 600)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (9, 9), 0)

    # Apply Canny edge detection
    edged = cv2.Canny(blur, 30, 50)

    # Apply adaptive thresholding
    thresholded = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=5)

    # Find contours and identify the document contour
    contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.015 * peri, True)
            if area > max_area and len(approx) == 4:
                document_contour = approx
                max_area = area


    # Reshape and sort document contour points
    document_contour = document_contour.reshape(4, 2)
    document_contour = np.float32(document_contour)
    document_contour = sort_points_clockwise(document_contour)

    # Calculate the perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(document_contour, np.array([[0, 0], [output_size[0], 0], [output_size[0], output_size[1]], [0, output_size[1]]], dtype=np.float32))

    # Apply the perspective transformation to the image
    warped = cv2.warpPerspective(image, matrix, output_size)
    # Define the size of the added border (e.g., 10 pixels)
    border_size = 5

    # Get the dimensions of the original image without the border
    original_height, original_width = warped.shape[:2]

    # Crop the image to remove the border
    warped =  warped[border_size:original_height-border_size, border_size:original_width-border_size]

    warped=cv2.resize(warped,(600,600))

    warped_sharpened=sharpen_image(warped)
    return warped_sharpened

def solution(image_path):
    width, height = 600, 600
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background (BGR format)

# Define colors
    orange = (51, 153, 255)   # BGR value for orange
    green = (0, 128, 0)     # BGR value for green
    navy = (255,0, 0 )      # BGR value for navy blue
    white = (255, 255, 255) # BGR value for white

# Draw orange rectangle
    cv2.rectangle(image, (0, 0), (600, 200), orange, -1)

# Draw white rectangle
    cv2.rectangle(image, (0, 200), (600, 400), white, -1)

# Draw big blue circle
    cv2.circle(image, (300, 300),100, navy, -1)

# Draw big white circle
    cv2.circle(image, (300, 300),98, white, -1)

    angle = 0

# Draw lines at different angles using a while loop
    while angle < 360:
        x1 = int(300 + 99 * np.cos(np.radians(angle)))
        y1 = int(300 + 99 * np.sin(np.radians(angle)))
        x2 = 300
        y2 = 300
        cv2.line(image, (x1, y1), (x2, y2), navy, 1)
    
    # Increment the angle
        angle += 15
# Draw green rectangle
    cv2.rectangle(image, (0, 400), (600,600), green, -1)
    strtimg=image

# Rotate the image 90 degrees clockwise
    lftgreen = cv2.transpose(image)
    lftgreen = cv2.flip(lftgreen, flipCode=1)

# Rotate the image 90 degrees clockwise
    lftorange = cv2.transpose(image)
    lftorange = cv2.flip(lftorange, flipCode=0)

# Rotate the image 180 degrees
    invimg = cv2.flip(image, flipCode=-1)
 
    # cv2_imshow( image)
    # Load the input image
    input_image = cv2.imread(image_path)
    input=trans(input_image)

    # specific_color = np.array([255, 165, 0])  # Replace with your specific color
    orange = np.array([51, 153, 255])   # BGR value for orange
    green = np.array([0, 128, 0])   # BGR value for green
    navy = np.array([255,0, 0 ])      # BGR value for navy blue
    white = np.array([255, 255, 255]) # BGR value for white
# distance1 = color_distance(color_at_point, specific_color)

# point 1
    x1,y1=300,100
    x2,y2=100,300

    color_at_point1 = input[y1, x1]
    color_at_point2 = input[y2, x2]
    distance1withorange = color_distance(color_at_point1, orange)
    distance1withwhite = color_distance(color_at_point1, white)
    distance1withgreen = color_distance(color_at_point1, green)

    distance2withorange = color_distance(color_at_point2, orange)
    distance2withwhite = color_distance(color_at_point2, white)
    distance2withgreen = color_distance(color_at_point2, green)
    distance1withorange = color_distance(color_at_point1, orange)
    distance1withwhite = color_distance(color_at_point1, white)
    distance1withgreen = color_distance(color_at_point1, green)

    distance2withorange = color_distance(color_at_point2, orange)
    distance2withwhite = color_distance(color_at_point2, white)
    distance2withgreen = color_distance(color_at_point2, green)

    threshold = 50

    output_image=image

    if distance1withorange <= threshold:
        output_image= strtimg

    elif distance1withgreen <= threshold:
        output_image= invimg

    elif distance1withwhite <= threshold:
  # 2 cases
        if distance2withorange <= threshold:
            output_image= lftorange
        elif distance2withgreen <= threshold:
            output_image= lftgreen

    return output_image
