import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(image):
    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(3,3),0)
    canny = cv2.Canny(blur,60,320)
    return canny

def region_of_interset(image):
    height = image.shape[0]
    polygons = np.array([[(0,height),(1000,height),(400,700)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

image = cv2.imread('sample.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interset(canny)
imS = cv2.resize(cropped_image, (960, 540))                # Resize image
# cv2.imshow("result", imS)
# cv2.waitKey(0)

# cv2.imshow("result", imS)
# cv2.waitKey(0)
