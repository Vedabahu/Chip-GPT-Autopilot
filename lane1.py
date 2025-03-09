import cv2
import numpy as np
import matplotlib.pyplot as plt


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    plt.imshow(blur)
    canny = cv2.Canny(blur, 100, 320)
    return canny


def region_of_interset(image):
    height, width = image.shape[0], image.shape[1]
    polygons = np.array([[(0, 600), (width, 600), (width, 1300), (0, 1300)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_space(canny_image):
    lines = cv2.HoughLinesP(
        canny_image, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150
    )

    line_image = np.zeros_like(canny_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)

    return line_image


image = cv2.imread("sample.jpg")
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interset(canny)
final_img = hough_space(cropped_image)
# imS = cv2.resize(cropped_image, (960, 540))  # Resize image

plt.imshow(final_img)
plt.show()

# cv2.imshow("result", imS)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("result", imS)
# cv2.waitKey(0)
