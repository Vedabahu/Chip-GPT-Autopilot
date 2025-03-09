import cv2
import numpy as np
from picam2 import Picamera2
import traceback
from time import sleep
# import matplotlib.pyplot as plt


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    # plt.imshow(blur)
    canny = cv2.Canny(blur, 100, 270)
    # canny = cv2.Canny(blur, 100, 250)
    return canny


def filter_hsv(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color range (Replace with actual values)
    lower_bound = np.array([20, 100, 100])  # Example for yellow
    upper_bound = np.array([50, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    filtered = cv2.bitwise_and(image, image, mask=mask)

    return filtered, mask


def region_of_interset(image):
    height, width = image.shape[0], image.shape[1]
    polygons = np.array([[(0, 300), (width, 300), (width, height), (0, height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_space(canny_image):
    lines = cv2.HoughLinesP(
        canny_image, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=25
    )

    line_image = np.zeros_like(canny_image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)

    return line_image, lines


# image = cv2.imread("sample.jpg")

# cap = cv2.VideoCapture("./videos/dataset_5.mp4")

picam2 = Picamera2()
video_config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(video_config)
picam2.start()

# canny_img = canny(frame)
# cropped_image = region_of_interset(canny_img)
# final_img = hough_space(canny_img)

try:
    while True:
        image = picam2.capture_array()

        # canny_img = filter_hsv(image)
        # cropped_image = region_of_interset(canny_img)
        # final_img = hough_space(canny_img)

        filtered, mask = filter_hsv(image)
        cropped_image = region_of_interset(mask)
        final_img, lines = hough_space(mask)

        #TODO: Lines above will be used for steering

        cv2.imshow('Lane detection', final_img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(traceback.format_exc())


finally:
    cv2.destroyAllWindows()

# imS = cv2.resize(cropped_image, (960, 540))  # Resize image

# cv2.imshow("result", imS)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("result", imS)
# cv2.waitKey(0)
