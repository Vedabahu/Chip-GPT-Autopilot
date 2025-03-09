import cv2
import numpy as np
from picamera2 import Picamera2
import traceback
from time import sleep
import RPi.GPIO as GPIO

# import matplotlib.pyplot as plt

# GPIO Pin Setup
LEFT_MOTOR_FORWARD = 17
LEFT_MOTOR_BACKWARD = 18
RIGHT_MOTOR_FORWARD = 22
RIGHT_MOTOR_BACKWARD = 23
LEFT_MOTOR_ENABLE = 13
RIGHT_MOTOR_ENABLE = 12

# LEFT_MOTOR_FORWARD = 22
# LEFT_MOTOR_BACKWARD = 23
# RIGHT_MOTOR_FORWARD = 17
# RIGHT_MOTOR_BACKWARD = 18
# LEFT_MOTOR_ENABLE = 12
# RIGHT_MOTOR_ENABLE = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_MOTOR_FORWARD, GPIO.OUT)
GPIO.setup(LEFT_MOTOR_BACKWARD, GPIO.OUT)
GPIO.setup(LEFT_MOTOR_ENABLE, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_FORWARD, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_BACKWARD, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_ENABLE, GPIO.OUT)

left_pwm = GPIO.PWM(LEFT_MOTOR_ENABLE, 1000)
right_pwm = GPIO.PWM(RIGHT_MOTOR_ENABLE, 1000)
speed = 30
left_pwm.start(speed)
right_pwm.start(speed)

def timed_stop(time=0.5):
    sleep(time)
    stop()

def move_forward():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)
    timed_stop()

def move_backward():
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    timed_stop()

def turn_left():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)
    timed_stop(0.25)

def turn_right():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)
    timed_stop(0.25)

def stop():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)

# def calculate_lane_position(lines, image_width):
#     """
#     Determines the average x-position of the detected lane lines and calculates deviation.
#     """
#     if lines is None:
#         return None  # No lane detected
    
#     lane_positions = []
    
#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         lane_positions.append((x1 + x2) // 2)  # Average x-coordinate of the line

#     lane_center = np.mean(lane_positions)  # Average lane center
#     frame_center = image_width // 2  # Center of the frame

#     deviation = lane_center - frame_center  # Positive → Right, Negative → Left
#     return deviation

def calculate_lane_position(lines, image_width, estimated_lane_width=200):
    """
    Determines the deviation of the vehicle from the lane center.
    If both left and right lane markings are detected, the lane center is 
    computed as the average of their centers.
    If only one marking is detected, the lane center is estimated by offsetting 
    the marking by an expected lane width.
    
    Parameters:
        lines (ndarray): Detected lane lines (each line as [x1,y1,x2,y2]).
        image_width (int): Width of the frame.
        estimated_lane_width (int): Assumed width of the lane in pixels.
        
    Returns:
        deviation (float): Difference between estimated lane center and frame center.
                           Positive value means lane center is to the right.
    """
    if lines is None:
        return None  # No lane detected

    left_positions = []
    right_positions = []
    frame_center = image_width // 2

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        x_center = (x1 + x2) // 2
        # Classify lane marking based on its position relative to the frame center.
        if x_center < frame_center:
            left_positions.append(x_center)
        else:
            right_positions.append(x_center)

    # If both lane markings are detected, compute the lane center as the midpoint.
    if left_positions and right_positions:
        left_avg = np.mean(left_positions)
        right_avg = np.mean(right_positions)
        lane_center = (left_avg + right_avg) / 2
    # If only the left marking is detected, assume it's the left lane marking.
    elif left_positions:
        left_avg = np.mean(left_positions)
        lane_center = left_avg + estimated_lane_width / 2
    # If only the right marking is detected, assume it's the right lane marking.
    elif right_positions:
        right_avg = np.mean(right_positions)
        lane_center = right_avg - estimated_lane_width / 2
    else:
        return None

    deviation = lane_center - frame_center
    return deviation


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
    lower_bound = np.array([20, 100, 100])import cv2
import numpy as np
from picamera2 import Picamera2
import traceback
from time import sleep
import RPi.GPIO as GPIO

# import matplotlib.pyplot as plt

# GPIO Pin Setup
LEFT_MOTOR_FORWARD = 17
LEFT_MOTOR_BACKWARD = 18
RIGHT_MOTOR_FORWARD = 22
RIGHT_MOTOR_BACKWARD = 23
LEFT_MOTOR_ENABLE = 13
RIGHT_MOTOR_ENABLE = 12

# LEFT_MOTOR_FORWARD = 22
# LEFT_MOTOR_BACKWARD = 23
# RIGHT_MOTOR_FORWARD = 17
# RIGHT_MOTOR_BACKWARD = 18
# LEFT_MOTOR_ENABLE = 12
# RIGHT_MOTOR_ENABLE = 13

GPIO.setmode(GPIO.BCM)
GPIO.setup(LEFT_MOTOR_FORWARD, GPIO.OUT)
GPIO.setup(LEFT_MOTOR_BACKWARD, GPIO.OUT)
GPIO.setup(LEFT_MOTOR_ENABLE, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_FORWARD, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_BACKWARD, GPIO.OUT)
GPIO.setup(RIGHT_MOTOR_ENABLE, GPIO.OUT)

left_pwm = GPIO.PWM(LEFT_MOTOR_ENABLE, 1000)
right_pwm = GPIO.PWM(RIGHT_MOTOR_ENABLE, 1000)
speed = 30
left_pwm.start(speed)
right_pwm.start(speed)

def timed_stop(time=0.3):
    sleep(time)
    stop()

def move_forward():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)
    timed_stop()

def move_backward():
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    timed_stop(0.15)

def turn_left():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)
    timed_stop(0.25)

def turn_right():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.HIGH)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)
    timed_stop(0.25)

def stop():
    GPIO.output(LEFT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_FORWARD, GPIO.LOW)
    GPIO.output(LEFT_MOTOR_BACKWARD, GPIO.LOW)
    GPIO.output(RIGHT_MOTOR_BACKWARD, GPIO.LOW)

# def calculate_lane_position(lines, image_width):
#     """
#     Determines the average x-position of the detected lane lines and calculates deviation.
#     """
#     if lines is None:
#         return None  # No lane detected
    
#     lane_positions = []
    
#     for line in lines:
#         x1, y1, x2, y2 = line.reshape(4)
#         lane_positions.append((x1 + x2) // 2)  # Average x-coordinate of the line

#     lane_center = np.mean(lane_positions)  # Average lane center
#     frame_center = image_width // 2  # Center of the frame

#     deviation = lane_center - frame_center  # Positive → Right, Negative → Left
#     return deviation

def calculate_lane_position(lines, image_width, estimated_lane_width=200):
    """
    Determines the deviation of the vehicle from the lane center.
    If both left and right lane markings are detected, the lane center is 
    computed as the average of their centers.
    If only one marking is detected, the lane center is estimated by offsetting 
    the marking by an expected lane width.
    
    Parameters:
        lines (ndarray): Detected lane lines (each line as [x1,y1,x2,y2]).
        image_width (int): Width of the frame.
        estimated_lane_width (int): Assumed width of the lane in pixels.
        
    Returns:
        deviation (float): Difference between estimated lane center and frame center.
                           Positive value means lane center is to the right.
    """
    if lines is None:
        return None  # No lane detected

    left_positions = []
    right_positions = []
    frame_center = image_width // 2

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        x_center = (x1 + x2) // 2
        # Classify lane marking based on its position relative to the frame center.
        if x_center < frame_center:
            left_positions.append(x_center)
        else:
            right_positions.append(x_center)

    # If both lane markings are detected, compute the lane center as the midpoint.
    if left_positions and right_positions:
        left_avg = np.mean(left_positions)
        right_avg = np.mean(right_positions)
        lane_center = (left_avg + right_avg) / 2
    # If only the left marking is detected, assume it's the left lane marking.
    elif left_positions:
        left_avg = np.mean(left_positions)
        lane_center = (left_avg + estimated_lane_width) / 2
    # If only the right marking is detected, assume it's the right lane marking.
    elif right_positions:
        right_avg = np.mean(right_positions)
        lane_center = (right_avg - estimated_lane_width) / 2
    else:
        return None

    deviation = lane_center - frame_center
    return deviation


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
    # filtered = cv2.bitwise_and(image, image, mask=mask)
    filtered = None

    return filtered, mask


def region_of_interset(image):
    height, width = image.shape[0], image.shape[1]
    polygons = np.array([[(0, 250), (width, 250), (width, height), (0, height)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_space(canny_image):
    lines = cv2.HoughLinesP(
        canny_image, 1, np.pi / 180, threshold=80, minLineLength=100, maxLineGap=25
    )

    # line_image = np.zeros_like(canny_image)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line.reshape(4)
    #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
    line_image = None

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

# try:
#     while True:
#         image = picam2.capture_array()
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # canny_img = filter_hsv(image)
#         # cropped_image = region_of_interset(canny_img)
#         # final_img = hough_space(canny_img)

#         filtered, mask = filter_hsv(image)
#         cropped_image = region_of_interset(mask)
#         final_img, lines = hough_space(mask)

#         #TODO: Lines above will be used for steering

#         cv2.imshow('Lane detection', final_img)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         sleep(0.1)

# except Exception as e:
#     print(traceback.format_exc())


# finally:
#     cv2.destroyAllWindows()

# imS = cv2.resize(cropped_image, (960, 540))  # Resize image

# cv2.imshow("result", imS)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("result", imS)
# cv2.waitKey(0)

try:
    while True:
        image = picam2.capture_array()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        _, mask = filter_hsv(image)
#        cropped_image = region_of_interset(mask)
        _, lines = hough_space(mask)

        deviation = calculate_lane_position(lines, image.shape[1])

        if deviation is None:
            stop()  # No lanes detected → Stop
            move_backward()
            print("Stop")
        elif deviation < -30:
            turn_left()  # Turn left
            print("Left")
        elif abs(deviation) < 30:
            move_forward()  # Move straight
            print("Forward")
        elif deviation > 30:
            turn_right()  # Turn right
            print("Right")

        # cv2.imshow('Lane detection', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(0.1)

except Exception as e:
    print(traceback.format_exc())

finally:
    stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()

  # Example for yellow
    upper_bound = np.array([50, 255, 255])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    # filtered = cv2.bitwise_and(image, image, mask=mask)
    filtered = None

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

    # line_image = np.zeros_like(canny_image)
    # if lines is not None:
    #     for line in lines:
    #         x1, y1, x2, y2 = line.reshape(4)
    #         cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 5)
    line_image = None

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

# try:
#     while True:
#         image = picam2.capture_array()
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         # canny_img = filter_hsv(image)
#         # cropped_image = region_of_interset(canny_img)
#         # final_img = hough_space(canny_img)

#         filtered, mask = filter_hsv(image)
#         cropped_image = region_of_interset(mask)
#         final_img, lines = hough_space(mask)

#         #TODO: Lines above will be used for steering

#         cv2.imshow('Lane detection', final_img)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#         sleep(0.1)

# except Exception as e:
#     print(traceback.format_exc())


# finally:
#     cv2.destroyAllWindows()

# imS = cv2.resize(cropped_image, (960, 540))  # Resize image

# cv2.imshow("result", imS)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# cv2.imshow("result", imS)
# cv2.waitKey(0)

try:
    while True:
        image = picam2.capture_array()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        _, mask = filter_hsv(image)
        _, lines = hough_space(mask)

        deviation = calculate_lane_position(lines, image.shape[1])

        if deviation is None:
            stop()  # No lanes detected → Stop
            print("Stop")
        elif abs(deviation) < 30:
            move_forward()  # Move straight
            print("Forward")
        elif deviation < -30:
            turn_left()  # Turn left
            print("Left")
        elif deviation > 30:
            turn_right()  # Turn right
            print("Right")

        cv2.imshow('Lane detection', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        sleep(0.2)

except Exception as e:
    print(traceback.format_exc())

finally:
    stop()
    GPIO.cleanup()
    cv2.destroyAllWindows()

