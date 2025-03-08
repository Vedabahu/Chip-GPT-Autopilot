import RPi.GPIO as GPIO
import cv2
import numpy as np
# import curses
# from pynput import keyboard
from time import sleep
from picamera2 import Picamera2
from libcamera import Transform
import traceback

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
    # sleep(time)
    # stop()
    pass

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

def process_frame(frame):
    """Process the frame to detect yellow lane lines."""
    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # #b39e24

    # Define the yellow color range in HSV
    lower_yellow = np.array([45, 50, 30])  # Adjusted for darker yellow
    upper_yellow = np.array([60, 255, 255])  # Keep upper limit wide

    # lower_yellow = np.array([80, 150, 150])  # Adjusted for darker yellow
    # upper_yellow = np.array([100, 180, 200])  # Keep upper limit wide


    # Create a mask for yellow color
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Apply Gaussian blur to the mask
    blurred = cv2.GaussianBlur(mask, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 10, 100)

    # Define a region of interest (ROI) polygon
    height, width = edges.shape
    roi = np.array([[
        (0, height),
        (width // 2 - 50, height // 2 + 50),
        (width // 2 + 50, height // 2 + 50),
        (width, height)
    ]], dtype=np.int32)

    # Create a mask for the ROI and apply it to the edges image
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Perform Hough Line Transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=150)

    line_image = np.copy(frame)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)

    return line_image, lines


def compute_steering_angle(lines, width):
    """Compute the steering angle based on detected lane lines."""
    if lines is None:
        return None

    left_lines = []
    right_lines = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if slope < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if not left_lines or not right_lines:
        return None

    left_line = np.mean(left_lines, axis=0).astype(int)
    right_line = np.mean(right_lines, axis=0).astype(int)

    left_x2 = left_line[0][2]
    right_x2 = right_line[0][2]

    lane_center = (left_x2 + right_x2) / 2
    frame_center = width / 2

    steering_angle = (lane_center - frame_center) / frame_center

    return steering_angle

def main():
    # Initialize the camera
    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    picam2 = Picamera2()
    video_config = picam2.create_preview_configuration(main={"size": (640, 480)}, transform=Transform(vflip=True))
    picam2.configure(video_config)
    picam2.start()

    try:
        while True:
            # ret, frame = cap.read()
            # if not ret:
            #     break

            frame = picam2.capture_array()

            # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            processed_frame, lines = process_frame(frame)
            steering_angle = compute_steering_angle(lines, frame.shape[1])

            if steering_angle is None:
                # stop()
                print("stop")
            elif steering_angle < -0.1:
                # turn_left()
                print("turn_left")
            elif steering_angle > 0.1:
                # turn_right()
                print("turn_right")
            else:
                # move_forward()
                print("move_forward")

            cv2.imshow('Lane Detection', processed_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            sleep(0.1)
    except Exception as e:
        print(traceback.format_exc())

    finally:
        stop()
        left_pwm.stop()
        right_pwm.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#     stdscr.clear()
#     stdscr.addstr("Use Arrow Keys to Control the Car\nPress 'q' to Quit\n")
#     stdscr.refresh()

#     curses.cbreak()  # Enable immediate key response
#     stdscr.keypad(True)  # Enable keypad mode
#     # stdscr.nodelay(True) # No delay

#     try:
#         while True:
#             key = stdscr.getch()  # Read a keypress
#             # print(key)
#             if key == curses.KEY_UP:
#                 stdscr.addstr(2, 0, "Moving Forward  ")
#                 move_forward()
#             elif key == curses.KEY_DOWN:
#                 stdscr.addstr(2, 0, "Moving Backward ")
#                 move_backward()
#             elif key == curses.KEY_LEFT:
#                 stdscr.addstr(2, 0, "Turning Left    ")
#                 turn_left()
#             elif key == curses.KEY_RIGHT:
#                 stdscr.addstr(2, 0, "Turning Right   ")
#                 turn_right()
#             elif key == ord('q'):
#                 break  # Exit on 'q' press
#             else:
#                 stop()  # Stop if no key is pressed

#             curses.flushinp()
#             stdscr.refresh()
#     except KeyboardInterrupt:
#         pass  # Allow clean exit

#     finally:
#         stop()
#         left_pwm.stop()
#         right_pwm.stop()
#         GPIO.cleanup()
#         curses.nocbreak()
#         stdscr.keypad(False)
#         curses.echo()
#         curses.endwin()

# curses.wrapper(main)
