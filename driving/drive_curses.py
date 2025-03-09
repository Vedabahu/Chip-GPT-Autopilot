import RPi.GPIO as GPIO
import curses
# from pynput import keyboard
from time import sleep

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

# def on_press(key):
#     print(key)
#     try:
#         if key == keyboard.Key.up:
#             print("Up")
#             move_forward()
#         elif key == keyboard.Key.down:
#             print("Up")
#             move_backward()
#         elif key == keyboard.Key.left:
#             print("Left")
#             turn_left()
#         elif key == keyboard.Key.right:
#             print("Right")
#             turn_right()
#     except AttributeError:
#         pass

# def on_release(key):
#     if key in {keyboard.Key.up, keyboard.Key.down, keyboard.Key.left, keyboard.Key.right}:
#         stop()
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False

# if __name__ == "__main__":
#     print("Use Arrow Keys to Control the Car\nPress 'Esc' to Quit\n")
#     # Collect events until released
#     with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#         listener.join()

#     # Cleanup GPIO settings
#     GPIO.cleanup()

def main(stdscr):
    stdscr.clear()
    stdscr.addstr("Use Arrow Keys to Control the Car\nPress 'q' to Quit\n")
    stdscr.refresh()

    curses.cbreak()  # Enable immediate key response
    stdscr.keypad(True)  # Enable keypad mode
    # stdscr.nodelay(True) # No delay

    try:
        while True:
            key = stdscr.getch()  # Read a keypress
            # print(key)
            if key == curses.KEY_UP:
                stdscr.addstr(2, 0, "Moving Forward  ")
                move_forward()
            elif key == curses.KEY_DOWN:
                stdscr.addstr(2, 0, "Moving Backward ")
                move_backward()
            elif key == curses.KEY_LEFT:
                stdscr.addstr(2, 0, "Turning Left    ")
                turn_left()
            elif key == curses.KEY_RIGHT:
                stdscr.addstr(2, 0, "Turning Right   ")
                turn_right()
            elif key == ord('q'):
                break  # Exit on 'q' press
            else:
                stop()  # Stop if no key is pressed

            curses.flushinp()
            stdscr.refresh()
    except KeyboardInterrupt:
        pass  # Allow clean exit

    finally:
        stop()
        left_pwm.stop()
        right_pwm.stop()
        GPIO.cleanup()
        curses.nocbreak()
        stdscr.keypad(False)
        curses.echo()
        curses.endwin()

curses.wrapper(main)
