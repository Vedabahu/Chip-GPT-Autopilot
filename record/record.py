import cv2
from picamera2 import Picamera2
import time

def record_video(filename, duration, resolution=(640, 480), framerate=20.0):
    # Initialize Picamera2
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": resolution})
    picam2.configure(video_config)
    picam2.start()

    # Allow the camera to warm up
    time.sleep(2)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    out = cv2.VideoWriter(filename, fourcc, framerate, resolution)

    start_time = time.time()
    while (time.time() - start_time) < duration:
        frame = picam2.capture_array()
        out.write(frame)
        cv2.imshow('Recording...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release everything if job is finished
    out.release()
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    filename = 'track_with_obstacles.mp4'  # Output video file
    duration = 20  # Duration in seconds
    record_video(filename, duration)
