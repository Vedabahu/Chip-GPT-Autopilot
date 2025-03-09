from picamera2.encoders import H264Encoder
from picamera2 import Picamera2, Preview
from libcamera import Transform
import time

picam2 = Picamera2()
# video_config = picam2.create_video_configuration()
video_config = picam2.create_preview_configuration(main={"size": (640, 480)}, transform=Transform(vflip=True))
picam2.configure(video_config)
# picam2.start_preview(Preview.QT)

encoder = H264Encoder()
output = "testing_4.mp4"

picam2.start_recording(encoder, output)
time.sleep(10)
# picam2.stop_preview()
picam2.stop_recording()

# picam2.start_and_record_video("testing.mp4", duration=20)