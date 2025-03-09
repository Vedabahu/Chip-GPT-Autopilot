# Hackathon : Hack Maze

> Topic : Autopilot

Chip GPT team code for the raspberry pi to run an autonomous car.

## Prerequsites

> Most of it is run on a Raspberry Pi 3 B+. Some codes can be run locally without a RPi.

1. Raspberry Pi Model 3 B+
1. Camera module
1. Chasis
1. Motors
1. H Brige

## Directory structure

- `auto_drivin` : for the code related to automatic driving
    - `auto_driving/optimized-lane-detect-drive.py` : final working code for autonomous driving
- `driving` : code related to drive the car manually using the arrow keys.
    - `driving/drive_curses.py` : final good working code
- `record` : code related to recording the video using RPi.
    - `record/record_native_library.py` : final working code to get videos
- `datasets` : consists of the datasets used to optimize the threshold in the code.