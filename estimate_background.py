import sys
import argparse
import numpy as np
import cv2
from skimage import data, filters

argparse = argparse.ArgumentParser()
argparse.add_argument("-i", "--input", help="Input video", required=True)
argparse.add_argument("-o", "--output", help="Output path (image)", required=True)
argparse.add_argument("-s", "--show", help="Show the result in a window", action="store_true")
args = argparse.parse_args()

# Open Video
cap = cv2.VideoCapture(args.input)

# Randomly select 25 frames
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)

# Store selected frames in an array
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)

# Calculate the median along the time axis
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    

cv2.imwrite(args.output, medianFrame)

if args.show:
    cv2.imshow('frame', medianFrame)
    cv2.waitKey(0)
