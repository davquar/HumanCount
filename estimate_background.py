import argparse
import numpy as np
import cv2

argparse = argparse.ArgumentParser()
argparse.add_argument("-i", "--input", help="Input video", required=True)
argparse.add_argument("-o", "--output", help="Output path (image)", required=True)
argparse.add_argument(
    "-s", "--show", help="Show the result in a window", action="store_true"
)
args = argparse.parse_args()

cap = cv2.VideoCapture(args.input)

# The median frame computation is based on 100 random frames
frame_ids = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)
frames = []
for frame_id in frame_ids:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    _, frame = cap.read()
    frames.append(frame)

median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)

cv2.imwrite(args.output, median_frame)

if args.show:
    cv2.imshow("frame", median_frame)
    cv2.waitKey(0)
