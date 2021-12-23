# import the necessary packages
import sys
import argparse
import numpy as np
import cv2

from subtract import BackgroundSubtractor
from contours import CountoursDetector
from utils import *

argparse = argparse.ArgumentParser()
argparse.add_argument("-i", "--input", help="Input video", required=True)
argparse.add_argument(
    "-b", "--background", help="Background of the current video", required=True
)
argparse.add_argument(
    "-s", "--show", help="Show the result in a window", action="store_true"
)
args = argparse.parse_args()

# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(args.input)

# the output will be written to output.avi
# out = cv2.VideoWriter("./output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (640, 480))

subtractor = BackgroundSubtractor()
contours_detector = CountoursDetector(50, 255, cv2.THRESH_BINARY)

background = cv2.imread(args.background)
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
canvas_background, contours_background = contours_detector.work(background)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    segmented = subtractor.work(gray)
    canvas_segmented, contours_segmented = contours_detector.work(
        segmented, mode=cv2.RETR_LIST
    )

    diff = cv2.subtract(canvas_segmented, canvas_background)
    diff_contours = np.copy(contours_segmented)

    # remove common contours
    for contour in contours_background:
        if np.any(np.isin(contour, diff_contours)):
            np.delete(diff_contours, contour)

    draw_bounding_boxes(frame, diff_contours, (0, 255, 0), 200)

    cv2.imshow("frame", frame)
    # cv2.resizeWindow("frame", 1920, 720)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
# and release the output
# out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
