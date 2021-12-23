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
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    # original_frame = cv2.resize(original_frame, (640, 480))
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # segmentation via classical MOG2
    segment = subtractor.work(gray)

    # compute difference between foreground and background contours
    canvas_gray, contours_gray = contours_detector.work(gray)
    diff = cv2.subtract(canvas_gray, canvas_background)

    diff_contours = np.delete(
        contours_gray, np.argwhere(np.isin(contours_gray, contours_background))
    )

    # draw bounding boxes around previously found contours
    draw_bounding_boxes(frame, diff_contours, (0, 255, 0), 200, len(frame) ** 2 / 2)

    # draw bounding boxes from MOG2
    _, segment_contours = contours_detector.work(segment)
    draw_bounding_boxes(frame, segment_contours, (0, 0, 255), 200, len(frame) ** 2 / 2)

    # returns the bounding boxes for the detected objects
    # boxes, weights = hog.detectMultiScale(original_frame, winStride=(8, 8))

    # boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    # for (xA, yA, xB, yB) in boxes:
    #     # display the detected boxes in the colour picture
    #     cv2.rectangle(original_frame, (xA, yA), (xB, yB), (0, 255, 0), 2)

    # Write the output video
    # out.write(original_frame.astype("uint8"))
    # Display the resulting frame
    cv2.imshow("frame", frame)
    cv2.resizeWindow("frame", 1920, 720)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release the capture
cap.release()
# and release the output
# out.release()
# finally, close the window
cv2.destroyAllWindows()
cv2.waitKey(1)
