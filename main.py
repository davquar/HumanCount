# import the necessary packages
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

cap = cv2.VideoCapture(args.input)

subtractor = BackgroundSubtractor()
contours_detector = CountoursDetector(50, 255, cv2.THRESH_BINARY)

background = cv2.imread(args.background)
background = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
background = cv2.resize(background, (350, 300))
canvas_background, contours_background = contours_detector.work(background)


def do_hog_svm():
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))
    draw_hog_bounding_boxes(frame, boxes, (255, 0, 0))


def do_mog():
    segmented = cv2.absdiff(gray, background)
    canvas_segmented, contours_segmented = contours_detector.work(
        segmented, mode=cv2.RETR_EXTERNAL, remove_shadows=True
    )

    diff = cv2.subtract(canvas_segmented, canvas_background)
    diff_contours = np.copy(contours_segmented)

    # remove common contours
    for contour in contours_background:
        if np.any(np.isin(contour, diff_contours)):
            try:
                np.delete(diff_contours, contour)
            except IndexError as error:
                print(error)

    draw_bounding_boxes(frame, diff_contours, (0, 255, 0), 200)


while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (350, 300))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    do_hog_svm()
    do_mog()

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
cv2.waitKey(1)
